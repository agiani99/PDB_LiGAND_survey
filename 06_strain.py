"""Step 2 - xtb strain energy of the deposited ligand pose.

For every selected ligand instance:
  E_pose   GFN2-xTB/ALPB(water): all heavy-atom torsions (rotatable, ring, amide...) held at their X-ray
           values with a harmonic constraint; bond lengths, angles and hydrogens relax
  E_relax  full unconstrained xTB relaxation started from there (nearest local minimum)
  E_ref    lowest xTB energy of (ETKDG conformer ensemble -> MMFF prefilter -> xTB optimisation) and E_relax
           (the ensemble is cached per ligand = isomeric SMILES + charge, so it is computed once)
  delta_e_kcal      = E_pose - E_ref     (strain vs. the lowest-energy minimised 3D structure)
  local_strain_kcal = E_pose - E_relax   (strain vs. the nearest local minimum)

Caveat: coordinate error in low-resolution models still inflates the torsion-constrained energy; interpret
strain per resolution class (the app can colour/filter by it).  (A first version froze all heavy atoms:
that froze X-ray bond-length errors too and gave ~16-20 kcal/mol even for 1.0 A ligands.)

Output: data/strain.parquet (resumable via data/strain_parts/).
Usage (pilot first!):
    python 06_strain.py --sample 200 --workers 8
    python 06_strain.py --scope unique            # one best-resolution instance per ligand id
    python 06_strain.py --scope all
Env: XTB_EXE to override the xtb executable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from common import DATA, ccd_mol, ccd_path, ligand_xyz, fetch_cif, cif_path

STRAIN_PARTS = DATA / "strain_parts"
REF_DIR = DATA / "strain_ref"
TMP = DATA / "tmp"
for _d in (STRAIN_PARTS, REF_DIR, TMP):
    _d.mkdir(parents=True, exist_ok=True)

EH2KCAL = 627.5095
XTB = (os.environ.get("XTB_EXE") or shutil.which("xtb")
       or r"C:\Users\andrea.DESKTOP-26V6UN4\xtb-bleed-windows\bin\xtb.exe")
SETTINGS = dict(method="gfn2", solvent="water")  # part of the reference cache key
_E_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
Z = Chem.GetPeriodicTable()


# ------------------------------------------------------------------ xtb wrapper
def write_xyz(path: Path, mol, conf_id: int = 0, xyz: np.ndarray | None = None):
    c = mol.GetConformer(conf_id)
    pos = xyz if xyz is not None else np.array([list(c.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    lines = [str(mol.GetNumAtoms()), "ligand"]
    lines += [f"{a.GetSymbol():2s} {pos[i,0]:12.6f} {pos[i,1]:12.6f} {pos[i,2]:12.6f}" for i, a in enumerate(mol.GetAtoms())]
    path.write_text("\n".join(lines) + "\n")


def read_xyz(path: Path) -> np.ndarray:
    L = path.read_text().splitlines()
    return np.array([[float(v) for v in ln.split()[1:4]] for ln in L[2:2 + int(L[0])]])


TORSION_FC = 0.5  # xtb $constrain force constant; holds torsions within ~2 deg of the X-ray values


def torsion_constraints(mh, conf_id: int = 0):
    """One heavy-atom dihedral (a,b,c,d,deg) per non-terminal bond b-c: pins the X-ray conformation
    (rotatable bonds, ring torsions, amide/double-bond planarity) while bonds and angles relax.
    Freezing all heavy atoms instead freezes X-ray bond-length errors and gives ~10-20 kcal/mol noise."""
    from rdkit.Chem import rdMolTransforms as T
    conf = mh.GetConformer(conf_id)
    out = []
    for b in mh.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        A, B = mh.GetAtomWithIdx(i), mh.GetAtomWithIdx(j)
        if A.GetAtomicNum() == 1 or B.GetAtomicNum() == 1:
            continue
        na = [n.GetIdx() for n in A.GetNeighbors() if n.GetIdx() != j and n.GetAtomicNum() > 1]
        nb = [n.GetIdx() for n in B.GetNeighbors() if n.GetIdx() != i and n.GetAtomicNum() > 1]
        if not na or not nb:
            continue
        a, d = na[0], nb[0]
        if T.GetAngleDeg(conf, a, i, j) > 175 or T.GetAngleDeg(conf, i, j, d) > 175:  # linear: undefined
            continue
        out.append((a + 1, i + 1, j + 1, d + 1, T.GetDihedralDeg(conf, a, i, j, d)))
    return out


def run_xtb(workdir: Path, mol, charge: int, *, conf_id=0, xyz=None, opt: bool = True,
            torsions: list | None = None, timeout: int = 600):
    """Return (energy_Eh, optimised_xyz_or_None); (None, None) on failure.
    torsions = [(a,b,c,d,deg)] 1-based atom indices, held with a harmonic constraint."""
    workdir.mkdir(parents=True, exist_ok=True)
    write_xyz(workdir / "in.xyz", mol, conf_id, xyz)
    cmd = [XTB, "in.xyz", "--gfn", "2", "--chrg", str(charge), "--alpb", SETTINGS["solvent"],
           "--parallel", "1", "--norestart"]
    if opt:
        cmd += ["--opt", "normal"]
    if torsions:
        lines = ["$constrain", f" force constant={TORSION_FC}"]
        lines += [" dihedral: %d,%d,%d,%d,%.4f" % t for t in torsions] + ["$end"]
        (workdir / "c.inp").write_text("\n".join(lines) + "\n")
        cmd += ["--input", "c.inp"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, None
    out = r.stdout
    if r.returncode != 0 or "normal termination" not in (out + r.stderr):
        return None, None
    if opt and "FAILED TO CONVERGE" in out:
        return None, None
    e = _E_RE.findall(out)
    if not e:
        return None, None
    return float(e[-1]), (read_xyz(workdir / "xtbopt.xyz") if opt and (workdir / "xtbopt.xyz").exists() else None)


# ------------------------------------------------------------------ molecule prep
# Rule-based protonation state at pH ~7.  CCD ideal ligands are neutral (COOH, PO3H2, amidines...), but in the
# protein these are mostly charged; a neutral acid/base folds into an internally H-bonded conformer in the xtb
# ensemble and makes the extended bound pose look very strained.
_ACID_SMARTS = ["[CX3](=O)[OX2H1]",       # carboxylic acid (last atom = OH oxygen)
                "[SX4](=O)(=O)[OX2H1]"]    # sulfonic acid
_P_OH = Chem.MolFromSmarts("[PX4](=O)[OX2H1]")
_AMIDINE = Chem.MolFromSmarts("[NX2;H1,H2;!a]=[CX3;!a]([#6,#1,N])[NX3;!$(NC=O);!$(NS=O)]")
_AMINE = Chem.MolFromSmarts("[NX3;H0,H1,H2;!a;!$(N~[!#6]);!$(N-[#6;!X4]);!$(N-[CX4]-[F,Cl]);"
                            "!$(N1CCOCC1);!$(N-[CX4]-C(F)(F)F)]")


def ionize(tmpl, amines: bool = False):
    """Return a copy of the heavy-atom template with pH 7 charges (acids -> anions, amidines/amines -> cations)."""
    m = Chem.RWMol(tmpl)
    m.UpdatePropertyCache(strict=False)
    for smarts in _ACID_SMARTS:
        for mt in m.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            at = m.GetAtomWithIdx(mt[-1])
            if at.GetFormalCharge() == 0:
                at.SetFormalCharge(-1)
    p_oh: dict[int, list[int]] = {}
    for mt in m.GetSubstructMatches(_P_OH):
        p_oh.setdefault(mt[0], []).append(mt[2])
    for oxy in p_oh.values():  # phosphate / phosphonate: 1 OH -> mono-anion, >=2 OH -> di-anion
        for k in oxy[:2]:
            m.GetAtomWithIdx(k).SetFormalCharge(-1)
    for mt in m.GetSubstructMatches(_AMIDINE):  # amidine / guanidine cation
        if m.GetAtomWithIdx(mt[0]).GetFormalCharge() == 0:
            m.GetAtomWithIdx(mt[0]).SetFormalCharge(1)
    for mt in (m.GetSubstructMatches(_AMINE) if amines else []):  # aliphatic amine cation (opt-in)
        at = m.GetAtomWithIdx(mt[0])
        if at.GetFormalCharge() == 0 and not any(n.GetFormalCharge() > 0 for n in at.GetNeighbors()):
            at.SetFormalCharge(1)
    for at in m.GetAtoms():  # only the atoms we charged get their H count recomputed ([nH] keeps its H)
        if at.GetFormalCharge() != 0 and not at.GetIsAromatic():
            at.SetNoImplicit(False)
            at.SetNumExplicitHs(0)
    mol = m.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def build_pose_mol(template, xyz: np.ndarray):
    """Heavy-atom template + X-ray coordinates -> (mol with H, formal charge). Stereo from 3D."""
    m = Chem.Mol(template)
    for a in m.GetAtoms():
        a.ClearProp("_name")  # keep canonicalisation independent of atom names
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, p.tolist())
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    mh = Chem.AddHs(m, addCoords=True)
    return mh, Chem.GetFormalCharge(m)


def ligand_key(mh, charge: int) -> str:
    smi = Chem.MolToSmiles(Chem.RemoveHs(mh))
    return hashlib.sha1(f"{smi}|{charge}|{SETTINGS}".encode()).hexdigest()[:20]


# ------------------------------------------------------------------ reference ensemble (cached)
def ensemble_reference(mh, charge: int, key: str, a, wd_root: Path):
    """Lowest xTB energy over an ETKDG ensemble; cached on disk (REF_DIR/<key>.json)."""
    cache = REF_DIR / f"{key}.json"
    lock = REF_DIR / f"{key}.lock"
    if cache.exists():
        return json.loads(cache.read_text())
    try:  # another worker is already computing this ligand -> wait for it
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        t0 = time.time()
        while time.time() - t0 < 1800 and not cache.exists():
            time.sleep(5)
        if cache.exists():
            return json.loads(cache.read_text())
    try:
        res = _compute_ensemble(mh, charge, a, wd_root)
        res["key"] = key
        tmp = cache.with_suffix(".tmp%d" % os.getpid())
        tmp.write_text(json.dumps(res))
        os.replace(tmp, cache)
        return res
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _compute_ensemble(mh, charge: int, a, wd_root: Path):
    m = Chem.Mol(mh)
    m.RemoveAllConformers()
    n_rot = Chem.rdMolDescriptors.CalcNumRotatableBonds(m)
    n_conf = int(min(a.max_confs, 20 + 10 * n_rot))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    ps.pruneRmsThresh = 0.3
    ps.numThreads = 1
    ids = list(AllChem.EmbedMultipleConfs(m, n_conf, ps))
    if not ids:
        ps.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(m, n_conf, ps))
    if not ids:
        return dict(e_ref_eh=None, n_conf=0, n_opt=0)
    try:
        r = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=2000)
        if any(x[0] == -1 for x in r) and not AllChem.MMFFHasAllMoleculeParams(m):
            raise ValueError
    except Exception:
        r = AllChem.UFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=2000)
    order = sorted(range(len(ids)), key=lambda i: r[i][1])
    # keep the lowest-MMFF conformers that differ by > 0.5 A heavy-atom RMSD
    heavy = Chem.RemoveHs(m)
    keep: list[int] = []
    for i in order:
        if all(AllChem.GetConformerRMS(heavy, ids[i], ids[j], prealigned=False) > 0.5 for j in keep):
            keep.append(i)
        if len(keep) >= a.n_opt:
            break
    es = []
    wd = Path(tempfile.mkdtemp(dir=wd_root))
    try:
        for i in keep:
            e, _ = run_xtb(wd, m, charge, conf_id=ids[i])
            if e is not None:
                es.append(e)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    return dict(e_ref_eh=min(es) if es else None, n_conf=len(ids), n_opt=len(es))


# ------------------------------------------------------------------ geometry QC of the deposited pose
QC_MAX_BOND_DEV = 0.08   # A, X-ray bond length vs. the xtb geometry of the same pose
QC_MAX_ANGLE_DEV = 10.0  # deg
QC_MAX_N_BOND = 3        # allowed number of deviating bonds
QC_MAX_RMSD = 0.5        # A heavy-atom RMSD X-ray vs. torsion-constrained xtb optimum
QC_MIN_CONTACT = 2.0     # A closest approach to any other (non-ligand) atom
_POLAR = ("N", "O", "S", "F", "Cl")


def _ccd_stereo(comp_id: str) -> dict:
    doc = gemmi.cif.read(str(ccd_path(comp_id)))
    blk = doc.sole_block() if len(doc) == 1 else doc[0]
    return {r[0].strip('"'): r[1] for r in blk.find("_chem_comp_atom.", ["atom_id", "pdbx_stereo_config"])}


def geometry_qc(model, row: dict, tmpl, xyz: np.ndarray, mh, xyz_opt: np.ndarray) -> dict:
    """Flags for coordinate errors that inflate strain (RSCC does not catch them).
    xyz = X-ray heavy-atom coordinates (template order); xyz_opt = xtb torsion-constrained optimum (all atoms)."""
    from rdkit.Chem import rdCIPLabeler
    n = tmpl.GetNumAtoms()
    names = [a.GetProp("_name") for a in tmpl.GetAtoms()]
    el = [a.GetSymbol() for a in tmpl.GetAtoms()]
    xo = xyz_opt[:n]
    qc = dict(qc_rmsd=float(np.sqrt(((xyz - xo) ** 2).sum(1).mean())))
    # bond lengths / angles of the X-ray pose vs. the relaxed geometry of the same conformation
    bd = [abs(np.linalg.norm(xyz[i] - xyz[j]) - np.linalg.norm(xo[i] - xo[j]))
          for i, j in ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mh.GetBonds()) if i < n and j < n]
    qc["qc_max_bond_dev"] = max(bd) if bd else 0.0
    qc["qc_n_bond_dev"] = int(sum(x > QC_MAX_BOND_DEV for x in bd))

    def _ang(P, i, j, k):
        u, v = P[i] - P[j], P[k] - P[j]
        return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    na = 0
    for a in mh.GetAtoms():
        i = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors() if x.GetIdx() < n]
        if i >= n:
            continue
        for p in range(len(nb)):
            for r in range(p + 1, len(nb)):
                na += abs(_ang(xyz, nb[p], i, nb[r]) - _ang(xo, nb[p], i, nb[r])) > QC_MAX_ANGLE_DEV
    qc["qc_n_angle_dev"] = int(na)
    # aromatic ring planarity of the X-ray pose
    pl = []
    for ring in mh.GetRingInfo().AtomRings():
        if max(ring) < n and all(mh.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            P = xyz[list(ring)] - xyz[list(ring)].mean(0)
            pl.append(float(np.abs(P @ np.linalg.svd(P)[2][2]).max()))
    qc["qc_arom_nonplanar"] = max(pl) if pl else 0.0
    # stereochemistry of the deposited coordinates vs. the CCD
    mm = Chem.Mol(mh)
    bad = 0
    try:
        rdCIPLabeler.AssignCIPLabels(mm)
        ccd = _ccd_stereo(row["comp_id"])
        for a in mm.GetAtoms():
            if a.GetIdx() < n and a.HasProp("_CIPCode"):
                c = ccd.get(names[a.GetIdx()], "N")
                bad += c in ("R", "S") and c != a.GetProp("_CIPCode")
    except Exception:  # noqa: BLE001
        pass
    qc["qc_stereo_mismatch"] = int(bad)
    # contacts with everything that is not this ligand
    oth = [(at.pos.tolist(), at.element.name) for ch in model for r in ch if r.subchain != row["asym_id"] for at in r]
    mind, clash = 9.0, 0
    if oth:
        O = np.array([o[0] for o in oth])
        oe = [o[1] for o in oth]
        for i in range(n):
            d = np.linalg.norm(O - xyz[i], axis=1)
            mind = min(mind, float(d.min()))
            for j in np.where(d < 3.0)[0]:
                clash += not (el[i] in _POLAR and oe[j] in _POLAR) and oe[j] != "H"
    qc["qc_min_contact"] = mind
    qc["qc_n_clash"] = int(clash)
    qc["qc_pass"] = bool(qc["qc_n_bond_dev"] < QC_MAX_N_BOND and qc["qc_rmsd"] <= QC_MAX_RMSD
                         and mind >= QC_MIN_CONTACT and qc["qc_stereo_mismatch"] == 0)
    return qc


# ------------------------------------------------------------------ one instance
def strain_instance(model, row: dict, a, wd_root: Path) -> dict:
    out = dict(pdb_id=row["pdb_id"], comp_id=row["comp_id"], asym_id=row["asym_id"], status="ok")
    t0 = time.time()
    tmpl = ccd_mol(row["comp_id"])
    if tmpl is None:
        return {**out, "status": "no_template"}
    tmpl0 = tmpl  # neutral CCD template (atom names) for the QC
    xyz = ligand_xyz(model, row["asym_id"], tmpl)
    if xyz is None:
        return {**out, "status": "incomplete"}
    if getattr(a, "ionize", False):
        out["charge_neutral"] = Chem.GetFormalCharge(tmpl)
        tmpl = ionize(tmpl, getattr(a, "ionize_amines", False))
    if tmpl.GetNumHeavyAtoms() > a.max_heavy:
        return {**out, "status": "too_large"}
    if any(at.GetAtomicNum() in (3, 4, 11, 12, 13, 19, 20) or at.GetAtomicNum() > 35 for at in tmpl.GetAtoms()):
        return {**out, "status": "unsupported_element"}
    mh, q = build_pose_mol(tmpl, xyz)
    n_el = sum(at.GetAtomicNum() for at in mh.GetAtoms()) - q
    if n_el % 2:
        return {**out, "status": "open_shell"}
    key = ligand_key(mh, q)
    out.update(key=key, charge=q, n_heavy=tmpl.GetNumHeavyAtoms())
    wd = Path(tempfile.mkdtemp(dir=wd_root))
    try:
        e_pose, xyz_h = run_xtb(wd, mh, q, torsions=torsion_constraints(mh))
        if e_pose is None:
            return {**out, "status": "xtb_pose_failed"}
        try:
            qc = geometry_qc(model, row, tmpl0, xyz, mh, xyz_h)
        except Exception:  # noqa: BLE001
            qc = {}
        e_relax, _ = run_xtb(wd, mh, q, xyz=xyz_h)
        ref = ensemble_reference(mh, q, key, a, wd_root)
        cands = [e for e in (ref.get("e_ref_eh"), e_relax) if e is not None]
        if not cands:
            return {**out, "status": "no_reference"}
        e_ref = min(cands)
        out.update(e_pose_eh=e_pose, e_relax_eh=e_relax, e_ref_eh=e_ref,
                   delta_e_kcal=(e_pose - e_ref) * EH2KCAL,
                   local_strain_kcal=(e_pose - e_relax) * EH2KCAL if e_relax is not None else np.nan,
                   n_conf=ref.get("n_conf"), n_opt=ref.get("n_opt"), t_sec=round(time.time() - t0, 1), **qc)
        return out
    except Exception as e:  # noqa: BLE001
        return {**out, "status": f"error:{str(e)[:80]}"}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def run_chunk(chunk, a):
    res = []
    for pid, rows in chunk:
        try:
            st = gemmi.read_structure(str(fetch_cif(pid)))
            st.setup_entities()
            model = st[0]
            for r in rows:
                res.append(strain_instance(model, r, a, TMP))
        except Exception as e:  # noqa: BLE001
            res += [dict(pdb_id=pid, comp_id=r["comp_id"], asym_id=r["asym_id"], status=f"entry_error:{str(e)[:60]}")
                    for r in rows]
        finally:
            if not a.keep_cif:
                cif_path(pid).unlink(missing_ok=True)
    return res


# ------------------------------------------------------------------ main
def druglike_mask(df: pd.DataFrame):
    """Drug-like window (Lipinski/Veber-style, slightly relaxed): excludes lipids, sugars,
    nucleotides, heme-like cofactors, peptides and other large/polar/flexible ligands."""
    return (df.mw.between(200, 700) & df.logp.between(-3, 6) & (df.hbd <= 6) & (df.hba <= 12)
            & (df.tpsa <= 200) & (df.n_rotb <= 15) & (df.n_rings >= 1))


def select(a) -> pd.DataFrame:
    df = pd.read_parquet(DATA / "survey.parquet")
    df = df[(df.status == "ok") & ~df.covalent.fillna(False) & ~df.has_metal.fillna(False)
            & ~df.metal_contact.fillna(False) & (df.heavy_atoms <= a.max_heavy)
            & (df.buried_frac >= a.min_buried) & (df.get('n_missing', 0) == 0)]
    if a.druglike:
        df = df[druglike_mask(df)]
    if a.scope == "unique":
        df = df.sort_values(["resolution", "rscc"], ascending=[True, False]).drop_duplicates("comp_id")
    if a.sample:
        df = df.sample(min(a.sample, len(df)), random_state=0)
    return df[["pdb_id", "comp_id", "asym_id"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["unique", "all"], default="unique")
    ap.add_argument("--sample", type=int, default=0, help="random subset of N instances (pilot)")
    ap.add_argument("--max-heavy", type=int, default=60)
    ap.add_argument("--min-buried", type=float, default=0.3, help="drop ligands with buried_frac below this")
    ap.add_argument("--druglike", action=argparse.BooleanOptionalAction, default=True,
                    help="restrict to drug-like ligands (MW 200-700, logP -3..6, HBD<=6, HBA<=12, TPSA<=200, "
                         "rotb<=15, >=1 ring); --no-druglike disables")
    ap.add_argument("--ionize", action=argparse.BooleanOptionalAction, default=True,
                    help="assign pH 7 charges (carboxylate, phosphate, sulfonate, amidine, aliphatic amine) before "
                         "xtb (amidines too); --no-ionize reproduces the neutral-CCD runs")
    ap.add_argument("--ionize-amines", action="store_true",
                    help="also protonate aliphatic amines (off by default: raised dE for most outliers in a test)")
    ap.add_argument("--max-confs", type=int, default=60)
    ap.add_argument("--n-opt", type=int, default=8, help="conformers optimised with xTB per ligand")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--chunk", type=int, default=5, help="entries per task")
    ap.add_argument("--keep-cif", action="store_true")
    a = ap.parse_args()
    if not Path(XTB).exists() and not shutil.which(XTB):
        raise SystemExit(f"xtb not found: {XTB} (set XTB_EXE)")

    sel = select(a)
    # chunk names depend on the selection -> keep parts of different settings apart (resume = same settings)
    parts_dir = STRAIN_PARTS / f"{a.scope}_n{a.sample}_h{a.max_heavy}_b{a.min_buried}_d{int(a.druglike)}_i{int(a.ionize)}{int(a.ionize_amines)}_o{a.n_opt}_c{a.max_confs}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in parts_dir.glob("*.parquet")}
    groups = [(pid, g.to_dict("records")) for pid, g in sel.groupby("pdb_id")]
    chunks = [(f"s{i // a.chunk:05d}", groups[i:i + a.chunk]) for i in range(0, len(groups), a.chunk)]
    chunks = [c for c in chunks if c[0] not in done]
    n_inst = sum(len(g[1]) for c in chunks for g in c[1])
    print(f"{len(sel)} instances selected ({a.scope}); {n_inst} still to run in {len(chunks)} chunks, "
          f"{a.workers} workers, xtb={XTB}")
    try:
        from tqdm import tqdm
    except ImportError:
        class tqdm:  # noqa: N801
            def __init__(self, **kw):
                self.n, self.total = 0, kw.get("total")

            def update(self, k=1):
                self.n += k
                print(f"  {self.n}/{self.total} instances", flush=True)

            def __enter__(self):
                return self

            def __exit__(self, *e):
                pass

    with ProcessPoolExecutor(a.workers) as ex, tqdm(total=n_inst, desc="strain", unit="lig") as bar:
        futs = {ex.submit(run_chunk, c[1], a): (c[0], sum(len(g[1]) for g in c[1])) for c in chunks}
        for f in as_completed(futs):
            name, n = futs[f]
            pd.DataFrame(f.result()).to_parquet(parts_dir / f"{name}.parquet", index=False)
            bar.update(n)

    parts = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet"))], ignore_index=True)
    parts.to_parquet(DATA / "strain.parquet", index=False)
    print(parts.status.value_counts().to_string())
    ok = parts[parts.status == "ok"]
    if len(ok):
        print(ok[["delta_e_kcal", "local_strain_kcal", "t_sec"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
