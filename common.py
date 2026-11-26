"""Shared helpers for the PDB ligand survey pipeline."""
from __future__ import annotations

import gzip
import time
from pathlib import Path

import gemmi
import numpy as np
import requests
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Lipinski

RDLogger.DisableLog("rdApp.*")

import os

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SURVEY_DATA", ROOT / "data"))
CIF_DIR = DATA / "cif"
CCD_DIR = DATA / "ccd"
PARTS = DATA / "parts"
for _d in (DATA, CIF_DIR, CCD_DIR, PARTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- thresholds
MAX_RES = 5.0
MW_MIN, MW_MAX = 200.0, 2000.0
RES_CLASSES = [(0.0, 2.0, "high (<2)"), (2.0, 3.5, "medium (2-3.5)"), (3.5, 5.01, "low (3.5-5)")]


def res_class(res: float) -> str:
    for lo, hi, name in RES_CLASSES:
        if lo <= res < hi:
            return name
    return "NA"


# ---------------------------------------------------------------- http
_session = requests.Session()


def http_get(url: str, tries: int = 4, timeout: int = 60) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
            if r.status_code == 404:
                break
        except requests.RequestException as e:  # noqa: PERF203
            last = e
        time.sleep(1.5 * (i + 1))
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------- CCD -> RDKit
_BOND = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "QUAD": Chem.BondType.QUADRUPLE,
}


def ccd_path(comp_id: str) -> Path:
    return CCD_DIR / f"{comp_id}.cif"


def fetch_ccd(comp_id: str) -> Path:
    p = ccd_path(comp_id)
    if not p.exists():
        r = http_get(f"https://files.rcsb.org/ligands/download/{comp_id}.cif")
        p.write_bytes(r.content)
    return p


def ccd_mol(comp_id: str):
    """Heavy-atom RDKit mol from the CCD with CCD atom names in prop '_name'.

    Bond orders and formal charges come from the CCD (Kekule); hydrogens are left
    implicit because X-ray models normally lack them.  Returns None on failure.
    """
    doc = gemmi.cif.read(str(fetch_ccd(comp_id)))
    block = doc.sole_block() if len(doc) == 1 else doc[0]
    atoms = block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "charge"])
    bonds = block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"])
    if not atoms:
        return None
    rw = Chem.RWMol()
    idx: dict[str, int] = {}
    for row in atoms:
        el = row[1].strip('"').capitalize()
        if el in ("H", "D"):
            continue
        a = Chem.Atom(el)
        try:
            a.SetFormalCharge(int(float(row[2])))
        except ValueError:
            pass
        i = rw.AddAtom(a)
        rw.GetAtomWithIdx(i).SetProp("_name", row[0].strip('"'))
        idx[row[0].strip('"')] = i
    for row in bonds:
        a1, a2 = row[0].strip('"'), row[1].strip('"')
        if a1 in idx and a2 in idx:
            rw.AddBond(idx[a1], idx[a2], _BOND.get(row[2], Chem.BondType.SINGLE))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:  # partial sanitisation (metal complexes, odd valences)
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,
                Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE,
                catchErrors=True,
            )
        except Exception:
            return None
    return mol


def rotatable_bonds(mol) -> list[tuple[int, int]]:
    return [(a, b) for a, b in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)]


# ---------------------------------------------------------------- SASA
_VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47,
        "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90, "B": 1.92, "SI": 2.10}


def vdw(symbols) -> np.ndarray:
    return np.array([_VDW.get(s.upper(), 1.80) for s in symbols])


def _sphere_points(n: int = 400) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


_SPH = _sphere_points()


def sasa_atoms(xyz: np.ndarray, radii: np.ndarray, targets: np.ndarray, probe: float = 1.4) -> np.ndarray:
    """Shrake-Rupley SASA (A^2) for atom indices ``targets`` among all atoms in ``xyz``."""
    from scipy.spatial import cKDTree

    r = radii + probe
    tree = cKDTree(xyz)
    out = np.zeros(len(targets))
    rmax = r.max()
    for k, i in enumerate(targets):
        pts = xyz[i] + _SPH * r[i]
        nbr = [j for j in tree.query_ball_point(xyz[i], r[i] + rmax) if j != i]
        if nbr:
            d = np.linalg.norm(pts[:, None, :] - xyz[nbr][None, :, :], axis=2)
            free = (d >= r[nbr][None, :]).all(axis=1)
        else:
            free = np.ones(len(pts), bool)
        out[k] = 4 * np.pi * r[i] ** 2 * free.mean()
    return out


def ligand_xyz(model, asym_id: str, mol):
    """Heavy-atom coordinates of one ligand instance ordered like the CCD template ``mol``.

    Same altloc rule as 03_instances.py (lowest altloc per residue).  None if atoms are missing.
    """
    name2idx = {a.GetProp("_name"): a.GetIdx() for a in mol.GetAtoms()}
    xyz = np.full((mol.GetNumAtoms(), 3), np.nan)
    for ch in model:
        for res in ch:
            if res.subchain != asym_id:
                continue
            alts = sorted({a.altloc for a in res if a.altloc != "\0"})
            keep = alts[0] if alts else "\0"
            for at in res:
                if at.altloc not in ("\0", keep):
                    continue
                j = name2idx.get(at.name)
                if j is not None and np.isnan(xyz[j, 0]):
                    xyz[j] = at.pos.tolist()
    return None if np.isnan(xyz).any() else xyz


# ---------------------------------------------------------------- mmCIF
def cif_path(pdb_id: str) -> Path:
    return CIF_DIR / f"{pdb_id.upper()}.cif"


def fetch_cif(pdb_id: str) -> Path:
    p = cif_path(pdb_id)
    if not p.exists():
        r = http_get(f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz", timeout=120)
        p.write_bytes(gzip.decompress(r.content))
    return p
