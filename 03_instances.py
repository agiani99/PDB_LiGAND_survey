"""Step 1c - 3D features for every ligand instance from the deposited mmCIF.

For each instance:
  * match model atoms to the CCD template by atom name (heavy atoms only)
  * burial: Shrake-Rupley SASA of the ligand alone vs. in the complex
        buried_frac        1 - SASA_complex / SASA_free  (ligand heavy atoms)
        n_buried_atoms     atoms with per-atom burial >= 0.5 (zero-free-SASA atoms count as buried)
  * rotatable bonds: n_rotb_buried = rotatable bonds whose two atoms are both buried
  * contacts: protein residues within 4 A, main contacting chain (used for family assignment)
  * flags: incomplete (template atoms missing), covalent (heavy-atom contact < 2.0 A, non-metal)

Resumable: results are written per chunk in data/parts/.
Usage: python 03_instances.py [--workers 10] [--chunk 200] [--keep-cif] [--limit N]
"""
import argparse
import os
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import DATA, PARTS, ccd_mol, cif_path, fetch_cif, rotatable_bonds, sasa_atoms, vdw

_TEMPLATES: dict = {}
SHELL = 12.0  # A around the ligand used as occluders


def template(comp_id):
    if comp_id not in _TEMPLATES:
        m = ccd_mol(comp_id)
        _TEMPLATES[comp_id] = None if m is None else (
            m, {a.GetProp("_name"): a.GetIdx() for a in m.GetAtoms()}, rotatable_bonds(m))
    return _TEMPLATES[comp_id]


def process_entry(pdb_id: str, rows: list[dict], keep_cif: bool, max_missing: float = 0.10):
    out = []
    try:
        path = fetch_cif(pdb_id)
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        model = st[0]
        # all non-water atoms with their residue context
        coords, syms, keys, is_het, chain_of = [], [], [], [], []
        for ch in model:
            for res in ch:
                if res.is_water():
                    continue
                het = res.het_flag == "H"
                alts = sorted({a.altloc for a in res if a.altloc != "\0"})
                keep_alt = alts[0] if alts else "\0"
                for at in res:
                    if at.altloc not in ("\0", keep_alt):
                        continue
                    coords.append(at.pos.tolist())
                    syms.append(at.element.name)
                    keys.append((res.subchain, ch.name, str(res.seqid.num), res.name, at.name))
                    is_het.append(het)
                    chain_of.append(ch.name)
        coords = np.array(coords)
        syms = np.array(syms)
        asym_arr = np.array([k[0] for k in keys])
        name_arr = np.array([k[4] for k in keys])
        het_arr = np.array(is_het)
        chain_arr = np.array(chain_of)
        resid_arr = np.array([f"{k[1]}{k[2]}" for k in keys])
        for r in rows:
            rec = dict(pdb_id=pdb_id, comp_id=r["comp_id"], asym_id=r["asym_id"], status="ok")
            try:
                t = template(r["comp_id"])
                if t is None:
                    rec["status"] = "no_template"
                    out.append(rec)
                    continue
                mol, name2idx, rotb = t
                sel = np.where(asym_arr == r["asym_id"])[0]
                if len(sel) == 0:
                    rec["status"] = "not_found"
                    out.append(rec)
                    continue
                lig_xyz = np.full((mol.GetNumAtoms(), 3), np.nan)
                lig_sym = [a.GetSymbol() for a in mol.GetAtoms()]
                for i in sel:
                    j = name2idx.get(name_arr[i])
                    if j is not None and np.isnan(lig_xyz[j, 0]):
                        lig_xyz[j] = coords[i]
                present = ~np.isnan(lig_xyz[:, 0])
                rec["n_atoms_model"] = int(present.sum())
                rec["n_atoms_template"] = mol.GetNumAtoms()
                n_tpl = mol.GetNumAtoms()
                rec["n_missing"] = int(n_tpl - present.sum())
                rec["frac_missing"] = float(1 - present.sum() / n_tpl)
                if present.sum() == 0 or rec["frac_missing"] > max_missing + 1e-9:
                    rec["status"] = "incomplete"
                    out.append(rec)
                    continue
                # up to max_missing of the template atoms may be unresolved: geometry is computed on the
                # atoms that are present (buried fraction and buried rotatable bonds are then lower bounds)
                idx = np.where(present)[0]
                lig_xyz = lig_xyz[idx]
                lig_sym = [lig_sym[k] for k in idx]
                # environment: everything except this ligand, within SHELL of any ligand atom
                env_mask = (asym_arr != r["asym_id"])
                env_idx = np.where(env_mask)[0]
                lt = cKDTree(lig_xyz)
                et = cKDTree(coords[env_idx])
                d, nn = lt.query(coords[env_idx], k=1)
                near = env_idx[d < SHELL]
                dd = np.linalg.norm(lig_xyz[:, None, :] - coords[near][None, :, :], axis=2)
                # contacts / covalent
                prot_near = ~het_arr[near]
                cont = dd.min(axis=0) < 4.0
                rec["n_contact_res"] = int(len(set(resid_arr[near][cont & prot_near])))
                ch_cnt = Counter(chain_arr[near][cont & prot_near])
                rec["main_chain"] = ch_cnt.most_common(1)[0][0] if ch_cnt else ""
                rec["n_contact_chains"] = len(ch_cnt)
                metal = np.isin(np.char.upper(syms[near]), ["ZN", "FE", "CU", "MN", "CO", "NI", "MG", "CA", "CD", "HG", "NA", "K"])
                rec["covalent"] = bool(((dd < 2.0) & ~metal[None, :] & (~het_arr[near])[None, :]).any())
                rec["metal_contact"] = bool(((dd < 2.8) & metal[None, :]).any())
                # SASA
                lr = vdw(lig_sym)
                allxyz = np.vstack([lig_xyz, coords[near]])
                allr = np.concatenate([lr, vdw(syms[near])])
                tg = np.arange(len(idx))
                s_free = sasa_atoms(lig_xyz, lr, tg)
                s_cplx = sasa_atoms(allxyz, allr, tg)
                with np.errstate(divide="ignore", invalid="ignore"):
                    b = np.where(s_free > 0.01, 1 - s_cplx / s_free, 1.0)
                b = np.clip(b, 0, 1)
                rec["sasa_free"] = float(s_free.sum())
                rec["sasa_complex"] = float(s_cplx.sum())
                rec["buried_frac"] = float(1 - s_cplx.sum() / s_free.sum()) if s_free.sum() > 0 else np.nan
                rec["n_buried_atoms"] = int((b >= 0.5).sum())
                rec["frac_buried_atoms"] = float((b >= 0.5).mean())
                rec["n_rotb"] = len(rotb)
                pos = {int(k): n for n, k in enumerate(idx)}  # template index -> row in the present-atom arrays
                rec["n_rotb_buried"] = int(sum(1 for a, c in rotb
                                               if a in pos and c in pos and b[pos[a]] >= 0.5 and b[pos[c]] >= 0.5))
            except Exception as e:  # noqa: BLE001
                rec["status"] = f"error:{str(e)[:80]}"
            out.append(rec)
    except Exception as e:  # noqa: BLE001
        print(f"entry {pdb_id} failed: {e}", file=sys.stderr)
        traceback.print_exc()
        for r in rows:
            out.append(dict(pdb_id=pdb_id, comp_id=r["comp_id"], asym_id=r["asym_id"], status=f"entry_error:{str(e)[:60]}"))
    finally:
        if not keep_cif:
            try:
                cif_path(pdb_id).unlink(missing_ok=True)
            except OSError:
                pass
    return out


def run_chunk(chunk, keep_cif, max_missing=0.10):
    res = []
    for pid, rows in chunk:
        res += process_entry(pid, rows, keep_cif, max_missing)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--chunk", type=int, default=100, help="entries per task")
    ap.add_argument("--keep-cif", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-missing", type=float, default=0.10,
                    help="accept ligands with up to this fraction of template atoms unresolved")
    ap.add_argument("--redo", action="store_true",
                    help="only (re)process instances absent from geometry.parquet and 'incomplete' ones within "
                         "--max-missing; results go to parts r*.parquet and override older rows")
    a = ap.parse_args()

    inst = pd.read_parquet(DATA / "instances.parquet")
    comps = pd.read_parquet(DATA / "components.parquet")
    inst = inst[inst.comp_id.isin(comps[comps.ok].comp_id)]
    prefix = "c"
    if a.redo:
        prefix = "r"
        key = ["pdb_id", "comp_id", "asym_id"]
        geo = pd.read_parquet(DATA / "geometry.parquet").drop_duplicates(key, keep="last")
        m = inst.merge(geo[key + ["status", "n_atoms_model", "n_atoms_template"]], on=key, how="left")
        fm = 1 - m.n_atoms_model / m.n_atoms_template
        todo = m[m.status.isna() | ((m.status == "incomplete") & (m.n_atoms_model > 0) & (fm <= a.max_missing + 1e-9))]
        print(f"redo: {m.status.isna().sum()} instances missing from geometry, "
              f"{len(todo) - m.status.isna().sum()} incomplete within {a.max_missing:.0%} cap")
        inst = todo
        for old in PARTS.glob("r*.parquet"):  # a new selection starts clean (names depend on it)
            old.unlink()
    done = {p.stem for p in PARTS.glob("*.parquet")}
    groups = [(pid, g[["comp_id", "asym_id"]].to_dict("records")) for pid, g in inst.groupby("pdb_id")]
    if a.limit:
        groups = groups[: a.limit]
    chunks = [(f"{prefix}{i // a.chunk:05d}", groups[i:i + a.chunk]) for i in range(0, len(groups), a.chunk)]
    chunks = [c for c in chunks if c[0] not in done]
    print(f"{len(groups)} entries, {len(chunks)} chunks to run, {a.workers} workers")
    try:
        from tqdm import tqdm
    except ImportError:  # minimal fallback if tqdm is not installed
        class tqdm:  # noqa: N801
            def __init__(self, total=None, **kw):
                self.n, self.total = 0, total

            def update(self, k=1):
                self.n += k
                print(f"  {self.n}/{self.total} entries", flush=True)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

    n_entries = sum(len(c[1]) for c in chunks)
    with ProcessPoolExecutor(a.workers) as ex, tqdm(total=n_entries, desc="entries", unit="entry") as bar:
        futs = {ex.submit(run_chunk, c[1], a.keep_cif, a.max_missing): (c[0], len(c[1])) for c in chunks}
        for f in as_completed(futs):
            name, n = futs[f]
            pd.DataFrame(f.result()).to_parquet(PARTS / f"{name}.parquet", index=False)
            bar.update(n)

    parts = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("*.parquet"))], ignore_index=True)
    # r* parts sort after c* parts, so redone rows override the older ones
    parts = parts.drop_duplicates(["pdb_id", "comp_id", "asym_id"], keep="last")
    parts.to_parquet(DATA / "geometry.parquet", index=False)
    print(parts.status.value_counts().to_string())


if __name__ == "__main__":
    main()
