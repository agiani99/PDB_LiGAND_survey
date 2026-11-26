"""Step 7 - validate the xtb strain energy (06_strain.py) against DFT single points with ORCA.

Design (same ligand, same geometries, only the energy method changes):
  prep     samples N ligands from strain.parquet (stratified by xtb dE bin, small + non-extreme charge),
           re-creates the geometries with xtb and writes them to data/orca/<id>/:
             pose.xyz   torsion-constrained xtb geometry of the deposited pose  (-> E_pose)
             relax.xyz  unconstrained xtb relaxation of the pose
             conf1..k.xyz  the k lowest xtb-optimised ETKDG conformers          (-> E_ref candidates)
           plus manifest.csv with the xtb energies of all of them.
  run      ORCA single points (default r2SCAN-3c + CPCM(water)) on every xyz; resumable (data/orca/results.jsonl).
  analyse  DFT dE = E_pose - min(E_relax, E_conf1..k)   versus the same quantity from xtb and versus the
           production delta_e_kcal; correlation, MAE, bias, triage-bin agreement, scatter plot, csv.

Usage:
    python 07_orca_validate.py prep --n 150 --workers 8
    python 07_orca_validate.py run --jobs 5 --nprocs 4          # heavy: run it yourself
    python 07_orca_validate.py analyse
Smoke test first:  prep --n 2 --out-name test ; run --out-name test --jobs 1 --nprocs 2
Env: ORCA_EXE overrides C:\\ORCA\\orca.exe ; XTB_EXE as in 06_strain.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

S = importlib.import_module("06_strain")
DATA = S.DATA
EH2KCAL = S.EH2KCAL
ORCA = os.environ.get("ORCA_EXE") or r"C:\ORCA\orca.exe"
_ORCA_E = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
BINS = [-1, 5, 10, 20, 1e9]
BIN_LABELS = ["<5", "5-10", "10-20", ">20"]


def odir(name: str) -> Path:
    return DATA / name


# ------------------------------------------------------------------ prep
def sample_ligands(n: int, max_heavy: int, seed: int) -> pd.DataFrame:
    sv = pd.read_parquet(DATA / "survey.parquet")
    st = sv[sv.delta_e_kcal.notna() & (sv.heavy_atoms <= max_heavy)].copy()
    if "charge" in pd.read_parquet(DATA / "strain.parquet").columns:
        ch = pd.read_parquet(DATA / "strain.parquet")[["pdb_id", "comp_id", "asym_id", "charge"]]
        st = st.merge(ch, on=["pdb_id", "comp_id", "asym_id"], how="left")
        st = st[st.charge.abs().fillna(0) <= 1]
    st["bin"] = pd.cut(st.delta_e_kcal, BINS, labels=BIN_LABELS)
    per = int(np.ceil(n / len(BIN_LABELS)))
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in st.groupby("bin", observed=True)]
    out = pd.concat(parts).sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
    return out


def ensemble_geoms(mh, charge: int, max_confs: int, n_opt: int, k: int, wd_root: Path):
    """Like 06_strain._compute_ensemble but keeps the optimised geometries; returns [(E_Eh, xyz)] lowest-first."""
    m = Chem.Mol(mh)
    m.RemoveAllConformers()
    n_rot = Chem.rdMolDescriptors.CalcNumRotatableBonds(m)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    ps.pruneRmsThresh = 0.3
    ps.numThreads = 1
    ids = list(AllChem.EmbedMultipleConfs(m, int(min(max_confs, 20 + 10 * n_rot)), ps))
    if not ids:
        ps.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(m, int(min(max_confs, 20 + 10 * n_rot)), ps))
    if not ids:
        return []
    try:
        r = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=2000)
        if any(x[0] == -1 for x in r) and not AllChem.MMFFHasAllMoleculeParams(m):
            raise ValueError
    except Exception:
        r = AllChem.UFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=2000)
    order = sorted(range(len(ids)), key=lambda i: r[i][1])
    heavy = Chem.RemoveHs(m)
    keep: list[int] = []
    for i in order:
        if all(AllChem.GetConformerRMS(heavy, ids[i], ids[j], prealigned=False) > 0.5 for j in keep):
            keep.append(i)
        if len(keep) >= n_opt:
            break
    res = []
    wd = Path(tempfile.mkdtemp(dir=wd_root))
    try:
        for i in keep:
            e, xyz = S.run_xtb(wd, m, charge, conf_id=ids[i])
            if e is not None and xyz is not None:
                res.append((e, xyz))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    res.sort(key=lambda t: t[0])
    return res[:k]


def prep_one(row: dict, out_root: Path, k: int, max_confs: int, n_opt: int) -> dict:
    import gemmi
    lid = f"{row['pdb_id']}_{row['comp_id']}_{row['asym_id']}"
    d = out_root / lid
    base = dict(id=lid, pdb_id=row["pdb_id"], comp_id=row["comp_id"], asym_id=row["asym_id"],
                delta_e_prod=row["delta_e_kcal"])
    try:
        if (d / "done").exists():
            return json.loads((d / "meta.json").read_text())
        st = gemmi.read_structure(str(S.fetch_cif(row["pdb_id"])))
        st.setup_entities()
        tmpl = S.ccd_mol(row["comp_id"])
        xyz = S.ligand_xyz(st[0], row["asym_id"], tmpl)
        if xyz is None:
            return {**base, "status": "incomplete"}
        tmpl = S.ionize(tmpl, False)
        mh, q = S.build_pose_mol(tmpl, xyz)
        d.mkdir(parents=True, exist_ok=True)
        wd = Path(tempfile.mkdtemp(dir=S.TMP))
        try:
            e_pose, xyz_p = S.run_xtb(wd, mh, q, torsions=S.torsion_constraints(mh))
            if e_pose is None:
                return {**base, "status": "xtb_pose_failed"}
            e_relax, xyz_r = S.run_xtb(wd, mh, q, xyz=xyz_p)
            geoms = {"pose": (e_pose, xyz_p)}
            if e_relax is not None:
                geoms["relax"] = (e_relax, xyz_r)
            for i, (e, x) in enumerate(ensemble_geoms(mh, q, max_confs, n_opt, k, S.TMP), 1):
                geoms[f"conf{i}"] = (e, x)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        for name, (e, x) in geoms.items():
            S.write_xyz(d / f"{name}.xyz", mh, xyz=x)
        meta = {**base, "status": "ok", "charge": int(q), "n_atoms": mh.GetNumAtoms(),
                "smiles": Chem.MolToSmiles(Chem.RemoveHs(mh)),
                "xtb": {n: e for n, (e, _) in geoms.items()}}
        (d / "meta.json").write_text(json.dumps(meta))
        (d / "done").write_text("1")
        return meta
    except Exception as e:  # noqa: BLE001
        return {**base, "status": f"error:{str(e)[:80]}"}
    finally:
        if not os.environ.get("KEEP_CIF"):
            S.cif_path(row["pdb_id"]).unlink(missing_ok=True)


def cmd_prep(a):
    out = odir(a.out_name)
    out.mkdir(parents=True, exist_ok=True)
    lig = sample_ligands(a.n, a.max_heavy, a.seed)
    print(f"{len(lig)} ligands sampled:\n{lig['bin'].value_counts().sort_index().to_string()}")
    rows = lig.to_dict("records")
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(prep_one, r, out, a.k, a.max_confs, a.n_opt) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 10 == 0 or i == len(futs):
                print(f"  prep {i}/{len(futs)}", flush=True)
    man = pd.DataFrame(res)
    ok = man[man.status == "ok"]
    print(man.status.value_counts().to_string())
    ok.drop(columns=["xtb"], errors="ignore").to_csv(out / "manifest.csv", index=False)
    (out / "xtb_energies.json").write_text(json.dumps({r["id"]: r["xtb"] for r in ok.to_dict("records")}))
    print(f"wrote {len(ok)} ligands to {out}")


# ------------------------------------------------------------------ run ORCA
def orca_input(method: str, nprocs: int, maxcore: int, charge: int, xyzname: str) -> str:
    pal = f"%pal nprocs {nprocs} end\n" if nprocs > 1 else ""
    return (f"! {method}\n{pal}%maxcore {maxcore}\n* xyzfile {charge} 1 {xyzname}\n")


def run_orca(job: tuple, method: str, nprocs: int, maxcore: int, timeout: int):
    out_root, lid, name, charge = job
    d = out_root / lid
    wd = d / f"orca_{name}"
    wd.mkdir(exist_ok=True)
    shutil.copy(d / f"{name}.xyz", wd / "mol.xyz")
    (wd / "mol.inp").write_text(orca_input(method, nprocs, maxcore, charge, "mol.xyz"))
    t0 = time.time()
    try:
        r = subprocess.run([ORCA, "mol.inp"], cwd=wd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        out = r.stdout
        (wd / "mol.out").write_text(out, encoding="utf-8")
        e = _ORCA_E.findall(out)
        ok = "ORCA TERMINATED NORMALLY" in out and bool(e)
        res = dict(id=lid, geom=name, e_dft_eh=float(e[-1]) if ok else None, t_sec=round(time.time() - t0, 1),
                   err=None if ok else (out.strip().splitlines() or [r.stderr[:200]])[-1][:200])
    except subprocess.TimeoutExpired:
        res = dict(id=lid, geom=name, e_dft_eh=None, t_sec=timeout, err="timeout")
    for f in list(wd.glob("*.gbw")) + list(wd.glob("*.tmp")) + list(wd.glob("*.densities")) + list(wd.glob("*_atom*")):
        f.unlink(missing_ok=True)
    return res


def cmd_run(a):
    out = odir(a.out_name)
    man = pd.read_csv(out / "manifest.csv")
    resf = out / "results.jsonl"
    done = set()
    if resf.exists():
        for ln in resf.read_text().splitlines():
            r = json.loads(ln)
            if r["e_dft_eh"] is not None:
                done.add((r["id"], r["geom"]))
    jobs = []
    for r in man.to_dict("records"):
        for g in ["pose", "relax"] + [f"conf{i}" for i in range(1, a.k + 1)]:
            if (out / r["id"] / f"{g}.xyz").exists() and (r["id"], g) not in done:
                jobs.append((out, r["id"], g, int(r["charge"])))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} ORCA single points to run ({a.method}), {a.jobs} jobs x {a.nprocs} procs, orca={ORCA}",
          flush=True)
    with ProcessPoolExecutor(a.jobs) as ex, open(resf, "a") as fh:
        futs = [ex.submit(run_orca, j, a.method, a.nprocs, a.maxcore, a.timeout) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"  {i}/{len(jobs)} {r['id']} {r['geom']} {r['t_sec']}s {'ok' if r['e_dft_eh'] else r['err']}",
                  flush=True)


# ------------------------------------------------------------------ analyse
def cmd_analyse(a):
    out = odir(a.out_name)
    if not (out / "results.jsonl").exists():
        raise SystemExit(f"{out / 'results.jsonl'} not found: run the ORCA step first "
                         f"(python 07_orca_validate.py run --jobs 16 --nprocs 1)")
    xt = json.loads((out / "xtb_energies.json").read_text())
    man = pd.read_csv(out / "manifest.csv").set_index("id")
    dft: dict[str, dict] = {}
    for ln in (out / "results.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["e_dft_eh"] is not None:
            dft.setdefault(r["id"], {})[r["geom"]] = r["e_dft_eh"]
    rows = []
    for lid, e in dft.items():
        if "pose" not in e:
            continue
        cands = [g for g in e if g != "pose"]
        if not cands:
            continue
        x = xt[lid]
        cx = [g for g in cands if g in x]
        e_dft_ref = min(e[g] for g in cands)
        e_xtb_ref = min(x[g] for g in cx)
        rows.append(dict(
            id=lid, comp_id=man.loc[lid, "comp_id"], n_atoms=man.loc[lid, "n_atoms"], charge=man.loc[lid, "charge"],
            n_cands=len(cands), delta_e_prod=man.loc[lid, "delta_e_prod"],
            de_xtb=(x["pose"] - e_xtb_ref) * EH2KCAL, de_dft=(e["pose"] - e_dft_ref) * EH2KCAL,
            ref_geom_dft=min(cands, key=lambda g: e[g]), ref_geom_xtb=min(cx, key=lambda g: x[g]),
            local_xtb=(x["pose"] - x["relax"]) * EH2KCAL if "relax" in e and "relax" in x else np.nan,
            local_dft=(e["pose"] - e["relax"]) * EH2KCAL if "relax" in e else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no complete ligands yet")
    df.to_csv(out / "validation.csv", index=False)

    def stats(x, y, label):
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        slope, icpt = np.polyfit(x, y, 1)
        print(f"{label:34s} n={len(x):3d}  Pearson {x.corr(y):.2f}  Spearman {x.corr(y, method='spearman'):.2f}  "
              f"MAE {np.abs(y - x).mean():.2f}  bias(dft-xtb) {np.mean(y - x):+.2f}  "
              f"RMSE {np.sqrt(np.mean((y - x) ** 2)):.2f}  fit dft = {slope:.2f}*xtb {icpt:+.2f}")

    print(f"\n{len(df)} ligands with DFT pose + reference ({a.method_label})")
    stats(df.de_xtb, df.de_dft, "dE xtb vs DFT (same candidates)")
    stats(df.delta_e_prod, df.de_dft, "dE production xtb vs DFT")
    stats(df.local_xtb, df.local_dft, "local strain xtb vs DFT")
    t = pd.crosstab(pd.cut(df.de_xtb, BINS, labels=BIN_LABELS), pd.cut(df.de_dft, BINS, labels=BIN_LABELS))
    print("\ntriage bins, rows = xtb, cols = DFT\n", t.to_string())
    agree = (pd.cut(df.de_xtb, BINS, labels=BIN_LABELS) == pd.cut(df.de_dft, BINS, labels=BIN_LABELS)).mean()
    print(f"bin agreement {agree:.0%};  ref geometry differs xtb/DFT in "
          f"{(df.ref_geom_dft != df.ref_geom_xtb).mean():.0%} of ligands")
    print("\nmedian dE: xtb %.2f, DFT %.2f" % (df.de_xtb.median(), df.de_dft.median()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
        for axx, (x, y, t) in zip(ax, [(df.de_xtb, df.de_dft, "dE (pose - lowest conformer)"),
                                       (df.local_xtb, df.local_dft, "local strain (pose - relaxed)")]):
            axx.scatter(x, y, s=14, alpha=.7)
            m = float(np.nanmax([x.max(), y.max()]))
            axx.plot([0, m], [0, m], "k--", lw=.8)
            axx.set_xlabel("xtb GFN2/ALPB (kcal/mol)")
            axx.set_ylabel(f"DFT {a.method_label} (kcal/mol)")
            axx.set_title(t)
        fig.tight_layout()
        fig.savefig(out / "validation.png", dpi=150)
        print(f"wrote {out / 'validation.csv'} and validation.png")
    except ImportError:
        print(f"wrote {out / 'validation.csv'} (matplotlib missing: no plot)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "run", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--out-name", default="orca", help="folder under data/")
        p.add_argument("--k", type=int, default=2, help="lowest xtb conformers carried to DFT (plus 'relax')")
        if name == "prep":
            p.add_argument("--n", type=int, default=150)
            p.add_argument("--max-heavy", type=int, default=35)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--max-confs", type=int, default=60)
            p.add_argument("--n-opt", type=int, default=8)
        if name == "run":
            p.add_argument("--method", default="r2SCAN-3c CPCM(water) TightSCF")
            p.add_argument("--jobs", type=int, default=5, help="parallel ORCA jobs")
            p.add_argument("--nprocs", type=int, default=4, help="MPI processes per ORCA job")
            p.add_argument("--maxcore", type=int, default=2000, help="MB per process")
            p.add_argument("--timeout", type=int, default=6 * 3600)
            p.add_argument("--limit", type=int, default=0)
        if name == "analyse":
            p.add_argument("--method-label", default="r2SCAN-3c/CPCM")
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
