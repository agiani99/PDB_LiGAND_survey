"""Step 10 - which force-field energy terms go with the xtb strain energy?  Output only in data/terms/.

For a random sample of ligands from strain.parquet:
  run      re-creates the xtb geometries of 06_strain.py (same settings and random seed):
             pose   torsion-constrained optimum of the deposited pose
             relax  unconstrained relaxation of the pose
             ref    lowest of relax and the xtb-optimised ETKDG ensemble
           and evaluates MMFF94 term by term on each of them, twice:
             held   after an MMFF minimisation that keeps every heavy-atom torsion at the xtb value, so that bond
                    lengths and angles are the force field's own (analyse default)
             raw    directly on the xtb geometry
           Resumable.  The geometries are saved (geoms/*.npz), so other decompositions need no new xtb run.
  analyse  differences per term:  total = pose - ref,  local = pose - relax,  basin = relax - ref.

Terms: bond, angle, stretch-bend, out-of-plane, torsion, van der Waals, electrostatic.

Read with care:
  * MMFF is used only to apportion the strain; its sum is NOT the xtb dE, and on a 35-ligand test the two agreed
    poorly (Spearman about 0 for raw, about 0.3 for held).  Check the agreement analyse prints first: if it is
    weak, the term shares describe MMFF and say little about the xtb strain.
  * The torsion term is not all of torsional strain: 1-4 and longer-range interactions (vdW, electrostatic)
    also change when a torsion turns.
  * Electrostatics have no solvent here.  They are computed at dielectric 1 and divided by --eps in analyse
    (default 4, an arbitrary choice; the sensitivity to eps is printed).
  * All geometries are ligand-only: "van der Waals" means contacts within the ligand, not with the protein.

Usage:
    python 10_strain_terms.py run --n 2000 --workers 12
    python 10_strain_terms.py analyse
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

S = importlib.import_module("06_strain")
V = importlib.import_module("07_orca_validate")  # ensemble_geoms: the ensemble of 06, keeping geometries
DATA = S.DATA
EH2KCAL = S.EH2KCAL

TERMS = {"bond": "Bond", "angle": "Angle", "strbend": "StretchBend", "oop": "Oop", "torsion": "Torsion",
         "vdw": "VdW", "ele": "Ele"}
GROUPS = {"torsion": ["torsion"], "angles+bonds": ["bond", "angle", "strbend", "oop"], "vdw": ["vdw"],
          "electrostatic": ["ele"]}
# DataFrame.max/median over several columns die with a Windows stack overflow on this installation when pandas
# uses bottleneck
pd.set_option("compute.use_bottleneck", False)
KEY = ["pdb_id", "comp_id", "asym_id"]
BINS = [-1e9, 5, 10, 20, 1e9]
BIN_LABELS = ["<5", "5-10", "10-20", ">20"]


# ------------------------------------------------------------------ run
def mmff_terms(mh, xyz: np.ndarray) -> dict | None:
    """MMFF94 energy of each term (kcal/mol, dielectric 1) for one geometry; None without parameters."""
    out = {}
    for name, rd in TERMS.items():
        m = Chem.Mol(mh)  # MMFF perception edits aromaticity flags: always work on a copy
        m.RemoveAllConformers()
        conf = Chem.Conformer(m.GetNumAtoms())
        for i, p in enumerate(xyz):
            conf.SetAtomPosition(i, p.tolist())
        m.AddConformer(conf, assignId=True)
        props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94")
        if props is None:
            return None
        for other in TERMS.values():
            getattr(props, f"SetMMFF{other}Term")(other == rd)
        ff = AllChem.MMFFGetMoleculeForceField(m, props, nonBondedThresh=1000.0, ignoreInterfragInteractions=False)
        if ff is None:
            return None
        out[name] = ff.CalcEnergy()
    return out


def held_geometry(mh, xyz: np.ndarray, eps: float = 4.0, fc: float = 10.0):
    """MMFF94 minimisation with every heavy-atom torsion restrained to its value in xyz.
    Returns (coordinates, largest torsion drift in degrees)."""
    from rdkit.Chem import rdMolTransforms as T
    m = Chem.Mol(mh)
    conf = m.GetConformer()
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, p.tolist())
    torsions = S.torsion_constraints(m)
    props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94")
    props.SetMMFFDielectricConstant(eps)
    ff = AllChem.MMFFGetMoleculeForceField(m, props, nonBondedThresh=1000.0, ignoreInterfragInteractions=False)
    for i, j, k, l, ang in torsions:
        ff.MMFFAddTorsionConstraint(i - 1, j - 1, k - 1, l - 1, False, ang - 0.5, ang + 0.5, fc)
    ff.Initialize()
    ff.Minimize(maxIts=10000)
    y = np.array(ff.Positions()).reshape(-1, 3)
    for i, p in enumerate(y):
        conf.SetAtomPosition(i, p.tolist())
    drift = [abs((T.GetDihedralDeg(conf, i - 1, j - 1, k - 1, l - 1) - ang + 180) % 360 - 180)
             for i, j, k, l, ang in torsions]
    return y, max(drift, default=0.0)


def one_ligand(model, row: dict, a, geom_dir: Path) -> dict:
    out = {k: row[k] for k in KEY}
    out.update(status="ok", delta_e_prod=row["delta_e_kcal"])
    t0 = time.time()
    tmpl = S.ccd_mol(row["comp_id"])
    if tmpl is None:
        return {**out, "status": "no_template"}
    xyz = S.ligand_xyz(model, row["asym_id"], tmpl)
    if xyz is None:
        return {**out, "status": "incomplete"}
    mh, q = S.build_pose_mol(S.ionize(tmpl, False), xyz)
    if not AllChem.MMFFHasAllMoleculeParams(Chem.Mol(mh)):
        return {**out, "status": "no_mmff_parameters"}  # checked first: saves the xtb time
    wd = Path(tempfile.mkdtemp(dir=S.TMP))
    try:
        e_pose, x_pose = S.run_xtb(wd, mh, q, torsions=S.torsion_constraints(mh))
        if e_pose is None:
            return {**out, "status": "xtb_pose_failed"}
        e_relax, x_relax = S.run_xtb(wd, mh, q, xyz=x_pose)
        if e_relax is None:
            return {**out, "status": "xtb_relax_failed"}
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    best = V.ensemble_geoms(mh, q, a.max_confs, a.n_opt, 1, S.TMP)
    e_ref, x_ref = min([(e_relax, x_relax)] + best, key=lambda t: t[0])
    geoms = {"pose": x_pose, "relax": x_relax, "ref": x_ref}
    for g, x in geoms.items():
        t = mmff_terms(mh, x)
        if t is None:
            return {**out, "status": "no_mmff_parameters"}
        out.update({f"{g}_{k}": v for k, v in t.items()})
        y, drift = held_geometry(mh, x)
        out.update({f"held_{g}_{k}": v for k, v in mmff_terms(mh, y).items()})
        out[f"held_{g}_drift"] = drift
    np.savez_compressed(geom_dir / ("{pdb_id}_{comp_id}_{asym_id}.npz".format(**row)),
                        z=np.array([at.GetAtomicNum() for at in mh.GetAtoms()]), charge=q, xray=xyz, **geoms)
    out.update(charge=int(q), n_atoms=mh.GetNumAtoms(), ref_is_relax=bool(e_ref == e_relax),
               xtb_total=(e_pose - e_ref) * EH2KCAL, xtb_local=(e_pose - e_relax) * EH2KCAL,
               xtb_basin=(e_relax - e_ref) * EH2KCAL, t_sec=round(time.time() - t0, 1))
    return out


def one_entry(pid: str, rows: list, a, geom_dir: Path) -> list:
    import gemmi
    had_cif = S.cif_path(pid).exists()
    res = []
    try:
        st = gemmi.read_structure(str(S.fetch_cif(pid)))
        st.setup_entities()
        for r in rows:
            try:
                res.append(one_ligand(st[0], r, a, geom_dir))
            except Exception as e:  # noqa: BLE001
                res.append({**{k: r[k] for k in KEY}, "status": f"error:{str(e)[:80]}"})
    except Exception as e:  # noqa: BLE001
        res = [{**{k: r[k] for k in KEY}, "status": f"entry_error:{str(e)[:60]}"} for r in rows]
    finally:
        if not had_cif:
            S.cif_path(pid).unlink(missing_ok=True)
    return res


def read_results(f: Path) -> pd.DataFrame:
    if not f.exists():
        return pd.DataFrame()
    rows = []
    for ln in f.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:  # half-written last line after an interruption
            pass
    return pd.DataFrame(rows).drop_duplicates(KEY, keep="last") if rows else pd.DataFrame()


def cmd_run(a):
    if not Path(S.XTB).exists() and not shutil.which(S.XTB):
        raise SystemExit(f"xtb not found: {S.XTB} (set XTB_EXE)")
    out = DATA / a.out_name
    geom_dir = out / "geoms"
    geom_dir.mkdir(parents=True, exist_ok=True)
    st = pd.read_parquet(DATA / "strain.parquet")
    st = st[st.status == "ok"].sort_values(KEY)
    # a few ligands (e.g. bridged cages) take hours in the conformer embedding: skip them by their production time
    slow = st.t_sec > a.max_sec
    print(f"{int(slow.sum())} of {len(st)} ligands left out of the sampling pool: the strain run needed more than "
          f"{a.max_sec:g} s for them")
    st = st[~slow]
    # fixed random order, so a larger --n later extends the same sample
    st = st.iloc[np.random.default_rng(a.seed).permutation(len(st))].head(a.n)[KEY + ["delta_e_kcal"]]
    resf = out / "results.jsonl"
    old = read_results(resf)
    if len(old):
        old = old[~old.status.str.startswith("entry_error")]  # download problems are retried
        st = st.merge(old[KEY], how="left", on=KEY, indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    groups = [(pid, g.to_dict("records")) for pid, g in st.groupby("pdb_id", sort=False)]
    print(f"{len(st)} ligands to run, {a.workers} workers; output {out}", flush=True)
    if resf.exists() and resf.stat().st_size and not resf.read_bytes().endswith(b"\n"):
        resf.open("a").write("\n")
    t0, n = time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex, open(resf, "a") as fh:
        futs = [ex.submit(one_entry, pid, rows, a, geom_dir) for pid, rows in groups]
        for f in as_completed(futs):
            for r in f.result():
                fh.write(json.dumps(r) + "\n")
                n += 1
                if n % 25 == 0 or n == len(st):
                    el = (time.time() - t0) / 60
                    print(f"  {n}/{len(st)}  {el:.1f} min  (about {el / n * (len(st) - n):.0f} min left)", flush=True)
            fh.flush()
    res = read_results(resf)
    if len(res):
        print(res.status.value_counts().to_string())


# ------------------------------------------------------------------ analyse
def differences(d: pd.DataFrame, part: str, eps: float, geometry: str = "held") -> pd.DataFrame:
    pre = "held_" if geometry == "held" else ""
    hi, lo = {"total": ("pose", "ref"), "local": ("pose", "relax"), "basin": ("relax", "ref")}[part]
    t = pd.DataFrame({g: sum(d[f"{pre}{hi}_{k}"] - d[f"{pre}{lo}_{k}"] for k in ks) for g, ks in GROUPS.items()})
    t["electrostatic"] /= eps
    t["mmff_sum"] = t[list(GROUPS)].sum(axis=1)
    t["xtb"] = d[f"xtb_{part}"]
    return t


def shares(t: pd.DataFrame, target: str) -> pd.Series:
    """cov(term, target) / var(target).  With target = mmff_sum the four shares add up to 1."""
    return pd.Series({g: np.cov(t[g], t[target])[0, 1] / t[target].var() for g in GROUPS})


def cmd_analyse(a):
    out = DATA / a.out_name
    res = read_results(out / "results.jsonl")
    if res.empty:
        raise SystemExit(f"no results in {out}: run `python 10_strain_terms.py run` first")
    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    say(res.status.str.split(":").str[0].value_counts().to_string())
    d = res[res.status == "ok"].reset_index(drop=True)
    if len(d) < 10:
        raise SystemExit("fewer than 10 successful ligands: nothing to analyse yet")
    dev = (d.xtb_total - d.delta_e_prod).abs()
    say(f"\nreproduction of the production dE: median |difference| {dev.median():.3f} kcal/mol, "
        f"{(dev > 1).mean():.1%} differ by more than 1")

    # drop absurd force-field values (typically an MMFF parameter problem), report how many
    bad = pd.Series(np.abs(differences(d, "total", 1.0, a.geometry)[list(GROUPS)].to_numpy()).max(axis=1) > a.max_term)
    say(f"excluded for an MMFF term difference > {a.max_term:g} kcal/mol (at eps 1): {int(bad.sum())}")
    d = d[~bad].reset_index(drop=True)
    say(f"analysed: {len(d)} ligands; geometry '{a.geometry}'; electrostatics divided by eps = {a.eps:g}")
    drift = np.maximum.reduce([d[f"held_{g}_drift"].to_numpy() for g in ("pose", "relax", "ref")])
    say(f"torsion drift in the held minimisation: median of the per-ligand maximum {np.median(drift):.1f} deg, "
        f"{(drift > 10).mean():.1%} of ligands above 10 deg")
    say("\nagreement of the MMFF sum with xtb (Spearman) - read this before the shares")
    say(pd.DataFrame({g: {p: differences(d, p, a.eps, g).mmff_sum.corr(d[f"xtb_{p}"], method="spearman")
                          for p in ("total", "local", "basin")} for g in ("held", "raw")}).round(2).to_string())

    for part, label in [("total", "pose - reference (dE)"), ("local", "pose - relaxed (local strain)"),
                        ("basin", "relaxed - reference (change of conformer)")]:
        t = differences(d, part, a.eps, a.geometry)
        say(f"\n===== {label} =====")
        say(f"agreement MMFF sum vs xtb: Pearson {t.mmff_sum.corr(t.xtb):.2f}, "
            f"Spearman {t.mmff_sum.corr(t.xtb, method='spearman'):.2f}; "
            f"median xtb {t.xtb.median():.1f}, median MMFF sum {t.mmff_sum.median():.1f}")
        tab = pd.DataFrame({
            "mean": t[list(GROUPS)].mean(), "median": t[list(GROUPS)].median(),
            "share_of_MMFF_var": shares(t, "mmff_sum"),
            "spearman_with_xtb": t[list(GROUPS)].corrwith(t.xtb, method="spearman")})
        say(tab.round(2).to_string())

    t = differences(d, "total", a.eps, a.geometry)
    boot = np.random.default_rng(0)
    bs = pd.DataFrame([shares(t.iloc[boot.integers(0, len(t), len(t))], "mmff_sum") for _ in range(1000)])
    say("\n95% bootstrap interval of the variance shares (total)")
    say(bs.quantile([0.025, 0.975]).T.round(2).to_string())

    d["bin"] = pd.cut(d.xtb_total, BINS, labels=BIN_LABELS)
    d["charged"] = np.where(d.charge == 0, "neutral", "charged")
    sv = pd.read_parquet(DATA / "survey.parquet")[KEY + ["n_rotb", "resolution"]]
    d = d.merge(sv, on=KEY, how="left")
    d["rotb"] = pd.cut(d.n_rotb, [-1, 2, 5, 8, 99], labels=["0-2", "3-5", "6-8", ">8"])
    d["res"] = pd.cut(d.resolution, [0, 2, 3.5, 9], labels=["high", "medium", "low"])
    t = differences(d, "total", a.eps, a.geometry)
    for col, name in [("bin", "xtb dE bin"), ("charged", "charge"), ("rotb", "rotatable bonds"),
                      ("res", "resolution class")]:
        g = t.groupby(d[col], observed=True)
        say(f"\nmedian term difference (total) by {name}")
        say(g[list(GROUPS) + ["mmff_sum", "xtb"]].median().assign(n=g.size()).round(2).to_string())

    largest = pd.Series(np.array(list(GROUPS))[t[list(GROUPS)].to_numpy().argmax(axis=1)], name="largest")
    say("\nlargest positive term per ligand (total), fraction of ligands by xtb dE bin")
    say(pd.crosstab(d["bin"], largest, normalize="index").round(2).to_string())

    say("\nsensitivity to the dielectric constant (variance shares, total)")
    say(pd.DataFrame({f"eps {e:g}": shares(differences(d, "total", e, a.geometry), "mmff_sum") for e in (1, 4, 80)})
        .round(2).to_string())
    say("agreement (Spearman) of MMFF sum with xtb dE: " + ", ".join(
        f"eps {e:g}: {differences(d, 'total', e, a.geometry).mmff_sum.corr(d.xtb_total, method='spearman'):.2f}"
        for e in (1, 4, 80)))

    pd.concat([d, t.drop(columns="xtb").add_prefix("d_")], axis=1).to_csv(out / f"terms_{a.geometry}.csv", index=False)
    (out / f"stats_{a.geometry}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        for axx, part in zip(ax, ["total", "local", "basin"]):
            m = differences(d, part, a.eps, a.geometry)[list(GROUPS)].groupby(d["bin"], observed=True).mean()
            m.plot.bar(ax=axx, rot=0, legend=part == "total")
            axx.set_title(part)
            axx.set_xlabel("xtb dE bin (kcal/mol)")
            axx.axhline(0, color="k", lw=.6)
        ax[0].set_ylabel(f"mean MMFF94 difference (kcal/mol, eps {a.eps:g})")
        fig.tight_layout()
        fig.savefig(out / f"terms_{a.geometry}.png", dpi=150)
        print(f"\nwrote terms_{a.geometry}.csv, stats_{a.geometry}.txt and terms_{a.geometry}.png in {out}")
    except ImportError:
        print(f"\nwrote terms_{a.geometry}.csv and stats_{a.geometry}.txt in {out} (matplotlib missing: no plot)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--out-name", default="terms", help="folder under data/")
        if name == "run":
            p.add_argument("--n", type=int, default=2000)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--workers", type=int, default=12)
            p.add_argument("--max-sec", type=float, default=900, help="skip ligands slower than this in strain.parquet")
            p.add_argument("--max-confs", type=int, default=60)  # as in the production strain run
            p.add_argument("--n-opt", type=int, default=8)
        else:
            p.add_argument("--eps", type=float, default=4.0, help="electrostatic term is divided by this")
            p.add_argument("--max-term", type=float, default=500.0)
            p.add_argument("--geometry", choices=["held", "raw"], default="held")
    a = ap.parse_args()
    {"run": cmd_run, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
