"""Step 8 - potency (ChEMBL binding assays) versus ligand strain.

link:     for every ligand with a strain energy (06_strain.py) find ChEMBL activities with
            * the same compound (CCD InChIKey; searched on the connectivity block, stereo match flagged, protonation
              character ignored; salt/parent forms included)
            * a SINGLE PROTEIN target whose UniProt accession is one of the accessions of a PDB entry in survey.parquet
              that contains this ligand
            * binding assay (assay_type 'B'), confidence score 8-9 (--min-conf), standard_relation '=', pChEMBL present,
              no data-validity comment, not a potential duplicate, types IC50 / Ki / Kd / Km
          -> data/potency/activities.parquet
analyse:  one row per (ligand, target, type) = median pChEMBL; main set requires that the protein of the instance
          whose strain was computed IS the assay target.  Raw, partial (size/flexibility/resolution), within-target
          and quality-filtered correlations -> data/potency/pairs.csv, stats.txt, potency_vs_strain.png

Usage:  python 08_potency.py            (link + analyse)     python 08_potency.py analyse
Env: CHEMBL_DB overrides the database path.  The database is opened read-only.
"""
from __future__ import annotations

import argparse
import os
import sqlite3

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from common import DATA, ccd_path

RDLogger.DisableLog("rdApp.*")
DB = os.environ.get("CHEMBL_DB") or (r"C:\Users\andrea.DESKTOP-26V6UN4\Documents\NEMO_AI\act_cliff\SQLITE"
                                     r"\chembl_37_sqlite\chembl_37.db")
OUT = DATA / "potency"
TYPES = ("Ki", "Kd", "IC50", "Km")


def accessions(u) -> set:
    return {x.strip() for x in str(u).split(",") if x.strip() and x != "None" and x != "nan"}


def ccd_inchikey(comp_id: str, smiles: str) -> str:
    """InChIKey with stereochemistry from the CCD file (survey SMILES are stereo-free); RDKit fallback."""
    p = ccd_path(comp_id)
    if p.exists():
        block = gemmi.cif.read(str(p)).sole_block()
        for r in block.find("_pdbx_chem_comp_descriptor.", ["type", "descriptor"]):
            if gemmi.cif.as_string(r[0]).lower() == "inchikey":
                return gemmi.cif.as_string(r[1])
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m) if m is not None else ""


def cmd_link(a):
    sv = pd.read_parquet(DATA / "survey.parquet", columns=["pdb_id", "comp_id", "smiles", "uniprot", "delta_e_kcal"])
    st = sv[sv.delta_e_kcal.notna()].drop_duplicates("comp_id")
    all_acc = sv[sv.comp_id.isin(st.comp_id)].groupby("comp_id").uniprot.agg(
        lambda s: set().union(*map(accessions, s)))
    db = sqlite3.connect("file:" + DB.replace("\\", "/") + "?mode=ro", uri=True)
    q = f"""
      SELECT cs.standard_inchi_key, act.activity_id, act.standard_type, act.pchembl_value, ass.assay_id, ass.doc_id,
             ass.confidence_score, ass.variant_id, td.chembl_id, td.pref_name, cseq.accession
      FROM compound_structures cs
      CROSS JOIN molecule_hierarchy h0 ON h0.molregno = cs.molregno
      CROSS JOIN molecule_hierarchy h ON h.parent_molregno = h0.parent_molregno
      CROSS JOIN activities act ON act.molregno = h.molregno
      CROSS JOIN assays ass ON ass.assay_id = act.assay_id
      CROSS JOIN target_dictionary td ON td.tid = ass.tid
      CROSS JOIN target_components tc ON tc.tid = td.tid
      CROSS JOIN component_sequences cseq ON cseq.component_id = tc.component_id
      WHERE cs.standard_inchi_key >= ? AND cs.standard_inchi_key < ?
        AND +ass.assay_type = 'B' AND ass.confidence_score BETWEEN ? AND 9 AND +td.target_type = 'SINGLE PROTEIN'
        AND +act.standard_relation = '=' AND act.pchembl_value IS NOT NULL
        AND act.data_validity_comment IS NULL AND COALESCE(act.potential_duplicate, 0) = 0
        AND +act.standard_type IN ({",".join("?" * len(TYPES))})
    """
    rows, n_key = [], 0
    for i, r in enumerate(st.itertuples(), 1):
        key = ccd_inchikey(r.comp_id, r.smiles)
        if not key:
            continue
        n_key += 1
        k14 = key[:14]
        for res in db.execute(q, (k14, k14 + "~", a.min_conf, *TYPES)):
            acc = res[-1]
            if acc not in all_acc.get(r.comp_id, set()):
                continue
            rows.append(dict(comp_id=r.comp_id, inchikey=key, chembl_key=res[0], stereo_exact=res[0][:25] == key[:25],
                             activity_id=res[1], type=res[2], pchembl=float(res[3]), assay_id=res[4], doc_id=res[5],
                             confidence=res[6], variant=res[7] is not None, target_chembl=res[8], target=res[9],
                             accession=acc, same_protein_as_strain=acc in accessions(r.uniprot)))
        if i % 2000 == 0 or i == len(st):
            print(f"  link {i}/{len(st)} ligands, {len(rows)} activities", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(["comp_id", "activity_id", "accession"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "activities.parquet", index=False)
    print(f"{n_key}/{len(st)} ligands with InChIKey; {df.comp_id.nunique()} matched, {len(df)} activities")


def partial_spearman(d: pd.DataFrame, x: str, y: str, covars: list) -> tuple:
    d = d[[x, y, *covars]].dropna()
    if len(d) < 10:
        return np.nan, len(d)
    R = d.rank()
    X = np.column_stack([np.ones(len(R)), R[covars].to_numpy()])
    res = [R[c].to_numpy() - X @ np.linalg.lstsq(X, R[c].to_numpy(), rcond=None)[0] for c in (x, y)]
    return float(np.corrcoef(*res)[0, 1]), len(d)


def boot_ci(d: pd.DataFrame, x: str, y: str, group: str | None = None, n: int = 1000):
    """95% bootstrap CI of Spearman rho; resamples whole groups (targets) when `group` is given."""
    rng = np.random.default_rng(0)
    idx = [g.index.to_numpy() for _, g in d.groupby(group)] if group else [np.array([i]) for i in d.index]
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        s = d.loc[pick]
        vals.append(s[x].rank().corr(s[y].rank()))
    return np.nanpercentile(vals, [2.5, 97.5])


def cmd_analyse(a):
    act = pd.read_parquet(OUT / "activities.parquet")
    act = act[act.stereo_exact & ~act.variant]
    sv = pd.read_parquet(DATA / "survey.parquet")
    st = sv[sv.delta_e_kcal.notna()].drop_duplicates("comp_id")
    cols = ["comp_id", "pdb_id", "family", "resolution", "res_class", "rscc", "heavy_atoms", "n_rotb", "n_rotb_buried",
            "buried_frac", "logp", "mw", "formal_charge", "delta_e_kcal", "local_strain_kcal", "qc_pass"]
    agg = (act.groupby(["comp_id", "accession", "target", "type", "same_protein_as_strain"])
              .agg(pchembl=("pchembl", "median"), n_meas=("pchembl", "size"), spread=("pchembl", "std"),
                   n_docs=("doc_id", "nunique")).reset_index())
    pairs = agg.merge(st[cols], on="comp_id")
    pairs["conf_energy"] = pairs.delta_e_kcal - pairs.local_strain_kcal
    pairs["LE"] = 1.364 * pairs.pchembl / pairs.heavy_atoms
    pairs["dE_per_HA"] = pairs.delta_e_kcal / pairs.heavy_atoms
    pairs.to_csv(OUT / "pairs.csv", index=False)

    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"activities (exact stereo, wild type): {len(act)}; ligand-target-type rows: {len(pairs)}")
    main = pairs[pairs.same_protein_as_strain].copy()
    say(f"main set (strain structure is of the assay target): {len(main)} rows, {main.comp_id.nunique()} ligands, "
        f"{main.accession.nunique()} targets")
    for t in TYPES:
        d = main[main.type == t].reset_index(drop=True)
        say(f"\n===== {t}: n={len(d)}, targets={d.accession.nunique()} =====")
        if len(d) < 30:
            say("  too few data for statistics")
            continue
        say(f"  pChEMBL median {d.pchembl.median():.2f} (IQR {d.pchembl.quantile(.25):.2f}-"
            f"{d.pchembl.quantile(.75):.2f}); dE median {d.delta_e_kcal.quantile(.5):.2f}")
        for y in ["delta_e_kcal", "local_strain_kcal", "conf_energy", "dE_per_HA", "heavy_atoms", "n_rotb"]:
            rho = d.pchembl.rank().corr(d[y].rank())
            lo, hi = boot_ci(d, "pchembl", y, "accession")
            say(f"  Spearman pChEMBL vs {y:18s} {rho:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  (resampling targets)")
        r, n = partial_spearman(d, "pchembl", "delta_e_kcal", ["heavy_atoms", "n_rotb", "resolution"])
        say(f"  partial Spearman (dE | heavy atoms, rotb, resolution) {r:+.3f}  n={n}")
        say(f"  Spearman LE vs dE {d.LE.rank().corr(d.delta_e_kcal.rank()):+.3f}   (LE has heavy atoms in it)")
        sub = d[(d.resolution <= 2.5) & d.qc_pass.fillna(False).astype(bool)]
        if len(sub) >= 30:
            r, n = partial_spearman(sub, "pchembl", "delta_e_kcal", ["heavy_atoms", "n_rotb", "resolution"])
            say(f"  quality subset (res<=2.5, qc_pass) n={len(sub)}: raw "
                f"{sub.pchembl.rank().corr(sub.delta_e_kcal.rank()):+.3f}, partial {r:+.3f}")
        # within-target: centre ranks inside each target with >= min_n ligands
        g = d.groupby("accession").filter(lambda x: len(x) >= a.min_n)
        if len(g):
            per = g.groupby("accession").apply(
                lambda x: pd.Series(dict(target=x.target.iloc[0], n=len(x),
                                         rho=x.pchembl.rank().corr(x.delta_e_kcal.rank()),
                                         rho_part=partial_spearman(x, "pchembl", "delta_e_kcal",
                                                                   ["heavy_atoms", "n_rotb"])[0])),
                include_groups=False)
            w = per.n.astype(float)
            say(f"  within-target (n>={a.min_n}): {len(per)} targets, weighted mean rho "
                f"{np.average(per.rho.astype(float), weights=w):+.3f}, partial "
                f"{np.average(per.rho_part.astype(float), weights=w):+.3f}; "
                f"{(per.rho.astype(float) < 0).mean():.0%} of targets negative")
            per.sort_values("n", ascending=False).to_csv(OUT / f"per_target_{t}.csv")
        b = pd.cut(d.delta_e_kcal, [-1, 2.5, 5, 10, 20, 1e9], labels=["<2.5", "2.5-5", "5-10", "10-20", ">20"])
        tab = d.groupby(b, observed=True).agg(
            n=("pchembl", "size"), pChEMBL_median=("pchembl", "median"),
            frac_ge8=("pchembl", lambda x: (x >= 8).mean()), heavy_median=("heavy_atoms", "median"),
            LE_median=("LE", "median"))
        say("  by dE bin:\n" + tab.round(2).to_string())
    (OUT / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [t for t in TYPES if (main.type == t).sum() >= 30]
        fig, ax = plt.subplots(2, len(ts), figsize=(4.5 * len(ts), 8), squeeze=False)
        for j, t in enumerate(ts):
            d = main[main.type == t]
            ax[0, j].scatter(d.delta_e_kcal.clip(upper=40), d.pchembl, s=5, alpha=.3)
            ax[0, j].set_title(f"{t} (n={len(d)})")
            ax[0, j].set_xlabel("xtb dE, kcal/mol (clipped at 40)")
            ax[0, j].set_ylabel("pChEMBL")
            b = pd.cut(d.delta_e_kcal, [-1, 2.5, 5, 10, 20, 1e9], labels=["<2.5", "2.5-5", "5-10", "10-20", ">20"])
            groups = [(k, g.pchembl.to_numpy()) for k, g in d.groupby(b, observed=True)]
            ax[1, j].boxplot([v for _, v in groups])
            ax[1, j].set_xticklabels([f"{k}\nn={len(v)}" for k, v in groups])
            ax[1, j].set_xlabel("dE bin, kcal/mol")
            ax[1, j].set_ylabel("pChEMBL")
        fig.tight_layout()
        fig.savefig(OUT / "potency_vs_strain.png", dpi=140)
    except ImportError:
        say("matplotlib missing: no plot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["all", "link", "analyse"], default="all")
    ap.add_argument("--min-conf", type=int, default=8, help="lowest ChEMBL assay confidence score (upper limit is 9)")
    ap.add_argument("--min-n", type=int, default=15, help="min ligands for a within-target correlation")
    a = ap.parse_args()
    if a.cmd in ("all", "link"):
        cmd_link(a)
    if a.cmd in ("all", "analyse"):
        cmd_analyse(a)


if __name__ == "__main__":
    main()
