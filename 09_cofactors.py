"""Step 9 - strain of natural nucleotide ligands / cofactors, and its relation to Km.  Kept apart from the survey:
everything is written to data/cofactors/ (strain.parquet and survey.parquet are not touched).

Differences from the main strain run (06_strain.py, whose energy protocol is reused unchanged):
  * the survey filters that remove these ligands are not applied (drug-like window, metal-contact filter);
    metal-contacting instances are KEPT and flagged (the energy is computed without the metal ion)
  * several instances per ligand and protein (--per, best resolution, distinct entries), not one per ligand
  * larger reference ensemble (--max-confs 300 --n-opt 30), computed once per ligand
  * analysis uses one common reference per ligand: the lowest energy found by the ensemble or any relaxation

select   show what would be run
run      xtb strain (heavy job, resumable)
km       Km values for these ligands from ChEMBL -> data/cofactors/km_chembl.csv (very sparse)
brenda   Km values from the BRENDA JSON download -> km_brenda.csv (same UniProt accession) and, kept separate,
         km_brenda_ecorg.csv (same EC number + organism only); mutant enzymes and ranges are skipped
analyse  per-ligand strain summary, reproducibility, and strain vs Km.  Km table: --km FILE.csv with columns
         comp_id, uniprot, km_uM, source  default km_brenda.csv

Usage:
    python 09_cofactors.py select
    python 09_cofactors.py run --workers 12
    python 09_cofactors.py brenda
    python 09_cofactors.py analyse
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem

S = importlib.import_module("06_strain")
DATA = S.DATA
OUT = DATA / "cofactors"
S.REF_DIR = OUT / "ref"  # set at import time so that worker processes use it too
PARTS = OUT / "parts"
for _d in (OUT, S.REF_DIR, PARTS):
    _d.mkdir(parents=True, exist_ok=True)


def _ligand_key(mh, charge: int) -> str:
    """One reference per ligand: stereo is left out of the cache key.  With stereo in the key, the arbitrary 3D
    labelling of phosphate oxygens made RDKit see new 'stereoisomers' and rebuild the ensemble for most instances."""
    smi = Chem.MolToSmiles(Chem.RemoveHs(mh), isomericSmiles=False)
    return hashlib.sha1(f"{smi}|{charge}|{S.SETTINGS}|cofactor".encode()).hexdigest()[:20]


S.ligand_key = _ligand_key

# CCD ids.  CMP = cAMP; NAI = NADH; NAP = NADP+; NDP = NADPH; FDA = FADH2
LIGANDS = ["ATP", "ADP", "AMP", "CMP", "NAD", "NAI", "NAP", "NDP", "FAD", "FDA", "FMN"]


def accessions(u) -> list:
    return [x.strip() for x in str(u).split(",") if x.strip() and x not in ("None", "nan")]


def select(a) -> pd.DataFrame:
    g = pd.read_parquet(DATA / "geometry.parquet")
    g = g[g.comp_id.isin(a.ligands) & (g.status == "ok") & (g.n_missing.fillna(0) == 0)
          & ~g.covalent.astype("boolean").fillna(False).astype(bool) & (g.buried_frac >= a.min_buried)]
    inst = pd.read_parquet(DATA / "instances.parquet")
    g = g.merge(inst[["pdb_id", "comp_id", "asym_id", "auth_asym_id", "resolution", "rscc"]],
                on=["pdb_id", "comp_id", "asym_id"])
    ch = pd.read_parquet(DATA / "chains.parquet").drop_duplicates(["pdb_id", "auth_asym_id"])
    g = g.merge(ch, on=["pdb_id", "auth_asym_id"], how="left")
    g = g[g.resolution <= a.max_res]
    g["uniprot"] = g.uniprot.fillna("")
    g = g.sort_values(["resolution", "rscc", "pdb_id", "asym_id"], ascending=[True, False, True, True])
    g = g.drop_duplicates(["comp_id", "uniprot", "pdb_id"])  # one copy per entry
    # proteins without accession cannot be grouped or matched to Km: skip them
    g = g[g.uniprot != ""].groupby(["comp_id", "uniprot"], sort=False).head(a.per)
    return g[["pdb_id", "comp_id", "asym_id", "uniprot", "description", "family", "resolution", "rscc",
              "metal_contact", "buried_frac", "n_contact_res"]].reset_index(drop=True)


def cmd_select(a):
    s = select(a)
    t = s.groupby("comp_id").agg(instances=("pdb_id", "size"), proteins=("uniprot", "nunique"),
                                 metal_contact=("metal_contact", "mean"), resolution=("resolution", "median"))
    print(t.round(2).to_string())
    print(f"total: {len(s)} instances, {s.drop_duplicates(['comp_id', 'uniprot']).shape[0]} ligand-protein pairs")


def work(chunk, a):
    return S.run_chunk(chunk, a)


def reference_mol(comp_id: str):
    """Ionised ligand with the correct (CCD ideal-coordinate) stereochemistry: (mol with H, charge)."""
    tmpl = S.ionize(S.ccd_mol(comp_id), False)
    block = gemmi.cif.read(str(S.ccd_path(comp_id)))[0]
    tags = ["pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal"]
    xyz = {r[0].strip('"'): [float(r[1]), float(r[2]), float(r[3])]
           for r in block.find("_chem_comp_atom.", ["atom_id", *tags]) if "?" not in (r[1], r[2], r[3])}
    return S.build_pose_mol(tmpl, np.array([xyz[at.GetProp("_name")] for at in tmpl.GetAtoms()]))


def ref_work(comp_id: str, a):
    mh, q = reference_mol(comp_id)
    return comp_id, S.ensemble_reference(mh, q, S.ligand_key(mh, q), a, S.TMP)


def cmd_run(a):
    if not Path(S.XTB).exists():
        raise SystemExit(f"xtb not found: {S.XTB} (set XTB_EXE)")
    for f in S.REF_DIR.glob("*.lock"):  # left over from an interrupted run
        f.unlink()
    sel = select(a)
    old = [pd.read_parquet(p) for p in PARTS.glob("*.parquet")]
    if old:
        done = pd.concat(old)[["pdb_id", "comp_id", "asym_id"]].drop_duplicates()
        sel = sel.merge(done, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    if a.sample:
        sel = sel.sample(min(a.sample, len(sel)), random_state=0)
    sel = sel[["pdb_id", "comp_id", "asym_id"]]
    ns = argparse.Namespace(max_heavy=60, ionize=True, ionize_amines=False, max_confs=a.max_confs,
                            n_opt=a.n_opt, keep_cif=False)
    print(f"{len(sel)} instances to run, {a.workers} workers; output {OUT}", flush=True)
    t0, done_n, stamp = time.time(), 0, int(time.time())
    with ProcessPoolExecutor(a.workers) as ex:
        # stage 1: one reference ensemble per ligand (slow for the large anions: 10-30 min each)
        for f in as_completed([ex.submit(ref_work, c, ns) for c in sorted(sel.comp_id.unique())]):
            c, r = f.result()
            print(f"  reference {c}: {r.get('n_opt')} of {r.get('n_conf')} conformers optimised  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        # stage 2: the instances
        groups = [(pid, g.to_dict("records")) for pid, g in sel.groupby("pdb_id")]
        futs = {ex.submit(work, groups[i:i + a.chunk], ns): i for i in range(0, len(groups), a.chunk)}
        for f in as_completed(futs):
            res = pd.DataFrame(f.result())
            res.to_parquet(PARTS / f"{stamp}_bulk_{futs[f]:05d}.parquet", index=False)
            done_n += len(res)
            print(f"  {done_n}/{len(sel)} instances  {(time.time() - t0) / 60:.1f} min", flush=True)
    parts = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("*.parquet"))], ignore_index=True)
    parts = parts.drop_duplicates(["pdb_id", "comp_id", "asym_id"], keep="last")
    parts.to_parquet(OUT / "strain.parquet", index=False)
    print(parts.status.value_counts().to_string())


def cmd_km(a):
    import sqlite3
    P = importlib.import_module("08_potency")
    db = sqlite3.connect("file:" + P.DB.replace("\\", "/") + "?mode=ro", uri=True)
    q = """SELECT cs.standard_inchi_key, act.standard_value, cseq.accession, ass.confidence_score, ass.chembl_id,
                  ass.description
           FROM compound_structures cs
           CROSS JOIN molecule_hierarchy h0 ON h0.molregno = cs.molregno
           CROSS JOIN molecule_hierarchy h ON h.parent_molregno = h0.parent_molregno
           CROSS JOIN activities act ON act.molregno = h.molregno
           CROSS JOIN assays ass ON ass.assay_id = act.assay_id
           CROSS JOIN target_dictionary td ON td.tid = ass.tid
           CROSS JOIN target_components tc ON tc.tid = td.tid
           CROSS JOIN component_sequences cseq ON cseq.component_id = tc.component_id
           WHERE cs.standard_inchi_key >= ? AND cs.standard_inchi_key < ?
             AND +act.standard_type = 'Km' AND +act.standard_relation = '=' AND +act.standard_units = 'nM'
             AND act.standard_value > 0 AND ass.confidence_score BETWEEN 8 AND 9
             AND +td.target_type = 'SINGLE PROTEIN' AND ass.variant_id IS NULL
             AND act.data_validity_comment IS NULL"""
    rows = []
    for c in a.ligands:
        key = P.ccd_inchikey(c, "")
        for r in db.execute(q, (key[:14], key[:14] + "~")):
            if r[0][:25] == key[:25]:
                rows.append(dict(comp_id=c, uniprot=r[2], km_uM=r[1] / 1000.0, source="ChEMBL " + r[4],
                                 confidence=r[3], description=r[5]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "km_chembl.csv", index=False)
    print(f"{len(df)} Km values -> {OUT / 'km_chembl.csv'}  (check the descriptions: some 'Km' entries are not "
          f"substrate Km measurements)")


# BRENDA substrate names (compared case-insensitively) per CCD id
BRENDA_NAMES = {
    "ATP": ["ATP", "MgATP", "MgATP2-", "Mg-ATP", "Mg2+-ATP", "ATP4-"],
    "ADP": ["ADP", "MgADP", "MgADP-", "Mg-ADP", "ADP3-"],
    "AMP": ["AMP", "5'-AMP", "adenosine 5'-phosphate", "adenosine 5'-monophosphate"],
    "CMP": ["cAMP", "3',5'-cAMP", "adenosine 3',5'-cyclic phosphate", "adenosine 3',5'-cyclic monophosphate",
            "cyclic AMP", "3',5'-cyclic AMP"],
    "NAD": ["NAD+", "NAD", "beta-NAD+", "beta-NAD"],
    "NAI": ["NADH", "beta-NADH"],
    "NAP": ["NADP+", "NADP", "beta-NADP+", "beta-NADP"],
    "NDP": ["NADPH", "beta-NADPH"],
    "FAD": ["FAD"],
    "FDA": ["FADH2"],
    "FMN": ["FMN", "riboflavin 5'-phosphate"],
}
_MUTANT = re.compile(r"mutant|mutated|mutation|variant|\b[ACDEFGHIKLMNPQRSTVWY]\d{1,4}[ACDEFGHIKLMNPQRSTVWY]\b")


def _ec_organism(accs: list) -> dict:
    """UniProt accession -> (organism, {EC numbers}) from ChEMBL (covers only proteins ChEMBL knows)."""
    import sqlite3
    P = importlib.import_module("08_potency")
    db = sqlite3.connect("file:" + P.DB.replace("\\", "/") + "?mode=ro", uri=True)
    q = """SELECT cs.organism, s.component_synonym FROM component_sequences cs
           CROSS JOIN component_synonyms s ON s.component_id = cs.component_id
           WHERE cs.accession = ? AND s.syn_type = 'EC_NUMBER'"""
    out = {}
    for acc in accs:
        r = db.execute(q, (acc,)).fetchall()
        if r:
            out[acc] = (r[0][0], {x[1] for x in r})
    return out


def cmd_brenda(a):
    src = Path(a.brenda) if a.brenda else next(iter(sorted((OUT / "brenda" / "json").glob("brenda_*.json"))), None)
    if src is None or not src.exists():
        raise SystemExit(f"BRENDA JSON not found; unpack it into {OUT / 'brenda' / 'json'}")
    print(f"reading {src} ...", flush=True)
    data = json.loads(src.read_text(encoding="utf-8"))["data"]
    names = {n.lower(): c for c in a.ligands for n in BRENDA_NAMES.get(c, [])}
    rows, n_mut = [], 0
    for ec, entry in data.items():
        prot = entry.get("protein") or {}
        for k in entry.get("km_value") or []:
            m = re.match(r"\s*([\d.eE+-]+)\s*\{(.*)\}\s*$", k.get("value", ""))
            if not m or m.group(2).strip().lower() not in names:
                continue  # other substrate, or a range / unspecified value
            try:
                mm = float(m.group(1))
            except ValueError:
                continue
            if mm <= 0:
                continue
            comment = k.get("comment") or ""
            low = comment.lower()
            if _MUTANT.search(comment) and not ("wild-type" in low or "wild type" in low or "wildtype" in low):
                n_mut += 1
                continue
            for pid in k.get("proteins") or []:
                pr = prot.get(pid, {})
                rows.append(dict(comp_id=names[m.group(2).strip().lower()], km_uM=mm * 1000.0, ec=ec,
                                 organism=pr.get("organism", ""), accessions=",".join(pr.get("accessions") or []),
                                 substrate=m.group(2), source=f"BRENDA EC {ec} refs {','.join(k.get('references', []))}",
                                 comment=comment))
    km = pd.DataFrame(rows).drop_duplicates(["comp_id", "ec", "organism", "accessions", "km_uM", "source"])
    km.to_csv(OUT / "km_brenda_all.csv", index=False)
    print(f"{len(km)} Km values for these ligands ({n_mut} mutant entries skipped)")
    print(km.comp_id.value_counts().to_string())

    sel = select(a)
    sel = sel.assign(uniprot=sel.uniprot.map(accessions)).explode("uniprot")
    pairs = sel[["comp_id", "uniprot"]].drop_duplicates()
    # 1) the BRENDA protein carries the same UniProt accession
    ex = km.assign(uniprot=km.accessions.str.split(",")).explode("uniprot")
    ex = ex.merge(pairs, on=["comp_id", "uniprot"])
    ex.to_csv(OUT / "km_brenda.csv", index=False)
    # 2) weaker: same EC number and organism, BRENDA entry without accession, pair not matched above
    info = _ec_organism(sorted(pairs.uniprot.unique()))
    cand = pd.DataFrame([dict(uniprot=u, organism_key=o.lower(), ec=e) for u, (o, ecs) in info.items() for e in ecs],
                        columns=["uniprot", "organism_key", "ec"])
    cand = pairs.merge(cand, on="uniprot")
    done = set(map(tuple, ex[["comp_id", "uniprot"]].to_numpy()))
    cand = cand[[(c, u) not in done for c, u in zip(cand.comp_id, cand.uniprot)]]
    weak = km[km.accessions == ""].assign(organism_key=lambda d: d.organism.str.lower())
    weak = weak.merge(cand, on=["comp_id", "ec", "organism_key"])
    weak.to_csv(OUT / "km_brenda_ecorg.csv", index=False)
    for label, d, f in [("UniProt match", ex, "km_brenda.csv"), ("EC + organism match", weak, "km_brenda_ecorg.csv")]:
        n = d.drop_duplicates(["comp_id", "uniprot"])
        print(f"\n{label}: {len(n)} ligand-protein pairs, {len(d)} values -> {OUT / f}")
        if len(n):
            print(n.comp_id.value_counts().to_string())


def cmd_analyse(a):
    st = pd.read_parquet(OUT / "strain.parquet")
    print(st.status.value_counts().to_string())
    st = st[st.status == "ok"].copy()
    # one reference per ligand chemistry (key = stereo SMILES + charge): lowest of ensemble and every relaxation
    st["e_common"] = st.groupby("key").e_ref_eh.transform("min")
    st["delta_e_common"] = (st.e_pose_eh - st.e_common) * S.EH2KCAL
    wrong = st.get("qc_stereo_mismatch", pd.Series(0, index=st.index)).fillna(0).astype(float) > 0
    print(f"instances with stereo differing from the CCD (excluded): {int(wrong.sum())}")
    st = st[~wrong]
    sel_args = argparse.Namespace(**{**vars(a), "per": 10 ** 9, "max_res": 99.0, "min_buried": 0.0})
    d = st.merge(select(sel_args), on=["pdb_id", "comp_id", "asym_id"], how="left")
    d.to_csv(OUT / "instances_strain.csv", index=False)
    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    q = lambda p: (lambda x: x.quantile(p))  # noqa: E731
    say("\nstrain per ligand (kcal/mol; common reference)")
    say(d.groupby("comp_id").agg(n=("pdb_id", "size"), proteins=("uniprot", "nunique"),
                                 dE_q25=("delta_e_common", q(.25)), dE_med=("delta_e_common", q(.5)),
                                 dE_q75=("delta_e_common", q(.75)),
                                 local_med=("local_strain_kcal", q(.5))).round(1).to_string())
    say("\nmedian dE with / without a metal contact (metal not included in the energy)")
    say(d.pivot_table(index="comp_id", columns="metal_contact", values="delta_e_common",
                      aggfunc=q(.5)).round(1).to_string())
    say("\nmedian dE by protein family")
    say(d.pivot_table(index="family", columns="comp_id", values="delta_e_common", aggfunc=q(.5)).round(0)
        .fillna("-").to_string())
    pr = d.groupby(["comp_id", "uniprot"]).agg(
        n=("pdb_id", "size"), dE=("delta_e_common", q(.5)), dE_min=("delta_e_common", "min"),
        local=("local_strain_kcal", q(.5)), sd=("delta_e_common", "std"), resolution=("resolution", q(.5)),
        buried=("buried_frac", q(.5)), metal=("metal_contact", "mean"), family=("family", "first"),
        description=("description", "first")).reset_index()
    pr.to_csv(OUT / "protein_strain.csv", index=False)
    say("\nreproducibility: pooled within-protein SD vs between-protein SD of the median dE")
    for c, g in pr.groupby("comp_id"):
        m = g[g.n > 1]
        if len(m) >= 5:
            say(f"  {c}: within {np.sqrt((m.sd ** 2).mean()):.1f}   between {g.dE.std():.1f}   "
                f"(proteins with >1 structure: {len(m)})")

    kmf = Path(a.km) if a.km else OUT / "km_brenda.csv"
    if kmf.exists() and kmf.stat().st_size > 5:
        km = pd.read_csv(kmf)
        km = km[km.km_uM > 0].copy()
        km["pKm"] = 6 - np.log10(km.km_uM)
        km = km.groupby(["comp_id", "uniprot"]).agg(pKm=("pKm", q(.5)), n_km=("pKm", "size")).reset_index()
        pr2 = pr.assign(uniprot=pr.uniprot.map(accessions)).explode("uniprot")
        m = pr2.merge(km, on=["comp_id", "uniprot"])
        m.to_csv(OUT / f"km_vs_strain_{kmf.stem}.csv", index=False)
        say(f"\nKm table {kmf.name}: {len(km)} ligand-protein values, {len(m)} matched to structures")
        for c, g in [("all", m)] + list(m.groupby("comp_id")):
            if len(g) < 10:
                say(f"  {c}: n={len(g)} - too few for a correlation")
                continue
            for y in ["dE", "dE_min", "local", "buried"]:
                rho = g.pKm.rank().corr(g[y].rank())
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(2000):
                    s = g.iloc[rng.integers(0, len(g), len(g))]
                    bs.append(s.pKm.rank().corr(s[y].rank()))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                say(f"  {c}: n={len(g)}  Spearman pKm vs {y:7s} {rho:+.2f}  [95% CI {lo:+.2f}, {hi:+.2f}]")
    else:
        say(f"\nno Km table found ({kmf})")
    (OUT / f"stats_{kmf.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run", "km", "brenda", "analyse"])
    ap.add_argument("--ligands", nargs="+", default=LIGANDS)
    ap.add_argument("--per", type=int, default=3, help="structures per ligand and protein")
    ap.add_argument("--max-res", type=float, default=2.5)
    ap.add_argument("--min-buried", type=float, default=0.3)
    ap.add_argument("--max-confs", type=int, default=300)
    ap.add_argument("--n-opt", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=3, help="entries per task")
    ap.add_argument("--sample", type=int, default=0, help="random subset (pilot)")
    ap.add_argument("--km", help="CSV with comp_id, uniprot, km_uM, source (default: km_brenda.csv)")
    ap.add_argument("--brenda", help="BRENDA JSON file (default: data/cofactors/brenda/json/brenda_*.json)")
    a = ap.parse_args()
    {"select": cmd_select, "run": cmd_run, "km": cmd_km, "brenda": cmd_brenda, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
