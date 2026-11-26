"""Step 1d - merge instances + components + geometry + family -> data/survey.parquet / .csv"""
import sys

import pandas as pd

from common import DATA

KEEP_METALS = "--keep-metals" in sys.argv


def main():
    inst = pd.read_parquet(DATA / "instances.parquet")
    comps = pd.read_parquet(DATA / "components.parquet")
    geo = pd.read_parquet(DATA / "geometry.parquet").drop(columns=["n_rotb"], errors="ignore")
    print(geo.status.value_counts().to_string())
    chains = pd.read_parquet(DATA / "chains.parquet")

    df = inst.merge(comps.drop(columns=["ok", "error"], errors="ignore"), on="comp_id", how="inner")
    df = df.merge(geo, on=["pdb_id", "comp_id", "asym_id"], how="left")
    df = df[df.status == "ok"].copy()
    # no metals at all: ligands that contain a metal or coordinate one (use --keep-metals to keep them)
    if not KEEP_METALS:
        n0 = len(df)
        df = df[~df.has_metal.fillna(False).astype(bool) & ~df.metal_contact.fillna(False).astype(bool)].copy()
        print(f"metal filter: removed {n0 - len(df)} of {n0} instances")
    # unresolved template atoms (0 for complete ligands; older geometry rows lack the column)
    df["n_missing"] = (df.n_atoms_template - df.n_atoms_model).fillna(0).astype(int)
    df["frac_missing"] = df.n_missing / df.n_atoms_template
    ch = chains.rename(columns={"auth_asym_id": "main_chain"})
    df = df.merge(ch[["pdb_id", "main_chain", "description", "uniprot", "family"]],
                  on=["pdb_id", "main_chain"], how="left")
    sp = DATA / "strain.parquet"
    if sp.exists():  # xtb strain energies from 06_strain.py (only for the instances that were run)
        st = pd.read_parquet(sp)
        qc_cols = [c for c in st.columns if c.startswith("qc_")]  # geometry QC flags (06_strain.py)
        st = st[st.status == "ok"][["pdb_id", "comp_id", "asym_id", "delta_e_kcal", "local_strain_kcal",
                                    "e_pose_eh", "e_relax_eh", "e_ref_eh", "n_conf"] + qc_cols]
        df = df.merge(st, on=["pdb_id", "comp_id", "asym_id"], how="left")
        if "qc_pass" in df:
            df["qc_pass"] = df["qc_pass"].astype("boolean")
        print(f"strain energies merged for {df.delta_e_kcal.notna().sum()} instances")
    df["family"] = df["family"].fillna("Other")
    df["ligand_label"] = df.pdb_id + "_" + df.comp_id
    df.to_parquet(DATA / "survey.parquet", index=False)
    df.to_csv(DATA / "survey.csv", index=False)
    print(f"{len(df)} ligand instances, {df.comp_id.nunique()} comps")
    print(df.family.value_counts().to_string())
    print(df[["buried_frac", "n_rotb", "n_rotb_buried", "mw", "logp"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
