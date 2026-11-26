"""Step 1e - precompute UMAP embeddings for the app (run after 04_merge.py).

Two embeddings, both computed on ALL instances so filters in the app never change the layout:
  chem : Morgan fingerprint (r=2, 2048 bit, Jaccard) per unique ligand id -> mapped to instances
  desc : scaled 2D descriptors + burial / rotatable-bond features per instance

Output: data/umap.parquet  (pdb_id, comp_id, asym_id, umap_chem_x/y, umap_desc_x/y)
Usage:  python 05_umap.py [--data-dir data]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler

DESC = ["mw", "logp", "tpsa", "hbd", "hba", "n_rotb", "n_rings", "n_arom_rings", "fsp3",
        "formal_charge", "buried_frac", "frac_buried_atoms", "n_rotb_buried", "n_contact_res"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    a = ap.parse_args()
    d = Path(a.data_dir)
    df = pd.read_parquet(d / "survey.parquet")

    # --- chemistry space (unique comps)
    comps = df.drop_duplicates("comp_id")[["comp_id", "smiles"]].reset_index(drop=True)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = np.zeros((len(comps), 2048), dtype=bool)
    for i, s in enumerate(comps.smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps[i] = gen.GetFingerprintAsNumPy(m).astype(bool)
    print(f"UMAP chem on {len(comps)} unique ligands ...")
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard", random_state=42).fit_transform(fps)
    comps["umap_chem_x"], comps["umap_chem_y"] = xy[:, 0], xy[:, 1]

    # --- descriptor space (instances)
    X = df[DESC].astype(float)
    X = X.fillna(X.quantile(0.5))  # (DataFrame.median() crashes on this Windows numpy build)
    X = StandardScaler().fit_transform(X)
    print(f"UMAP desc on {len(df)} instances ...")
    xy2 = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(X)

    out = df[["pdb_id", "comp_id", "asym_id"]].copy()
    out = out.merge(comps[["comp_id", "umap_chem_x", "umap_chem_y"]], on="comp_id", how="left")
    out["umap_desc_x"], out["umap_desc_y"] = xy2[:, 0], xy2[:, 1]
    out.to_parquet(d / "umap.parquet", index=False)
    print("wrote", d / "umap.parquet")


if __name__ == "__main__":
    main()
