"""Draw the ligands with the highest xtb strain (default: dE > 30 kcal/mol) as a grid + csv."""
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--min-de", type=float, default=30.0)
ap.add_argument("--out", default=str(DATA / "strain_outliers"))
a = ap.parse_args()

s = pd.read_parquet(DATA / "strain.parquet")
s = s[(s.status == "ok") & (s.delta_e_kcal > a.min_de)]
d = pd.read_parquet(DATA / "survey.parquet")[["pdb_id", "comp_id", "asym_id", "smiles", "resolution", "rscc",
                                              "n_rotb", "formal_charge"]]
m = s.merge(d, on=["pdb_id", "comp_id", "asym_id"], suffixes=("", "_s")).sort_values("delta_e_kcal", ascending=False)
m.drop(columns=[c for c in m if c.endswith("_s")]).to_csv(a.out + ".csv", index=False)
mols = [Chem.MolFromSmiles(x) for x in m.smiles]
leg = [f"{r.pdb_id} {r.comp_id}  dE {r.delta_e_kcal:.0f} / loc {r.local_strain_kcal:.0f}\nRSCC {r.rscc:.2f}  res {r.resolution:.1f}  q {int(r.formal_charge)}"
       for r in m.itertuples()]
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(380, 300), legends=leg)
img.save(a.out + ".png")
print(len(m), "ligands ->", a.out + ".png")
