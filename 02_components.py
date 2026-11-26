"""Step 1b - per-component (CCD) SMILES + 2D descriptors for every unique ligand id.

SMILES carry stereochemistry (R/S and E/Z) taken from the CCD ideal coordinates; if the resulting InChIKey does not
match the CCD InChIKey, a CCD isomeric SMILES that does match is used instead.  Column stereo_check records which:
  ideal_3d | ccd_smiles | mismatch (3D-derived kept, stereo not confirmed) | no_ccd_key

Output: data/components.parquet
"""
from concurrent.futures import ThreadPoolExecutor

import gemmi
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from common import DATA, ccd_mol, ccd_path, rotatable_bonds

RDLogger.DisableLog("rdApp.*")


def _block(comp_id: str):
    doc = gemmi.cif.read(str(ccd_path(comp_id)))
    return doc[0]


def _mol_from_coords(m, block, tags: list):
    """Copy of the CCD mol with stereochemistry perceived from one set of CCD coordinates (None if incomplete)."""
    xyz = {}
    for r in block.find("_chem_comp_atom.", ["atom_id", *tags]):
        try:
            xyz[r[0].strip('"')] = [float(r[1]), float(r[2]), float(r[3])]
        except ValueError:  # '?' = coordinate not given
            pass
    out = Chem.Mol(m)
    if any(a.GetProp("_name") not in xyz for a in out.GetAtoms()):
        return None
    conf = Chem.Conformer(out.GetNumAtoms())
    for a in out.GetAtoms():
        conf.SetAtomPosition(a.GetIdx(), xyz[a.GetProp("_name")])
    out.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(out)
    return out


def stereo_smiles(comp_id: str, m) -> tuple[str, str]:
    """(isomeric SMILES, stereo_check)."""
    block = _block(comp_id)
    desc = [(gemmi.cif.as_string(r[0]).upper(), gemmi.cif.as_string(r[1]))
            for r in block.find("_pdbx_chem_comp_descriptor.", ["type", "descriptor"])]
    ref = next((d[:25] for t, d in desc if t == "INCHIKEY"), None)  # skeleton + stereo, not protonation

    mol3d = None
    for tags in (["pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal"],
                 ["model_Cartn_x", "model_Cartn_y", "model_Cartn_z"]):
        cand = _mol_from_coords(m, block, tags)
        if cand is None:
            continue
        mol3d = mol3d or cand
        if ref and Chem.MolToInchiKey(cand)[:25] == ref:
            return Chem.MolToSmiles(cand), "ideal_3d"
    if ref:
        for t, d in sorted(desc, key=lambda x: x[0] != "SMILES_CANONICAL"):
            if t.startswith("SMILES"):
                sm = Chem.MolFromSmiles(d)
                if sm is not None and Chem.MolToInchiKey(sm)[:25] == ref:
                    return Chem.MolToSmiles(sm), "ccd_smiles"
    return Chem.MolToSmiles(mol3d if mol3d is not None else m), "mismatch" if ref else "no_ccd_key"


def describe(comp_id: str) -> dict:
    row = dict(comp_id=comp_id, ok=False)
    try:
        m = ccd_mol(comp_id)
        if m is None:
            return row
        try:
            smiles, check = stereo_smiles(comp_id, m)
        except Exception:  # noqa: BLE001 - never lose a ligand because of stereo perception
            smiles, check = Chem.MolToSmiles(m), "mismatch"
        row.update(
            ok=True,
            smiles=smiles,
            stereo_check=check,
            mw=Descriptors.MolWt(m),
            heavy_atoms=m.GetNumHeavyAtoms(),
            logp=Crippen.MolLogP(m),
            tpsa=rdMolDescriptors.CalcTPSA(m),
            hbd=Lipinski.NumHDonors(m),
            hba=Lipinski.NumHAcceptors(m),
            n_rotb=len(rotatable_bonds(m)),
            n_rings=rdMolDescriptors.CalcNumRings(m),
            n_arom_rings=rdMolDescriptors.CalcNumAromaticRings(m),
            fsp3=rdMolDescriptors.CalcFractionCSP3(m),
            n_stereo=len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)),
            formal_charge=Chem.GetFormalCharge(m),
            has_metal=any(a.GetSymbol() in {"Zn", "Fe", "Cu", "Mn", "Co", "Ni", "Mg", "Ca", "Pt", "Ru", "Hg", "Cd", "Mo", "V", "Ir", "Rh", "Os", "Pd", "Au", "Ag"}
                          for a in m.GetAtoms()),
            n_hetero=sum(a.GetSymbol() not in ("C", "H") for a in m.GetAtoms()),
        )
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)[:100]
    return row


def main():
    inst = pd.read_parquet(DATA / "instances.parquet")
    comps = sorted(inst.comp_id.unique())
    print(f"{len(comps)} unique components")
    try:
        from tqdm import tqdm
    except ImportError:  # minimal fallback if tqdm is not installed
        def tqdm(it, total=None, **kw):
            for i, x in enumerate(it, 1):
                if i % 200 == 0 or i == total:
                    print(f"  {i}/{total}", flush=True)
                yield x

    with ThreadPoolExecutor(8) as ex:
        rows = list(tqdm(ex.map(describe, comps), total=len(comps), desc="components", unit="lig"))
    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "components.parquet", index=False)
    print(f"ok: {int(df.ok.sum())}/{len(df)}")
    print(df.stereo_check.value_counts().to_string())


if __name__ == "__main__":
    main()
