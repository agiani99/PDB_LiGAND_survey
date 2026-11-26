"""Protein family assignment from the RCSB polymer entity description.

Keyword rules adapted from TautomerQSAR/target_family_analysis.py (first match wins).
Changes vs. the original: CYP added, and Nuclear receptor / CYP / Protease are tested
BEFORE the generic 'receptor' GPCR keyword (otherwise 'estrogen receptor' became GPCR).
This is a first-pass heuristic; refine later with UniProt / ChEMBL target classes.
"""
RULES: list[tuple[str, list[str]]] = [
    ("CYP", ["cytochrome p450", "cytochrome p-450", "cyp", "p450"]),
    ("Kinase", ["kinase"]),
    ("Nuclear receptor", ["nuclear receptor", "estrogen receptor", "androgen receptor",
                          "glucocorticoid", "mineralocorticoid", "progesterone receptor",
                          "thyroid hormone receptor", "retinoic acid receptor", "retinoid x receptor",
                          "peroxisome proliferator", "vitamin d", "farnesoid", "liver x receptor",
                          "rar-", "ppar", "ror-gamma", "nuclear hormone"]),
    ("Protease", ["protease", "peptidase", "thrombin", "trypsin", "cathepsin", "caspase",
                  "elastase", "renin", "plasmin", "chymotrypsin", "kallikrein", "subtilisin",
                  "secretase", "factor xa", "coagulation factor", "metalloproteinase",
                  "thermolysin", "proteinase", "pepsin"]),
    ("GPCR", ["g-protein coupled", "g protein-coupled", "gpcr", "adrenergic", "muscarinic",
              "dopamine receptor", "serotonin receptor", "opioid receptor", "histamine receptor",
              "adenosine receptor", "cannabinoid", "chemokine receptor", "neuropeptide",
              "sphingosine", "rhodopsin", "orexin", "angiotensin", "glucagon", "smoothened",
              "metabotropic", "frizzled", "receptor"]),
    ("Ion channel", ["channel", "nicotinic", "gaba", "glutamate receptor", "glycine receptor"]),
    ("Transporter", ["transporter", "pump", "atpase", "solute carrier", "permease"]),
    ("Phosphodiesterase", ["phosphodiesterase", "pde"]),
    ("Oxidoreductase", ["reductase", "oxidase", "dehydrogenase", "monooxygenase", "hydroxylase",
                        "peroxidase", "cyclooxygenase", "lipoxygenase", "oxidoreductase"]),
    ("Transferase", ["transferase"]),
    ("Hydrolase", ["hydrolase", "esterase", "lipase", "amidase", "phosphatase", "glucosidase",
                   "neuraminidase", "deacetylase", "lactamase"]),
    ("Synthase/Lyase", ["synthase", "synthetase", "lyase", "isomerase", "epimerase", "racemase"]),
    ("Chaperone/Binding", ["heat shock", "hsp90", "bromodomain", "chaperone", "binding protein"]),
    ("Other enzyme", ["ase"]),
]
ORDER = [r[0] for r in RULES] + ["Other"]


def assign(description: str) -> str:
    d = (description or "").lower()
    for fam, kws in RULES:
        if any(k in d for k in kws):
            return fam
    return "Other"
