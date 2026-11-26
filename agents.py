"""Crystallisation / cryo / buffer additives to exclude (CCD component ids).

Built-in seed list of the usual suspects.  To extend it with the Hauptman-Woodward
crystallizationDatabase, map its chemical names to CCD ids and pass a text file (one id
per line) with  `01_select.py --agents-file my_ids.txt`.
"""
from pathlib import Path

BUILTIN = {
    # polyols / PEGs / cryoprotectants
    "GOL", "EDO", "PEG", "PGE", "PG4", "P6G", "PE4", "PE3", "PE5", "PE8", "1PE", "2PE", "12P", "15P",
    "PGO", "PGR", "MPD", "MRD", "IPA", "EOH", "MOH", "ACT", "ACY", "FMT", "DMS", "DMF", "BU3", "BTB",
    "TRS", "TAM", "EPE", "MES", "HEZ", "OCT", "OLC", "BOG", "LDA", "LMT", "DDM", "DGA", "C8E", "LHG",
    "PEU", "PE6", "PG0", "PG5", "PG6", "7PE", "XPE", "NHE", "CXS", "CIT", "FLC", "TLA", "MLI", "MLT",
    "SIN", "TAR", "BCT", "CO3", "IMD", "BME", "DTT", "DTD", "TCE", "GBL", "TFA",
    # inorganic ions / small
    "SO4", "PO4", "NO3", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE", "FE2", "NI", "CU", "CO", "CD",
    "BR", "IOD", "SCN", "AZI", "NH4", "HG", "PT", "AU", "CS", "RB", "SR", "BA", "LI", "F", "OH", "PI",
    "GD", "YB", "TB", "EU", "SM", "PB", "TL", "W", "MO", "VO4", "AF3", "ALF", "BEF", "MGF", "SO3",
    "CAC", "ARS", "HOH", "DOD",
}


def load(extra_file: str | None = None) -> set[str]:
    ids = set(BUILTIN)
    if extra_file:
        ids |= {ln.strip().upper() for ln in Path(extra_file).read_text().splitlines() if ln.strip()}
    return ids
