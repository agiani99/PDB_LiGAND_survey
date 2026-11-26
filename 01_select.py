"""Step 1a - select X-ray entries (res < 5 A) and list candidate ligand instances.

Outputs (data/):
  instances.parquet  one row per ligand instance (pdb_id, comp_id, asym_id, auth chain/seq, MW, RSCC...)
  chains.parquet     one row per polymer chain (pdb_id, auth_asym_id, description, uniprot, family)

Usage:  python 01_select.py [--limit N] [--agents-file ids.txt]
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import agents
import family
from common import DATA, MAX_RES, MW_MAX, MW_MIN, http_get, res_class, _session

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GQL = "https://data.rcsb.org/graphql"

QUERY = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined", "operator": "less", "value": MAX_RES}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.nonpolymer_entity_count", "operator": "greater", "value": 0}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein", "operator": "greater", "value": 0}},
        ],
    },
    "return_type": "entry",
    "request_options": {"return_all_hits": True},
}

GQL_Q = """
query($ids:[String!]!){ entries(entry_ids:$ids){
  rcsb_id
  rcsb_entry_info{ resolution_combined }
  polymer_entities{
    entity_poly{ rcsb_entity_polymer_type }
    rcsb_polymer_entity{ pdbx_description }
    rcsb_polymer_entity_container_identifiers{
      auth_asym_ids
      reference_sequence_identifiers{ database_name database_accession } }
  }
  nonpolymer_entities{
    nonpolymer_comp{ chem_comp{ id formula_weight } }
    nonpolymer_entity_instances{
      rcsb_nonpolymer_entity_instance_container_identifiers{ asym_id auth_asym_id auth_seq_id comp_id }
      rcsb_nonpolymer_instance_validation_score{ RSCC ranking_model_fit }
    }
  }
}}
"""


def all_ids() -> list[str]:
    r = _session.post(SEARCH, json=QUERY, timeout=120)
    r.raise_for_status()
    return [h["identifier"] for h in r.json()["result_set"]]


def fetch_batch(ids: list[str]):
    for attempt in range(4):
        r = _session.post(GQL, json={"query": GQL_Q, "variables": {"ids": ids}}, timeout=120)
        if r.status_code == 200 and "data" in r.json() and r.json()["data"]:
            return r.json()["data"]["entries"]
    print("batch failed", ids[:2], file=sys.stderr)
    return []


def parse(entries, excluded: set[str]):
    inst, chains = [], []
    for e in entries:
        if not e:
            continue
        pid = e["rcsb_id"]
        res = (e["rcsb_entry_info"]["resolution_combined"] or [None])[0]
        for pe in e.get("polymer_entities") or []:
            if (pe["entity_poly"] or {}).get("rcsb_entity_polymer_type") != "Protein":
                continue
            desc = (pe["rcsb_polymer_entity"] or {}).get("pdbx_description") or ""
            cid = pe["rcsb_polymer_entity_container_identifiers"]
            ups = sorted({x["database_accession"] for x in (cid.get("reference_sequence_identifiers") or [])
                          if x["database_name"] == "UniProt"})
            for ch in cid["auth_asym_ids"] or []:
                chains.append(dict(pdb_id=pid, auth_asym_id=ch, description=desc,
                                   uniprot=",".join(ups), family=family.assign(desc)))
        for ne in e.get("nonpolymer_entities") or []:
            cc = ne["nonpolymer_comp"]["chem_comp"]
            comp, mw = cc["id"], cc["formula_weight"]
            if comp in excluded or mw is None or not (MW_MIN < mw < MW_MAX):
                continue
            for it in ne.get("nonpolymer_entity_instances") or []:
                ids_ = it["rcsb_nonpolymer_entity_instance_container_identifiers"]
                val = it.get("rcsb_nonpolymer_instance_validation_score") or [{}]
                val = val[0] if val else {}
                inst.append(dict(pdb_id=pid, comp_id=comp, mw_ccd=mw, resolution=res,
                                 res_class=res_class(res) if res is not None else "NA",
                                 asym_id=ids_["asym_id"], auth_asym_id=ids_["auth_asym_id"],
                                 auth_seq_id=ids_["auth_seq_id"],
                                 rscc=val.get("RSCC"), model_fit_rank=val.get("ranking_model_fit")))
    return inst, chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N entries (testing)")
    ap.add_argument("--agents-file", default=None)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    excluded = agents.load(a.agents_file)
    ids = all_ids()
    print(f"{len(ids)} X-ray entries (res<{MAX_RES}) with protein + non-polymer")
    if a.limit:
        ids = ids[: a.limit]
    batches = [ids[i:i + a.batch] for i in range(0, len(ids), a.batch)]
    try:
        from tqdm import tqdm
    except ImportError:  # minimal fallback if tqdm is not installed
        def tqdm(it, total=None, **kw):
            for k, x in enumerate(it, 1):
                if k % 20 == 0 or k == total:
                    print(f"  {k}/{total} batches", flush=True)
                yield x

    inst, chains = [], []
    with ThreadPoolExecutor(a.threads) as ex:
        bar = tqdm(ex.map(fetch_batch, batches), total=len(batches), desc="entries (batches of %d)" % a.batch,
                   unit="batch")
        for ents in bar:
            i, c = parse(ents, excluded)
            inst += i
            chains += c
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(instances=len(inst))
    inst = pd.DataFrame(inst).drop_duplicates()
    chains = pd.DataFrame(chains).drop_duplicates()
    inst.to_parquet(DATA / "instances.parquet", index=False)
    chains.to_parquet(DATA / "chains.parquet", index=False)
    print(f"instances: {len(inst)}  unique comps: {inst.comp_id.nunique()}  entries: {inst.pdb_id.nunique()}")
    print(inst.res_class.value_counts().to_string())


if __name__ == "__main__":
    main()
