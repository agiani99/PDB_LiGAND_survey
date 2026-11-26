"""PDB ligand survey - Streamlit app.   Run:  streamlit run app.py

Data dir: ./data (override with env SURVEY_DATA).  Needs survey.parquet (04_merge.py);
umap.parquet (05_umap.py) is optional - without it a PCA is shown instead.
Optional strain column `delta_e_kcal` (xtb/DFT step, kcal/mol) unlocks the energy colourings.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bokeh.embed import file_html
from bokeh.models import (BooleanFilter, CDSView, ColorBar, ColumnDataSource, HoverTool,
                          LinearColorMapper, Range1d)
from bokeh.palettes import Turbo256, Viridis256
from bokeh.plotting import figure
from bokeh.resources import INLINE

import family as fam
import molserver

st.set_page_config(page_title="PDB Ligand Survey", layout="wide")
DATA = Path(os.environ.get("SURVEY_DATA", Path(__file__).resolve().parent / "data"))

RES_ORDER = ["high (<2)", "medium (2-3.5)", "low (3.5-5)"]
TRIAGE_BINS = [-np.inf, 5, 10, 20, np.inf]  # kcal/mol; edit to taste
TRIAGE_LABELS = ["<5 (low strain)", "5-10 (moderate)", "10-20 (high)", ">20 (very high)"]
_DISTINCT = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
             "#bcbd22", "#393b79", "#ad494a", "#637939", "#a55194", "#7f7f7f"]
FAMILY_COLORS = {f_: _DISTINCT[i % len(_DISTINCT)] for i, f_ in enumerate(fam.ORDER)}
FAMILY_COLORS["Other"] = "#b0b0b0"
RES_COLORS = {"high (<2)": "#2a9d8f", "medium (2-3.5)": "#e9c46a", "low (3.5-5)": "#e76f51"}
TRIAGE_COLORS = dict(zip(TRIAGE_LABELS, ["#2a9d8f", "#e9c46a", "#f4a261", "#d62828"]))

NUMERIC_LABELS = {
    "n_rotb": "# rotatable bonds", "n_rotb_buried": "# buried rotatable bonds",
    "buried_frac": "buried fraction (SASA)", "frac_buried_atoms": "fraction buried atoms",
    "n_buried_atoms": "# buried atoms", "delta_e_kcal": "ΔE strain (kcal/mol)",
    "mw": "MW", "logp": "logP", "tpsa": "TPSA", "hbd": "HBD", "hba": "HBA", "heavy_atoms": "heavy atoms",
    "n_rings": "# rings", "n_arom_rings": "# aromatic rings", "fsp3": "Fsp3",
    "n_contact_res": "# contact residues", "resolution": "resolution (Å)", "rscc": "RSCC",
}
CATEGORICAL = {"family": "protein family", "res_class": "resolution class", "delta_triage": "ΔE triage"}


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading survey ...")
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "survey.parquet")
    up = DATA / "umap.parquet"
    if up.exists():
        df = df.merge(pd.read_parquet(up), on=["pdb_id", "comp_id", "asym_id"], how="left")
    else:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        cols = [c for c in ["mw", "logp", "tpsa", "hbd", "hba", "n_rotb", "n_rings", "fsp3",
                            "buried_frac", "n_rotb_buried"] if c in df]
        X = df[cols].astype(float)
        # NB: DataFrame.median() hard-crashes in worker threads on this Windows numpy build -> quantile
        X = StandardScaler().fit_transform(X.fillna(X.quantile(0.5)))
        xy = PCA(2, random_state=0).fit_transform(X)
        df["umap_desc_x"], df["umap_desc_y"] = xy[:, 0], xy[:, 1]
        df.attrs["pca_only"] = True
    if "delta_e_kcal" in df:
        df["delta_triage"] = pd.cut(df.delta_e_kcal, TRIAGE_BINS, labels=TRIAGE_LABELS).astype(str)
        df.loc[df.delta_e_kcal.isna(), "delta_triage"] = "n/a"
    df["family"] = df["family"].fillna("Other")
    return df


@st.cache_resource
def mol_port() -> int:
    return molserver.start()


if not (DATA / "survey.parquet").exists():
    st.error(f"`{DATA / 'survey.parquet'}` not found - run steps 01-04 first (or set SURVEY_DATA).")
    st.stop()

df = load()
pca_only = bool(df.attrs.get("pca_only")) or "umap_chem_x" not in df
molserver.SMILES.update(dict(zip(df.comp_id, df.smiles)))
port = mol_port()

# ----------------------------------------------------------------------------- sidebar filters
with st.sidebar:
    st.header("Filters")
    res_sel = st.multiselect("Resolution class", RES_ORDER, default=RES_ORDER)
    fams = [f for f in fam.ORDER if f in set(df.family)]
    fam_sel = st.multiselect("Protein family", fams, default=fams)
    mw_lo, mw_hi = float(df.mw.min()), float(df.mw.max())
    mw_rng = st.slider("MW", mw_lo, mw_hi, (mw_lo, mw_hi))
    bur_rng = st.slider("Buried fraction", 0.0, 1.0, (0.3, 1.0))
    only_complete = st.checkbox("Only fully resolved ligands (no missing atoms)", False,
                                help="Off: ligands with up to 10% unresolved atoms are included; their buried "
                                     "fraction / buried rotatable bonds are computed on the resolved atoms.")
    qc_ok = st.checkbox("Drop strain outliers failing geometry QC (bond/angle errors, clashes, wrong chirality)",
                        False) if "qc_pass" in df else False
    druglike = st.checkbox("Drug-like only (MW 200-700, logP -3..6, HBD<=6, HBA<=12, TPSA<=200, rotb<=15, ring)", True)
    rotb_rng = st.slider("# rotatable bonds", 0, int(df.n_rotb.max()), (0, int(df.n_rotb.max())))
    if "delta_e_kcal" in df:
        tri_sel = st.multiselect("ΔE triage", TRIAGE_LABELS + ["n/a"], default=TRIAGE_LABELS + ["n/a"])
    st.subheader("Quality / scope")
    drop_cov = st.checkbox("Exclude covalent ligands", True)
    drop_metal = st.checkbox("No metals (ligand contains a metal or touches one)", True)
    min_rscc = st.slider("Min RSCC (0 = off)", 0.0, 1.0, 0.0, 0.05)
    excl = st.text_input("Exclude ligand ids (comma separated)", "HEM,CLR,CLA,PLM,STE,OLA")
    dedupe = st.selectbox("Redundancy", ["all instances", "one per ligand id + family", "one per ligand id"])
    st.subheader("Display")
    max_pts = st.slider("Max points plotted", 2000, 100000, 30000, 1000)
    alpha = st.slider("Point opacity", 0.1, 1.0, 0.6, 0.05)
    size = st.slider("Point size", 2, 14, 6)

m = df.res_class.isin(res_sel) & df.family.isin(fam_sel)
m &= df.mw.between(*mw_rng) & df.buried_frac.between(*bur_rng) & df.n_rotb.between(*rotb_rng)
if qc_ok:
    m &= df.qc_pass.fillna(True).astype(bool)
if only_complete and "n_missing" in df:
    m &= df.n_missing == 0
if druglike:
    m &= (df.mw.between(200, 700) & df.logp.between(-3, 6) & (df.hbd <= 6) & (df.hba <= 12)
          & (df.tpsa <= 200) & (df.n_rotb <= 15) & (df.n_rings >= 1))
if "delta_e_kcal" in df:
    m &= df.delta_triage.isin(tri_sel)
if drop_cov:
    m &= ~df.covalent.fillna(False)
if drop_metal:
    m &= ~df.metal_contact.fillna(False) & ~df.has_metal.fillna(False)
if min_rscc > 0:
    m &= df.rscc.fillna(0) >= min_rscc
ex = {x.strip().upper() for x in excl.split(",") if x.strip()}
if ex:
    m &= ~df.comp_id.isin(ex)
f = df[m]
if dedupe == "one per ligand id + family":
    f = f.sort_values("resolution").drop_duplicates(["comp_id", "family"])
elif dedupe == "one per ligand id":
    f = f.sort_values("resolution").drop_duplicates("comp_id")
n_all = len(f)
if n_all > max_pts:
    f = f.sample(max_pts, random_state=0)
f = f.reset_index(drop=True)


# ----------------------------------------------------------------------------- plotting
def color_options() -> dict[str, str]:
    opts = {CATEGORICAL[k]: k for k in CATEGORICAL if k in f}
    opts.update({NUMERIC_LABELS[k]: k for k in NUMERIC_LABELS if k in f and f[k].notna().any()})
    return opts


def hover_tool(port: int) -> HoverTool:
    de = "<br>ΔE: <b>@delta_e_kcal{0.1} kcal/mol</b>" if "delta_e_kcal" in f else ""
    html = f"""
    <div style="display:flex;gap:10px;font:12px sans-serif;max-width:430px">
      <img src="http://127.0.0.1:{port}/mol/@comp_id.svg" width="220" height="160"
           style="background:#fff;border:1px solid #ccc;border-radius:4px">
      <div><b style="font-size:14px">@comp_id</b> &nbsp;<span style="color:#666">@pdb_id</span><br>
      @family<br>res @resolution{{0.00}} Å (@res_class)<br>
      MW @mw{{0.0}} · logP @logp{{0.0}}<br>
      rotB @n_rotb (buried @n_rotb_buried)<br>
      buried @buried_frac{{0.00}}{de}</div>
    </div>"""
    return HoverTool(tooltips=html)


def make_figure(xcol: str, ycol: str, color_by: str, xlabel: str, ylabel: str, log: tuple[bool, bool] = (False, False)):
    src = ColumnDataSource(f)
    # wheel_zoom is available in the toolbar but not active, so the page can still be scrolled over the plot
    p = figure(width=1000, height=640, tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select",
               output_backend="webgl",
               x_axis_type="log" if log[0] else "linear", y_axis_type="log" if log[1] else "linear")
    p.xaxis.axis_label, p.yaxis.axis_label = xlabel, ylabel
    p.add_tools(hover_tool(port))
    kw = dict(x=xcol, y=ycol, size=size, alpha=alpha, line_color=None)
    if color_by in CATEGORICAL:
        cmap = {"family": FAMILY_COLORS, "res_class": RES_COLORS, "delta_triage": TRIAGE_COLORS}[color_by]
        cats = [c for c in cmap if c in set(f[color_by])] + sorted(set(f[color_by]) - set(cmap))
        for i, c in enumerate(cats):
            view = CDSView(filter=BooleanFilter((f[color_by] == c).tolist()))
            p.scatter(source=src, view=view, color=cmap.get(c, "#999999"), legend_label=c, **kw)
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "11px"
        p.add_layout(p.legend[0], "right")
    else:
        v = f[color_by].astype(float)
        lo, hi = np.nanpercentile(v, [1, 99]) if v.notna().any() else (0, 1)
        pal = Turbo256 if color_by in ("delta_e_kcal",) else Viridis256
        mapper = LinearColorMapper(palette=pal, low=lo, high=hi if hi > lo else lo + 1, nan_color="#cccccc")
        p.scatter(source=src, color={"field": color_by, "transform": mapper}, **kw)
        p.add_layout(ColorBar(color_mapper=mapper, title=NUMERIC_LABELS.get(color_by, color_by)), "right")
    return p


def show(p, height=760):
    components.html(file_html(p, INLINE, "plot"), height=height, scrolling=False)


# ----------------------------------------------------------------------------- layout
st.title("PDB ligand survey")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ligand instances (filtered)", f"{n_all:,}")
c2.metric("Unique ligand ids", f"{f.comp_id.nunique():,}")
c3.metric("PDB entries", f"{f.pdb_id.nunique():,}")
c4.metric("Plotted", f"{len(f):,}")
if f.empty:
    st.warning("No ligands match the filters.")
    st.stop()

tab_umap, tab_xy, tab_dist, tab_tab = st.tabs(["UMAP", "Scatter explorer", "Distributions by family", "Table"])
copts = color_options()

with tab_umap:
    a, b = st.columns([1, 2])
    color_label = a.selectbox("Colour by", list(copts), index=list(copts).index("protein family"))
    spaces = {"Descriptors + burial (per instance)": "desc"}
    if not pca_only:
        spaces["Chemistry: Morgan FP (per ligand id)"] = "chem"
    space = b.radio("Embedding", list(spaces), horizontal=True)
    if pca_only:
        st.info("`umap.parquet` not found - showing a PCA of descriptors. Run `python 05_umap.py` for UMAP.")
    key = spaces[space]
    xc, yc = f"umap_{key}_x", f"umap_{key}_y"
    if key == "chem":  # instances of one ligand id share coordinates -> small jitter so they stay visible
        jit = st.slider("Jitter (fraction of axis range)", 0.0, 0.03, 0.004, 0.001)
        rng = np.random.default_rng(0)
        for c in (xc, yc):
            f[c] = f[c] + rng.normal(0, jit * (f[c].max() - f[c].min() + 1e-9), len(f))
    show(make_figure(xc, yc, copts[color_label], "UMAP 1" if not pca_only else "PC1",
                     "UMAP 2" if not pca_only else "PC2"))
    st.caption("Hover a point for the structure (drawn on the fly from the ligand SMILES) and key values. "
               "Click legend entries to hide classes; lasso-select is available in the toolbar.")

with tab_xy:
    nums = {NUMERIC_LABELS[k]: k for k in NUMERIC_LABELS if k in f}
    a, b, c = st.columns(3)
    xl = a.selectbox("X", list(nums), index=list(nums).index("MW"))
    yl = b.selectbox("Y", list(nums), index=list(nums).index("buried fraction (SASA)"))
    cl = c.selectbox("Colour", list(copts), index=list(copts).index("protein family"), key="xy_col")
    show(make_figure(nums[xl], nums[yl], copts[cl], xl, yl))

with tab_dist:
    from bokeh.models import FactorRange
    nums = {NUMERIC_LABELS[k]: k for k in NUMERIC_LABELS if k in f}
    vl = st.selectbox("Property", list(nums), index=list(nums).index("buried fraction (SASA)"))
    v = nums[vl]
    g = f.groupby("family")[v].describe()[["count", "mean", "25%", "50%", "75%"]].round(2)
    g = g.loc[[x for x in fam.ORDER if x in g.index]]
    order = list(g.index)
    p = figure(x_range=FactorRange(*order), width=1000, height=450, tools="pan,wheel_zoom,reset,save",
               y_axis_label=vl)
    p.vbar(x=order, top=g["75%"].values, bottom=g["25%"].values, width=0.6, color="#4c78a8", alpha=0.6)
    p.scatter(x=order, y=g["50%"].values, color="black", size=8, legend_label="median")
    p.scatter(x=order, y=g["mean"].values, color="#e45756", size=7, marker="diamond", legend_label="mean")
    p.xaxis.major_label_orientation = 0.8
    show(p, 480)
    st.caption("Bars: interquartile range per family.")
    st.dataframe(g, use_container_width=True)

with tab_tab:
    cols = [c for c in ["pdb_id", "comp_id", "family", "description", "uniprot", "resolution", "res_class", "mw",
                        "logp", "tpsa", "n_rotb", "n_rotb_buried", "buried_frac", "delta_e_kcal", "covalent",
                        "rscc", "smiles"] if c in f]
    st.dataframe(f[cols], use_container_width=True, height=600)
    st.download_button("Download filtered table (CSV)", f[cols].to_csv(index=False).encode(), "survey_filtered.csv")
