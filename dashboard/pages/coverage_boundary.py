"""Coverage boundary — the flagship view: where coverage falls below nominal, and why."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from components.loaders import load_results
from components.ui import (ACCENT, DANGER, MUTED, badges, caveat, footer, kpis, page_head,
                           section, source_note, style_chart)

page_head("Coverage Boundary",
          "Empirical coverage, the proved bound and the ρ → 1 limiting prediction — three "
          "different things, drawn differently.", "empirical", "theory")

df = load_results("gate2_boundary.csv")
q = load_results("gate2_quantiles.csv")

with st.container(border=True):
    a, b = st.columns([1, 3], gap="large")
    n_sel = a.selectbox("gold labels  n", sorted(df.n.unique()),
                        index=len(df.n.unique()) - 1)
    rho_opts = sorted(df.rho.unique())
    rho_sel = b.multiselect("predictor quality  ρ", rho_opts,
                            default=[r for r in rho_opts if r >= 0.9])
    X_AXES = {"N (log scale)": ("N", "log"),
              "N / (n − 1)": ("N_over_n_minus_1", "log"),
              "N − (n − 1)": ("N_minus_n_plus_1", "symlog")}
    axis_label = st.radio("horizontal axis", list(X_AXES), horizontal=True)

sub = df[(df.n == n_sel) & (df.rho.isin(rho_sel))].copy()
if sub.empty:
    st.info("Choose at least one ρ.")
    st.stop()
xcol, xscale = X_AXES[axis_label]
sub["ρ"] = sub.rho.astype(str)

tm = q[(q.candidate == "C2_t_min") & (q.n == n_sel) & (q.rho.isin(rho_sel))][
    ["n", "N", "rho", "coverage"]].rename(columns={"coverage": "tmin_coverage"})
sub = sub.merge(tm, on=["n", "N", "rho"], how="left")

under = int((sub.coverage_actual < 0.95).sum())
kpis([
    {"label": "cells shown", "value": f"{len(sub)}"},
    {"label": "B9 below nominal", "value": f"{under}",
     "sub": "of the cells shown", "tone": "warn" if under else "good"},
    {"label": "boundary", "value": f"N = {int(n_sel) - 1}", "sub": "N = n − 1", "tone": "accent"},
    {"label": "worst B9 coverage", "value": f"{sub.coverage_actual.min():.4f}", "tone": "warn"},
])

base = alt.Chart(sub).encode(
    x=alt.X(f"{xcol}:Q", scale=alt.Scale(type=xscale), title=axis_label))
tips = [alt.Tooltip("n:Q"), alt.Tooltip("N:Q"), alt.Tooltip("rho:Q", title="ρ"),
        alt.Tooltip("coverage_actual:Q", format=".5f", title="B9 coverage"),
        alt.Tooltip("tmin_coverage:Q", format=".5f", title="B9-tmin coverage"),
        alt.Tooltip("bound_B:Q", format=".5f", title="bound B"),
        alt.Tooltip("limit_pred_rho_to_1:Q", format=".5f", title="ρ→1 limit")]

band = alt.Chart(sub.head(1)).mark_rect(color=DANGER, opacity=0.06).encode(
    y=alt.datum(0.60), y2=alt.datum(0.95))
b9 = base.mark_line(point=alt.OverlayMarkDef(size=38, filled=True), strokeWidth=2.0).encode(
    y=alt.Y("coverage_actual:Q", scale=alt.Scale(domain=[0.60, 1.0]),
            title="empirical coverage"),
    color=alt.Color("ρ:N", scale=alt.Scale(scheme="reds"),
                    legend=alt.Legend(orient="top-right", title="B9, by ρ")),
    tooltip=tips)
tmin_line = base.mark_line(strokeDash=[3, 3], strokeWidth=1.5, opacity=0.9,
                           color="#3fb950").encode(y="tmin_coverage:Q", tooltip=tips)
limit = base.mark_line(strokeDash=[7, 4], color="#ffffff", strokeWidth=2.0).encode(
    y="limit_pred_rho_to_1:Q", tooltip=tips)
nominal = alt.Chart(sub.head(1)).mark_rule(color=ACCENT, strokeDash=[6, 4],
                                           strokeWidth=1.5).encode(y=alt.datum(0.95))
layers = band + b9 + tmin_line + limit + nominal
if xcol == "N":
    layers = layers + alt.Chart(pd.DataFrame({"v": [float(n_sel - 1)]})).mark_rule(
        color="#a371f7", strokeWidth=2).encode(x="v:Q")
st.altair_chart(style_chart(layers, 470), width='stretch')

lg = st.columns(4)
for col, (swatch, label, desc) in zip(lg, [
        ("#f85149", "B9 — solid", "what the deployed interval achieved"),
        ("#3fb950", "B9-tmin — dotted", "the finite-pool safeguard"),
        ("#ffffff", "ρ → 1 limit — dashed", "computed from the limit, not fitted"),
        ("#a371f7", "N = n − 1", "the boundary; the limit crosses nominal here")]):
    col.markdown(
        f"<div class='srcnote'><span style='color:{swatch};font-size:1.1rem'>—</span> "
        f"<b style='color:#e6edf3'>{label}</b><br>{desc}</div>", unsafe_allow_html=True)
st.caption("The shaded band below the blue dashed line is under-coverage. Hover any point for "
           "the exact coverage, the bound, and the limiting prediction at that cell.")
source_note("results/gate2_boundary.csv", "results/gate2_quantiles.csv")

section("How close is the limiting prediction?",
        "The prediction is derived from the ρ → 1 limit, then compared — never fitted to these "
        "numbers.")
strong = sub[sub.rho == max(rho_sel)].sort_values("N")
if not strong.empty:
    comp = strong[["N", "coverage_actual", "limit_pred_rho_to_1", "bound_B"]].copy()
    comp["|measured − predicted|"] = (comp.coverage_actual - comp.limit_pred_rho_to_1).abs()
    comp = comp.rename(columns={"coverage_actual": f"measured at ρ = {max(rho_sel)}",
                                "limit_pred_rho_to_1": "ρ → 1 prediction", "bound_B": "bound B"})
    st.dataframe(comp.style.format({c: "{:.5f}" for c in comp.columns if c != "N"}),
                 width='stretch', height=270)

section("The proved bound across the whole grid",
        "Points below the diagonal respect Theorem 1.")
allc = df.copy()
allc["|Δ_cov|"] = allc.delta_cov.abs()
scat = alt.Chart(allc).mark_circle(size=40, opacity=0.6).encode(
    x=alt.X("bound_B:Q", scale=alt.Scale(type="log"), title="theoretical bound  B  (log)"),
    y=alt.Y("|Δ_cov|:Q", scale=alt.Scale(type="log"), title="|Δ_cov| measured  (log)"),
    color=alt.Color("rho:Q", scale=alt.Scale(scheme="viridis"), title="ρ"),
    tooltip=[alt.Tooltip("n:Q"), alt.Tooltip("N:Q"), alt.Tooltip("rho:Q", title="ρ"),
             alt.Tooltip("delta_cov:Q", format=".5f"), alt.Tooltip("bound_B:Q", format=".5f")])
diag = alt.Chart(pd.DataFrame({"v": [allc.bound_B.min(), allc.bound_B.max()]})).mark_line(
    strokeDash=[5, 4], color=MUTED).encode(x="v:Q", y="v:Q")
st.altair_chart(style_chart(scat + diag, 360), width='stretch')
viol = int((allc["|Δ_cov|"] > allc.bound_B).sum())
st.caption(f"All {len(allc)} preregistered cells. **Violations: {viol}.** The bound is never "
           "attained — the tightest cell keeps a factor of about four in reserve.")
caveat()
footer()
