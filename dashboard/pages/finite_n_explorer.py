"""Finite-N explorer — the safeguard, the proved bound, and the boundary verdict."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import streamlit as st
from scipy import stats

from components.loaders import load_results  # noqa: E402
from components.ui import (DANGER, OK, badge, boundary_diagram, caveat, footer,  # noqa: E402
                           kpis, page_head, section, source_note)

page_head("Finite-N Explorer",
          "How the quantile, the safeguard and the proved bound change with the configuration.",
          "theory")

with st.container(border=True):
    c = st.columns(4)
    n = c[0].number_input("gold labels  n", 4, 200, 50, 1)
    N = c[1].number_input("pool size  N", 2, 1_000_000, 25, 1)
    rho = c[2].slider("predictor quality  ρ", 0.0, 0.99, 0.9, 0.01)
    level = c[3].selectbox("confidence level", [0.95, 0.90, 0.99], 0)
alpha = 1 - level
nu1, nu2 = n - 2, N - 1
active = N < n - 1

section("Finite-pool boundary")
st.markdown(badge("boundary") if active else badge("safe"), unsafe_allow_html=True)
if active:
    st.markdown(
        f"<div class='note scope'><b>N &lt; n − 1 ⇒ finite-pool risk active.</b> "
        f"<code>N = {N} &lt; n − 1 = {n-1}</code>, so the pool carries fewer degrees of freedom "
        f"(<code>N−1 = {nu2}</code>) than the gold sample (<code>n−2 = {nu1}</code>). In the "
        "strong-predictor limit the statistic behaves like <code>t_(N−1)</code> while the standard "
        "interval supplies <code>t_(n−2)</code> — the interval is then <b>too short</b>, and "
        "coverage falls below nominal.</div>", unsafe_allow_html=True)
else:
    st.markdown(
        f"<div class='note' style='border-color:rgba(63,185,80,.4);background:rgba(63,185,80,.07)'>"
        f"<b>N ≥ n − 1 ⇒ standard quantile regime.</b> <code>N = {N} ≥ n − 1 = {n-1}</code>, so "
        f"<code>min(n−2, N−1) = n−2</code> and the safeguard is <b>identical</b> to the standard "
        "interval — same estimate, same width, same coverage.</div>", unsafe_allow_html=True)

boundary_diagram(int(n), int(N))

t_n2 = float(stats.t.ppf(1 - alpha / 2, nu1))
t_min = float(stats.t.ppf(1 - alpha / 2, min(nu1, nu2)))
kpis([
    {"label": "B9 quantile · t(n−2)", "value": f"{t_n2:.4f}", "sub": f"df = {nu1}"},
    {"label": "B9-tmin quantile · t(min(n−2, N−1))", "value": f"{t_min:.4f}",
     "sub": f"df = {min(nu1, nu2)}", "tone": "accent" if active else ""},
    {"label": "width ratio · safeguard ÷ standard", "value": f"{t_min / t_n2:.4f}×",
     "sub": "wider" if t_min > t_n2 else "identical",
     "tone": "warn" if t_min > t_n2 else "good"},
])

section("The proved bound",
        "Theorem 1 bounds how far the deployed interval's coverage can sit from the classical "
        "idealisation: |Δ_cov| ≤ B(n, ρ, α, N). B is explicit and non-asymptotic. It is evaluated "
        "here from the formula — it is not fitted to any simulation.")
try:
    from ppi_finite.theory_gate1 import coverage_bound
    b = coverage_bound(int(n), float(rho), float(N), float(alpha))
    kpis([
        {"label": "B(n, ρ, α, N)", "value": f"{b.total:.5f}",
         "tone": "good" if b.total < 0.05 else ""},
        {"label": "C = N × B", "value": f"{b.C:.3f}", "sub": "finite as N grows"},
        {"label": "status", "value": "informative" if b.total < 0.05 else "vacuous",
         "sub": "B < 0.05" if b.total < 0.05 else "B ≥ 0.05",
         "tone": "good" if b.total < 0.05 else "warn"},
    ])
    if b.total >= 1.0:
        st.markdown(
            "<div class='note caveat'>At this configuration the bound is <b>vacuous</b> "
            "(<code>B ≥ 1</code>): mathematically valid but saying nothing, because the proof "
            "divides by σ twice and the constant grows without bound as ρ → 1.</div>",
            unsafe_allow_html=True)
except Exception as exc:                                    # pragma: no cover - display only
    st.warning(f"Bound unavailable: {exc}")

section("Measured coverage at this cell, where it was run")
g2 = load_results("gate2_boundary.csv")
hit = g2[(g2.n == n) & (g2.N == float(N)) & (g2.rho.round(3) == round(float(rho), 3))]
if hit.empty:
    st.info("This exact (n, N, ρ) is not on the saved Gate-2 grid. The quantiles and the bound "
            "above are formula evaluations and remain valid; the coverage table is shown only for "
            "cells that were actually simulated.")
else:
    r = hit.iloc[0]
    q = load_results("gate2_quantiles.csv")
    qh = q[(q.n == n) & (q.N == float(N)) & (q.rho.round(3) == round(float(rho), 3))]
    rows = [{"rule": "B9 · t(n−2)", "empirical coverage": float(r.coverage_actual)}]
    tm = qh[qh.candidate == "C2_t_min"]
    if not tm.empty:
        rows.append({"rule": "B9-tmin · t(min(n−2, N−1))",
                     "empirical coverage": float(tm.iloc[0].coverage)})
    rows.append({"rule": "idealised statistic", "empirical coverage": float(r.coverage_ideal)})
    st.dataframe(pd.DataFrame(rows).style.format({"empirical coverage": "{:.5f}"}),
                 width='stretch')
    respected = abs(float(r.delta_cov)) <= float(r.bound_B)
    kpis([
        {"label": "Δ_cov = actual − idealised", "value": f"{float(r.delta_cov):+.5f}"},
        {"label": "|Δ_cov| ÷ B", "value": f"{float(r.ratio_abs_delta_over_B):.4f}",
         "sub": "bound respected" if respected else "VIOLATION",
         "tone": "good" if respected else "warn"},
    ])
    source_note("results/gate2_boundary.csv", "results/gate2_quantiles.csv")

caveat()
footer()
