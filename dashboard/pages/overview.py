"""Overview — the methods, the finite-N problem, and the headline result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import streamlit as st

from components.loaders import load_results
from components.ui import (ACCENT, DANGER, MUTED, OK, badges, caveat, footer, page_head,
                           section, source_note, style_chart)

page_head("Overview",
          "What the methods are, what goes wrong at small gold-label counts, and the fix.",
          "empirical")

section("The methods")
cols = st.columns(3, gap="medium")
cards = [
    ("Classical", "Ignore the model. Use the gold labels alone: `Ȳ ± t·s/√n`. Always valid, "
                  "but wastes the predictions."),
    ("PPI", "Correct the pool's mean prediction by the gold-sample bias: `θ̂ = f̄_N + (Ȳ − f̄_n)`. "
            "Valid, but can be *worse* than classical when the predictor is poor."),
    ("PPI++", "Shrink the correction by a tuning coefficient `λ̂` estimated from the gold sample. "
              "Its guarantee — never worse than classical — is **asymptotic in the number of gold "
              "labels**."),
]
for col, (name, body) in zip(cols, cards):
    with col, st.container(border=True):
        st.markdown(f"#### {name}")
        st.markdown(body)

c1, c2 = st.columns(2, gap="medium")
with c1, st.container(border=True):
    st.markdown("#### LC-PPI — this work's correction")
    st.markdown(
        "Recognise the estimator as an ordinary-least-squares fitted value. Two changes follow: a "
        "residual variance on `n−2` degrees of freedom carrying the **leverage** term the PPI++ "
        "variance omits, and a Student rather than normal quantile.")
with c2, st.container(border=True):
    st.markdown("#### B9-tmin — the finite-pool safeguard")
    st.markdown(
        "The pool's variance `σ̂_f²` is itself estimated, from `N−1` degrees of freedom. Using "
        "`t_{min(n−2, N−1)}` instead of `t_{n−2}` accounts for that. The rule is classical — "
        "Hsu (1938); Mickey & Brown (1966) — and nothing is claimed for it here.")

section("The headline result",
        "PPI++ can substantially under-cover at very small gold-label sizes. "
        "At four gold labels a nominal 95 % interval covers about 68 %.")

df = load_results("C_sample_size.csv")
cov = df[(df.metric == "coverage") & (df.method.isin(["B1", "B3", "B9"]))].copy()
cov["Method"] = cov.method.map({"B1": "Classical", "B3": "PPI++", "B9": "LC-PPI"})
cov["ρ"] = cov.rho.astype(str)

band = alt.Chart(cov.head(1)).mark_rect(color=DANGER, opacity=0.055).encode(
    y=alt.datum(0.60), y2=alt.datum(0.95))
line = alt.Chart(cov).mark_line(point=alt.OverlayMarkDef(size=42, filled=True),
                                strokeWidth=2.1).encode(
    x=alt.X("n:Q", scale=alt.Scale(type="log"), title="gold labels  n  (log scale)"),
    y=alt.Y("value:Q", scale=alt.Scale(domain=[0.60, 1.0]), title="empirical coverage"),
    color=alt.Color("Method:N", scale=alt.Scale(
        domain=["Classical", "PPI++", "LC-PPI"], range=[MUTED, DANGER, OK]),
        legend=alt.Legend(orient="top-right", title=None)),
    detail="ρ:N",
    tooltip=[alt.Tooltip("Method:N"), alt.Tooltip("n:Q"), alt.Tooltip("rho:Q", title="ρ"),
             alt.Tooltip("value:Q", format=".4f", title="coverage")],
)
nominal = alt.Chart(cov.head(1)).mark_rule(strokeDash=[6, 4], color=ACCENT,
                                           strokeWidth=1.4).encode(y=alt.datum(0.95))
st.altair_chart(style_chart(band + line + nominal, 400), width='stretch')
st.caption("One line per predictor quality ρ ∈ {0, 0.5, 0.9}. Blue dashed line is the nominal "
           "level; the shaded band below it is under-coverage. The PPI++ deficit is nearly "
           "invariant in ρ — a good predictor does not protect against it.")
source_note("results/C_sample_size.csv")

section("The finite-`N` problem",
        "When the unlabelled pool is finite, two further things happen, and they pull in "
        "opposite directions.")
d1, d2 = st.columns(2, gap="medium")
with d1, st.container(border=True):
    st.markdown("##### 1 · A bias, not a variance error")
    st.markdown(
        "The corrected statistic is exactly Student-`t` pivotal — but **for a shifted quantity** "
        "whose offset involves the unknown population mean of the predictions. The shortfall is a "
        "**bias, not a mis-specified variance**, which is why every variance-side repair tested "
        "here fails.")
with d2, st.container(border=True):
    st.markdown("##### 2 · The pool's own degrees of freedom")
    st.markdown(
        "The pool's variance is estimated. As the predictor approaches perfection the statistic "
        "inherits the **pool's** degrees of freedom while the interval still supplies the gold "
        "sample's, so coverage falls below nominal **exactly when `N < n − 1`**. The safeguard "
        "removes this.")

caveat()
footer()
