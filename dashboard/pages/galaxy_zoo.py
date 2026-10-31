"""Galaxy Zoo — the real-data case study, with its scope stated plainly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from components.loaders import METHOD_LABELS, load_galaxies, load_results, pivot_metrics  # noqa: E402
from components.ui import (ACCENT, DANGER, GALAXIES_SCOPE, MUTED, OK, caveat,  # noqa: E402
                           footer, kpis, live_note, page_head, section, source_note,
                           style_chart)

page_head("Galaxy Zoo",
          "Galaxy Zoo 2 morphological classifications — a fully observed population.",
          "realdata")
caveat(GALAXIES_SCOPE, kind="scope")

Y, Yhat = load_galaxies()
theta_pop = float(Y.mean())
df = load_results("I_real_data.csv")

with st.container(border=True):
    n_opts = sorted(df.n.unique())
    n_sel = st.select_slider("gold-label sample size  n", options=n_opts, value=n_opts[0])
row = df[df.n == n_sel].iloc[0]
N_sel = float(row.N)

kpis([
    {"label": "population size", "value": f"{len(Y):,}", "sub": "fully observed"},
    {"label": "gold-label sample  n", "value": f"{int(n_sel):,}"},
    {"label": "prediction pool  N", "value": f"{int(N_sel):,}", "sub": "population − gold sample"},
    {"label": "estimand  θ", "value": "mean of Y", "sub": "population mean"},
    {"label": "population value", "value": f"{theta_pop:.4f}", "tone": "accent"},
])

section("Saved results at this sample size",
        "The column is labelled containment fraction, not coverage, deliberately — see the "
        "notice above.")
METRICS = ["theta_hat_mean", "coverage", "width_mean", "width_ratio_b1", "seconds_per_interval"]
tbl = pivot_metrics(df, METRICS, n=n_sel)
if not tbl.empty:
    show = tbl.copy()
    show.insert(0, "method", [METHOD_LABELS.get(m, m) for m in show.index])
    show = show.rename(columns={
        "theta_hat_mean": "mean point estimate", "coverage": "containment fraction",
        "width_mean": "mean interval width", "width_ratio_b1": "width ÷ classical",
        "seconds_per_interval": "sec / interval"})
    fmt = {c: "{:.4f}" for c in show.columns if c != "method"}
    if "sec / interval" in show.columns:
        fmt["sec / interval"] = "{:.2e}"
    st.dataframe(show.style.format(fmt), width='stretch')

    pts = tbl.reset_index().dropna(subset=["coverage", "width_mean"])
    if not pts.empty:
        bars = alt.Chart(pts).mark_bar(size=18, opacity=0.9).encode(
            y=alt.Y("method:N", sort=alt.EncodingSortField("coverage", order="descending"),
                    title=None),
            x=alt.X("coverage:Q", scale=alt.Scale(domain=[0.7, 1.0]),
                    title="containment fraction"),
            color=alt.condition(alt.datum.coverage < 0.95, alt.value(DANGER), alt.value(OK)),
            tooltip=[alt.Tooltip("method:N"), alt.Tooltip("coverage:Q", format=".4f",
                                                          title="containment"),
                     alt.Tooltip("width_mean:Q", format=".4f", title="mean width")])
        nominal = alt.Chart(pts.head(1)).mark_rule(color=ACCENT, strokeDash=[6, 4],
                                                   strokeWidth=1.5).encode(x=alt.datum(0.95))
        st.altair_chart(style_chart(bars + nominal, 320), width='stretch')
source_note("results/I_real_data.csv", "data/galaxies.npz")

section("One split, computed live")
live_note("A single gold/unlabelled split of the real population, so that an actual interval can "
          "be shown rather than an average.")
seed = st.number_input("split seed", 0, 10**6, 0, 1)
try:
    from ppi_finite import registry
    from ppi_finite.estimators import labelled_summaries
    from ppi_finite.types import make_data

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(Y))
    gold, pool = idx[:int(n_sel)], idx[int(n_sel):]
    data = make_data(Y[gold][None, :], Yhat[gold][None, :],
                     fbar_N=np.array([Yhat[pool].mean()]), N=float(len(pool)),
                     sigma2_f=np.array([Yhat[pool].var(ddof=1)]),
                     theta_true=np.array([theta_pop]))
    view = data.view()
    summ = labelled_summaries(view.Y, view.f)
    rows = []
    for mid in ["B1", "B3", "B9", "B9-tmin", "B9-HC"]:
        if mid not in registry.METHODS:
            continue
        r = registry.get(mid)(view, 0.05, None, summaries=summ)
        lo, hi = float(np.ravel(r.ci_low)[0]), float(np.ravel(r.ci_high)[0])
        rows.append({"method": mid, "label": METHOD_LABELS.get(mid, mid),
                     "estimate": float(np.ravel(r.theta_hat)[0]), "lower": lo, "upper": hi,
                     "width": hi - lo, "contains": lo <= theta_pop <= hi})
    live = pd.DataFrame(rows)

    lc, rc = st.columns([3, 2], gap="large")
    with lc:
        rules = alt.Chart(live).mark_rule(strokeWidth=7, opacity=0.85).encode(
            y=alt.Y("method:N", sort=list(live.method), title=None),
            x=alt.X("lower:Q", title="interval for θ"), x2="upper:Q",
            color=alt.Color("contains:N", scale=alt.Scale(domain=[True, False],
                                                          range=[OK, DANGER]),
                            legend=alt.Legend(orient="top", title="contains θ")),
            tooltip=[alt.Tooltip("label:N", title="method"),
                     alt.Tooltip("estimate:Q", format=".4f"),
                     alt.Tooltip("lower:Q", format=".4f"), alt.Tooltip("upper:Q", format=".4f"),
                     alt.Tooltip("width:Q", format=".4f")])
        dots = alt.Chart(live).mark_point(size=70, filled=True, color="#ffffff").encode(
            y=alt.Y("method:N", sort=list(live.method)), x="estimate:Q")
        truth = alt.Chart(live.head(1)).mark_rule(color=ACCENT, strokeDash=[5, 4],
                                                  strokeWidth=1.6).encode(x=alt.datum(theta_pop))
        st.altair_chart(style_chart(rules + dots + truth, 280), width='stretch')
    with rc:
        st.dataframe(live.drop(columns=["label"]).set_index("method").style.format(
            {c: "{:.4f}" for c in ["estimate", "lower", "upper", "width"]}), width='stretch')
    st.caption(f"Population value θ = {theta_pop:.4f}. Whether one interval happens to contain it "
               "says nothing about the procedure; the saved table above is the evidence.")
except Exception as exc:                                    # pragma: no cover - display only
    st.warning(f"Live split unavailable: {exc}")

section("What this can and cannot establish")
c1, c2 = st.columns(2, gap="medium")
with c1, st.container(border=True):
    st.markdown("##### Can")
    st.markdown("That the deficit seen in simulation appears on real, non-Gaussian, "
                "heteroskedastic data, and by more than the Gaussian model predicts.")
with c2, st.container(border=True):
    st.markdown("##### Cannot")
    st.markdown("Any statement about superpopulation coverage, about other datasets, or about the "
                "theory of the paper — the simulations carry that. The classical interval is "
                "itself unreliable on this dataset at ten gold labels, which is part of the "
                "finding.")
caveat()
footer()
