"""Method comparison — saved research summaries, and a live single-draw illustration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from components.loaders import METHOD_LABELS, load_results, pivot_metrics  # noqa: E402
from components.ui import (ACCENT, DANGER, MUTED, OK, badges, caveat, footer, kpis,  # noqa: E402
                           live_note, page_head, section, source_note, style_chart)

page_head("Method Comparison",
          "Every method side by side on a configuration you choose.", "empirical")

SOURCES = {
    "Experiment A — reproduction (N = 10⁶)": "A_reproduction.csv",
    "Experiment C — gold-sample sweep": "C_sample_size.csv",
    "Experiment E — finite pool": "E_finite_N.csv",
    "Experiment I — Galaxies (real data)": "I_real_data.csv",
}

with st.container(border=True):
    a, b = st.columns([2, 3], gap="large")
    src_label = a.selectbox("Saved experiment", list(SOURCES))
    df = load_results(SOURCES[src_label])
    sel = {}
    with b:
        sliders = st.columns(3)
        for col, label, sl in zip(("n", "N", "rho"),
                                  ("gold labels  n", "pool size  N", "predictor quality  ρ"),
                                  sliders):
            if col in df.columns and df[col].nunique() > 1:
                vals = sorted(df[col].dropna().unique())
                sel[col] = sl.select_slider(label, options=vals,
                                            value=vals[min(len(vals) // 2, len(vals) - 1)])
            elif col in df.columns and df[col].nunique() == 1:
                sel[col] = df[col].dropna().iloc[0]

    present = sorted(df.method.unique())
    chosen = st.multiselect("Methods", present, default=present,
                            format_func=lambda m: f"{m} · {METHOD_LABELS.get(m, m)}")

METRICS = ["theta_hat_mean", "coverage", "width_mean", "var_estimate_mean",
           "unlabeled_share_mean", "seconds_per_interval"]
tbl = pivot_metrics(df, METRICS, **sel)

if tbl.empty or not chosen:
    st.info("No saved rows for that combination.")
else:
    tbl = tbl.loc[[m for m in chosen if m in tbl.index]]
    if "coverage" in tbl.columns and not tbl["coverage"].dropna().empty:
        cov = tbl["coverage"].dropna()
        best, worst = cov.idxmax(), cov.idxmin()
        kpis([
            {"label": "methods compared", "value": f"{len(tbl)}"},
            {"label": "closest to nominal", "value": f"{cov.loc[(cov - 0.95).abs().idxmin()]:.4f}",
             "sub": (cov - 0.95).abs().idxmin(), "tone": "good"},
            {"label": "lowest coverage", "value": f"{cov.min():.4f}", "sub": worst,
             "tone": "warn" if cov.min() < 0.94 else ""},
            {"label": "highest coverage", "value": f"{cov.max():.4f}", "sub": best,
             "tone": "accent"},
        ])

    section("Saved results",
            "Averages over the experiment's Monte-Carlo replications, exactly as generated. "
            "Blank cells are metrics that experiment did not record.")
    show = tbl.copy()
    show.insert(0, "method", [METHOD_LABELS.get(m, m) for m in show.index])
    show = show.rename(columns={
        "theta_hat_mean": "point estimate", "coverage": "empirical coverage",
        "width_mean": "mean width", "var_estimate_mean": "variance estimate",
        "unlabeled_share_mean": "unlabelled share", "seconds_per_interval": "sec / interval"})
    fmt = {c: "{:.4f}" for c in show.columns if c != "method"}
    if "sec / interval" in show.columns:
        fmt["sec / interval"] = "{:.2e}"
    st.dataframe(show.style.format(fmt), width='stretch')

    if "coverage" in tbl.columns and "width_mean" in tbl.columns:
        section("Coverage against width",
                "Up and to the left is better: nominal coverage at the smallest width.")
        pts = tbl.reset_index().dropna(subset=["coverage", "width_mean"])
        pts["label"] = pts.method.map(lambda m: METHOD_LABELS.get(m, m))
        scat = alt.Chart(pts).mark_circle(size=170, opacity=0.9).encode(
            x=alt.X("width_mean:Q", title="mean interval width"),
            y=alt.Y("coverage:Q", title="empirical coverage"),
            color=alt.Color("method:N", legend=alt.Legend(orient="right", title=None)),
            tooltip=[alt.Tooltip("label:N", title="method"),
                     alt.Tooltip("coverage:Q", format=".4f"),
                     alt.Tooltip("width_mean:Q", format=".4f", title="width")])
        text = alt.Chart(pts).mark_text(dy=-15, color=MUTED, fontSize=10).encode(
            x="width_mean:Q", y="coverage:Q", text="method:N")
        nominal = alt.Chart(pts.head(1)).mark_rule(
            strokeDash=[6, 4], color=ACCENT, strokeWidth=1.4).encode(y=alt.datum(0.95))
        st.altair_chart(style_chart(scat + text + nominal, 380), width='stretch')

source_note(f"results/{SOURCES[src_label]}")
caveat()

section("A single interval, computed live",
        "The saved experiments store averages over replications, not individual intervals — so a "
        "lower and upper limit has to be computed to be shown.")
live_note("One draw from the bivariate Gaussian model at the settings below.")

with st.container(border=True):
    d = st.columns(5)
    n_live = d[0].number_input("gold labels  n", 4, 200, 10, 1)
    N_live = d[1].number_input("pool size  N", 2, 1_000_000, 1000, 1)
    rho_live = d[2].slider("predictor quality  ρ", 0.0, 0.99, 0.9, 0.01)
    alpha_live = d[3].selectbox("confidence level", [0.95, 0.90, 0.99], 0)
    seed_live = d[4].number_input("draw seed", 0, 10**6, 0, 1,
                                  help="Draws a different sample. It cannot affect any saved "
                                       "research result.")

try:
    import warnings

    from ppi_finite import registry
    from ppi_finite.estimators import labelled_summaries
    from ppi_finite.simulation import dgp_gaussian

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = dgp_gaussian(np.random.default_rng(int(seed_live)), R=1, n=int(n_live),
                            rho=float(rho_live), N=float(N_live), unlabeled_mode="explicit")
    view = data.view()
    summ = labelled_summaries(view.Y, view.f)
    theta_true = float(np.ravel(data.theta_true)[0])
    rows = []
    for mid in ["B1", "B2", "B3", "B9", "B9-tmin", "B9-HC"]:
        if mid not in registry.METHODS:
            continue
        r = registry.get(mid)(view, 1 - float(alpha_live), None, summaries=summ)
        lam = r.extras.get("lambda_hat") if isinstance(r.extras, dict) else None
        lo, hi = float(np.ravel(r.ci_low)[0]), float(np.ravel(r.ci_high)[0])
        rows.append({"method": mid, "label": METHOD_LABELS.get(mid, mid),
                     "estimate": float(np.ravel(r.theta_hat)[0]), "lower": lo, "upper": hi,
                     "width": hi - lo, "variance estimate": float(np.ravel(r.se)[0]) ** 2,
                     "df": float(np.ravel(r.dof)[0]) if np.ndim(r.dof) else float(r.dof),
                     "λ̂": float(np.ravel(lam)[0]) if lam is not None else np.nan,
                     "covers": lo <= theta_true <= hi})
    live = pd.DataFrame(rows)

    lc, rc = st.columns([3, 2], gap="large")
    with lc:
        rules = alt.Chart(live).mark_rule(strokeWidth=6, opacity=0.85).encode(
            y=alt.Y("method:N", sort=list(live.method), title=None),
            x=alt.X("lower:Q", title="interval"), x2="upper:Q",
            color=alt.Color("covers:N", scale=alt.Scale(domain=[True, False],
                                                        range=[OK, DANGER]),
                            legend=alt.Legend(orient="top", title="contains θ")),
            tooltip=[alt.Tooltip("label:N", title="method"),
                     alt.Tooltip("estimate:Q", format=".4f"),
                     alt.Tooltip("lower:Q", format=".4f"), alt.Tooltip("upper:Q", format=".4f"),
                     alt.Tooltip("width:Q", format=".4f")])
        dots = alt.Chart(live).mark_point(size=70, filled=True, color="#ffffff").encode(
            y=alt.Y("method:N", sort=list(live.method)), x="estimate:Q")
        truth = alt.Chart(live.head(1)).mark_rule(color=ACCENT, strokeDash=[5, 4],
                                                  strokeWidth=1.6).encode(x=alt.datum(theta_true))
        st.altair_chart(style_chart(rules + dots + truth, 300), width='stretch')
        st.caption(f"Blue dashed line is the true θ = {theta_true:.4f}. White dot is the point "
                   "estimate.")
    with rc:
        st.dataframe(
            live.drop(columns=["label", "covers"]).set_index("method")
                .style.format({c: "{:.4f}" for c in
                               ["estimate", "lower", "upper", "width", "variance estimate",
                                "df", "λ̂"]}),
            width='stretch')
    st.caption("Whether an individual interval covers is a coin toss, not evidence: coverage is a "
               "property of the procedure over many draws, which is what the saved table reports.")
except Exception as exc:                                    # pragma: no cover - display only
    st.warning(f"Live illustration unavailable: {exc}")

footer()
