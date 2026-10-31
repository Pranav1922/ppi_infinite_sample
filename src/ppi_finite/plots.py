"""Figure generation (L5).  Extension E2, built for E6.

`15` §L5: this module **consumes `results/` only** — the tidy CSV and the per-replication `.npz`
raw store — and never recomputes a statistic.  Every figure is therefore traceable to rows of a
CSV that has a provenance sidecar.

Conventions: one colour per method held fixed across figures, nominal level drawn as a reference
line wherever a coverage axis appears, and **no figure asserts empirical coverage for a quantity
the protocol cannot support** — on real data the coverage panels are explicitly labelled as
coverage under repeated subsampling from a fixed finite population, not superpopulation coverage.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

__all__ = ["METHOD_COLOURS", "load_raw", "figure_estimates", "figure_widths",
           "figure_intervals", "figure_lambda", "figure_method_comparison", "make_all",
           "coverage_panels", "figure_deficit_vs_inverse_n", "figure_cost_vs_coverage",
           "figure_tail_miscoverage", "make_all_experiments"]

#: Stable across every figure so a reader can carry the legend between panels.
METHOD_COLOURS = {
    "B1": "#4c4c4c", "B2": "#8c6d31", "B3": "#c1272d", "B4": "#e08214", "B5": "#b07aa1",
    "B7": "#6a9fb5", "B8": "#1f78b4", "B9": "#1a9850", "B9-R2": "#66bd63", "B9-HC": "#006837",
}
NOMINAL = 0.95

#: Journal rendering, for the Statistics and Computing artwork rules: no titles inside the figure,
#: lettered panels, a marker per method so colour is never the only cue, and all lettering at 8 pt
#: at the final printed width. Switched on only by `experiments/make_journal_figures.py`; with it
#: off every figure is drawn exactly as before.
JOURNAL = False
JOURNAL_WIDTH_IN = 6.2          # inside the manuscript's 16.6 cm text width
METHOD_MARKERS = {
    "B1": "s", "B2": "v", "B3": "o", "B4": "^", "B5": "<", "B6": "P", "B7": "h", "B8": "X",
    "B9": "D", "B9-nolev": ">", "B9-R1": "p", "B9-R2": "*", "B9-R5": "d", "B9-HC": "H",
}


def _fs(size: float) -> float:
    return 8 if JOURNAL else size


def _marker(m: str) -> str:
    return METHOD_MARKERS.get(m, "o") if JOURNAL else "o"


def _save(fig, out: Path):
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
    if JOURNAL:
        fig.savefig(Path(out).with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=_fs(10), loc="left")
    ax.set_xlabel(xlabel, fontsize=_fs(9))
    ax.set_ylabel(ylabel, fontsize=_fs(9))
    ax.tick_params(labelsize=8)
    if JOURNAL:   # EPS has no transparency: the same tint (grey #b0b0b0 at 25 % on white), opaque
        ax.grid(color="#ebebeb", linewidth=0.5)
    else:
        ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_raw(raw_dir: Path, n: int) -> dict:
    """Per-replication arrays for one cell, from the `.npz` raw store."""
    hits = [p for p in sorted(Path(raw_dir).glob("*.npz")) if f"__n-{n}__" in p.name]
    if not hits:
        raise FileNotFoundError(f"no raw npz for n={n} in {raw_dir}")
    with np.load(hits[0]) as z:
        return {k: z[k] for k in z.files}


def _methods_in(df, wanted):
    present = set(df.method.unique())
    return [m for m in wanted if m in present]


def figure_estimates(df, theta, methods, out: Path):
    """1 — point-estimate comparison against the known population value."""
    g = df[df.metric == "theta_hat_mean"].pivot_table(index="n", columns="method", values="value")
    sd = df[df.metric == "theta_hat_sd"].pivot_table(index="n", columns="method", values="value")
    R = int(df.R.iloc[0])
    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
    xs = np.arange(len(g.index))
    for i, m in enumerate(methods):
        off = (i - len(methods) / 2) * 0.09
        ax.errorbar(xs + off, g[m], yerr=1.96 * sd[m] / np.sqrt(R), fmt="o", ms=3.5,
                    capsize=2, lw=1, color=METHOD_COLOURS.get(m, "grey"), label=m)
    ax.axhline(theta, color="k", lw=1, ls="--")
    ax.annotate(f"population θ = {theta:.4f}", (len(xs) - 0.55, theta), fontsize=7.5,
                va="bottom", ha="right")
    ax.set_xticks(xs); ax.set_xticklabels([f"n = {v}" for v in g.index])
    _style(ax, "Mean point estimate across repeated gold subsamples (bars: ±1.96 MC s.e. of the mean)",
           "gold-label sample size", "estimate of θ")
    ax.legend(fontsize=7, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figure_widths(df, methods, out: Path):
    """2 — interval-width comparison, absolute and relative to the classical interval."""
    w = df[df.metric == "width_mean"].pivot_table(index="n", columns="method", values="value")
    r = df[df.metric == "width_ratio_b1"].pivot_table(index="n", columns="method", values="value")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=200)
    for m in methods:
        axes[0].plot(w.index, w[m], "o-", ms=3.5, lw=1.2, color=METHOD_COLOURS.get(m, "grey"), label=m)
        axes[1].plot(r.index, r[m], "o-", ms=3.5, lw=1.2, color=METHOD_COLOURS.get(m, "grey"))
    axes[1].axhline(1.0, color="k", lw=1, ls="--")
    for a in axes:
        a.set_xscale("log"); a.set_xticks(list(w.index)); a.set_xticklabels(list(w.index))
    axes[0].set_yscale("log")
    _style(axes[0], "Mean interval width", "gold-label sample size", "width")
    _style(axes[1], "Width relative to the classical interval", "gold-label sample size",
           "width ÷ width(B1)")
    axes[0].legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figure_intervals(raw_dir, theta, methods, ns, out: Path, n_show: int = 30, seed: int = 0):
    """3 — the intervals themselves, for a common random set of subsamples."""
    fig, axes = plt.subplots(1, len(ns), figsize=(3.6 * len(ns), 3.6), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(seed)
    xlabel = f"{n_show} random gold subsamples, sorted by the classical estimate"
    for ax, n in zip(axes, ns):
        raw = load_raw(raw_dir, n)
        pick = rng.choice(len(raw[f"{methods[0]}|theta_hat"]), n_show, replace=False)
        pick = pick[np.argsort(raw[f"{methods[0]}|theta_hat"][pick])]   # sorted: reads as a band
        for i, m in enumerate(methods):
            th = raw[f"{m}|theta_hat"][pick]
            half = raw[f"{m}|width"][pick] / 2
            x = np.arange(n_show) + (i - len(methods) / 2) * 0.16
            ok = (th - half <= theta) & (theta <= th + half)
            c = METHOD_COLOURS.get(m, "grey")
            ax.vlines(x, th - half, th + half, lw=1.1, color=c, alpha=np.where(ok, 0.85, 0.35))
            miss = ~ok
            if miss.any():
                ax.plot(x[miss], th[miss], "x", ms=3, color=c, mew=0.8)
        ax.axhline(theta, color="k", lw=1, ls="--")
        # one shared label below both panels: per-panel it is wider than the panel and the
        # neighbouring copies collide
        _style(ax, f"n = {n}", "", "interval for θ")
        ax.set_xticks([])
    axes[0].set_ylabel("interval for θ", fontsize=9)
    handles = [plt.Line2D([], [], color=METHOD_COLOURS.get(m, "grey"), lw=2, label=m)
               for m in methods]
    fig.legend(handles=handles, fontsize=7, ncol=len(methods), frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.supxlabel(xlabel, fontsize=9, y=0.02)
    fig.suptitle("Intervals on the same subsamples; × marks an interval missing the population θ",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figure_lambda(raw_dir, ns, out: Path, method: str = "B9"):
    """4 — the distribution of the estimated tuning coefficient."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(ns):
        raw = load_raw(raw_dir, n)
        key = f"{method}|lambda_hat"
        if key not in raw:
            continue
        ax.hist(raw[key], bins=90, density=True, histtype="step", lw=1.3,
                color=cmap(i / max(len(ns) - 1, 1)), label=f"n = {n}")
    ax.axvline(1.0, color="k", lw=1, ls="--")
    ax.annotate("λ = 1 (plain PPI)", (1.0, ax.get_ylim()[1] * 0.93), fontsize=7.5, ha="left")
    ax.axvline(0.0, color="grey", lw=1, ls=":")
    _style(ax, f"Estimated tuning coefficient λ̂ across repeated gold subsamples ({method})",
           "λ̂", "density")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figure_method_comparison(df, methods, out: Path, subtitle: str = ""):
    """5 — coverage under repeated subsampling, with the interpretation stated on the figure."""
    g = df[df.metric == "coverage"].pivot_table(index="n", columns="method", values="value")
    se = df[df.metric == "coverage"].pivot_table(index="n", columns="method", values="mc_se")
    fig, ax = plt.subplots(figsize=(JOURNAL_WIDTH_IN, 3.0) if JOURNAL else (7.0, 3.6), dpi=200)
    for m in methods:
        ax.errorbar(g.index, g[m], yerr=1.96 * se[m], fmt=f"{_marker(m)}-", ms=4, lw=1.2, capsize=2,
                    color=METHOD_COLOURS.get(m, "grey"), label=m)
    ax.axhline(NOMINAL, color="k", lw=1, ls="--")
    if not JOURNAL:   # journal: the caption identifies the dashed line; a label here hit a point
        ax.annotate("nominal 0.95", (g.index[-1], NOMINAL), fontsize=7.5, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_xticks(list(g.index)); ax.set_xticklabels(list(g.index))
    _style(ax, "" if JOURNAL else "Containment of the population θ under repeated gold subsampling",
           "gold-label sample size", "fraction of subsamples containing θ")
    if subtitle and not JOURNAL:
        ax.text(0.0, -0.30, subtitle, transform=ax.transAxes, fontsize=7.2, va="top", color="#333")
    ax.legend(fontsize=_fs(7), ncol=5, frameon=False, loc="lower right")
    _save(fig, out)


def make_all(csv: str | Path, out_dir: str | Path = "figures", prefix: str = "I") -> list[Path]:
    """Every E6 figure, from the results CSV and its raw store."""
    csv = Path(csv)
    df = pd.read_csv(csv)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = csv.parent / f"{csv.stem}_raw"
    theta = float(df[df.metric == "theta_hat_mean"].value.iloc[0])
    theta = 0.2592725318043361 if prefix == "I" else theta
    order = ["B1", "B2", "B3", "B4", "B5", "B8", "B9", "B9-R2", "B9-HC"]
    methods = _methods_in(df, order)
    ns = sorted(df.n.unique())
    caption = ("Coverage here is the fraction of repeated gold subsamples, drawn without\n"
               "replacement from one fixed population of 16,743 items, whose interval contains\n"
               "that population's θ. It is NOT superpopulation coverage and is not evidence for\n"
               "the theory; the synthetic experiments carry that (see docs/13_RESEARCH_LOG.md).")
    paths = []
    f = out_dir / f"{prefix}_point_estimates.png"; figure_estimates(df, theta, methods, f); paths.append(f)
    f = out_dir / f"{prefix}_interval_widths.png"; figure_widths(df, methods, f); paths.append(f)
    f = out_dir / f"{prefix}_intervals.png"
    figure_intervals(raw_dir, theta, [m for m in ("B1", "B3", "B9", "B9-HC") if m in methods],
                     [n for n in ns if n in (10, 50)] or ns[:2], f); paths.append(f)
    f = out_dir / f"{prefix}_lambda_distribution.png"; figure_lambda(raw_dir, ns, f); paths.append(f)
    f = out_dir / f"{prefix}_method_comparison.png"
    figure_method_comparison(df, methods, f, caption); paths.append(f)
    return paths


# ---------------------------------------------------------------------------------------------
# The pre-registered figure set of `07`, Experiments A-I.  Every function below reads a results
# CSV and nothing else: no statistic is recomputed (`15` §L5).
# ---------------------------------------------------------------------------------------------

def _panel_grid(n_panels, width=3.15, height=3.0):
    if JOURNAL:
        width, height = JOURNAL_WIDTH_IN / n_panels, 2.5
    fig, axes = plt.subplots(1, n_panels, figsize=(width * n_panels, height), dpi=200, sharey=True)
    return fig, np.atleast_1d(axes)


def coverage_panels(csv, out: Path, *, x: str, panel: str, methods, title: str,
                    xlabel: str, subtitle: str = "", logx: bool = False, ylim=None,
                    metric: str = "coverage", ylabel: str = "empirical coverage",
                    hline: float | None = NOMINAL, query: str | None = None):
    """Generic `metric` versus `x`, one panel per level of `panel`, one line per method."""
    df = pd.read_csv(csv)
    if query:
        df = df.query(query)
    g = df[df.metric == metric].pivot_table(index=[panel, x], columns="method", values="value")
    # mc_se is NaN for width and MSE metrics, and pandas drops all-NaN columns; treat it as absent.
    se = df[df.metric == metric].pivot_table(index=[panel, x], columns="method", values="mc_se",
                                             dropna=False)
    levels = sorted(g.index.get_level_values(0).unique())
    fig, axes = _panel_grid(len(levels))
    for i, (ax, lev) in enumerate(zip(axes, levels)):
        sub, ssub = g.xs(lev, level=0), se.xs(lev, level=0)
        for m in methods:
            if m not in sub:
                continue
            col = ssub[m] if m in ssub else None
            yerr = 1.96 * col if col is not None and col.notna().all() else None
            ax.errorbar(sub.index, sub[m], yerr=yerr, fmt=f"{_marker(m)}-", ms=3.2, lw=1.1, capsize=1.5,
                        color=METHOD_COLOURS.get(m, "grey"), label=m)
        if hline is not None:
            ax.axhline(hline, color="k", lw=1, ls="--")
        if logx:
            ax.set_xscale("log"); ax.set_xticks(list(sub.index))
            if JOURNAL:   # integer labels, turned when long enough to collide at the printed width
                labels = [f"{v:g}" for v in sub.index]
                turn = max(map(len, labels)) >= 4
                ax.set_xticks([], minor=True)
                ax.set_xticklabels(labels, rotation=45 if turn else 0, ha="right" if turn else "center")
            else:
                ax.set_xticklabels(list(sub.index))
        if ylim:
            ax.set_ylim(*ylim)
        if JOURNAL and not logx and any("=" in str(v) for v in sub.index):
            # categorical settings such as "heteroskedastic:gamma=0.25": the axis label names the
            # parameter, so the tick shows only its value
            ax.set_xticks(range(len(sub.index)) if not np.issubdtype(np.asarray(sub.index).dtype,
                          np.number) else list(sub.index))
            ax.set_xticklabels([str(v).split("=")[-1] for v in sub.index])
        # journal: the part letter only; what each panel shows is stated in the caption
        label = f"({'abcdefgh'[i]})" if JOURNAL else f"{panel} = {lev}"
        _style(ax, label, xlabel, ylabel if ax is axes[0] else "")
    axes[0].set_ylabel(ylabel, fontsize=_fs(9))
    handles = [plt.Line2D([], [], color=METHOD_COLOURS.get(m, "grey"), marker=_marker(m), ms=3.5,
                          lw=1.2, label=m) for m in methods if m in g]
    if JOURNAL:
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, fontsize=_fs(7), ncol=len(handles), frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.0 if JOURNAL else -0.06))
    if not JOURNAL:
        fig.suptitle(title, fontsize=10, x=0.005, ha="left")
        if subtitle:
            fig.text(0.005, -0.13, subtitle, fontsize=7.2, ha="left", color="#333")
        fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    if JOURNAL:
        fig.savefig(Path(out).with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)


def figure_deficit_vs_inverse_n(csv, out: Path, method: str = "B3"):
    """`07` Experiment C's diagnostic plot: the deficit against `1/n`.  Linearity through the
    origin is what the `O(1/n)` relative-distortion argument of `05` predicts."""
    df = pd.read_csv(csv)
    g = df[df.metric == "coverage"].pivot_table(index=["rho", "n"], columns="method", values="value")
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
    cmap = plt.get_cmap("viridis")
    rhos = sorted(g.index.get_level_values(0).unique())
    for i, rho in enumerate(rhos):
        sub = g.xs(rho, level=0)
        ax.plot(1 / sub.index, NOMINAL - sub[method], "o", ms=4.5,
                color=cmap(i / max(len(rhos) - 1, 1)), label=f"ρ = {rho}")
    allsub = g[method]
    inv = 1 / g.index.get_level_values(1).values
    sl, ic = np.polyfit(inv, NOMINAL - allsub.values, 1)
    xs = np.linspace(0, inv.max() * 1.05, 50)
    ax.plot(xs, sl * xs + ic, "k--", lw=1,
            label=f"fit: {sl:.3f}/n {ic:+.4f}")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    for n in sorted(set(g.index.get_level_values(1))):
        ax.annotate(f"n={n}", (1 / n, NOMINAL - g.xs(rhos[0], level=0)[method].loc[n]),
                    fontsize=6.5, xytext=(2, 4), textcoords="offset points")
    _style(ax, f"Coverage deficit of {method} against 1/n  (07 Experiment C diagnostic)",
           "1 / n", f"0.95 − coverage({method})")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figure_cost_vs_coverage(csv, out: Path, methods, n_show=(10, 20, 50)):
    """`07` Experiment B: wall-clock against coverage — "the argument that survives even if BCa
    works"."""
    df = pd.read_csv(csv)
    cov = df[df.metric == "coverage"].pivot_table(index=["n", "rho"], columns="method", values="value")
    sec = df[df.metric == "seconds_per_interval"].pivot_table(index=["n", "rho"], columns="method",
                                                              values="value")
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for m in methods:
        if m not in cov:
            continue
        xs = [sec.loc[(n, r), m] for (n, r) in cov.index if n in n_show]
        ys = [cov.loc[(n, r), m] for (n, r) in cov.index if n in n_show]
        ax.scatter(xs, ys, s=22, color=METHOD_COLOURS.get(m, "grey"), label=m, zorder=3)
    ax.axvline(np.nan)
    ax.axhline(NOMINAL, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    _style(ax, "Cost against coverage (Experiment B, N = 10⁶)",
           "wall-clock seconds per interval (log scale)", "empirical coverage")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    note = ("Four to five orders of magnitude separate the analytic intervals from the bootstrap "
            "ones,\nand neither bootstrap reaches nominal at n = 10.")
    fig.text(0.005, -0.10, note, fontsize=7.2, ha="left", color="#333")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figure_tail_miscoverage(csv, out: Path, methods, n: int = 10):
    """`07` Experiment G: stacked left/right miscoverage.  Total coverage hides asymmetry."""
    df = pd.read_csv(csv)
    df = df[df.n == n]
    fams = sorted(df.dgp.unique())
    lo = df[df.metric == "tail_miscoverage_left"].pivot_table(index=["dgp", "rho"], columns="method",
                                                              values="value")
    hi = df[df.metric == "tail_miscoverage_right"].pivot_table(index=["dgp", "rho"], columns="method",
                                                               values="value")
    rho = sorted(df.rho.unique())[0]
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=200)
    x = np.arange(len(fams)); w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        if m not in lo:
            continue
        l = [lo.loc[(f, rho), m] for f in fams]
        r = [hi.loc[(f, rho), m] for f in fams]
        c = METHOD_COLOURS.get(m, "grey")
        ax.bar(x + i * w - 0.4, l, w, color=c, alpha=0.55, label=f"{m} left" if i == 0 else None)
        ax.bar(x + i * w - 0.4, r, w, bottom=l, color=c, label=m)
    ax.axhline(0.05, color="k", lw=1, ls="--")
    ax.annotate("nominal total 0.05", (len(fams) - 0.6, 0.05), fontsize=7, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f.replace("skew:skew_family=", "") for f in fams],
                                         rotation=20, ha="right", fontsize=7.5)
    _style(ax, f"Left (pale) and right (solid) miscoverage, n = {n}, ρ = {rho} "
               "(07 Experiment G)", "", "miscoverage")
    ax.legend(fontsize=7, ncol=6, frameon=False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def make_all_experiments(results="results", out_dir="figures") -> list[Path]:
    """The whole pre-registered figure set of `07`, from saved results only."""
    R, O = Path(results), Path(out_dir)
    O.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    core = ["B1", "B3", "B4", "B5", "B9-nolev", "B9"]

    def add(fn, *a, **kw):
        try:
            fn(*a, **kw); made.append(a[1] if isinstance(a[1], Path) else a[0])
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  skipped {a[1] if len(a) > 1 else a[0]}: {type(e).__name__}: {e}")

    if (R / "A_reproduction.csv").exists():
        add(coverage_panels, R / "A_reproduction.csv", O / "A_coverage_vs_n.png",
            x="n", panel="rho", methods=["B1", "B2", "B3", "B9-nolev", "B9"],
            title="Experiment A — coverage against n, N = 10⁶ (R = 200,000)",
            xlabel="gold-label sample size n", logx=True, ylim=(0.86, 0.965),
            query="N == 1e6 and alpha_level == 0.05")
        add(coverage_panels, R / "A_reproduction.csv", O / "A_width_vs_rho.png",
            x="rho", panel="n", methods=["B2", "B3", "B9-nolev", "B9"],
            title="Experiment A — mean interval width relative to the classical interval",
            xlabel="ρ", metric="width_ratio_b1", ylabel="width ÷ width(B1)", hline=1.0,
            query="N == 1e6 and alpha_level == 0.05")
    if (R / "C_sample_size.csv").exists():
        add(coverage_panels, R / "C_sample_size.csv", O / "C_coverage_vs_n.png",
            x="n", panel="rho", methods=core,
            title="Experiment C — coverage down to n = 4 (R = 200,000)",
            xlabel="gold-label sample size n", logx=True,
            subtitle="PPI++ covers 0.681 at n = 4 against a nominal 0.95; the leverage-corrected "
                     "interval holds at 0.9495–0.9505 for every n ≥ 4.")
        add(figure_deficit_vs_inverse_n, R / "C_sample_size.csv", O / "C_deficit_vs_inverse_n.png")
    if (R / "D_predictor_quality.csv").exists():
        add(coverage_panels, R / "D_predictor_quality.csv", O / "D_coverage_vs_rho.png",
            x="rho", panel="n", methods=core + ["B6"],
            title="Experiment D — coverage against predictor quality ρ (R = 200,000)",
            xlabel="ρ = Corr(Y, f)",
            subtitle="The PPI++ deficit is invariant in ρ (spread ≤ 0.0018), the signature of an "
                     "estimated-coefficient cost rather than a predictor-quality problem.")
    if (R / "B_bootstrap.csv").exists():
        add(coverage_panels, R / "B_bootstrap.csv", O / "B_bootstrap_coverage.png",
            x="rho", panel="n", methods=["B1", "B3", "B7", "B8", "B9"],
            title="Experiment B — the bootstrap comparison (R = 20,000, B = 2,000)",
            xlabel="ρ")
        add(figure_cost_vs_coverage, R / "B_bootstrap.csv", O / "B_cost_vs_coverage.png",
            ["B1", "B3", "B7", "B8", "B9"])
    if (R / "E_finite_N.csv").exists():
        add(coverage_panels, R / "E_finite_N.csv", O / "E_coverage_vs_N.png",
            x="N", panel="n", methods=["B1", "B3", "B9", "B9-R1", "B9-R2", "B9-R5"],
            title="Experiment E — finite N, explicit unlabelled mode (R = 200,000)",
            xlabel="unlabelled pool size N", logx=True,
            query="rho == 0.9",
            subtitle="ρ = 0.9, the cell where the finite-N effect is largest. B9 over-covers; the "
                     "Satterthwaite variant R2 removes it; R5 (the exact quantile) does not "
                     "improve on R2.")
    if (R / "F_heavy_tails.csv").exists():
        add(coverage_panels, R / "F_heavy_tails.csv", O / "F_coverage_vs_nu.png",
            x="dgp", panel="n", methods=["B1", "B3", "B9", "B9-HC"],
            title="Experiment F — heavy-tailed residuals (R = 200,000)",
            xlabel="residual degrees of freedom ν",
            subtitle="The comparison that matters is B9 against B1, not B9 against 0.95 (10 H4): "
                     "both drift up together as ν falls.")
    if (R / "G_skew.csv").exists():
        add(figure_tail_miscoverage, R / "G_skew.csv", O / "G_coverage_by_tail.png",
            ["B1", "B3", "B9", "B9-HC"])
    if (R / "H_hetero.csv").exists():
        add(coverage_panels, R / "H_hetero.csv", O / "H_coverage_vs_gamma.png",
            x="dgp", panel="n", methods=["B1", "B3", "B9", "B9-HC"],
            title="Experiment H — heteroskedasticity σ(f) = exp(γf) (R = 200,000)",
            xlabel="heteroskedasticity γ", query="rho == 0.0",
            subtitle="B9 degrades as γ grows; the HC3 variant does not.")
    return made
