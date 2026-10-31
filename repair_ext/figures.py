r"""Figures for the repair extension. Reads the saved CSVs only; computes no new statistic."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_repair_extension"
FIG = OUT / "figures"

ORDER = ["B1", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B9-tmin", "Ridge-PPI", "PPBoot"]
LABEL = {"B1": "Classical", "B3": "PPI++", "B4": "PPI++ +t", "B5": "cross-fit",
         "B6": "threshold", "B7": "percentile boot", "B8": "BCa boot", "B9": "LC-PPI",
         "B9-tmin": "LC-PPI t_min", "Ridge-PPI": "Ridge-PPI", "PPBoot": "PPBoot"}
COLOUR = {"B1": "#777777", "B3": "#c1121f", "B4": "#e07a5f", "B5": "#f2cc8f", "B6": "#b08968",
          "B7": "#8ecae6", "B8": "#219ebc", "B9": "#2a9d8f", "B9-tmin": "#1b7f5f",
          "Ridge-PPI": "#7209b7", "PPBoot": "#3a0ca3"}
MAIN = ["B1", "B3", "B9", "B9-tmin", "Ridge-PPI", "PPBoot"]


def _style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load():
    cov = pd.read_csv(OUT / "coverage_results.csv")
    wid = pd.read_csv(OUT / "interval_width_results.csv")
    return cov, wid


def fig_coverage_vs_n(cov, pool=200):
    sub = cov[(cov.N == pool) & (cov.n <= 50)]
    rhos = sorted(sub.rho.unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.0 * len(rhos), 3.6), dpi=200, sharey=True)
    for ax, rho in zip(np.atleast_1d(axes), rhos):
        s = sub[sub.rho == rho]
        for m in MAIN:
            d = s[s.method == m].sort_values("n")
            if d.empty:
                continue
            ax.plot(d.n, d.coverage, "-o", ms=3.5, lw=1.5, color=COLOUR[m], label=LABEL[m])
        ax.axhline(0.95, color="k", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_ylim(0.55, 1.02)
        _style(ax, f"ρ = {rho}", "gold labels n (log)", "empirical coverage" if rho == rhos[0] else "")
    np.atleast_1d(axes)[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle(f"Coverage against gold-sample size (pool N = {pool})", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig1_coverage_vs_n.png", bbox_inches="tight")
    plt.close(fig)


def fig_coverage_error_vs_n(cov, pool=200):
    sub = cov[cov.N == pool]
    rhos = sorted(sub.rho.unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.0 * len(rhos), 3.6), dpi=200, sharey=True)
    for ax, rho in zip(np.atleast_1d(axes), rhos):
        s = sub[sub.rho == rho]
        for m in MAIN:
            d = s[s.method == m].sort_values("n")
            if d.empty:
                continue
            ax.plot(d.n, d.coverage_error, "-o", ms=3.5, lw=1.5, color=COLOUR[m], label=LABEL[m])
        ax.axhline(0.005, color="k", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        _style(ax, f"ρ = {rho}", "gold labels n (log)",
               "|coverage − 0.95| (log)" if rho == rhos[0] else "")
    np.atleast_1d(axes)[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Absolute coverage error; dotted line is the paper's ±0.005 band",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig2_coverage_error_vs_n.png", bbox_inches="tight")
    plt.close(fig)


def fig_width_vs_n(wid, pool=200):
    sub = wid[wid.N == pool]
    rhos = sorted(sub.rho.unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.0 * len(rhos), 3.6), dpi=200, sharey=True)
    for ax, rho in zip(np.atleast_1d(axes), rhos):
        s = sub[sub.rho == rho]
        for m in MAIN:
            d = s[s.method == m].sort_values("n")
            if d.empty:
                continue
            ax.plot(d.n, d.width_mean, "-o", ms=3.5, lw=1.5, color=COLOUR[m], label=LABEL[m])
        ax.set_xscale("log"); ax.set_yscale("log")
        _style(ax, f"ρ = {rho}", "gold labels n (log)",
               "mean interval width (log)" if rho == rhos[0] else "")
    np.atleast_1d(axes)[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"Mean interval width (pool N = {pool})", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_width_vs_n.png", bbox_inches="tight")
    plt.close(fig)


def fig_coverage_vs_rho_small_n(cov, pool=200):
    sub = cov[(cov.N == pool) & (cov.n <= 10)]
    ns = sorted(sub.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.3 * len(ns), 3.4), dpi=200, sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ns):
        s = sub[sub.n == n]
        for m in MAIN:
            d = s[s.method == m].sort_values("rho")
            if d.empty:
                continue
            ax.plot(d.rho, d.coverage, "-o", ms=4, lw=1.5, color=COLOUR[m], label=LABEL[m])
        ax.axhline(0.95, color="k", ls="--", lw=1)
        ax.set_ylim(0.55, 1.02)
        _style(ax, f"n = {n}", "predictor quality ρ",
               "empirical coverage" if n == ns[0] else "")
    np.atleast_1d(axes)[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle("Coverage against predictor quality, n ≤ 10", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig4_coverage_vs_rho_small_n.png", bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(cov, pool=200):
    sub = cov[cov.N == pool]
    methods = [m for m in ORDER if m in set(sub.method)]
    combos = sorted({(n, r) for n, r in zip(sub.n, sub.rho)})
    M = np.full((len(methods), len(combos)), np.nan)
    for i, m in enumerate(methods):
        for j, (n, r) in enumerate(combos):
            v = sub[(sub.method == m) & (sub.n == n) & (sub.rho == r)].coverage
            if len(v):
                M[i, j] = float(v.iloc[0]) - 0.95
    fig, ax = plt.subplots(figsize=(0.46 * len(combos) + 3.4, 0.42 * len(methods) + 2.0), dpi=200)
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([f"{n}\n{r}" for n, r in combos], fontsize=6.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABEL[m] for m in methods], fontsize=7.5)
    for i in range(len(methods)):
        for j in range(len(combos)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=5.2,
                        color="black" if abs(M[i, j]) < lim * 0.55 else "white")
    ax.set_xlabel("n  /  ρ", fontsize=9)
    ax.set_title(f"Coverage deviation from nominal (pool N = {pool}); blue = under-covers",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="coverage − 0.95")
    fig.tight_layout(); fig.savefig(FIG / "fig5_coverage_deviation_heatmap.png",
                                    bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    cov, wid = load()
    fig_coverage_vs_n(cov)
    fig_coverage_error_vs_n(cov)
    fig_width_vs_n(wid)
    fig_coverage_vs_rho_small_n(cov)
    fig_heatmap(cov)
    for f in sorted(FIG.glob("*.png")):
        print("wrote", f.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
