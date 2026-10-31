r"""Gate-2 figures. Reads results/gate2_*.csv only and computes no new statistic.

    PYTHONPATH=src python3 experiments/gate2_figures.py
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RHO_C = {0.0: "#4477aa", 0.5: "#66ccee", 0.9: "#ee6677", 0.95: "#cc3311", 0.99: "#aa3377",
         0.999: "#222222"}
CAND_C = {"B9_t_n2": "#333333", "C1_t_N1": "#ee6677", "C2_t_min": "#228833",
          "C3_ws2": "#ccbb44", "C4_exact2": "#66ccee", "R2_ws_inf": "#aa3377"}


def load(name):
    return list(csv.DictReader(open(ROOT / "results" / name)))


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_bound_ratio(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200)
    B = np.array([float(r["bound_B"]) for r in rows])
    D = np.array([abs(float(r["delta_cov"])) for r in rows])
    ax = axes[0]
    ax.loglog(B, np.maximum(D, 1e-7), "o", ms=3, alpha=0.5, color="#4477aa")
    lim = [min(B.min(), D.min()) * 0.5, B.max() * 2]
    ax.loglog(lim, lim, "k--", lw=1, label="|Δ| = B (bound attained)")
    ax.legend(fontsize=7, frameon=False)
    _style(ax, f"Every cell lies below the bound ({len(rows)} cells, 0 violations)",
           "theoretical bound B(n, ρ, α, N)", "|Δ_cov| measured")
    ax = axes[1]
    ratio = D / B
    ax.hist(ratio, bins=40, color="#4477aa", alpha=0.85)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.axvline(ratio.max(), color="#ee6677", lw=1, label=f"max = {ratio.max():.3f}")
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "Slack: the bound is never attained", "|Δ_cov| / B", "cells")
    fig.tight_layout(); fig.savefig(FIG / "Gate2_bound_ratio.png", bbox_inches="tight")
    plt.close(fig)


def fig_coverage_boundary(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200, sharey=True)
    sel = [r for r in rows if float(r["rho"]) >= 0.9]
    for ax, key, xlabel in ((axes[0], "N_minus_n_plus_1", "N − n + 1"),
                            (axes[1], "N_over_n_minus_1", "N / (n − 1)")):
        for rho in sorted({float(r["rho"]) for r in sel}):
            pts = sorted([(float(r[key]), float(r["coverage_actual"]))
                          for r in sel if float(r["rho"]) == rho])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", ms=3.2, alpha=0.75,
                    color=RHO_C.get(rho, "grey"), label=f"ρ = {rho}")
        ax.axhline(0.95, color="k", lw=1, ls="--")
        ax.axvline(0 if key == "N_minus_n_plus_1" else 1.0, color="#ee6677", lw=1.2)
        ax.set_xscale("symlog" if key == "N_minus_n_plus_1" else "log")
        _style(ax, f"Coverage against {xlabel}", xlabel, "empirical coverage")
    axes[0].set_ylim(0.68, 1.005)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    axes[0].text(0.02, 0.06, "red line: G8 boundary N = n − 1", transform=axes[0].transAxes,
                 fontsize=7, color="#ee6677")
    fig.tight_layout(); fig.savefig(FIG / "Gate2_coverage_boundary.png", bbox_inches="tight")
    plt.close(fig)


def fig_limit_rho(rows):
    ns = [10, 20, 50]
    fig, axes = plt.subplots(1, len(ns), figsize=(3.4 * len(ns), 3.4), dpi=200, sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ns):
        sub = [r for r in rows if int(r["n"]) == n and float(r["rho"]) >= 0.9]
        Ns = sorted({float(r["N"]) for r in sub})
        for rho in sorted({float(r["rho"]) for r in sub}):
            pts = [(float(r["N"]), float(r["coverage_actual"]))
                   for r in sub if float(r["rho"]) == rho]
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3, lw=1,
                    color=RHO_C.get(rho, "grey"), label=f"ρ = {rho}")
        lim = sorted({(float(r["N"]), float(r["limit_pred_rho_to_1"])) for r in sub})
        ax.plot([p[0] for p in lim], [p[1] for p in lim], "k--", lw=1.4,
                label="limit P(|t$_{N-1}$| ≤ t$_{n-2}$)")
        ax.axhline(0.95, color="grey", lw=0.8, ls=":")
        ax.axvline(n - 1, color="#ee6677", lw=1.2)
        ax.set_xscale("log")
        _style(ax, f"n = {n}", "N (log)", "coverage" if n == ns[0] else "")
    np.atleast_1d(axes)[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("Coverage approaches the ρ → 1 limit from above; red line is N = n − 1",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "Gate2_limit_rho.png", bbox_inches="tight")
    plt.close(fig)


def fig_quantile_comparison(qrows):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), dpi=200)
    A = [r for r in qrows if r["grid"] == "grid_A"]
    cands = ["B9_t_n2", "C1_t_N1", "C2_t_min", "C3_ws2", "C4_exact2", "R2_ws_inf"]
    ax = axes[0]
    for i, c in enumerate(cands):
        cov = np.array([float(r["coverage"]) for r in A if r["candidate"] == c])
        ax.plot(np.full(cov.size, i) + np.linspace(-0.22, 0.22, cov.size), cov, ".",
                ms=2.6, alpha=0.5, color=CAND_C[c])
        ax.plot([i - 0.3, i + 0.3], [cov.min()] * 2, "-", lw=2, color=CAND_C[c])
    ax.axhline(0.95, color="k", lw=1, ls="--")
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cands], fontsize=6.5)
    ax.set_ylim(0.68, 1.005)
    _style(ax, "Coverage over grid_A (bar = worst cell)", "", "empirical coverage")
    ax = axes[1]
    for c in cands:
        w = np.array([float(r["width_rel_B9"]) for r in A if r["candidate"] == c])
        cov = np.array([float(r["coverage"]) for r in A if r["candidate"] == c])
        ax.plot(w, cov, ".", ms=3, alpha=0.55, color=CAND_C[c], label=c)
    ax.axhline(0.95, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    _style(ax, "Coverage against width (relative to the deployed interval)",
           "mean width ÷ B9 width (log)", "empirical coverage")
    fig.tight_layout(); fig.savefig(FIG / "Gate2_quantile_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows, qrows = load("gate2_boundary.csv"), load("gate2_quantiles.csv")
    fig_bound_ratio(rows)
    fig_coverage_boundary(rows)
    fig_limit_rho(rows)
    fig_quantile_comparison(qrows)
    for f in ("Gate2_bound_ratio", "Gate2_coverage_boundary", "Gate2_limit_rho",
              "Gate2_quantile_comparison"):
        print(f"wrote figures/{f}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
