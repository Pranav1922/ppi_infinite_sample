r"""Gate-4 figures. Reads results/gate4_quantiles.csv only; computes no new statistic."""
from __future__ import annotations

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
C = {"B9_t_n2": "#333333", "B9_tmin": "#228833", "C5_bb": "#ee6677"}


def _style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> int:
    rows = list(csv.DictReader(open(ROOT / "results" / "gate4_quantiles.csv")))
    f = lambda r, k: float(r[k])

    # --- width vs coverage -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), dpi=200)
    for ax, grid, ttl in ((axes[0], "grid_primary", "Difficult grid (ρ ≥ 0.9)"),
                          (axes[1], "grid_supplementary", "Supplementary (ρ ∈ {0, 0.5})")):
        for cand in ("B9_t_n2", "B9_tmin", "C5_bb"):
            s = [r for r in rows if r["candidate"] == cand and r["grid"] == grid]
            ax.plot([f(r, "width_rel_B9") for r in s], [f(r, "coverage") for r in s],
                    "o", ms=3.4, alpha=0.6, color=C[cand], label=cand)
        ax.axhline(0.95, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        _style(ax, ttl, "mean width ÷ deployed B9 width (log)", "empirical coverage")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Protection costs width; Berger–Boos recovers almost none of it where it matters",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "Gate4_width_vs_coverage.png",
                                    bbox_inches="tight")
    plt.close(fig)

    # --- the quantile itself, against N ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), dpi=200, sharey=True)
    for ax, n in zip(axes, (10, 20, 50)):
        Ns = np.array(sorted({f(r, "N") for r in rows if int(r["n"]) == n}))
        ax.plot(Ns, [stats.t.ppf(0.975, n - 2)] * len(Ns), "-", lw=1.6, color=C["B9_t_n2"],
                label="B9  $t_{n-2}$")
        ax.plot(Ns, [stats.t.ppf(0.975, min(n - 2, N - 1)) for N in Ns], "-o", ms=3.2, lw=1.4,
                color=C["B9_tmin"], label="B9-tmin  $t_{\\min(n-2,N-1)}$")
        for rho, mk in ((0.0, "s"), (0.999, "^")):
            s = sorted([(f(r, "N"), f(r, "width_rel_tmin")) for r in rows
                        if r["candidate"] == "C5_bb" and int(r["n"]) == n and f(r, "rho") == rho])
            if s:
                base = np.array([stats.t.ppf(0.975, min(n - 2, N - 1)) for N, _ in s])
                ax.plot([x[0] for x in s], base * np.array([x[1] for x in s]), mk, ms=4,
                        color=C["C5_bb"], alpha=0.85,
                        label=f"C5 Berger–Boos, ρ={rho}")
        ax.axvline(n - 1, color="#aaaaaa", lw=1.2, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        _style(ax, f"n = {n}", "N (log)", "critical value" if n == 10 else "")
    axes[0].legend(fontsize=6.3, frameon=False)
    fig.suptitle("The safeguard tracks $t_{N-1}$ below N = n−1 (dotted); Berger–Boos rarely departs from it",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "Gate4_quantile_boundary.png",
                                    bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/Gate4_width_vs_coverage.png")
    print("wrote figures/Gate4_quantile_boundary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
