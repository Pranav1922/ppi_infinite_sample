r"""Figures for the repair-comparison extension. Reads saved CSVs only."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "repair_extension"
FIG = ROOT / "figures"

ORDER = ["B1", "B3", "B9", "B9-tmin", "Ridge-PPI", "PPBoot", "Efron-PPboot2"]
LAB = {"B1": "Classical", "B3": "PPI++", "B9": "LC-PPI", "B9-tmin": "LC-PPI t_min",
       "Ridge-PPI": "Ridge-PPI*", "PPBoot": "PPBoot", "Efron-PPboot2": "Efron PPboot2"}
COL = {"B1": "#777777", "B3": "#c1121f", "B9": "#2a9d8f", "B9-tmin": "#1b7f5f",
       "Ridge-PPI": "#7209b7", "PPBoot": "#3a0ca3", "Efron-PPboot2": "#e07a5f"}


def _style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


#: A marker per method, used in the journal rendering so colour is never the only cue.
MARK = {"B1": "s", "B3": "o", "B9": "D", "B9-tmin": "d", "Ridge-PPI": "^", "PPBoot": "v",
        "Efron-PPboot2": "P"}


def coverage_figure(df, out: Path, journal: bool = False) -> None:
    """Coverage against n, one panel per rho. `journal=True` follows the Statistics and Computing
    artwork rules (no in-figure title, lettered panels, per-method markers, 8 pt lettering at the
    16 cm printed width) and also writes an EPS; the default output is unchanged."""
    rhos = sorted(df.rho.unique())
    size = (6.2, 2.7) if journal else (4.1 * len(rhos), 3.7)
    fig, axes = plt.subplots(1, len(rhos), figsize=size, dpi=200, sharey=True)
    for i, (ax, rho) in enumerate(zip(np.atleast_1d(axes), rhos)):
        s = df[df.rho == rho]
        for m in ORDER:
            d = s[s.method == m].sort_values("n")
            if d.empty:
                continue
            ax.plot(d.n, d.coverage, "-" + (MARK[m] if journal else "o"), ms=4 if not journal else 3.5,
                    lw=1.6 if not journal else 1.2, color=COL[m], label=LAB[m])
        ax.axhline(0.95, color="k", ls="--", lw=1)
        # EPS has no transparency: the journal rendering uses the same tint as an opaque grey
        ax.axhspan(0.945, 0.955, color="#f0f0f0" if journal else "k", alpha=None if journal else 0.06,
                   zorder=0)
        ax.set_xscale("log"); ax.set_ylim(0.55, 1.0)
        _style(ax, f"({'abc'[i]})" if journal else f"ρ = {rho}", "gold labels n (log)",
               "empirical coverage" if rho == rhos[0] else "")
        if journal:
            ax.set_title(f"({'abc'[i]})", fontsize=8, loc="left")
            ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
            ns = sorted(s.n.unique())
            ax.set_xticks(ns); ax.set_xticks([], minor=True); ax.set_xticklabels([f"{v:g}" for v in ns])
            ax.grid(color="#ebebeb", lw=0.5, alpha=None)   # opaque: EPS has no transparency
    if journal:
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(h, lab, fontsize=8, ncol=4, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0))
        fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
        plt.close(fig)
        return
    np.atleast_1d(axes)[0].legend(fontsize=6.6, frameon=False, loc="lower right")
    fig.suptitle("Coverage of the published few-label repairs (shaded: the ±0.005 band). "
                 "* = our interval, the paper defines none", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    df = pd.read_csv(OUT / "summary.csv")
    rhos = sorted(df.rho.unique())

    # 1. coverage vs n
    coverage_figure(df, FIG / "repair_extension_coverage.png")

    # 2. width vs n
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.1 * len(rhos), 3.7), dpi=200, sharey=True)
    for ax, rho in zip(np.atleast_1d(axes), rhos):
        s = df[df.rho == rho]
        for m in ORDER:
            d = s[s.method == m].sort_values("n")
            if d.empty:
                continue
            ax.plot(d.n, d.width_mean, "-o", ms=4, lw=1.6, color=COL[m], label=LAB[m])
        ax.set_xscale("log"); ax.set_yscale("log")
        _style(ax, f"ρ = {rho}", "gold labels n (log)",
               "mean interval width (log)" if rho == rhos[0] else "")
    np.atleast_1d(axes)[0].legend(fontsize=6.6, frameon=False)
    fig.suptitle("Mean interval width", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "repair_extension_width.png", bbox_inches="tight")
    plt.close(fig)

    # 3. per-method summary: worst coverage and the coverage/MSE split
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=200)
    ax = axes[0]
    ms = [m for m in ORDER if m in set(df.method)]
    worst = [df[df.method == m].coverage.min() for m in ms]
    small = [df[(df.method == m) & (df.n <= 10)].coverage.min() for m in ms]
    y = np.arange(len(ms))
    ax.barh(y - 0.2, worst, 0.38, color=[COL[m] for m in ms], label="worst over the grid")
    ax.barh(y + 0.2, small, 0.38, color=[COL[m] for m in ms], alpha=0.5, label="worst at n ≤ 10")
    ax.axvline(0.95, color="k", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in ms], fontsize=8)
    ax.set_xlim(0.55, 1.0); ax.legend(fontsize=7, frameon=False, loc="lower right")
    _style(ax, "Worst-case coverage", "empirical coverage", "")
    ax = axes[1]
    for m in ms:
        d = df[df.method == m]
        ax.scatter(d.mse, d.coverage, s=34, alpha=0.75, color=COL[m], label=LAB[m])
    ax.axhline(0.95, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    _style(ax, "Point-estimation error against coverage", "MSE of the point estimate (log)",
           "empirical coverage")
    ax.legend(fontsize=6.6, frameon=False, ncol=2)
    fig.suptitle("A method can improve MSE and still under-cover — the two are separate questions",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "repair_extension_methods.png", bbox_inches="tight")
    plt.close(fig)

    for f in ("repair_extension_coverage", "repair_extension_width", "repair_extension_methods"):
        print("wrote figures/" + f + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
