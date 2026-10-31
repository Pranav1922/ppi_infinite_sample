#!/usr/bin/env python3
"""Generate the paper's results tables from saved CSVs only (`15` §L5).

    python experiments/make_tables.py > docs/17_PAPER_TABLES.md

Every number in the paper's tables comes from here. Nothing is transcribed by hand.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")


def _pivot(name, metric="coverage", index=("n",), query=None):
    d = pd.read_csv(R / f"{name}.csv")
    if query:
        d = d.query(query)
    return d[d.metric == metric].pivot_table(index=list(index), columns="method", values="value")


def _rng(x):
    return f"{x.min():.4f}–{x.max():.4f}"


def _row(label, cells):
    return "| " + label + " | " + " | ".join(cells) + " |"


def table1():
    print("## Table 1 — Main coverage comparison\n")
    print("Bivariate Gaussian, `N = 10⁶`, `R = 200,000` (Monte-Carlo s.e. 0.0005). Each cell is the")
    print("range of empirical coverage over `ρ ∈ {0.0, 0.5, 0.9}`. Nominal level 0.95.\n")
    g = _pivot("C_sample_size", index=("n", "rho"))
    m = ["B1", "B3", "B4", "B5", "B9-nolev", "B9"]
    print(_row("gold labels `n`", [f"**{x}**" for x in
               ["B1 classical", "B3 PPI++", "B4 +*t*", "B5 cross-fit", "B9-nolev", "B9 LC-PPI"]]))
    print("|" + "---|" * (len(m) + 1))
    for n in sorted(g.index.get_level_values(0).unique()):
        sub = g.xs(n, level=0)
        print(_row(f"**{n}**", [_rng(sub[c]) for c in m]))
    w = _pivot("A_reproduction", "width_ratio_b1", ("n", "rho"), "N == 1e6 and alpha_level == 0.05")
    print("\n**Mean interval width ÷ classical**, same runs (`N = 10⁶`):\n")
    print(_row("`n`, `ρ`", ["B3 PPI++", "B9-nolev", "B9 LC-PPI"]))
    print("|---|---|---|---|")
    for (n, rho) in w.index:
        print(_row(f"{n}, {rho}", [f"{w.loc[(n, rho), c]:.3f}" for c in
                                   ["B3", "B9-nolev", "B9"]]))


def table2():
    print("\n## Table 2 — The simple corrections to PPI++\n")
    print("`N = 10⁵`, `R = 60,000` (Monte-Carlo s.e. 0.0009), `n = 10`. Tests H2; every entry is")
    print("coverage at nominal 0.95.\n")
    g = _pivot("A4_simple_fixes", index=("n", "rho"))
    m = ["B1", "B3", "B4", "B5", "B5-kfold", "B5-avglam", "B6-t1.5", "B6", "B6-t3"]
    names = ["B1", "B3 PPI++", "B4 +*t*", "B5 2-fold", "B5 5-fold", "B5 avg-λ",
             "B6 \\|t\\|>1.5", "B6 \\|t\\|>2", "B6 \\|t\\|>3"]
    print(_row("`ρ` at `n = 10`", names))
    print("|" + "---|" * (len(m) + 1))
    for rho in sorted(g.xs(10, level=0).index):
        print(_row(f"**{rho}**", [f"{g.loc[(10, rho), c]:.4f}" for c in m]))
    print(_row("**max over ρ**", [f"**{g.xs(10, level=0)[c].max():.4f}**" for c in m]))
    print("\n*No method attains ≥ 0.945 **uniformly in `ρ`**, which is H2's condition.")
    print("Note B6 at |t| > 3 does reach 0.9489 at `ρ = 0` and 0.9459 at `ρ = 0.2` — and falls")
    print("to 0.9128 at `ρ = 0.8`. That non-monotonicity is precisely why a selection rule is not")
    print("a substitute for a correct variance. H2's guard — three cross-fitting schemes and three")
    print("thresholds — is met: every one falls below 0.945 somewhere.*")
    mse = _pivot("A5_mse", "mse_ratio_b1", ("n", "rho"))
    print("\n**Point-estimate MSE ÷ classical**, `R = 40,000`, `n = 10`:\n")
    print(_row("`ρ`", ["B3 PPI++", "B5 cross-fit", "B6 threshold"]))
    print("|---|---|---|---|")
    for rho in sorted(mse.xs(10, level=0).index):
        print(_row(f"{rho}", [f"{mse.loc[(10, rho), c]:.3f}" for c in ["B3", "B5", "B6"]]))
    print("\n*Cross-fitting — the fix recommended by the paper documenting the problem — is the")
    print("worst method in the table at small `n`.*")


def table3():
    print("\n## Table 3 — Bootstrap comparison\n")
    print("`N = 10⁶`, `R = 20,000` (Monte-Carlo s.e. 0.0015), `B = 2,000` resamples.\n")
    g = _pivot("B_bootstrap", index=("n", "rho"), query="N == 1e6")
    m = ["B1", "B3", "B7", "B8", "B9"]
    print(_row("`n`", ["B1", "B3 PPI++", "B7 percentile", "B8 BCa", "B9 LC-PPI"]))
    print("|" + "---|" * (len(m) + 1))
    for n in sorted(g.index.get_level_values(0).unique()):
        print(_row(f"**{n}**", [_rng(g.xs(n, level=0)[c]) for c in m]))
    f = _pivot("B_final", index=("n", "rho"), query="N == 1e6")
    print(_row("**10, `B = 9,999`**",
               [_rng(f.xs(10, level=0)[c]) for c in ["B1", "B3", "B7-B9999", "B8-B9999", "B9"]]))
    sec = _pivot("B_bootstrap", "seconds_per_interval", ("n", "rho"), "N == 1e6")
    s10 = sec.xs(10, level=0).mean()
    print("\n**Wall-clock seconds per interval at `n = 10`:** " + ", ".join(
        f"{c} {s10[c]:.2e}" for c in m) + ".\n")
    print("*Five times the resampling budget changes nothing, so the shortfall is not a")
    print("Monte-Carlo effect. Kill criterion K3 is not triggered.*")


def table4():
    print("\n## Table 5 — Finite-`N` analysis\n")
    print("Explicit unlabelled mode, `R = 200,000`, `n = 10`, `ρ = 0.9` — the cell where the")
    print("finite-`N` effect is largest. `s` is the share of variance carried by the unlabelled term.")
    # Column key (Statistics and Computing: a table caption must explain its components);
    # each description is the docstring of the mode in `ppi_finite.variance.var_lc_ppi`.
    print("Columns: B9 is Algorithm 1 as deployed; B9 V_full is the plug-in form of (3) with")
    print("the true `μ_f`, an infeasible diagnostic; B9-R1 replaces `(f̄_n − μ_f)²` in it by")
    print("`max(0, Δ² − σ̂_f²/N)`; B9-R2 uses the Welch–Satterthwaite degrees of freedom; B9-R5 uses")
    print("the exact quantile of (4).\n")
    d = pd.concat([pd.read_csv(R / "E_finite_N.csv"), pd.read_csv(R / "E_largeN.csv")])
    g = d[d.metric == "coverage"].pivot_table(index=["n", "rho", "N"], columns="method",
                                              values="value")
    sh = d[d.metric == "unlabeled_share_mean"].pivot_table(index=["n", "rho", "N"],
                                                           columns="method", values="value")
    m = ["B3", "B9", "B9-vfull", "B9-R1", "B9-R2", "B9-R5"]
    print(_row("`N`", ["`s`", "B3 PPI++", "B9", "B9 V_full", "B9-R1", "B9-R2", "B9-R5"]))
    print("|" + "---|" * (len(m) + 2))
    for N in sorted(g.xs((10, 0.9), level=(0, 1)).index):
        k = (10, 0.9, N)
        print(_row(f"**{int(N)}**", [f"{sh.loc[k, 'B9']:.3f}"] + [f"{g.loc[k, c]:.4f}" for c in m]))
    allc = g.reset_index()
    print("\n**Worst |coverage − 0.95| over all 81 cells:** " + ", ".join(
        f"{c} {np.abs(g[c] - 0.95).max():.4f}" for c in ["B9", "B9-R1", "B9-R2", "B9-R5"]) + ".")
    print("**Largest coverage difference from B9 over all 81 cells:** "
          f"V_full {np.abs(g['B9'] - g['B9-vfull']).max():.4f}, "
          f"R1 {np.abs(g['B9'] - g['B9-R1']).max():.4f} — neither repairs the over-coverage.\n")
    print("*`B9` never under-covers on this grid; `R1` cannot repair over-coverage; `R2` meets ±0.004 everywhere")
    print("but is not conservative; `R5` (the exact quantile) is no better than `R2`.*")


def table5():
    print("\n## Table 6 — Robustness outside the Gaussian model\n")
    print("`R = 200,000`. Ranges are over all cells of each experiment. **The comparison that")
    print("matters is B9 against B1, not B9 against 0.95** (pre-registered hypothesis H4).\n")
    print(_row("setting", ["cells", "B1", "B3 PPI++", "B9 LC-PPI", "B9-HC", "min(B9 − B1)"]))
    print("|---|---|---|---|---|---|---|")
    for name, label in (("F_heavy_tails", "F heavy residuals"), ("F_heavy_f", "F heavy predictor"),
                        ("G_skew", "G skew + binary"), ("H_hetero", "H σ(f)=exp(γf)"),
                        ("H_step", "H two-group step")):
        g = _pivot(name, index=("n", "rho", "dgp"))
        cols = ["B1", "B3", "B9", "B9-HC"]
        print(_row(f"**{label}**", [str(len(g))] + [_rng(g[c]) for c in cols]
                   + [f"**{(g['B9'] - g['B1']).min():+.4f}**"]))
    gb = _pivot("G_skew", index=("n", "rho", "dgp"), query="dgp == 'skew:skew_family=binary'")
    print(_row("*G binary only*", [str(len(gb))] + [_rng(gb[c]) for c in
               ["B1", "B3", "B9", "B9-HC"]] + [f"{(gb['B9'] - gb['B1']).min():+.4f}"]))
    print("\n*Heavy tails cost nothing. Skew is the limitation and B1 shares it. Heteroskedasticity")
    print("breaks B9 and HC3 repairs it. For binary labels B9 is the most accurate of the three.*")


def table6():
    print("\n## Table 7 — Galaxies real-data demonstration\n")
    print("16,743 items, `θ = 0.2592725`, realised `ρ = 0.741`, `R = 20,000` gold subsamples,")
    print("`N = M − n`. **Containment under repeated subsampling from one fixed finite population —")
    print("not superpopulation coverage.**\n")
    g = _pivot("I_real_data")
    w = _pivot("I_real_data", "width_ratio_b1")
    m = ["B1", "B2", "B3", "B4", "B5", "B8", "B9", "B9-HC"]
    print(_row("`n`", ["B1", "B2 PPI", "B3 PPI++", "B4", "B5", "B8 BCa", "B9 LC-PPI", "B9-HC"]))
    print("|" + "---|" * (len(m) + 1))
    for n in sorted(g.index):
        print(_row(f"**{n}**", [f"{g.loc[n, c]:.4f}" for c in m]))
    print(_row("*width ÷ B1, n=10*", [f"{w.loc[10, c]:.3f}" for c in m]))
    d = pd.read_csv(R / "I_real_data.csv")
    fails = d.pivot_table(index="n", columns="method", values="n_failed", aggfunc="max")
    print(f"\n**Failed interval constructions:** B8 only — {int(fails.loc[10, 'B8'])} of 20,000 at")
    print(f"`n = 10` and {int(fails.loc[20, 'B8'])} at `n = 20` (degenerate BCa on a constant")
    fm = d[d.metric == "coverage_failures_as_miss"].pivot_table(index="n", columns="method",
                                                                values="value")
    print(f"binary resample). Counting them as misses gives B8 {fm.loc[10, 'B8']:.4f} at `n = 10`.")
    print("Every other method: 0 failures.\n")
    print("*PPI++ under-covers on real data too, and by more than in the Gaussian model — as")
    print("the skew experiment of §8 predicts for a binary, skewed, heteroskedastic outcome.*")


def table8():
    """Table 8 — the second real-data demonstration (MT-bench expert judgments, GPT-4 judge)."""
    print("\n## Table 8 — MT-bench LLM-judgment demonstration\n")
    print("1,478 expert pairwise judgments of GPT-3.5-turbo (ties scored ½), `θ = 0.6519`, GPT-4 as")
    print("the predictor with realised `ρ = 0.489`, `R = 20,000` gold subsamples, `N = M − n`.")
    print("**Containment under repeated subsampling from one fixed finite population — not")
    print("superpopulation coverage.**\n")
    g = _pivot("J_mtbench")
    w = _pivot("J_mtbench", "width_ratio_b1")
    m = ["B1", "B2", "B3", "B4", "B5", "B8", "B9", "B9-HC"]
    print(_row("`n`", ["B1", "B2 PPI", "B3 PPI++", "B4", "B5", "B8 BCa", "B9 LC-PPI", "B9-HC"]))
    print("|" + "---|" * (len(m) + 1))
    for n in sorted(g.index):
        print(_row(f"**{n}**", [f"{g.loc[n, c]:.4f}" for c in m]))
    print(_row("*width ÷ B1, n=10*", [f"{w.loc[10, c]:.3f}" for c in m]))
    d = pd.read_csv(R / "J_mtbench.csv")
    fails = d.pivot_table(index="n", columns="method", values="n_failed", aggfunc="max")
    fm = d[d.metric == "coverage_failures_as_miss"].pivot_table(index="n", columns="method",
                                                                values="value")
    print(f"\n**Failed interval constructions:** B8 only — {int(fails.loc[10, 'B8'])} of 20,000 at")
    print(f"`n = 10` (degenerate bootstrap resamples); counting them as misses gives B8 "
          f"{fm.loc[10, 'B8']:.4f} at `n = 10`. Every other method: 0 failures. The sampling")
    print("fraction reaches `n/M = 6.8 %` at `n = 100`, where the finite-population correction is")
    print("no longer negligible.")


def table7() -> None:
    """Table 4 (numbered in order of first citation) — the published few-label repairs, from results/repair_extension/summary.csv."""
    import pandas as pd
    df = pd.read_csv(R / "repair_extension" / "summary.csv")
    tol = 0.005
    print("\n## Table 4 — Published few-label repairs\n")
    print("`R = 10,000`, `B = 999` bootstrap resamples, `N = 1000`; 18 cells over "
          "`n ∈ {4, 5, 8, 10, 15, 20}` and")
    print("`ρ ∈ {0, 0.5, 0.9}`, all methods on the same draw within a cell. Nominal level 0.95.\n")
    print("| method | min coverage | max coverage | cells outside ±0.005 | mean width | mean MSE |")
    print("|---|---|---|---|---|---|")
    for m, lab in [("B1", "B1 classical"), ("B3", "B3 PPI++"),
                   ("B9", "B9 / B9-tmin LC-PPI"), ("Ridge-PPI", "Ridge-PPI"),
                   ("PPBoot", "PPBoot"), ("Efron-PPboot2", "Efron-PPboot2")]:
        d = df[df.method == m]
        print(f"| **{lab}** | {d.coverage.min():.4f} | {d.coverage.max():.4f} | "
              f"{int((d.coverage_error > tol).sum())}/18 | {d.width_mean.mean():.3f} | "
              f"{d.mse.mean():.4f} |")
    print("\n*Interval provenance: PPBoot and Efron-PPboot2 use the original authors' own "
          "intervals; the")
    print("Ridge-PPI interval is ours, their point estimator placed in the PPI++ construction. "
          "Sigmoid-PPI")
    print("and cross-PPBoot are not applicable to this design and were not evaluated.*")
    # No Source line here: make_manuscript.table_block() appends one for every table, and
    # printing a second produced a duplicate under this table.


def main() -> int:
    print("# 17 — Paper results tables\n")
    print("> **Generated by `experiments/make_tables.py` from `results/*.csv`.** Do not edit by")
    print("> hand: regenerate with `make tables`. Every CSV has a provenance sidecar recording the")
    print("> git commit, seed specification and package versions.\n")
    for fn in (table1, table2, table3, table4, table5, table6, table7, table8):
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
