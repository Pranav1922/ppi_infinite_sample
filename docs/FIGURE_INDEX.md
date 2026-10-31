# Figure index

All 16 figures. Regenerate with `make figures`; `plots.py` reads `results/` only and recomputes no
statistic (`15` §L5). Paper placement follows `docs/PAPER_OUTLINE.md`.

---

## Experiment A — reproduction

### `A_coverage_vs_n.png`
* **Experiment:** A. `N = 10⁶`, `α = 0.05`, `R = 200,000`.
* **Purpose:** the headline failure and its repair, at the three canonical sample sizes.
* **x:** gold-label sample size `n` (log). **y:** empirical coverage. Panels: `ρ`.
* **Methods:** B1, B2, B3, B9-nolev, B9.
* **Conclusion:** PPI++ sits near 0.878 at `n = 10` regardless of `ρ`; the corrected interval sits
  on 0.95; the no-leverage control sits between, at ≈ 0.937.
* **Paper:** §9 Results. *Candidate Figure 1 if §3 does not take `C_coverage_vs_n`.*

### `A_width_vs_rho.png`
* **Experiment:** A. **Purpose:** the price of validity.
* **x:** `ρ`. **y:** mean width ÷ classical. Panels: `n`. **Methods:** B2, B3, B9-nolev, B9.
* **Conclusion:** the valid interval is 8 % wider than classical at `ρ = 0`, `n = 10` and 53 %
  narrower at `ρ = 0.9`. PPI++'s apparent narrowness is under-coverage in different units.
* **Paper:** §9 Results, next to Table 1's width block.

## Experiment C — sample-size sweep

### `C_coverage_vs_n.png`
* **Experiment:** C. `n = 4 … 50`, `R = 200,000`.
* **Purpose:** the strongest single result — behaviour below `n = 10`, which no published study
  covers.
* **x:** `n` (log). **y:** coverage. Panels: `ρ`. **Methods:** B1, B3, B4, B5, B9-nolev, B9.
* **Conclusion:** PPI++ collapses to **0.681 at `n = 4`**; the corrected interval holds at
  0.9495–0.9505 for every `n ≥ 4`.
* **Paper:** **§3 Introduction — Figure 1.** This is the figure that states the problem.

### `C_deficit_vs_inverse_n.png`
* **Experiment:** C. **Purpose:** the diagnostic `07` names — is the deficit `O(1/n)`?
* **x:** `1/n`. **y:** `0.95 − coverage(B3)`. **Methods:** B3, all three `ρ` overlaid.
* **Conclusion:** approximately linear in `1/n` over `n ∈ [8, 50]` with curvature at `n = 4, 5`;
  a free-intercept fit gives `1.127/n − 0.0295`, so the relationship is close to but not exactly
  proportional to `1/n`. The three `ρ` curves coincide — the `ρ`-invariance, seen directly.
* **Paper:** §9 Results, supporting the mechanism argument. *Report the intercept honestly.*

## Experiment D — predictor-quality sweep

### `D_coverage_vs_rho.png`
* **Experiment:** D. Ten-point `ρ` grid, `0 → 0.99`.
* **x:** `ρ`. **y:** coverage. Panels: `n`. **Methods:** B1, B3, B4, B5, B6, B9-nolev, B9.
* **Conclusion:** the PPI++ deficit is flat in `ρ` (spread ≤ 0.0018) — the signature of a cost of
  *estimating* `λ̂`. B6's threshold rule is visibly non-monotone, worst near `ρ = 0.7`.
* **Paper:** §9 Results, with the `ρ`-invariance claim.

## Experiment B — bootstrap

### `B_bootstrap_coverage.png`
* **x:** `ρ`. **y:** coverage. Panels: `n`. **Methods:** B1, B3, B7, B8, B9.
* **Conclusion:** BCa reaches only ≈ 0.90 at `n = 10` and is indistinguishable from the percentile
  bootstrap. Kill criterion K3 does not fire.
* **Paper:** §9 Results.

### `B_cost_vs_coverage.png`
* **x:** wall-clock seconds per interval (log). **y:** coverage. **Methods:** B1, B3, B7, B8, B9.
* **Conclusion:** four to five orders of magnitude separate the analytic intervals from the
  bootstraps, and neither bootstrap reaches nominal. This is the argument `07` says survives even
  if BCa had worked.
* **Paper:** §9 Results.

## Experiment E — finite `N`

### `E_coverage_vs_N.png`
* **Experiment:** E, `ρ = 0.9` (the cell where the effect is largest), explicit unlabelled mode.
* **x:** pool size `N` (log). **y:** coverage. Panels: `n`. **Methods:** B1, B3, B9, B9-R1, B9-R2,
  B9-R5.
* **Conclusion:** B9 over-covers as `N` falls, never under-covering; R1 does not repair it; R2
  removes it; **R5, the exact quantile, does not improve on R2** — which localises the residual in
  the idealisation, not the approximation.
* **Paper:** §6 Mathematical Characterization or §9 Results.

## Experiments F, G, H — robustness

### `F_coverage_vs_nu.png`
* **x:** residual degrees of freedom `ν`. **y:** coverage. Panels: `n`. **Methods:** B1, B3, B9,
  B9-HC.
* **Conclusion:** heavy tails cost nothing — B9 tracks B1 to within 0.0022; both drift up together
  as `ν` falls. **The caption must state that the comparison is B9 against B1, not against 0.95.**
* **Paper:** §10 Robustness.

### `G_coverage_by_tail.png`
* **x:** skew family. **y:** left (pale) and right (solid) miscoverage, stacked. **Methods:** B1, B3,
  B9, B9-HC. `n = 10`.
* **Conclusion:** skewed errors produce badly asymmetric miscoverage for *everything*, and B9's is
  no worse than B1's — `07` Experiment G's failure condition is not met.
* **Paper:** §10 Robustness.

### `H_coverage_vs_gamma.png`
* **x:** heteroskedasticity `γ`. **y:** coverage. Panels: `n`. **Methods:** B1, B3, B9, B9-HC.
* **Conclusion:** B9 degrades as `γ` grows (0.905 at `γ = 1`, `n = 10`); HC3 does not.
* **Paper:** §10 Robustness. *The figure that justifies recommending HC3.*

## Experiment I — Galaxies real data

> **Every caption in this group must state that the quantity is containment under repeated
> subsampling from one fixed finite population, not superpopulation coverage.**

### `I_method_comparison.png`
* **x:** `n` (log). **y:** fraction of subsamples containing the population `θ`. **Methods:** all
  nine. Caption already carries the interpretation note on the figure face.
* **Conclusion:** PPI++ at 0.800 at `n = 10`; B9-HC nominal from `n = 20`.
* **Paper:** **§11 — the real-data figure.**

### `I_intervals.png`
* **x:** 30 random gold subsamples, sorted by the classical estimate. **y:** the intervals
  themselves; × marks a miss. Panels `n = 10, 50`. **Methods:** B1, B3, B9, B9-HC.
* **Conclusion:** shows *why* — at `n = 10` the corrected intervals are occasionally very wide
  (small `S_ff` inflating leverage) while PPI++'s are uniformly too short.
* **Paper:** §11, or supplementary if space is short.

### `I_interval_widths.png`
* **x:** `n` (log). **y:** mean width (left, log) and width ÷ B1 (right). **Methods:** all nine.
* **Conclusion:** the width ordering on real data matches the simulation.
* **Paper:** §11.

### `I_point_estimates.png`
* **x:** `n`. **y:** mean estimate ± 1.96 MC s.e., against the known `θ = 0.2593`.
* **Conclusion:** every method is unbiased to four decimals — the differences are entirely in the
  *interval*, not the estimate.
* **Paper:** §11 or supplementary. *Useful for pre-empting "is the estimator biased?"*

### `I_lambda_distribution.png`
* **x:** `λ̂`. **y:** density, one curve per `n`.
* **Conclusion:** `λ̂` is centred near 0.98 but has sd 0.53 at `n = 10` and is negative in 2.1 % of
  subsamples — a direct picture of the estimated-coefficient variability that causes the problem.
* **Paper:** §11 or §6, whichever needs the intuition more.

---

## Numbering proposal

| paper figure | file |
|---|---|
| 1 | `C_coverage_vs_n.png` |
| 2 | `C_deficit_vs_inverse_n.png` |
| 3 | `D_coverage_vs_rho.png` |
| 4 | `A_width_vs_rho.png` |
| 5 | `B_cost_vs_coverage.png` |
| 6 | `E_coverage_vs_N.png` |
| 7 | `F_coverage_vs_nu.png` |
| 8 | `H_coverage_vs_gamma.png` |
| 9 | `I_method_comparison.png` |
| S1–S7 | `A_coverage_vs_n`, `B_bootstrap_coverage`, `G_coverage_by_tail`, `I_intervals`, `I_interval_widths`, `I_point_estimates`, `I_lambda_distribution` |
