# 07 — Experiment Plan

Nine experiments, A–I. Run them in order; each gates the next. Every experiment writes a CSV to
`results/` and a figure to `figures/` with the names given below.

**Universal conventions (apply to every experiment unless overridden):**

- Nominal confidence level 95% (`alpha_level = 0.05`). Experiment A additionally reports 90% and 99%.
- **Common random numbers**: within a cell, every baseline sees the *identical* simulated dataset.
  Violating this invalidates all paired comparisons. See `09` §Random number generation.
- Replications `R` as specified per experiment; the Monte-Carlo standard error of a coverage
  estimate near 0.95 is `√(0.95·0.05/R)` — 0.00049 at R = 200,000, 0.00069 at R = 100,000,
  0.0022 at R = 10,000.
- Report coverage to **four decimal places** with its MC standard error. Do not interpret a
  difference smaller than 3 combined standard errors.
- Baseline identifiers exactly as in `06`.

---

## Experiment A — Reproduction

**Objective.** Independently reproduce the preliminary Gaussian results before building anything on
top of them. This is a gate: if A fails, nothing downstream is trustworthy.

**Data-generating process.** `f ~ N(0,1)`; `Y = ρ f + √(1−ρ²) ε`, `ε ~ N(0,1)` independent. So
`Var(Y) = Var(f) = 1`, `Corr(Y,f) = ρ`, and the estimand is `θ = E[Y] = 0`.

**Variables.**

| Variable | Values |
|---|---|
| `n` | 10, 20, 50 (plus 100 for the extended table) |
| `ρ` | 0.0, 0.5, 0.9 (plus 0.2, 0.8, 0.95 for the extended table) |
| `N` | 10⁶ (primary), also 1000 and 200 |
| `alpha_level` | 0.05 (primary), 0.10 and 0.01 as a secondary check |
| `R` | 200,000 |

**Baselines.** B1, B3, B4, B5, B6, B9. (B2, B7, B8, B10 enter in later experiments.)

**Metrics.** Coverage; mean and median width; width ratio to B1; MSE ratio to B1.

**Expected plots.**
- `figures/A_coverage_vs_n.png` — coverage against n, one line per method, one panel per ρ,
  horizontal reference at 0.95.
- `figures/A_width_vs_rho.png` — width ratio to B1 against ρ, one line per method, one panel per n.

**Interpretation.** B1 must sit at nominal. B3 must sit near 0.878 at n = 10. B9 must sit near 0.950
at N = 10⁶.

**Failure conditions.**
- B1 outside 0.950 ± 0.003 → the harness is wrong; stop and debug.
- B3 at n = 10, N = 10⁶ outside 0.878 ± 0.005 → the B3 implementation differs from the preliminary
  one; find the difference before proceeding.
- B9 at n = 10, N = 10⁶ outside 0.950 ± 0.004 → the correction is not what was implemented before;
  compare against `exact.py` line by line.

---

## Experiment B — Bootstrap comparison

**Objective.** Determine whether a bootstrap already solves the problem. **This is kill criterion
K3** and is the highest-priority experiment after A.

**Variables.** `n ∈ {10, 20, 50}`; `ρ ∈ {0.0, 0.3, 0.5, 0.7, 0.9}`; `N = 10⁶`; `B` (bootstrap
resamples) `∈ {2000}` for exploration and `9999` for the final table; `R = 20,000` replications
(reduced because the bootstrap is expensive — MC standard error 0.0015, adequate to distinguish
0.92 from 0.95).

**Baselines.** B1, B3, B7, B8, B9.

**Metrics.** Coverage; width ratio to B1; **wall-clock seconds per interval**, reported explicitly.

**Expected plots.** `figures/B_bootstrap_coverage.png`; `figures/B_cost_vs_coverage.png` (a scatter
of wall-clock against coverage, which is the argument that survives even if BCa works).

**Interpretation.**
- If B8 attains ≥ 0.945 at n = 10 across all ρ → K3 is triggered. Do **not** abandon the project;
  reframe it (see `10` K3) and record the decision in `13_RESEARCH_LOG.md`.
- If B8 falls short → LC-PPI's case strengthens considerably, and B8's shortfall becomes a headline
  result in its own right.

**Failure conditions.** Bootstrap coverage above 0.96 at every cell indicates an implementation bug
(likely resampling the wrong thing or double-counting). Verify against B1 by bootstrapping the plain
mean, which should reproduce the classical interval closely.

---

## Experiment C — Sample-size sweep

**Objective.** Map the deficit and the correction as functions of n, especially in the very-small-n
region that no published study covers.

**Variables.** `n ∈ {4, 5, 8, 10, 15, 20, 50}`; `ρ ∈ {0.0, 0.5, 0.9}`; `N = 10⁶`; `R = 200,000`.

**Baselines.** B1, B3, B4, B5, B9.

**Metrics.** Coverage; width ratio; **the empirical deficit `0.95 − coverage(B3)` plotted against
`1/n`** — the leverage hypothesis predicts approximate linearity in `1/n`.

**Expected plots.**
- `figures/C_coverage_vs_n.png`.
- `figures/C_deficit_vs_inverse_n.png` — **the diagnostic plot.** If the B3 deficit is approximately
  linear in `1/n` and passes near the origin, that supports the `O(1/n)` relative-distortion argument
  in `05`.

**Interpretation.** B9 should remain at nominal for every n ≥ 4. If B9 degrades at n = 4 or 5, the
exactness argument has an unstated constraint that must be found.

**Failure conditions.** B9 outside 0.950 ± 0.004 at any n ≥ 4 → the pivot argument is wrong
somewhere; re-derive `04` §4.7 before continuing.

---

## Experiment D — Predictor-quality sweep

**Objective.** Confirm that the B3 deficit is invariant in ρ (which is the signature of an
estimated-parameter cost rather than a predictor-quality problem), and map where each fix breaks.

**Variables.** `ρ ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99}`; `n ∈ {10, 20, 50}`;
`N = 10⁶`; `R = 200,000`. Additionally, sweep the B6 threshold over `|t| ∈ {1.5, 2, 3}` and the B10
shrinkage constant over `c ∈ {1, 2, 3}`.

**Baselines.** B1, B2, B3, B4, B5, B6, B9, B10.

**Metrics.** Coverage; width ratio; **fraction of the oracle efficiency gain retained**, where the
oracle interval uses the population λ* and the true variance.

**Expected plots.**
- `figures/D_coverage_vs_rho.png` — the key figure; B3 should be a flat line well below 0.95, B6
  should be visibly non-monotone, B9 should be flat at 0.95.
- `figures/D_efficiency_frontier.png` — width ratio to B1 against ρ, with the oracle as a lower
  envelope. This is the picture of the validity–efficiency frontier.

**Interpretation.** The flatness of B3 in ρ is a *prediction of the leverage hypothesis*. If instead
B3's deficit varies strongly with ρ, the diagnosis in `04` §4.6 is incomplete.

**Failure conditions.** B3 coverage varying by more than 0.01 across ρ at fixed n → the diagnosis
needs revision; log it and re-derive.

---

## Experiment E — Finite-N sweep

**Objective.** Characterise the known inexactness at finite N and test the candidate repairs R1–R4
from `04` §4.8.4.

**Variables.** `N ∈ {50, 100, 200, 500, 1000, 5000, 10⁴, 10⁵, 10⁶}`; `n ∈ {10, 20, 50}`;
`ρ ∈ {0.0, 0.5, 0.9}`; `R = 200,000`. Also sweep `coef_mode ∈ {"ols", "ppi_plus"}` (see `05`).

**Baselines.** B1, B3, B9, plus **R1, R2, R3, R4 as variants of B9** (label them `B9-R1` … `B9-R4`).

**Metrics.** Coverage; width ratio; the **over-coverage** `coverage − 0.95` plotted against `N/n`
and against ρ.

**Expected plots.**
- `figures/E_coverage_vs_N.png` — one panel per n, one line per ρ, for B9 and each repair.
- `figures/E_overcoverage_vs_rho.png` — tests the conjecture in `04` §4.8.3 that over-coverage
  scales with ρ.

**Interpretation.**
- Reproduce first: B9 should over-cover, ≈ 0.9612 at (N=200, n=10, ρ=0.9) and ≈ 0.9513 at
  (N=200, n=10, ρ=0.0).
- Then verify (V_full) numerically: implement the variance of `04` §4.8.3 directly and check it
  against the empirical variance of `θ̂_LC` over replicates. **This is the first numerical test that
  (V_full) has ever received.**
- Then compare R1–R4 and select on coverage-closest-to-nominal subject to never under-covering.

> **RUN 2026-08-20 (E4).** The `N` grid is split by unlabelled mode, because `08` §T1.1 requires
> `explicit` mode for `N ≤ 5000` while `explicit` at `N = 10⁶` with `R = 200,000` would be 2·10¹¹
> draws per cell: `configs/experiment_E_finite_N.toml` covers `N ≤ 5000` in `explicit` mode and
> `configs/experiment_E_largeN.toml` covers `N ≥ 10⁴` in `mean_only`, where `08` says the
> approximation is harmless. Together they are `07`'s full grid, 81 cells at `R = 200,000`.
> **E4c addendum (2026-08-20).** The idealisation-error study is not an experiment in the
> `07` sense — it produces no `results/*.csv` and sweeps no method. It lives in
> `ppi_finite.theory.ablation_coverages` (in `src/`, because `11` forbids a variance
> formula in `experiments/`) and its numbers are recorded in `docs/theory_finite_N.md`
> Part II and in `13`. Its ablations must never be registered as methods: they use
> population quantities no practitioner has.
>
> A separate convergence arm (`configs/experiment_E_convergence*.toml`, `R = 20,000`) runs both
> modes on the same grid and finds them **indistinguishable at every N down to 200** (largest
> coverage difference 0.0047, ≈ 2.2 combined Monte-Carlo standard errors). `coef_mode` is swept via
> the registry entry `B9-ppiplus` rather than a config key. Results and verdicts: `04` §4.8.3-4.8.4
> E4 updates, `13`, `14` Q1.

**Failure conditions.**
- If B9 *under*-covers at any finite N, that contradicts the preliminary data and the conservativeness
  hypothesis (R4) is dead; log it prominently.
- If (V_full) does not match the empirical variance, `04` §4.8.3 contains an algebra error; fix `04`
  first, then re-run.

---

## Experiment F — Heavy tails

**Objective.** Test H4. Determine how fast the exactness of the `t` pivot degrades when the Gaussian
assumption A1 fails in the tails.

**Data-generating process.** Replace the Gaussian residual by a scaled Student-*t*:
`Y = ρ f + √(1−ρ²) · ε`, `ε = T_ν / √(ν/(ν−2))` (standardised to unit variance), with
`ν ∈ {3, 5, 10, 30}`. Also run a variant with heavy-tailed **`f`** (not just heavy-tailed residuals),
because leverage depends on the tails of `f` and this is where the correction is most exposed.

**Variables.** `ν` as above; `n ∈ {10, 20, 50}`; `ρ ∈ {0.0, 0.5, 0.9}`; `N = 10⁶`; `R = 200,000`.

**Baselines.** B1, B3, B8, B9.

**Metrics.** Coverage; width ratio. Also report B1's coverage, which will itself depart from nominal
under heavy tails — **the correct comparison is B9 against B1, not B9 against 0.95**, because a
method cannot be blamed for a failure the classical interval shares.

**Expected plots.** `figures/F_coverage_vs_nu.png`.

**Interpretation.** Degradation is expected and is not a failure; the question is the rate. Report
the smallest ν at which B9 falls below 0.94.

> **RUN 2026-08-20 (E5).** `configs/experiment_F_heavy_tails.toml` (heavy residuals) and
> `experiment_F_heavy_f.toml` (heavy predictor), both at the pre-registered `R = 200,000` with the
> analytic methods. **Deviation, stated:** `07` names B8 as a baseline *and* `R = 200,000`; a BCa
> interval costs ≈ 10⁴× an analytic one, so the two cannot both be honoured in one run. The B8
> baseline is in `experiment_F_bootstrap.toml` at `R = 20,000`, following Experiment B's own
> precedent and its stated MC s.e. of 0.0015. Results: `10` H4/K4 and `13`.

**Failure conditions.** This is kill criterion **K4**: if B9 drops below 0.93 at ν ≥ 5 and n ≥ 20,
and cannot be repaired by a studentisation or a robust scale estimate, the practical scope of the
method is much narrower than claimed and must be stated as such.

---

## Experiment G — Skew

**Objective.** Test H4 for asymmetric errors, which stress a symmetric interval differently from
heavy tails.

**Data-generating process.** Residuals from a standardised skew distribution: a centred and scaled
`Exponential(1)`, `Gamma(k, 1)` for `k ∈ {1, 2, 5}`, and a lognormal with mild and severe skew. Also
a **discrete** variant with `Y ∈ {0, 1}` (Bernoulli with `P(Y=1|f)` a logistic function of `f`) —
this is the case that matters most for real LLM-annotation applications, where the gold label is a
binary judgement.

**Variables.** skew family as above; `n ∈ {10, 20, 50, 100}`; `ρ ∈ {0.0, 0.5, 0.9}` (or the closest
achievable correlation for the discrete case); `N = 10⁶`; `R = 200,000`.

**Baselines.** B1, B3, B7, B8, B9.

**Metrics.** Coverage; **one-sided miscoverage in each tail separately** — with skewed errors a
symmetric interval can have nominal total coverage while being badly wrong in one tail, and total
coverage alone hides this.

**Expected plots.** `figures/G_coverage_by_tail.png` — stacked left-tail / right-tail miscoverage.

**Interpretation.** If B9 has correct total coverage but asymmetric tail errors, that motivates B8
(BCa is designed for exactly this) and should be reported as a limitation, not hidden.

> **RUN 2026-08-20 (E5).** `configs/experiment_G_skew.toml`, full 7-family × 4-n × 3-ρ grid at
> `R = 200,000`. **Deviations, stated:** (i) the B7/B8 baselines are in
> `experiment_G_bootstrap.toml` at `R = 20,000` and on a reduced grid (3 families × 2 n × 3 ρ), for
> the cost reason above; (ii) the binary family's `ρ = 0.9` cell is run at its closest achievable
> correlation, **0.7979**, because with Gaussian `f` and prevalence 0.5 the correlation is capped
> at `2φ(0)` — `07` provides for this ("or the closest achievable correlation for the discrete
> case") and the cap is proved in `tests/test_robustness.py`; (iii) `08` says "lognormal, mild and
> severe" without values, so `σ = 0.5` and `σ = 1.0` were chosen — an implementation decision
> logged in `13`.
>
> **On this failure condition specifically:** B9's tail miscoverage is **not** worse than B1's. At
> the worst cell (lognormal severe, n = 10, ρ = 0) B9 is 0.0032/0.1493 left/right against B1's
> 0.0010/0.1600 — B9's right tail is *better*. The condition is not met.

**Failure conditions.** B9 tail miscoverage worse than B1's at the same cell → the correction is
making skew *worse*, which would be a real problem.

---

## Experiment H — Heteroskedasticity

**Objective.** Test assumption A7. `s²` is a single pooled residual variance; if `Var(Y|f)` depends
on `f`, the leverage term is misweighted.

**Data-generating process.** `Y = ρ f + √(1−ρ²) · σ(f) · ε / c` with `σ(f) = exp(γ f)` for
`γ ∈ {0, 0.25, 0.5, 1.0}` and `c` a normalising constant keeping `Var(Y) = 1`. Also a variance
that is a step function of `f` (two-group heteroskedasticity), which is closer to realistic
annotation settings where the model is reliable on easy items and noisy on hard ones.

**Variables.** `γ` as above; `n ∈ {10, 20, 50}`; `ρ ∈ {0.0, 0.5, 0.9}`; `N = 10⁶`; `R = 200,000`.

**Baselines.** B1, B3, B8, B9, plus **B9-HC**: a variant of B9 using a heteroskedasticity-consistent
(HC3) sandwich estimate in place of `s²(1/n + leverage)`.

**Metrics.** Coverage; width ratio.

**Expected plots.** `figures/H_coverage_vs_gamma.png`.

**Interpretation.** If B9 degrades and B9-HC does not, the recommendation becomes "use the HC variant
when heteroskedasticity is suspected", and HC3 (which is itself a leverage-weighted estimator)
becomes part of the method rather than an afterthought.

> **OPEN — requires research decision.** Whether the HC variant retains any exactness claim. Almost
> certainly not; it would be an asymptotic refinement. Do not claim otherwise.
>
> **RUN 2026-08-20 (E5), and the recommendation `07` anticipated is now supported.**
> `configs/experiment_H_hetero.toml` (`σ(f) = exp(γf)`) and `experiment_H_step.toml` (two-group),
> both at `R = 200,000`. B9 degrades exactly as feared — 0.9047 at γ = 1, n = 10 — and **B9-HC does
> not**: its minimum over all 72 heteroskedastic cells is **0.9517**, and it is below 0.94 in none
> of them. So "use the HC variant when heteroskedasticity is suspected" is now an evidenced
> recommendation rather than a guess. **No exactness is claimed for it**; the `t_{n−2}` quantile it
> keeps is a convention for comparability, not a derivation. Note also that HC3 is *conservative*
> under homoskedasticity at small n (0.9584 vs B9's 0.9503 at n = 20), which is the price.

---

## Experiment I — Real AI-generated labels

**Objective.** Show that the phenomenon is not an artefact of simulation, on data where `f` is an
actual model's output and `Y` is an actual human label.

**Design.** Take a dataset with both gold labels and model predictions on the *same* items (see
`08`, Tier 2 and Tier 3). Treat the full dataset as the population, so that the true θ is known
exactly. Then repeatedly:

1. draw a random gold subsample of size n (without replacement);
2. treat a disjoint subset of size N as the "unlabelled" pool, using only its predictions;
3. construct all intervals;
4. check whether each contains the population θ.

Repeat `R = 20,000` times (subsampling replications, which is cheap once the dataset is in memory).

**Variables.** `n ∈ {10, 20, 50, 100}`; `N` as large as the dataset allows; dataset ∈ Tier 2/Tier 3
list; estimand = population mean of the gold label.

**Baselines.** B1, B3, B4, B5, B8, B9.

**Metrics.** Coverage against the known population θ; width ratio; the realised `ρ` of the dataset
(reported, not controlled).

**Expected plots.** `figures/I_real_data_coverage.png` — one panel per dataset.

**Interpretation.** This is a **confirmation** experiment, not a proof. Sampling without replacement
from a finite population is not the same as i.i.d. sampling from a superpopulation, and that
difference must be stated. If B3 under-covers here too, the practical claim is established; if it
does not, that is important and must be investigated rather than buried.

> **RUN 2026-08-20 (E6).** `configs/experiment_I_real_data.toml` on the Galaxies population
> (16,743 items, `θ = 0.2592725`, realised `ρ = 0.741`), `n ∈ {10, 20, 50, 100}`, `N = M − n`,
> `R = 20,000`. Dataset documented in `docs/tier2_galaxies.md`; adapter in
> `src/ppi_finite/realdata.py`. **Deviations, stated:** `[sweep]` omits `rho` and `N` because on
> real data they are observed, not controlled (the runner writes the realised values into the CSV);
> and `B2` and `B9-HC` were added to `07`'s list — `B2` because the user-facing comparison is
> classical / PPI / PPI++ / LC-PPI, `B9-HC` because heteroskedasticity here is **guaranteed**, not
> suspected (binary `Y` ⇒ `Var(Y|f) = p(f)(1−p(f))`; measured 12.6× variance ratio across
> quintiles of `f`).
>
> **On the "large discrepancy" clause: there is none, and the agreements are the interesting part.**
> B3 under-covers at small `n` here too (0.800 at `n = 10`), which is the phenomenon the project
> exists to document, and it under-covers *more* than in the Gaussian simulation (0.878) — as
> Experiment G would predict for a binary, skewed, heteroskedastic outcome. B9 improves on it
> (0.900) without reaching nominal, and **B9-HC does reach it from `n = 20`** (0.951 / 0.953 /
> 0.952), which is exactly the recommendation E5 arrived at. Results and interpretation: `13`.

**Failure conditions.** None that kill the project — Experiment I is deliberately **not** load-bearing
for the theory (see `08`). But a large discrepancy between simulation and real data must be explained.

---

## Experiment dependency graph

```
A (reproduction)  ─┬─►  C (n sweep)    ─┐
                   ├─►  D (ρ sweep)    ─┤
                   ├─►  E (N sweep)    ─┼─►  I (real data)
                   └─►  B (bootstrap)  ─┤
                                        │
                         F, G, H (robustness) ─┘
```

A gates everything. B should be run immediately after A because it can change the framing of the
whole project. E is the main theoretical contribution work. F/G/H can run in parallel with E.
