# 09 — Reproduction Protocol

> **This document is the reproduction source of truth.** The numbers below are the acceptance
> targets for Experiment A. If a reimplementation cannot hit them within the stated tolerances,
> **do not proceed to any other experiment** — find the discrepancy first.

> **All numbers in this document are existing preliminary results and must be independently
> reproduced by the implementing agent.** They were produced once, by three scripts, during project
> selection. They are not independently verified. Treat them as targets, not as truths.

---

## Environment

**Reference environment in which the preliminary numbers were produced:**

| Component | Version |
|---|---|
| Python | 3.11.15 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| OS | Linux (x86-64) |
| Threads | 2 |

**Requirements for reproduction:** Python ≥ **3.11**; NumPy ≥ 1.24; SciPy ≥ 1.10. Results should be
insensitive to these versions — the computations are elementary — but the exact versions used must
be recorded in every results CSV (see §Output files).

> **Amended 2026-08-19 (was: Python ≥ 3.10).** Raised to 3.11 because
> `15_ARCHITECTURE_SPECIFICATION.md` §7.2 uses TOML experiment configs parsed with the standard
> library `tomllib`, which is stdlib only from 3.11. This matches the reference environment above
> (3.11.15) and adds no dependency. See `15` §13 CONFLICT 1 for the alternatives that were rejected.

Reproduction does **not** require a GPU, network access, or more than ~2 GB of RAM if the mean-only
unlabelled mode is used (`08` §T1.1).

---

## Seed policy

**Rules that must not be violated:**

1. Use `numpy.random.Generator` (`np.random.default_rng(seed)`). Do **not** use the legacy
   `np.random.seed` / global state.
2. **One generator per experiment cell**, seeded deterministically from the cell coordinates. The
   preliminary scripts used:
   - `exact.py`: `seed = 31*n + int(100*rho) + N`
   - `cover.py`: `seed = 7717*n + int(1000*rho)`
   - `probe.py`: `seed = 1000*n + int(100*rho)`

   These formulas are recorded for provenance. A reimplementation **should not copy them** — they
   can collide (e.g. `exact.py` at different (n, ρ, N) combinations). Use instead a documented,
   collision-free scheme such as
   `SeedSequence(entropy=BASE_ENTROPY, spawn_key=(exp_id, n, rho_index, N_index, dgp_index))`.
   Record `BASE_ENTROPY` in the repository as a constant.
3. **Common random numbers are mandatory.** Within a cell, generate the data **once** and pass the
   same arrays to every baseline. Never regenerate per method. This is what makes paired comparisons
   valid, and it is the easiest thing in the whole project to get silently wrong.
4. Bootstrap resampling (B7/B8) needs its own child generator, spawned from the cell generator, so
   that adding a bootstrap method does not perturb the draws seen by the analytic methods.

---

## Simulation parameters — Experiment A (the reproduction target)

| Parameter | Value |
|---|---|
| DGP | T1.1 bivariate Gaussian (`08`) |
| `Var(Y)`, `Var(f)` | 1, 1 |
| `E[Y]`, `E[f]` | 0, 0 → estimand `θ = 0` |
| `n` | 10, 20, 50 |
| `ρ` | 0.0, 0.5, 0.9 |
| `N` | 10⁶ (primary); also 1000 and 200 |
| unlabelled mode | mean-only (`f̄_N ~ N(0, 1/N)`, `σ̂_f² = 1`) |
| `alpha_level` | 0.05 |
| `R` (replications) | 200,000 |
| coverage MC s.e. near 0.95 | 0.00049 |

**Baseline configuration details that affect the numbers and must match:**

- B3 uses `λ̂ = Ĉov/((1+n/N)·V̂ar(f))` with `ddof = 1` sample moments, variance
  `V̂ar(Y − λ̂f)/n + λ̂² V̂ar(f)/N` with `ddof = 1`, and a **normal** quantile.
- B4, B5, B6 use `t_{n−2}`.
- B9 uses the OLS slope `b` (not `b/(1+r)`), `s² = RSS/(n−2)`, the leverage term, and `t_{n−2}`.
- B1 uses `t_{n−1}` and `ddof = 1`.

---

## Expected results — Table A1 (primary reproduction target)

Coverage of nominal 95% intervals. **N = 10⁶, R = 200,000.** Source: `exact.py`.

| n | ρ | B1 classical *t* | B3 PPI++ (as shipped) | "no-leverage + *t*" control | **B9 LC-PPI** |
|---|---|---|---|---|---|
| 10 | 0.0 | 0.9496 | **0.8784** | 0.9369 | **0.9499** |
| 10 | 0.5 | 0.9490 | **0.8791** | 0.9364 | **0.9492** |
| 10 | 0.9 | 0.9511 | **0.8787** | 0.9370 | **0.9496** |
| 20 | 0.0 | 0.9489 | 0.9194 | 0.9432 | 0.9492 |
| 20 | 0.5 | 0.9502 | 0.9198 | 0.9434 | 0.9497 |
| 20 | 0.9 | 0.9492 | 0.9196 | 0.9437 | 0.9497 |
| 50 | 0.0 | 0.9502 | 0.9393 | 0.9478 | 0.9501 |
| 50 | 0.5 | 0.9497 | 0.9384 | 0.9471 | 0.9494 |
| 50 | 0.9 | 0.9501 | 0.9394 | 0.9477 | 0.9500 |

The **"no-leverage + *t*" control** is B9's point estimate with variance `s²/n + b²σ̂_f²/N` (i.e. the
leverage term deleted) and a `t_{n−2}` quantile. It exists to decompose the deficit: it shows the
quantile alone gets you to ≈ 0.937 and the leverage term supplies the remaining ≈ 0.013. Implement
it as `B9-nolev` in the code.

### Table A2 — mean interval width, relative to B1

Same runs. Source: `exact.py`.

| n | ρ | B3 | B9-nolev | **B9 LC-PPI** |
|---|---|---|---|---|
| 10 | 0.0 | 0.814 | 1.016 | **1.081** |
| 10 | 0.5 | 0.705 | 0.880 | **0.936** |
| 10 | 0.9 | 0.354 | 0.442 | **0.471** |
| 20 | 0.0 | 0.911 | 1.003 | **1.031** |
| 20 | 0.5 | 0.789 | 0.869 | **0.893** |
| 20 | 0.9 | 0.397 | 0.437 | **0.449** |
| 50 | 0.0 | 0.965 | 1.000 | **1.011** |
| 50 | 0.5 | 0.836 | 0.866 | **0.876** |
| 50 | 0.9 | 0.421 | 0.436 | **0.441** |

### Table A3 — finite-N behaviour of B9 (the known inexactness)

Coverage of B9 only. Source: `exact.py`.

| N | n | ρ = 0.0 | ρ = 0.5 | ρ = 0.9 |
|---|---|---|---|---|
| 10⁶ | 10 | 0.9499 | 0.9492 | 0.9496 |
| 1000 | 10 | 0.9513 | 0.9509 | 0.9528 |
| 200 | 10 | 0.9513 | 0.9527 | **0.9612** |
| 1000 | 20 | 0.9505 | 0.9503 | 0.9521 |
| 200 | 20 | 0.9514 | 0.9520 | 0.9577 |
| 1000 | 50 | 0.9506 | 0.9508 | 0.9524 |
| 200 | 50 | 0.9507 | 0.9510 | 0.9554 |

The monotone increase with ρ at fixed N is the pattern the conjecture in `04` §4.8.3 must explain.

---

## Expected results — Table A4 (secondary: the simple-fix comparison)

Coverage of nominal 95% intervals. **N = 10⁵, R = 60,000** (MC s.e. ≈ 0.00089). Source: `cover.py`.

Because `R` and `N` differ from Table A1, B1/B3 values differ slightly from Table A1 — that is
expected Monte-Carlo and configuration variation, not an inconsistency.

| n | ρ | B1 | B3 (z) | B4 (*t*) | B5 cross-fit | B6 threshold | B10-ish `c=1` | B10-ish `c=3` |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.00 | 0.9506 | 0.8783 | 0.9241 | 0.9254 | 0.9409 | 0.9373 | 0.9469 |
| 10 | 0.20 | 0.9501 | 0.8793 | 0.9247 | 0.9261 | 0.9367 | 0.9348 | 0.9441 |
| 10 | 0.50 | 0.9490 | 0.8777 | 0.9234 | 0.9245 | 0.9237 | 0.9255 | 0.9332 |
| 10 | 0.80 | 0.9492 | 0.8796 | 0.9240 | 0.9248 | **0.9194** | 0.9217 | **0.9211** |
| 10 | 0.95 | 0.9503 | 0.8795 | 0.9249 | 0.9258 | 0.9248 | 0.9245 | 0.9230 |
| 20 | 0.00 | 0.9525 | 0.9225 | 0.9401 | 0.9403 | 0.9485 | 0.9455 | 0.9504 |
| 20 | 0.50 | 0.9500 | 0.9198 | 0.9389 | 0.9387 | 0.9349 | 0.9379 | 0.9380 |
| 20 | 0.95 | 0.9500 | 0.9196 | 0.9375 | 0.9377 | 0.9375 | 0.9375 | 0.9373 |
| 50 | 0.00 | 0.9504 | 0.9395 | 0.9461 | 0.9459 | 0.9488 | 0.9479 | 0.9496 |
| 50 | 0.50 | 0.9517 | 0.9399 | 0.9461 | 0.9457 | 0.9455 | 0.9456 | 0.9450 |
| 50 | 0.95 | 0.9506 | 0.9404 | 0.9460 | 0.9458 | 0.9460 | 0.9458 | 0.9456 |
| 100 | 0.00 | 0.9506 | 0.9461 | 0.9489 | 0.9489 | 0.9500 | 0.9497 | 0.9506 |
| 100 | 0.50 | 0.9510 | 0.9447 | 0.9478 | 0.9478 | 0.9478 | 0.9477 | 0.9475 |
| 100 | 0.95 | 0.9499 | 0.9422 | 0.9453 | 0.9453 | 0.9453 | 0.9453 | 0.9454 |

> **Amended 2026-08-19 — which cross-fitting scheme this column is.** `cover.py` averages the two
> fold λ̂'s and applies the average to the whole gold sample. That is **not** the estimator `06` B5
> writes down (average the two fold-level *estimates*), which is what `probe.py` — and therefore
> Table A5 — uses. The two differ materially at n = 10: 0.9243 versus 0.9092. This column is the
> `cover.py` scheme, registered as **`B5-avglam`**; `06` B5's own formula is registered as `B5`.
> See `06` B5 and `13`.

**Note on the last two columns.** These are the shrunk-λ̃ rules applied on top of the **B3** variance
— *not* on top of B9. They are **not** the B10 defined in `06`. Label them `B6-shrink-c1` and
`B6-shrink-c3` in code to avoid the confusion. They are recorded because they document a rule that
was tried and **failed** (non-monotone in ρ, worst at ρ = 0.8).

---

## Expected results — Table A5 (point-estimate MSE, ratio to classical)

**N = 10⁵, R = 40,000.** Source: `probe.py`. Values > 1 mean the method is *worse* than ignoring the
model entirely.

| n | ρ | oracle λ* | B3 plug-in | B5 cross-fit | B6 threshold |
|---|---|---|---|---|---|
| 10 | 0.00 | 1.0000 | 1.1416 | **1.4792** | 1.0440 |
| 10 | 0.10 | 0.9904 | 1.1341 | **1.4826** | 1.0519 |
| 10 | 0.30 | 0.9143 | 1.0438 | 1.3734 | 1.0368 |
| 10 | 0.50 | 0.7497 | 0.8523 | 1.1170 | 0.9388 |
| 10 | 0.90 | 0.1861 | 0.2137 | 0.2766 | 0.2195 |
| 20 | 0.00 | 1.0000 | 1.0586 | 1.1430 | 1.0163 |
| 20 | 0.50 | 0.7524 | 0.7966 | 0.8598 | 0.8546 |
| 50 | 0.00 | 1.0000 | 1.0224 | 1.0494 | 1.0074 |
| 100 | 0.00 | 1.0000 | 1.0099 | 1.0213 | 1.0021 |
| 500 | 0.00 | 1.0000 | 1.0019 | 1.0033 | 1.0008 |

The headline here: **cross-fitting — the fix recommended by the paper that documented the problem —
is the worst method in the table at small n by MSE.**

---

## Experiment B results (E3, 2026-08-19) — **new results, not reproduction targets**

Everything above this point is a *preliminary* number to be reproduced. The table below is the
opposite: Experiment B had no preliminary run at all (`06` B7/B8 both said "not yet run"), so these
are **first measurements**, produced by this repository, and they become the targets that a future
reimplementation must reproduce.

Coverage of nominal 95% intervals. **N = 10⁶, R = 20,000** (MC s.e. 0.0015), `mean_only`,
`seed_entropy = 20260819`. Source: `configs/experiment_B_bootstrap.toml` → `results/B_bootstrap.csv`.

| n | ρ | B1 | B3 | B7 percentile | B8 BCa | B9-nolev | B9 |
|---|---|---|---|---|---|---|---|
| 10 | 0.0 | 0.9499 | 0.8794 | 0.9010 | 0.9026 | 0.9382 | 0.9506 |
| 10 | 0.3 | 0.9505 | 0.8787 | 0.9016 | 0.8996 | 0.9373 | 0.9506 |
| 10 | 0.5 | 0.9476 | 0.8784 | 0.9015 | 0.9020 | 0.9375 | 0.9501 |
| 10 | 0.7 | 0.9488 | 0.8751 | 0.9012 | 0.8983 | 0.9342 | 0.9482 |
| 10 | 0.9 | 0.9492 | 0.8770 | 0.8991 | 0.9001 | 0.9362 | 0.9497 |
| 20 | 0.0 | 0.9495 | 0.9234 | 0.9286 | 0.9284 | 0.9445 | 0.9512 |
| 20 | 0.3 | 0.9515 | 0.9222 | 0.9280 | 0.9264 | 0.9448 | 0.9498 |
| 20 | 0.5 | 0.9490 | 0.9188 | 0.9237 | 0.9248 | 0.9428 | 0.9487 |
| 20 | 0.7 | 0.9504 | 0.9176 | 0.9247 | 0.9228 | 0.9424 | 0.9493 |
| 20 | 0.9 | 0.9468 | 0.9185 | 0.9250 | 0.9256 | 0.9421 | 0.9486 |
| 50 | 0.0 | 0.9470 | 0.9353 | 0.9382 | 0.9375 | 0.9447 | 0.9469 |
| 50 | 0.3 | 0.9496 | 0.9401 | 0.9425 | 0.9419 | 0.9488 | 0.9513 |
| 50 | 0.5 | 0.9498 | 0.9392 | 0.9404 | 0.9394 | 0.9476 | 0.9499 |
| 50 | 0.7 | 0.9489 | 0.9386 | 0.9399 | 0.9409 | 0.9469 | 0.9489 |
| 50 | 0.9 | 0.9497 | 0.9366 | 0.9389 | 0.9387 | 0.9467 | 0.9490 |

Final-table setting `B = 9999`, n = 10 only (`configs/experiment_B_final.toml` →
`results/B_final.csv`): B7 0.9022 / 0.9016 / 0.9056 / 0.9026 / 0.8998 and B8 0.9029 / 0.9011 /
0.9038 / 0.9023 / 0.9012 at ρ = 0 / 0.3 / 0.5 / 0.7 / 0.9. **Five times the resampling budget moves
nothing**, which is the evidence that the shortfall is not bootstrap Monte-Carlo error.

Mean width relative to B1 at n = 10 (B = 2000): B3 0.814, B7 0.928, B8 0.933, B9-nolev 1.016,
B9 1.082 at ρ = 0. Wall-clock per interval at n = 10: B1 4.4·10⁻⁸ s, B9 2.9·10⁻⁸ s,
B7 2.9·10⁻⁴ s, B8 3.2·10⁻⁴ s (B = 2000) and ≈ 3.0·10⁻³ s at B = 9999 — four to five orders of
magnitude. `n_failed = 0` for every method in every cell.

**Bootstrap seed policy (extends §Seed policy rule 4).** Each stochastic method receives a
*freshly constructed* generator from the same bootstrap child seed, rather than one shared
generator object. A shared object would make whichever bootstrap ran first consume the stream, so
adding B8 would silently change B7's intervals. The consequence is also desirable: B7 and B8 are
built from the **identical** bootstrap resamples, which is the same paired-comparison logic rule 3
applies to the simulated data. `tests/test_crn.py` asserts both properties.

---

## Experiment E results (E4, 2026-08-20) — **new results, not reproduction targets**

Source: `configs/experiment_E_finite_N.toml` (explicit mode, `N ≤ 5000`) and
`experiment_E_largeN.toml` (mean_only, `N ≥ 10⁴`), `R = 200,000`, MC s.e. 0.00049, 81 cells.

Coverage summary over all 81 `(n, ρ, N)` cells:

| method | min | max | worst \|cov − 0.95\| | cells < 0.946 | cells > 0.954 |
|---|---|---|---|---|---|
| B1 | 0.9489 | 0.9510 | 0.0011 | 0 | 0 |
| B3 | 0.8773 | 0.9451 | 0.0727 | 81 | 0 |
| B9-nolev | 0.9357 | 0.9637 | 0.0143 | 43 | 4 |
| **B9** (`v482`) | 0.9488 | 0.9711 | 0.0211 | **0** | 15 |
| B9-vfull | 0.9488 | 0.9706 | 0.0206 | 0 | 13 |
| B9-R1 | 0.9488 | 0.9713 | 0.0213 | 0 | 16 |
| **B9-R2** | 0.9472 | 0.9527 | **0.0028** | 0 | 0 |
| B9-R3 | 0.6225 | 0.9511 | 0.3275 | 22 | 0 |
| B9-ppiplus | 0.8938 | 0.9616 | 0.0562 | 5 | 6 |

The worst cell for B9 is `n = 10, ρ = 0.9, N = 50` at 0.9711. The `N = 200, n = 10, ρ = 0.9` cell
of Table A3 reproduces **in explicit mode** at **0.9613** against the recorded 0.9612.

Variance diagnostic, `mean(se²)/Var(θ̂)` across replications: `[0.988, 1.025]` for both `v482` and
`v_full` over all 81 cells. **The variance expression is not the problem**; see `04` §4.8.3.

Mechanism check (derived, no free parameters): predicted coverage
`P(|t_{ν_eff}| ≤ t_{n−2,0.975})` with `ν_eff = (n−2)/(1−s)²` and `s` the unlabelled share of the
variance, matches observed B9 coverage to **0.0029** across all 81 cells.

**Tolerance note.** The variance-ratio check uses ±0.05. That catches a structurally wrong
expression — deleting `b²σ̂_f²/N` gives ratios of 0.20–0.58 — but **not** the `σ²σ_f²/(N·S_ff)`
term, which is ≈ 1 % of the variance and sits below Monte-Carlo resolution at `R = 200,000`.
That term is verified algebraically in `tests/test_finite_n.py`, not by simulation. Do not read the
±0.05 as the precision of the verification.

---

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| Coverage, R = 200,000 | **± 0.004** absolute | ≈ 8 Monte-Carlo standard errors; generous enough to absorb a different RNG stream and minor `ddof` conventions |
| Coverage, R = 60,000 | ± 0.005 absolute | ≈ 5.6 MC s.e. |
| Coverage, R = 40,000 | ± 0.006 absolute | ≈ 5.5 MC s.e. |
| Coverage, R = 20,000 | ± 0.006 absolute | ≈ 4 MC s.e.; added 2026-08-19 for Experiment B, which `07` fixes at R = 20,000 |
| Width ratio | ± 0.010 relative | width estimates are far more stable than coverage |
| MSE ratio | ± 0.020 relative | heavier-tailed statistic, converges more slowly |

**A deviation outside tolerance is a signal, not a nuisance.** Do not widen a tolerance to make a
test pass. Investigate, and record the finding in `13_RESEARCH_LOG.md`.

---

## Output files

Every experiment writes a **tidy long-format CSV** to `results/`, one row per (experiment, method,
cell, metric):

```
results/A_reproduction.csv
results/B_bootstrap.csv
results/C_sample_size.csv
results/D_predictor_quality.csv
results/E_finite_N.csv
results/F_heavy_tails.csv
results/G_skew.csv
results/H_heteroskedastic.csv
results/I_real_data.csv
```

**Required columns:**

```
experiment_id, cell_id, method, dgp, n, N, rho, alpha_level, R, unlabeled_mode, coef_mode,
variance_mode, metric, value, mc_se, n_failed, seed_spec, timestamp_utc,
python_version, numpy_version, scipy_version, git_commit
```

`metric ∈ {coverage, coverage_failures_as_miss, width_mean, width_median, width_ratio_b1, mse,
mse_ratio_b1, oracle_gain_fraction, tail_miscoverage_left, tail_miscoverage_right,
theta_hat_mean, theta_hat_sd, leverage_frac_mean, seconds_per_interval}`.

> **Reproducibility caveat, added 2026-08-19.** `seconds_per_interval` is wall-clock and is
> therefore the one metric that is **not** bit-reproducible; the determinism test excludes it
> exactly as it excludes the timestamp columns. Every other row must be byte-identical for a
> fixed seed specification.

> **Amended 2026-08-19.** The first column was `experiment`; it is now `experiment_id`, and
> `cell_id` and `variance_mode` are added, matching `15` §8.1 — one identifier spelled one way
> everywhere (`15` §1.2). `theta_hat_mean` and `theta_hat_sd` are added because **V2 requires
> them** and the original enumeration did not contain them; `theta_hat_sd` is what makes V2's
> z-test recomputable from the CSV. `coverage_failures_as_miss` is the "failures count as
> non-covering" figure required by `06` §Reporting item 6.

Also write, for every experiment, `results/<X>_raw/` containing per-replication outcomes
(covered/not, width) in compressed `.npz`, so that tail behaviour can be recomputed without re-running.

---

## Verification checks

Run these in order. Each has a binary pass condition.

| # | Check | Pass condition | If it fails |
|---|---|---|---|
| V1 | **Harness sanity.** B1 coverage at every cell of Table A1. | within 0.950 ± 0.004 | The simulator or the coverage counter is wrong. Nothing else is interpretable. Stop. |
| V2 | **Estimand sanity.** Mean of `θ̂` for B1 and B9 across replications. | `\|mean(θ̂)\| ≤ z_crit · sd(θ̂)/√R`, a two-sided z-test with `z_crit` Bonferroni-corrected across the sub-checks at family-wise level 0.001 (**amended 2026-08-19, was: within ±0.002 of 0**) | An estimator is biased — likely a sign error in `(f̄_N − f̄_n)`. |
| V3 | **B3 deficit reproduces.** B3 coverage at n = 10. | 0.878 ± 0.004 at every ρ | The B3 implementation differs. Check: normal vs *t* quantile, `ddof`, the `(1 + n/N)` factor. |
| V4 | **ρ-invariance of the B3 deficit.** Spread of B3 coverage across ρ at fixed n. | max − min ≤ 0.006 | The diagnosis in `04` §4.6 may be incomplete. Log before proceeding. |
| V5 | **Decay in n.** B3 coverage at n = 10/20/50. | 0.878 / 0.919 / 0.939, each ± 0.004 | As V3. |
| V6 | **B9 exactness at N → ∞.** B9 coverage at every cell of Table A1. | 0.950 ± 0.004 | The correction is not what `04` §4.8.1 specifies. Compare term by term against `exact.py`. |
| V7 | **Decomposition.** B9-nolev coverage at n = 10. | 0.937 ± 0.005 | The quantile/leverage split is different from the preliminary finding. |
| V8 | **Width honesty.** B9 width ratio to B1 at n = 10, ρ = 0. | 1.081 ± 0.010 | Check that the leverage term is present and `s²` uses `n − 2`. |
| V9 | **Finite-N over-coverage.** B9 at N = 200, n = 10, ρ = 0.9. | 0.9612 ± 0.004 | The finite-N term differs. Check `b² σ̂_f² / N` and whether `σ̂_f²` is estimated or fixed at 1. |
| V10 | **Cross-fit MSE penalty.** B5 MSE ratio at n = 10, ρ = 0. | 1.479 ± 0.030 | The fold construction differs; document which scheme is used. |
| V11 | **Analytic identity.** For a single dataset, check numerically that `Ȳ + b(μ_f − f̄_n)` equals the OLS fitted value at `μ_f` from an independent regression implementation (e.g. `numpy.polyfit` or `scipy.stats.linregress`). | agreement to 1e-10 | The algebraic identity in `04` §4.7 is mis-implemented. |
| V12 | **Exact-pivot check.** Under the N → ∞ Gaussian DGP, the studentised statistic `T` from `04` §4.7 compared against `t_{n−2}` by a Kolmogorov–Smirnov test on 200,000 draws at n = 10. | KS p-value not significant at 0.001 after Bonferroni across cells | This is the strongest single test of the theory. Failure means the pivot claim is wrong. **Run this before writing any proof.** |

> **Amendment 2026-08-19 — V2 restated as a z-test.** V2 was originally a fixed absolute
> tolerance of ±0.002. That is not a bias test: its stringency depends entirely on `R` and on the
> estimator's own dispersion. At `R = 200,000` and `sd(θ̂) ≈ 0.338` (n = 10, ρ = 0) it is only 2.65
> Monte-Carlo standard errors, so a **correct** implementation fails it with probability ≈ 13 %
> across the 18 sub-checks — which is exactly what happened on the first full reproduction run
> (B9 at n = 10, ρ = 0 came out at −3.01 MC s.e. while every other method in the same cell moved
> with it under common random numbers). At `R = 10⁶` the same ±0.002 would be 13 s.e. and could
> never fire at all.
>
> The replacement is the test V2 was evidently meant to be: `z = mean(θ̂)/(sd(θ̂)/√R)` against a
> two-sided critical value, Bonferroni-corrected across the sub-checks at family-wise level 0.001
> — the same family-wise level this document already sets for V12. With 18 sub-checks that is
> `z_crit = 4.03`. Note this is **not** a blanket relaxation: it is *looser* than ±0.002 at
> n = 10 (0.0031) and *stricter* at n = 50 (0.0013), because it scales with the estimator's
> dispersion instead of ignoring it. Any real bias — a sign error, a wrong estimand — is tens of
> standard errors and still fires.
>
> `sd(θ̂)` is available as the `theta_hat_sd` metric in every results CSV, so the test is
> recomputable from the CSV alone. See `13` (2026-08-19) and `14` Q14, now closed.

V12 deserves emphasis: it tests the *distributional* claim rather than just one quantile of it. A
method can hit 95% coverage by accident while having the wrong shape; V12 rules that out.

---

## Reproduction command

The repository must expose a single entry point that regenerates Table A1–A5 and runs V1–V12:

```
python -m experiments.reproduce --experiment A --output results/ --seed-entropy <BASE_ENTROPY>
python -m experiments.verify   --tables results/A_reproduction.csv
```

`verify` must exit non-zero if any of V1–V12 fails, print which check failed, and print the observed
versus expected value with the tolerance. No manual eyeballing.

---

## Provenance of these numbers

Produced during project selection by three standalone scripts, shipped with this package as
`preliminary_code/`:

| Script | Produced | R | N |
|---|---|---|---|
| `exact.py` | Tables A1, A2, A3 | 200,000 | 10⁶, 1000, 200 |
| `cover.py` | Table A4 | 60,000 | 10⁵ |
| `probe.py` | Table A5 | 40,000 | 10⁵ |

These scripts are **exploratory code, not a reference implementation.** They should be read for
their formulas and then replaced, not extended. Known issues in them, all recorded here so that the
reimplementation does not inherit them:

1. `probe.py` and `cover.py` compute the shrinkage statistic `t²` using `λ̂` rather than `b` in the
   numerator (`04` §4.9). Immaterial at `N ≫ n`; wrong in general.
2. `exact.py` hard-codes `σ_f² = 1` in the `b²σ̂_f²/N` term because the DGP has unit variance. A
   general implementation must estimate it.
3. The seed formulas can collide across cells (see §Seed policy).
4. The mean-only unlabelled mode means `σ_f²` is treated as known everywhere, including at `N = 200`
   where it should not be.
5. B2, B7, B8 are absent entirely.
