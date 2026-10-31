# 11 — Implementation Specification

> **This document is the implementation source of truth for module contracts** — function
> signatures, inputs, outputs, equations implemented, and required tests. Equations referenced here
> are defined in `04_MATHEMATICAL_FORMULATION.md`, which wins in any conflict.

> **Superseded in three places by `15_ARCHITECTURE_SPECIFICATION.md`** (added 2026-08-19). `15` owns
> the *system* — layering, data flow, orchestration, storage, MVP order — and extends this document
> as follows. No module responsibility, equation, or test requirement below changes.
>
> 1. **Repository tree.** `15` §2 is a strict superset of the tree below: it adds `configs/`,
>    `data/`, `requirements.txt`, `Makefile`, and three modules (`config.py`, `io.py`, `plots.py`,
>    plus `types.py` split out of the data-contract section here). Use `15` §2.
> 2. **Experiment scripts.** The nine per-experiment scripts listed below are replaced by **one
>    config-driven runner** (`experiments/run.py --config configs/experiment_X.toml`), because
>    Experiments A–I differ only in parameter grids and would otherwise be nine copies of the same
>    loop. The nine names survive as `Makefile` targets. See `15` §7.1 and §13 CONFLICT 2.
> 3. **Variance-mode naming.** `compute_variance` in `15` §6.3 uses prefixed mode names
>    (`"lc_v482"`, …) because it also serves the non-LC variances; `lc_ppi_interval` below keeps the
>    unprefixed names (`"v482"`, …) and maps internally. Neither spelling may appear in a results
>    file — only the method IDs from `06`. See `15` §13 CONFLICT 4.

---

## Programming language

**Python.** No other language on the critical path. No compiled extensions. No JIT dependency.

Rationale: every computation in this project is a handful of arithmetic operations on arrays of
length `n ≤ 500`, vectorised over replications. NumPy is more than fast enough, and the project's
deliverable is a *drop-in replacement for `ppi-py`*, which is a Python package.

---

## Core packages

| Package | Use | Minimum |
|---|---|---|
| `numpy` | all array computation, RNG (`default_rng`, `SeedSequence`) | 1.24 |
| `scipy` | `scipy.stats` for `t`, `norm` quantiles; KS test for V12 | 1.10 |
| `pandas` | tidy results tables, CSV I/O | 2.0 |
| `scikit-learn` | Tier-2 datasets and the weak predictors used as AI annotators | 1.3 |
| `matplotlib` | figures | 3.7 |
| `pytest` | tests | 7.0 |

**Not permitted on the critical path:** PyTorch, TensorFlow, JAX, any deep-learning framework, any
RL library, `statsmodels` for the core estimators (it may be used in a *test* as an independent
implementation to check against, which is a legitimate and valuable use).

---

## Repository structure

```text
ppi-finite-sample/
├── README.md
├── pyproject.toml
├── docs/                          # the research package: 01..14 + this README
├── src/
│   └── ppi_finite/
│       ├── __init__.py
│       ├── estimators.py          # point estimators
│       ├── variance.py            # variance expressions, incl. the leverage term
│       ├── confidence_intervals.py# B1..B6, B9, B10 interval constructors
│       ├── bootstrap.py           # B7, B8
│       ├── ppi.py                 # B1 (classical) and B2 (plain PPI)
│       ├── ppi_plus.py            # B3..B6, the PPI++ family
│       ├── lc_ppi.py              # B9, B10 — the proposed method
│       ├── simulation.py          # DGPs T1.1..T1.5, replication driver
│       ├── metrics.py             # coverage, width, MSE, oracle gain, MC standard errors
│       ├── registry.py            # the B1..B10 method registry
│       └── realdata.py            # Tier 2/3 loaders and the subsampling protocol
├── experiments/
│   ├── reproduce.py               # Experiment A
│   ├── exp_b_bootstrap.py
│   ├── exp_c_sample_size.py
│   ├── exp_d_predictor_quality.py
│   ├── exp_e_finite_n.py
│   ├── exp_f_heavy_tails.py
│   ├── exp_g_skew.py
│   ├── exp_h_heteroskedastic.py
│   ├── exp_i_real_data.py
│   └── verify.py                  # V1..V12
├── tests/
├── results/                       # CSVs (git-tracked), raw npz (git-ignored if large)
├── figures/
└── notebooks/                     # optional, never load-bearing
```

**Rule:** nothing in `experiments/` may contain an estimator or a variance formula. Experiments
compose things from `src/ppi_finite/`. If a formula appears in two places it will eventually
disagree with itself.

---

## Data contracts

A single dataclass carries a simulated or real dataset through the whole system:

```python
@dataclass(frozen=True)
class PPIData:
    Y: np.ndarray            # (n,)   gold labels on the labelled sample
    f: np.ndarray            # (n,)   predictions on the labelled sample
    f_unlabeled: np.ndarray | None   # (N,) predictions on the unlabelled pool
    fbar_N: float | None     # supplied directly in mean-only mode
    sigma2_f: float | None   # variance of f; estimated from f_unlabeled if None
    N: int | float           # np.inf allowed
    theta_true: float | None # known in simulation; None on real data
```

Exactly one of `f_unlabeled` / (`fbar_N`, `N`) must be supplied. Enforce in `__post_init__`.

For vectorised Monte Carlo the same fields appear with a leading replication axis:
`Y: (R, n)`, `f: (R, n)`, `fbar_N: (R,)`. Every estimator must work on the batched form — **write
the batched form first and derive the single-dataset form from it**, not the other way round, or the
Monte Carlo will be unusably slow.

The universal return type:

```python
@dataclass(frozen=True)
class IntervalResult:
    method: str              # "B9", "B3", ...
    theta_hat: np.ndarray    # (R,) or scalar
    ci_low: np.ndarray
    ci_high: np.ndarray
    se: np.ndarray
    dof: np.ndarray | float
    extras: dict             # method-specific diagnostics; see 05 for LC-PPI's list
    n_failed: int            # replications where construction failed
```

---

## Module specifications

### `src/ppi_finite/estimators.py`

Point estimators and shared sufficient statistics.

| Function | Signature | Returns | Implements |
|---|---|---|---|
| `labelled_summaries` | `(Y, f) -> dict` | `Ybar, fbar_n, S_ff, S_fY, b, s2, se_b, n` | `04` §4.1, §4.7 |
| `theta_classical` | `(Y) -> array` | `Ȳ` | `04` §4.2 |
| `theta_ppi` | `(Y, f, fbar_N) -> array` | `Ȳ − (f̄_n − f̄_N)` | `04` §4.3 |
| `theta_ppi_plus` | `(Y, f, fbar_N, lam) -> array` | `Ȳ − λ(f̄_n − f̄_N)` | `04` §4.4 |
| `lambda_plugin` | `(Y, f, n, N) -> array` | `b/(1+n/N)` | `04` §4.4 |
| `lambda_crossfit` | `(Y, f, n, N, scheme) -> array` | fold-split λ̂ | `06` B5 |
| `lambda_threshold` | `(Y, f, n, N, t_crit) -> array` | λ̂ or 0 | `06` B6 |
| `lambda_shrunk` | `(Y, f, n, N, c) -> array` | `λ̂(1 − c/t²)₊` | `04` §4.9 |

`labelled_summaries` must use `S_ff` as a **sum of squares**, not a variance. This is the single most
likely place for an off-by-`(n−1)` error; guard it with a test.

`lambda_crossfit` must support `scheme ∈ {"2fold", "kfold"}` with `k` a parameter, per `06` B5.

**Tests required.** `b` matches `scipy.stats.linregress(f, Y).slope` to 1e-12. `s2` matches
`RSS/(n−2)` computed independently. `se_b` matches `linregress(...).stderr`. `theta_ppi_plus` with
λ = 0 equals `theta_classical`, and with λ = 1 equals `theta_ppi`.

### `src/ppi_finite/variance.py`

| Function | Returns | Implements |
|---|---|---|
| `var_classical(Y)` | `s_Y²/n` | `04` §4.2 |
| `var_ppi_plus(Y, f, lam, N)` | `V̂ar(Y − λf)/n + λ²V̂ar(f)/N` | `04` §4.5 |

> **Amended 2026-08-19.** `var_ppi_plus` previously took a `sigma2_f` argument, which implied the dataset-level σ̂_f². `04` §4.5 uses the `V̂ar(·)` notation defined in `04` §4.4 as the sample moments of the **same n gold pairs**, and both preliminary scripts compute `S_ff/(n−1)`. The argument is therefore removed and the moment computed from `f`. `var_lc_ppi` keeps `sigma2_f`: `04` §4.8.2 defines *its* `σ̂_f²` as estimated from the N unlabelled predictions. The two families differ deliberately. See `13` entry L1.

| `leverage_term(f, fbar_N)` | `(f̄_N − f̄_n)²/S_ff` | `04` §4.6 |
| `var_lc_ppi(Y, f, fbar_N, sigma2_f, N)` | `s²(1/n + lev) + b²σ̂_f²/N` | `04` §4.8.2 |
| `var_lc_ppi_full(...)` | (V_full) | `04` §4.8.3 — **currently unverified; must be flagged in its docstring** |
| `var_lc_ppi_r1/r2/r3(...)` | the candidate finite-N repairs | `04` §4.8.4 |
| `var_hc3(...)` | heteroskedasticity-consistent variant | Experiment H |

Every function's docstring must cite the exact equation number in `04` that it implements. A function
whose docstring cannot cite an equation should not exist.

**Tests required.** `leverage_term` is zero when `f̄_N == f̄_n`. `var_lc_ppi` with `N = inf` reduces
to `s²(1/n + lev)`. `var_lc_ppi` ≥ `s²/n` always.

### `src/ppi_finite/confidence_intervals.py`

| Function | Implements |
|---|---|
| `ci_from_se(theta_hat, se, dof, alpha_level, dist)` | generic; `dist ∈ {"normal","t"}` |
| `ci_classical` | B1 |
| `ci_ppi` | B2 |
| `ci_ppi_plus` | B3 (normal quantile) |
| `ci_ppi_plus_t` | B4 |
| `ci_ppi_plus_crossfit` | B5 |
| `ci_ppi_plus_threshold` | B6 |

**Tests required.** B3 with `alpha_level = 0.05` reproduces the `z = 1.959964` multiplier. B1
matches `scipy.stats.t.interval` on the same data to 1e-12.

### `src/ppi_finite/lc_ppi.py`

The proposed method. Public entry point:

```python
def lc_ppi_interval(
    Y, f, f_unlabeled=None, *, mu_f=None, N=None, sigma2_f=None,
    alpha_level=0.05, coef_mode="ols", variance_mode="v482",
    shrinkage_c=None,
) -> IntervalResult
```

- `coef_mode ∈ {"ols", "ppi_plus"}` — `b` versus `b/(1+r)`. Default `"ols"` (`05`).
- `variance_mode ∈ {"v482", "v_full", "r1", "r2", "r3", "hc3"}` — selects the variance expression.
  Default `"v482"`, which is what produced the preliminary numbers.
- `shrinkage_c=None` gives B9; a float gives B10.
- Must populate `extras` with every diagnostic listed in `05` (Outputs), including `leverage_frac`
  and the `finite_N_regime` flag when `N/n < 1000`.

**Tests required.**
- `mu_f` supplied vs `f_unlabeled` with `N = 10⁶` agree to 4 decimals.
- The identity check V11: `theta_hat` equals the OLS fitted value at `μ_f`.
- Degenerate `S_ff = 0` falls back to B1 and sets the flag.
- `n = 3` warns; `n = 2` raises.

### `src/ppi_finite/bootstrap.py`

| Function | Implements |
|---|---|
| `bootstrap_replicates(Y, f, fbar_N, B, rng, resample_unlabeled=False)` | resampling of the gold pairs |
| `ci_percentile` | B7 |
| `ci_bca` | B8 — bias correction `ẑ₀` and jackknife acceleration `â` |

`ci_bca` must be validated against an independent implementation on a known example (e.g. bootstrap
the plain mean and compare against `scipy.stats.bootstrap(method="BCa")`) before it is trusted. A
wrong BCa would silently decide kill criterion K3.

**Performance note.** With `R = 20,000` replications × `B = 2000` resamples × `n = 10`, a naive loop
is ~4×10⁸ operations — fine if vectorised over both axes, unusable if written as a Python loop.
Draw the resample index matrix once as `(R, B, n)` in chunks.

### `src/ppi_finite/simulation.py`

| Function | Implements |
|---|---|
| `dgp_gaussian(rng, R, n, rho, N, mode)` | T1.1 |
| `dgp_heavy_tail(rng, ..., nu, heavy_f=False)` | T1.2 |
| `dgp_skew(rng, ..., family, params)` | T1.3, incl. the binary-label case |
| `dgp_heteroskedastic(rng, ..., gamma, form)` | T1.4 |
| `dgp_misspecified(rng, ..., g)` | T1.5 |
| `run_cell(dgp_spec, methods, R, seed_spec)` | draws **once**, evaluates **all** methods on the same draw |

`run_cell` is where common random numbers are enforced. It must be **impossible** for a caller to
evaluate a method on a fresh draw: the method registry receives arrays, never a generator.

Each DGP returns the batched `PPIData` plus `theta_true`.

**Tests required.** For each DGP, empirical `Var(Y)`, `Var(f)`, `Corr(Y,f)`, `E[Y]` match the target
to within Monte-Carlo error at `R = 10⁶`. This catches standardisation errors in the skew and
heavy-tail families, which are easy to get wrong.

### `src/ppi_finite/metrics.py`

| Function | Returns |
|---|---|
| `coverage(ci_low, ci_high, theta_true)` | proportion, plus its MC standard error |
| `width_stats(...)` | mean, median, and ratio to a reference method |
| `mse(theta_hat, theta_true)` | MSE and its ratio to a reference |
| `oracle_gain_fraction(...)` | `(V_ref − V_method)/(V_ref − V_oracle)` |
| `tail_miscoverage(...)` | left and right separately (Experiment G) |
| `mc_se_proportion(p, R)` | `√(p(1−p)/R)` |

`coverage` must **never** silently drop failed replications. Return `n_failed` alongside, and provide
both the "among successes" and the "failures count as non-covering" figures.

### `src/ppi_finite/registry.py`

A dict mapping `"B1"…"B10"` (plus variants `"B9-nolev"`, `"B9-R1"`, `"B3b"`, `"B6-shrink-c1"`, …) to
callables with a uniform signature `(data: PPIData, alpha_level: float, rng) -> IntervalResult`.
Experiments iterate over the registry; adding a method must not require editing any experiment.

The registry is also the pre-registration record: adding an entry after results have been seen
requires a log entry in `13` (`10`, standing prohibition 7).

### `src/ppi_finite/realdata.py`

| Function | Implements |
|---|---|
| `load_tier2(name)` | bundled sklearn datasets; returns features, gold labels |
| `fit_weak_predictor(X, y, strength)` | a deliberately weak model, with `strength` tuning ρ |
| `subsample_protocol(population, n, N, rng)` | the disjoint gold/unlabelled split of `08` |
| `load_tier3(name)` | real AI-annotation data; must fail loudly and informatively if unavailable |

`load_tier2` must catch download failures (they occurred in the environment where the preliminary
work was done), log them, and continue with bundled loaders.

---

## Testing requirements

| Level | Requirement |
|---|---|
| Unit | Every function in `estimators.py`, `variance.py`, `confidence_intervals.py` has a test comparing against an independent computation (`scipy.stats.linregress`, `statsmodels.OLS`, a hand-computed 5-point example). |
| Property | `var_lc_ppi ≥ var` without leverage; interval contains the point estimate; width increases as `alpha_level` decreases; `λ = 0` recovers classical. |
| Reduction | `lc_ppi_interval` with `f` constant (`S_ff = 0`) equals the classical interval. `lc_ppi_interval` with `ρ = 1` exactly (noiseless) produces a degenerate interval — check it does not divide by zero. |
| Statistical | The exact-pivot KS test (V12) as a slow-marked test at `R = 200,000`, `n = 10`. |
| Regression | Tables A1–A5 with reduced `R = 20,000` as a fast smoke test in CI; full `R` as a nightly/manual job. |
| Determinism | Two runs with the same seed spec produce byte-identical CSVs (excluding timestamps). |

Mark the expensive tests `@pytest.mark.slow` and keep the default `pytest` run under 60 seconds.

---

## Logging and provenance

Every experiment run must record: the git commit, the seed specification, all package versions, the
full parameter grid, the wall-clock time, and the exact command line. Write it as a sidecar JSON next
to each CSV (`results/A_reproduction.meta.json`).

A results file without provenance is not a result.
