# 05 — Proposed Method: LC-PPI

**Working name: Leverage-Corrected Prediction-Powered Inference (LC-PPI).**

> **The name is provisional.** No name was established during project selection. It may be replaced
> once the prior-art check (`03`, K1) determines whether the correction is a new construction or a
> translation of an existing survey-sampling result — in the latter case a name that credits the
> lineage would be more honest. Do not build the name into public API surfaces in a way that is
> painful to change; keep it to module and function names that a single rename can update.

---

## Inputs

| Input | Type | Required | Notes |
|---|---|---|---|
| `Y` | array, length n | yes | gold labels on the labelled sample |
| `f` | array, length n | yes | model predictions on the **same** labelled units, aligned index-wise |
| `f_unlabeled` | array, length N | yes* | model predictions on the unlabelled pool |
| `alpha_level` | float | no (default 0.05) | nominal error level; `1 − alpha_level` is the confidence level |
| `mu_f` | float | no | if the population prediction mean is genuinely known, pass it and the `N → ∞` branch is used |
| `sigma2_f` | float | no | variance of `f`; if omitted, estimated from `f_unlabeled` |

\* Either `f_unlabeled` **or** `mu_f` must be supplied. If `mu_f` is given, `N` is treated as
infinite. If `f_unlabeled` is given, `N = len(f_unlabeled)`, `f̄_N = mean(f_unlabeled)` and
`σ̂_f² = var(f_unlabeled, ddof=1)`.

**Preconditions that must be checked and raised on, not silently handled:**

- `len(Y) == len(f)`; both 1-D; no NaNs.
- `n >= 4`. For `n == 3` the `t_{n−2}` has one degree of freedom and the interval is degenerate in
  practice; raise a warning. For `n < 3` raise an error.
- `S_ff > 0` (the predictions on the labelled sample are not all identical). If `S_ff == 0` the
  slope is undefined; fall back to the classical interval and set a diagnostic flag.
- The labelled and unlabelled samples are disjoint. **This cannot be checked from the arrays and
  must be documented as a caller responsibility.**

---

## Outputs

Return a single record (dataclass or dict) containing:

| Field | Meaning |
|---|---|
| `theta_hat` | point estimate `θ̂_LC` |
| `ci_low`, `ci_high` | the corrected confidence interval |
| `width` | `ci_high − ci_low` |
| `se` | `√V̂_LC` |
| `dof` | degrees of freedom used for the quantile |
| `lambda_hat` | the plug-in tuning coefficient actually used |
| `b` | the OLS slope |
| `leverage` | the value of the leverage term `(f̄_N − f̄_n)² / S_ff` |
| `leverage_frac` | `leverage / (1/n + leverage)` — the share of the variance attributable to the correction; a useful diagnostic for how much this matters in a given dataset |
| `s2_resid` | `s²` |
| `var_components` | dict with `mean_term`, `leverage_term`, `unlabeled_term` |
| `n`, `N` | sample sizes |
| `rho_hat` | sample correlation of `Y` and `f` — reported as a diagnostic, **not** used in construction |
| `flags` | list of strings: e.g. `"degenerate_S_ff"`, `"small_n"`, `"finite_N_regime"` |

`flags` should include `"finite_N_regime"` whenever `N / n < 1000`, because that is the regime where
the interval is known to be inexact (see `04` §4.8.2–4.8.3). This is a factual warning, not a
failure.

---

## Algorithm

```
INPUT: Y[1..n], f[1..n], f_unlabeled[1..N] (or mu_f), alpha_level

 1. VALIDATE
      assert n >= 4, len(Y) == len(f), no NaNs
      compute S_ff = sum_i (f_i - fbar_n)^2 ; assert S_ff > 0

 2. LABELLED-SAMPLE SUMMARIES
      Ybar   = mean(Y)
      fbar_n = mean(f)
      S_ff   = sum_i (f_i - fbar_n)^2
      S_fY   = sum_i (f_i - fbar_n)(Y_i - Ybar)

 3. UNLABELLED-SAMPLE SUMMARY
      if mu_f supplied:  fbar_N = mu_f ;  N = infinity ;  sigma2_f = 0 contribution
      else:              fbar_N = mean(f_unlabeled) ; N = len(f_unlabeled)
                         sigma2_f = var(f_unlabeled, ddof=1)

 4. SLOPE
      b = S_fY / S_ff                                  # OLS slope
      r = n / N   (0 if N infinite)
      lambda_hat = b / (1 + r)                         # reported for comparability with PPI++

 5. RESIDUAL VARIANCE
      resid = (Y_i - Ybar) - b * (f_i - fbar_n)
      s2    = sum_i resid_i^2 / (n - 2)

 6. LEVERAGE TERM
      leverage = (fbar_N - fbar_n)^2 / S_ff

 7. CORRECTED VARIANCE
      V = s2 * (1/n + leverage)  +  b^2 * sigma2_f / N        # last term is 0 if N infinite

 8. INTERVAL
      theta_hat = Ybar + b * (fbar_N - fbar_n)
      q         = t_quantile(1 - alpha_level/2, dof = n - 2)
      CI        = theta_hat +/- q * sqrt(V)

 9. DIAGNOSTICS
      report leverage_frac, var_components, rho_hat, flags
```

**Note on step 4 vs step 8.** The point estimate in step 8 uses `b`, not `λ̂ = b/(1+r)`. This is
deliberate: the exactness argument in `04` §4.7 is for the OLS fitted value, which uses `b`. The
`(1 + r)` shrinkage in PPI++'s λ* is a variance-optimality adjustment derived under a *known*-λ
analysis, and combining it with the leverage correction has **not** been analysed.

> **OPEN — requires implementation/research decision.** Whether the finite-N optimal coefficient
> should be `b`, `b/(1+r)`, or something else once the leverage cost of estimating it is accounted
> for. In all preliminary experiments `r ≤ 0.05` so the two coincide numerically. The implementation
> must expose this as a parameter `coef_mode ∈ {"ols", "ppi_plus"}` defaulting to `"ols"`, log which
> was used, and Experiment E must report both.

**Do not add steps.** In particular: do not add a shrinkage step to the default path (see §4.9 — it
is a separate, optional method, B10), do not add a calibration step, and do not add any data-driven
model selection.

---

## Complexity

| Quantity | Cost |
|---|---|
| Time | `O(n + N)` — two passes over the labelled data, one over the unlabelled predictions |
| Memory | `O(1)` beyond the inputs; only `mean(f_unlabeled)` and `var(f_unlabeled)` are needed, so the unlabelled pool can be streamed |
| Comparison: PPI++ | `O(n + N)` — **identical order**; the correction adds a constant number of arithmetic operations |
| Comparison: bootstrap (B7/B8) | `O(B(n + N))` for `B` resamples, typically `B = 2000`; BCa adds a jackknife pass, `O(n²)` in the naive implementation |

The practical point: LC-PPI costs the same as PPI++ to three significant figures, whereas a bootstrap
alternative costs three to four orders of magnitude more. If both attain nominal coverage, this is
the argument that survives (see K3 in `10`).

---

## Comparison to PPI++ — exactly what changes

| Component | PPI++ | LC-PPI |
|---|---|---|
| Point estimate | `Ȳ − λ̂ (f̄_n − f̄_N)`, `λ̂ = b/(1+r)` | `Ȳ + b (f̄_N − f̄_n)` — same form, `coef_mode` selects `b` or `b/(1+r)` |
| Variance, gold-sample term | `V̂ar(Y − λ̂ f) / n` | `s² / n` where `s² = RSS/(n−2)` |
| Variance, leverage term | **absent** | `s² · (f̄_N − f̄_n)² / S_ff` |
| Variance, unlabelled term | `λ̂² V̂ar(f) / N` | `b² σ̂_f² / N` |
| Quantile | normal, `z_{1−α/2}` | Student, `t_{n−2, 1−α/2}` |
| Degrees-of-freedom accounting | none | `n − 2` (two parameters estimated: intercept and slope) |

Three changes, all small, all in the same direction (wider): a residual variance with the correct
denominator, an added leverage term, and a heavier-tailed quantile. Note that
`V̂ar(Y − λ̂f)/n` and `s²/n` differ only in the degrees-of-freedom convention (`n−1` vs `n−2`) when
λ̂ = b; the substantive additions are the leverage term and the quantile.

---

## Why the method could improve validity

Stated mathematically rather than by appeal to the simulation:

1. The PPI++ interval is built from the variance of `θ̂_λ` **at fixed λ**. The estimator actually
   computed is `θ̂_{λ̂}`, a different random variable, with strictly larger variance whenever
   `f̄_n ≠ μ_f` — which happens with probability 1.
2. The excess variance is exactly `σ² (μ_f − f̄_n)² / S_ff`, the squared extrapolation distance
   times the slope's sampling variance. This term is `O_p(σ²/(n · S_ff / σ_f²)) = O_p(σ²/n²)`
   relative to the leading `σ²/n` — hence a **relative** distortion of order `1/n`, which is
   precisely the observed decay pattern.
3. Restoring it and studentising by `s` with `n − 2` degrees of freedom turns the studentised
   statistic into an exact `t` pivot under M1 (`04` §4.7), so coverage becomes exactly nominal
   rather than approximately so.
4. Because the argument is a pivot argument and not an asymptotic one, it does not degrade as
   `n → 4`; the preliminary simulation is consistent with this (coverage at n = 10 is as exact as at
   n = 50).

**What this argument does not establish**, and must not be presented as establishing:

- Anything at finite N (see `04` §4.8.3 — there the pivot argument breaks and the measured interval
  over-covers).
- Anything outside the Gaussian working model M1.
- Anything about M-estimation.
- That the correction is novel (see the prior-art warning in `03`).
