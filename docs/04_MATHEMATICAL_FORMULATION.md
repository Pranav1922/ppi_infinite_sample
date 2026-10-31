# 04 — Mathematical Formulation

> **This document is the mathematical source of truth.** If any other document in this package
> contains an equation that disagrees with one here, this document wins and the discrepancy must be
> recorded in `13_RESEARCH_LOG.md`.

Every result below carries an explicit **status tag**. See §4.12 for the tag definitions. Nothing
untagged should be treated as established.

---

## 4.1 Notation

| Symbol | Meaning |
|---|---|
| `N` | number of **unlabelled** units (model predictions only) |
| `n` | number of **gold-labelled** units (both gold label and model prediction) |
| `Y_i`, i = 1..n | gold label on labelled unit i (real-valued in the core setting) |
| `f_i`, i = 1..n | model prediction on labelled unit i |
| `f_j^u`, j = 1..N | model prediction on unlabelled unit j |
| `θ` | estimand. Core setting: `θ = E[Y]`, the population mean of the gold label |
| `μ_f` | `E[f]`, the population mean of the prediction |
| `σ_f²` | `Var(f)` |
| `f̄_n` | `(1/n) Σ_{i=1}^n f_i` — mean prediction on the **labelled** sample |
| `f̄_N` | `(1/N) Σ_{j=1}^N f_j^u` — mean prediction on the **unlabelled** sample |
| `Ȳ` | `(1/n) Σ_{i=1}^n Y_i` |
| `r` | `n / N`, the labelled-to-unlabelled ratio |
| `ρ` | `Corr(Y, f)` |
| `λ` | PPI++ tuning coefficient (fixed, unknown) |
| `λ*` | variance-minimising value of λ |
| `λ̂` | plug-in estimate of λ* from the gold sample |
| `λ̃` | proposed **shrunk** tuning coefficient (§4.9) |
| `b` | OLS slope of Y on f in the gold sample, `b = S_fY / S_ff` |
| `β` | population linear-projection slope, `β = Cov(Y,f) / Var(f)` |
| `α` | population linear-projection intercept, `α = E[Y] − β μ_f` |
| `S_ff` | `Σ_{i=1}^n (f_i − f̄_n)²` (**note: a sum of squares, not a variance**) |
| `S_fY` | `Σ_{i=1}^n (f_i − f̄_n)(Y_i − Ȳ)` |
| `ε_i` | projection residual, `Y_i − α − β f_i` |
| `σ²` | `Var(ε)` — the **residual** variance (distinct from `Var(Y)`) |
| `s²` | `RSS / (n − 2)` where `RSS = Σ_i (Y_i − Ȳ − b(f_i − f̄_n))²` — unbiased estimate of `σ²` |
| `θ̂` | a generic estimator of θ |
| `α_lvl` | nominal error level; `1 − α_lvl` is the nominal confidence level (0.95 throughout) |
| `z_q` | standard-normal quantile |
| `t_{ν,q}` | Student-*t* quantile with ν degrees of freedom |
| `h` | leverage term, `h = (μ_f − f̄_n)² / S_ff` (or its finite-N analogue) |

**Naming caution.** `α` is used for the regression intercept; the nominal error level is written
`α_lvl` throughout to avoid collision. Implementations should use `alpha_level` for the error level
and `intercept` for the regression intercept.

**Sampling assumptions used throughout unless stated otherwise:** the n labelled pairs and the N
unlabelled predictions are drawn i.i.d. from the same population and are **mutually independent**
(disjoint samples). The predictor `f` is a fixed function not fitted on either sample.

---

## 4.2 Classical estimator

Ignore the model entirely:

```
θ̂_classical = Ȳ
Var(θ̂_classical) = σ_Y² / n
```

Interval (exact under normality of Y):

```
Ȳ  ±  t_{n−1, 1−α_lvl/2} · s_Y / √n ,      s_Y² = (1/(n−1)) Σ_i (Y_i − Ȳ)²
```

Status: `TEXTBOOK`. This is B1 and serves as the validity reference — its empirical coverage in
simulation must come out at the nominal level, and if it does not, the harness is broken.

---

## 4.3 PPI estimator

```
θ̂_PPI = f̄_N + (1/n) Σ_{i=1}^n (Y_i − f_i)
       = mean of predictions on the unlabelled pool  +  mean rectifier on the gold set
```

Equivalently, this is the λ = 1 case of §4.4.

Asymptotic variance (as used by the original method):

```
Var(θ̂_PPI) ≈ Var(Y − f)/n  +  Var(f)/N
```

Status: `TEXTBOOK` (restatement of the published method).

---

## 4.4 PPI++ estimator

For a fixed λ:

```
θ̂_λ = (1/n) Σ_{i=1}^n (Y_i − λ f_i)  +  λ f̄_N
     = Ȳ  −  λ (f̄_n − f̄_N)
```

The second line is the form used throughout this project; the two are algebraically identical.

The variance-minimising coefficient:

```
λ*  =  Cov(Y, f)  /  [ (1 + n/N) · Var(f) ]  =  β / (1 + r)
```

The plug-in estimate actually used:

```
λ̂  =  Ĉov(Y, f)  /  [ (1 + r) · V̂ar(f) ]  =  b / (1 + r)
```

with `Ĉov` and `V̂ar` the sample covariance and variance computed on the **same n gold pairs** (the
project's reference implementation uses the `ddof = 1` versions; the choice is immaterial because it
cancels in the ratio).

Status: `TEXTBOOK` (restatement), except that the exact `ddof` and quantile conventions of the
shipped `ppi-py` code path are **OPEN — TO BE VERIFIED** against source (see `03`).

---

## 4.5 PPI++ variance and interval, as used in the preliminary study

```
V̂_PPI++  =  (1/n) · V̂ar(Y − λ̂ f)  +  (λ̂² / N) · V̂ar(f)

CI_PPI++  =  θ̂_λ̂  ±  z_{1−α_lvl/2} · √V̂_PPI++
```

where `V̂ar(Y − λ̂ f)` is the `ddof = 1` sample variance of the n values `Y_i − λ̂ f_i`.

> **Disambiguated 2026-08-19 (no equation changed).** Both `V̂ar(·)` here — and the `V̂ar(f)` of
> §4.3 — are the `ddof = 1` sample moments **of the n gold pairs**, exactly as the `V̂ar` notation
> is defined in §4.4 above ("computed on the **same n gold pairs**"). So `V̂ar(f) = S_ff/(n−1)`.
> This is what `cover.py` (`var_f = (fc*fc).sum(1)/(n−1)`) and `exact.py` (`S_ff/(n−1)`) compute,
> and what `09` §Baseline configuration means by "`ddof = 1` sample moments".
>
> **The LC family is deliberately different.** §4.8.2's `σ̂_f²` is defined there as "the variance
> of `f` estimated from the N unlabelled predictions". The distinction is substantive, not an
> oversight: the PPI/PPI++ expressions are restatements of published methods that take every
> moment from the gold sample, whereas the LC variance's unlabelled term is the variance of `f̄_N`
> and is best estimated from the pool that produced `f̄_N`. Implementations must not unify them.
> See `13` entry L1 and `15` §5.

**The defect under investigation:** this expression is the variance θ̂_λ would have **if λ were the
fixed constant λ\***. It contains no term for the fact that λ̂ was estimated from the same n gold
labels used to form the mean.

Status: `IMPLEMENTED AND MEASURED` — this exact expression produced the ≈ 0.878 coverage at n = 10
reported in `02` and `09`.

---

## 4.6 Proposed leverage correction

The proposed variance, in the limiting case where `μ_f` is known (`N → ∞`):

```
V̂_LC  =  s² · [ 1/n  +  (μ_f − f̄_n)² / S_ff ]
```

**Where the second term comes from.** Write the estimator in the form
`θ̂ = Ȳ + b (μ_f − f̄_n)`, i.e. the sample mean adjusted by an *estimated* slope times the gap
between the labelled sample's prediction mean and the population prediction mean. Two independent
sources of error enter:

1. `Ȳ` deviates from `α + β f̄_n` — contributes `σ²/n`.
2. `b` deviates from `β`, and that error is **multiplied by the extrapolation distance**
   `(μ_f − f̄_n)`. Since `Var(b) = σ² / S_ff`, this contributes `σ² (μ_f − f̄_n)² / S_ff`.

The second term is exactly the classical **leverage** (or extrapolation) term of regression
prediction: the further the point at which you evaluate the fitted line is from the centre of the
design, the more the slope error costs you. The PPI++ variance retains only source 1.

Two structural consequences, both matching the preliminary observations:

- The term does **not** vanish as ρ → 0. Even a useless predictor has an estimated slope with
  sampling error, and `(μ_f − f̄_n)²` is `O(σ_f²/n)` regardless. Hence a coverage deficit
  approximately **invariant in ρ** — as observed (0.8784 / 0.8791 / 0.8787 at ρ = 0 / 0.5 / 0.9).
- Its relative size is `O(1/n)` compared with the `1/n` term, so the deficit decays with n — as
  observed (0.878 → 0.919 → 0.939 → 0.946 at n = 10 / 20 / 50 / 100).

Status: `DERIVED` + `NUMERICALLY VERIFIED` (N → ∞ case). Not yet `FORMALLY PROVED` in written form.

---

## 4.7 Regression interpretation

**Claim (the algebraic identity).** With `μ_f` known and `λ` taken to be the OLS slope `b`, the
PPI++ point estimate is exactly the OLS fitted value of the regression of Y on f, evaluated at
`f = μ_f`:

```
θ̂_b  =  Ȳ − b (f̄_n − μ_f)  =  Ȳ + b (μ_f − f̄_n)  =  α̂ + b μ_f  =  Ŷ(μ_f)
```

since the OLS line passes through `(f̄_n, Ȳ)`.

**Why this makes an exact result available.** Under the Gaussian working model (§4.10, M1),
conditional on `f_1..f_n`:

```
Ȳ − (α + β f̄_n)  =  ε̄            ~  N(0, σ²/n)
b − β                              ~  N(0, σ²/S_ff)
```
and these two are **independent** (the OLS slope is orthogonal to the intercept-at-the-centroid).
Therefore

```
θ̂_b − θ  =  ε̄ + (b − β)(μ_f − f̄_n)   ~  N( 0 ,  σ² [ 1/n + (μ_f − f̄_n)²/S_ff ] )
```

Furthermore `(n−2) s² / σ² ~ χ²_{n−2}`, independent of `(ε̄, b)`. Hence the studentised quantity

```
T  =  ( θ̂_b − θ )  /  ( s · √( 1/n + (μ_f − f̄_n)²/S_ff ) )
```

follows an **exact** `t_{n−2}` distribution, conditionally on the design `f_1..f_n` and therefore
also marginally.

**Note on where normality is and is not needed.** `θ = α + β μ_f` holds *exactly* for the linear
projection of Y on f whether or not the joint distribution is normal, because the projection
residual has mean zero and is uncorrelated with f. So the **point estimator** is fine without
normality. Normality (plus homoskedasticity) is needed only for the **exactness of the t pivot**.
This is what makes Experiments F/G/H in `07` meaningful rather than pointless: the question is how
fast exactness degrades, not whether the estimator breaks.

Status: `DERIVED`, `NUMERICALLY VERIFIED` (coverage 0.9492–0.9501 across all tested cells at
N = 10⁶). Not yet `FORMALLY PROVED` in written form; requires n ≥ 3 for `s²` to be defined and
n ≥ 4 for the `t` to have finite variance.

---

## 4.8 Proposed confidence interval

### 4.8.1 The `N → ∞` interval (the case with an exactness argument)

```
θ̂_LC   =  Ȳ + b (μ_f − f̄_n)

V̂_LC   =  s² · [ 1/n + (μ_f − f̄_n)² / S_ff ]

CI_LC  =  θ̂_LC  ±  t_{n−2, 1−α_lvl/2} · √V̂_LC
```

Status: `NUMERICALLY VERIFIED`, `PROOF PENDING`.

### 4.8.2 The interval as actually implemented in the preliminary study (finite N)

`μ_f` is replaced by the observed unlabelled mean `f̄_N`, and a term for the sampling error of
`f̄_N` is added:

```
θ̂_LC   =  Ȳ + b (f̄_N − f̄_n)

V̂_LC   =  s² · [ 1/n + (f̄_N − f̄_n)² / S_ff ]  +  b² σ̂_f² / N

CI_LC  =  θ̂_LC  ±  t_{n−2, 1−α_lvl/2} · √V̂_LC
```

where `σ̂_f²` is the variance of `f` estimated from the N unlabelled predictions (a large sample, so
essentially known). **In the preliminary code `σ̂_f² = 1` by construction of the simulation; a general
implementation must estimate it.**

**This version is not exact.** Measured over-coverage grew with ρ and shrank with N:

| N | n | ρ | coverage of `CI_LC` |
|---|---|---|---|
| 10⁶ | 10 | 0.0 / 0.5 / 0.9 | 0.9499 / 0.9492 / 0.9496 |
| 1000 | 10 | 0.0 / 0.5 / 0.9 | 0.9513 / 0.9509 / 0.9528 |
| 200 | 10 | 0.0 / 0.5 / 0.9 | 0.9513 / 0.9527 / **0.9612** |

Status: `IMPLEMENTED AND MEASURED`; **known to be inexact at finite N**.

### 4.8.3 Diagnosis of the finite-N inexactness — derivation

Decompose the finite-N estimator around the truth. Conditional on the labelled design `f_1..f_n`,
and writing `Δ = f̄_N − f̄_n`, `δ = f̄_N − μ_f`:

```
θ̂_LC − θ  =  ε̄  +  (b − β) Δ  +  β δ
```

The three components are mutually uncorrelated (`b − β` has mean zero and is independent of the
unlabelled sample, so it does not correlate with `β δ`). Taking expectations over both the residuals
and the unlabelled sample, and using `E[Δ²] = (f̄_n − μ_f)² + σ_f²/N`:

```
Var( θ̂_LC | f_1..f_n )  =  σ²/n
                          +  σ² · [ (f̄_n − μ_f)² + σ_f²/N ] / S_ff
                          +  β² σ_f² / N                                   ... (V_full)
```

Comparing (V_full) with the implemented §4.8.2 expression reveals **two distinct discrepancies**:

1. §4.8.2 substitutes the **realised** `Δ² = (f̄_N − f̄_n)²` for its expectation
   `(f̄_n − μ_f)² + σ_f²/N`. Realised-versus-expected substitution is not automatically wrong — it is
   the correct *conditional-on-`f̄_N`* quantity — but it cannot be combined with an unconditional term.
2. Having used the realised `Δ²` (which already contains the `σ_f²/N` fluctuation of `f̄_N`), the
   implementation then **adds `b² σ̂_f²/N` on top**, partially double-counting the unlabelled-sample
   variability.

Moreover, conditional on `f̄_N` the estimator is **not centred at θ**: it has conditional mean
`β (f̄_N − μ_f)`. So the finite-N problem is not purely a variance-inflation problem.

Additionally, even with the correct variance, the exact `t_{n−2}` pivot no longer holds: the
statistic is the sum of a `t`-pivotal regression component and an approximately normal component
`β δ` whose scale is proportional to `|β| ∝ ρ`, with a **different** effective degrees of freedom.
Applying the `t_{n−2}` quantile to the combined standard error is then conservative — which predicts
exactly the observed pattern of over-coverage increasing with ρ and vanishing at ρ = 0.

Status: (V_full) is `DERIVED` here but **`NOT YET NUMERICALLY VERIFIED`**. The double-counting
diagnosis is `DERIVED`. The degrees-of-freedom explanation of the residual over-coverage is
`CONJECTURED`.

> ## E4 UPDATE (2026-08-20) — status of this subsection after Experiment E
>
> ### Established (measured)
>
> **(V_full) is verified**, to the resolution the simulation can reach. `mean(V̂)/Var(θ̂_LC)` over
> 200,000 replications lies in `[0.988, 1.025]` across all 81 cells of Experiment E
> (`results/E_finite_N.csv`, `results/E_largeN.csv`), the residual being Monte-Carlo error in the
> empirical variance. **Caveat, stated because it matters:** that test resolves the leading terms
> but **not** the `σ² σ_f²/(N·S_ff)` interaction term, which is ≈ 1 % of the variance even in the
> most favourable cell — deleting it moves the ratio by less than the Monte-Carlo error. That term
> is established *algebraically* (below), not numerically.
>
> **The §4.8.2 expression is not measurably worse.** `v482`'s variance ratio is
> `[0.988, 1.025]` too — the same interval. Substituting the realised `Δ²` costs nothing
> detectable, and coverage under the two differs by at most 0.0007 in any cell.
>
> ### Derived (new, this package)
>
> **1. An exact decomposition.** For any `n`, `N`,
> ```
> θ̂_LC − θ  =  ε̄  +  (b − β)(μ_f − f̄_n)  +  b (f̄_N − μ_f)
> ```
> This is an algebraic identity — verified to `1e-12` in `tests/test_finite_n.py` — not an
> approximation. It replaces the `Δ`-based decomposition above, in which the middle term's
> multiplier is random.
>
> **2. (V_full) is exactly the conditional variance, and the regrouping shows why.** Given the
> labelled design the three terms are uncorrelated, `E[b] = β`, and `δ = f̄_N − μ_f` is independent
> of the labelled sample, so `Var(bδ | design) = E[b²]·σ_f²/N` with `E[b²] = β² + σ²/S_ff`. Hence
> ```
> Var(θ̂_LC | f_1..f_n)  =  σ²/n  +  σ²(μ_f − f̄_n)²/S_ff  +  (β² + σ²/S_ff)·σ_f²/N
> ```
> which is **algebraically identical to (V_full) as written above** (checked to `1e-15`). The
> `σ²σ_f²/(N·S_ff)` term is not an artefact; it is the cost of multiplying the estimated slope by
> the estimated gap.
>
> **3. The realised-`Δ²` substitution is unbiased, so discrepancy 1 above is not a defect.**
> `E[Δ² | design] = (f̄_n − μ_f)² + σ_f²/N` exactly, so the §4.8.2 leverage term has expectation
> equal to (V_full)'s.
>
> **4. Discrepancy 2 — the double count — is real but is the wrong size and the wrong sign in ρ.**
> Using `b²` where the exact expression has `β²` inflates the variance by exactly
> `σ²σ_f²/(N·S_ff)`, i.e. `O(1/N)` relative — under 1 % at `N = 200`, `n = 10`. And it carries a
> factor `σ² = 1 − ρ²`, so it *shrinks* as ρ grows, whereas the observed over-coverage *grows*
> sharply with ρ. **The double-counting diagnosis therefore cannot be the cause of the observed
> over-coverage**, and the measured variance ratios confirm it.
>
> **5. The cause is the quantile, and the effect is quantitatively predictable.** Write
> `s = (b²σ̂_f²/N)/V̂` for the share of the variance carried by the essentially-normal unlabelled
> component. Satterthwaite gives `ν_eff = (n−2)/(1 − s)²`. Applying the `t_{n−2}` quantile to a
> statistic behaving like `t_{ν_eff}` predicts coverage `P(|t_{ν_eff}| ≤ t_{n−2, 1−α/2})`. That
> prediction, which has **no free parameters**, matches the measured B9 coverage to within
> **0.0029 across all 81 cells** (mean absolute error 0.0007, correlation 0.973). The §4.8.3
> conjecture is thus promoted from `CONJECTURED` to `DERIVED + NUMERICALLY VERIFIED`, and it is the
> whole explanation, not part of one.
>
> ### Derived after E4b (2026-08-20) — see `docs/theory_finite_N.md` for the full argument
>
> * **Theorem A′ (`PROVED`).** `(θ̂_LC − θ − βδ)/(s√(1/n + Δ²/S_ff)) ~ t_{n−2}` exactly, for the
>   estimator **as implemented**, conditionally on `f̄_N` and marginally. The §4.8.2 interval is an
>   exact `1−α` interval for `θ + βδ`; the finite-N problem is an **unestimable bias**, not a
>   variance error.
> * **The exact marginal law of the studentised statistic is not a standard family** (`DERIVED`).
>   The numerator carries a product `Z₁W` of independent normals and `V̂` is not independent of it.
> * **Theorem B (`PROVED` under the stated idealisation).** Dropping the `O(1/N)` product term and
>   the `V̂`-numerator dependence, `T = Z/√((1−s)Q/ν + s)`, a normal variance mixture — the
>   Behrens–Fisher law with one degree of freedom infinite — with exact coverage integral
>   `E_Q[2Φ(q√((1−s)Q/ν + s)) − 1]`.
> * **`ν_eff = (n−2)/(1−s)²` is Welch–Satterthwaite instantiated** (`DERIVED`), matching the first
>   two moments of `V̂` — **not** of `T`, whose variance and kurtosis it misses.
> * **Theorem C (`PROVED` under the same idealisation).** Coverage is monotone non-decreasing in
>   `s`, hence never below nominal: the conservativeness claim R4, proved for the idealised
>   statistic.
>
> ### Open
>
> * Whether `t_{ν_eff}` is **exact** or merely a very good approximation. Satterthwaite is an
>   approximation by construction; nothing here proves an exact finite-N pivot exists, and R2 is
>   measurably *not* conservative (see §4.8.4). `OPEN`.
> * A proof of conservativeness for §4.8.2 (R4). 81 of 81 cells are at or above nominal, but that
>   is evidence, not a theorem. `OPEN`.

### 4.8.4 Candidate finite-N repairs (to be tested, not assumed)

- **R1 — expectation form.** Use (V_full) with `(f̄_n − f̄_N)²` replaced by
  `max(0, (f̄_n − f̄_N)² − σ̂_f²/N)` as an estimate of `(f̄_n − μ_f)²`, and no extra additive term
  beyond `b² σ̂_f²/N`. Status: `PROPOSED`.
- **R2 — Satterthwaite / Welch effective degrees of freedom.** Keep the §4.8.2 variance
  decomposition but compute an effective ν from the relative sizes of the `t_{n−2}` component and
  the (essentially known-variance, ν = ∞) component `b² σ̂_f²/N`, and use `t_{ν_eff}`.
  Status: `PROPOSED`.
- **R3 — conditional form.** Accept the conditional-on-`f̄_N` view, and correct the conditional bias
  rather than the variance. Status: `PROPOSED`, least developed.
- **R4 — do nothing and prove conservativeness.** If the §4.8.2 interval can be shown to be
  **conservative** (coverage ≥ nominal) for all N, n, ρ under M1, that is itself a publishable and
  practically sufficient result. Status: `PROPOSED`; supported by all three measured N values but
  **not proved**, and a counterexample search is required.

> ## E4 UPDATE (2026-08-20) — verdicts on R1–R4, measured over all 81 cells of Experiment E
>
> | | worst \|coverage − 0.95\| | cells below nominal | verdict |
> |---|---|---|---|
> | §4.8.2 `v482` (B9) | 0.0211 | 0 | conservative, never invalid; over-covers at small `N/n` |
> | **R1** expectation form | 0.0213 | 0 | **does not repair anything.** Algebraically `R1 = v482` whenever `Δ² ≥ σ̂_f²/N`, and wider otherwise, so it can only *increase* an over-coverage that is already the problem |
> | **R2** Satterthwaite | **0.0028** | 1 (0.9472) | **the repair that works.** Meets `14` Q1's success condition (a) of ±0.004 at every cell. Costs ≈ 0.8 % of interval width on average, up to 10 % at the worst cell |
> | **R3** conditional form | 0.3275 | 22 | **fails, and the reason is instructive** — see below |
> | **R4** prove conservativeness | — | 0 | supported by 81/81 cells; still `OPEN` as a theorem |
>
> **R1 was never going to work, and the algebra says so.** `max(0, Δ² − σ̂_f²/N) + σ̂_f²/N` equals
> `Δ²` exactly unless the max clips. R1 therefore differs from `v482` only where `Δ² < σ̂_f²/N`, and
> is *wider* there. Since the problem is over-coverage, a uniformly wider interval cannot be the
> repair. Measured: R1 and `v482` differ by at most 0.0004 in coverage anywhere.
>
> **R3 as specified is infeasible, which is itself the finding.** `04` gives no formula for R3, so
> the implementation takes the literal conditional view: conditional on the labelled design *and*
> `f̄_N`, `θ̂_LC` is exactly `t_{n−2}`-pivotal about `θ + β(f̄_N − μ_f)`, giving
> `V̂ = s²(1/n + h)` with no unlabelled term. That interval is exact — **for the wrong estimand**.
> Correcting the conditional bias `β(f̄_N − μ_f)` requires `μ_f`, which is exactly what is not
> known; there is nothing to plug in. Measured consequence: coverage collapses to 0.6225 at
> `n = 50, ρ = 0.9, N = 50`. **R3 should be closed as not viable** unless someone supplies an
> estimator of the bias, and `04` should say so rather than leaving it "least developed".
>
> **R2's cost and its one flaw.** `ν_eff = (n−2)/(1 − s)² ≥ n−2` always, so R2 only ever narrows.
> It is therefore **not conservative**: at `n = 50, ρ = 0.9, N = 50` it covers 0.9472, i.e. 5.7
> Monte-Carlo standard errors below nominal. The honest summary is a trade: `v482` is never invalid
> but wastes up to 2 % coverage; R2 is accurate to ±0.003 but can sit a few thousandths below
> nominal. Which to recommend is a research decision, recorded as `OPEN` in `14` Q1.
>
> **R5, added after the theory (E4b) and not one of `04`'s R1–R4.** Theorem B makes the exact
> quantile computable by inverting the coverage integral, which removes Satterthwaite's
> approximation error entirely. Measured over the same 81 cells: worst `|cov − 0.95| = 0.0028`,
> RMSE 0.0009, **five** cells more than 3 MC s.e. below nominal against R2's one. **R5 is not
> better than R2.** Since removing the quantile approximation does not improve coverage, the
> residual ≈ 0.003 is the cost of the *idealisation*, not of the approximation — and R2's accuracy
> is therefore partly fortuitous. See `docs/theory_finite_N.md` §9.
>
> **Practical note on how much this matters.** PPI is used when N ≫ n. The measured distortion at
N/n = 20 (N = 200, n = 10) is ≈ +0.011 coverage; at N/n = 100 it is ≈ +0.003; at N/n = 10⁵ it is
undetectable. R4 may therefore be entirely adequate in practice, but the project should determine
which of R1–R4 is correct rather than assume.

---

## 4.9 Shrunk tuning coefficient λ̃

> **Proposed / not yet theoretically finalized.**

Define the studentised slope statistic

```
se(b)  =  s / √S_ff            (the standard OLS slope standard error)
t²     =  ( b / se(b) )²
```

and the positive-part shrinkage rule

```
λ̃  =  λ̂ · ( 1 − c / t² )₊ ,        c > 0 a constant to be chosen
```

`(x)₊ = max(x, 0)`. As `t² → ∞` (a clearly useful predictor) `λ̃ → λ̂`; when the slope is not
distinguishable from zero, `λ̃ → 0` and the estimator falls back continuously to classical inference.
Unlike a hard threshold, the map is continuous in the data.

**Status of this object has changed and the change must be respected.** In the preliminary study
`λ̃` was introduced as a candidate *validity* fix and **it did not work**: at n = 10 with c = 3 the
resulting interval covered 0.9469 at ρ = 0 but only 0.9211 at ρ = 0.8 — non-uniform, i.e. it failed
for the same reason the hard threshold failed. The leverage correction of §4.6–4.8, by contrast,
achieved near-exact coverage **without any shrinkage of λ**.

Therefore:

- λ̃ is **demoted** from a validity mechanism to an **efficiency** mechanism. The open question is
  whether, *given* a valid interval, shrinking λ produces a narrower one on average, particularly
  in the low-ρ regime.
- The preliminary code computed `t²` using `λ̂` rather than `b` in the numerator. Since
  `λ̂ = b/(1+r)` and `r = n/N ≈ 0` in all tested settings, the difference was negligible, but a
  general implementation should use `b / se(b)` as written above. **Discrepancy logged.**
- The choice of `c` is `OPEN — requires research decision`. Values 1, 2 and 3 were probed. No
  principled derivation of `c` exists yet.

---

## 4.10 Assumptions, stated separately

**M1 — Gaussian working model** (used for the exactness claim in §4.7):

- A1. `(Y_i, f_i)`, i = 1..n, are i.i.d. draws from a **bivariate normal** distribution.
- A2. The N unlabelled predictions are i.i.d. from the same marginal distribution of `f`, and are
  **independent of the labelled sample**.
- A3. `Var(f) > 0`.
- A4. `n ≥ 3` (so that `s²` is defined); `n ≥ 4` recommended.
- A5. The predictor `f` is a fixed function, **not** fitted on the labelled sample used for
  inference.
- A6. `θ = E[Y]` — the estimand is a mean.
- A7. Homoskedasticity: `Var(Y | f)` does not depend on `f`. (Implied by A1.)
- A8. No missingness, no measurement error in `Y`, gold labels are the ground truth by definition.

**M2 — Linear-projection model** (the weaker model under which the *point estimator* remains valid):

- A1′. `(Y_i, f_i)` i.i.d. with finite second moments. Normality **not** assumed.
- A2′–A6′, A8′ as above.
- Under M2, `θ = α + β μ_f` still holds exactly for the linear projection, so `θ̂_LC` is consistent
  and asymptotically normal, but the `t_{n−2}` pivot is only approximate.

**M3 — What is deliberately not assumed anywhere:**

- Nothing about the *quality* of `f`. ρ may be 0.
- No calibration of `f`. `f` may be systematically biased by any amount.
- No relationship between `f` and `Y` beyond joint distribution — in particular `f` need not be an
  estimate of `Y` at all.

---

## 4.11 Exactness status

### Demonstrated / derived
- The `N → ∞` Gaussian case (§4.7). Exact `t_{n−2}` pivot, derived, and numerically confirmed at
  coverage 0.9492–0.9501 across every (n, ρ) cell tested.

### Open
- The finite-N correction. (V_full) is derived (§4.8.3) but unverified; R1–R4 are untested;
  the §4.8.2 implementation is known to over-cover, with the magnitude measured.

### Unverified
- Everything outside M1: heavy tails, skew, heteroskedasticity, discrete/binary `Y`, model
  misspecification of the conditional mean, estimands other than the mean, and any real data.
  **No experiment in this class has been run.**

---

## 4.12 Derivation status tags

| Tag | Meaning |
|---|---|
| `TEXTBOOK` | A restatement of published/standard material. No claim of contribution. |
| `DERIVED` | Worked out in this package's mathematics, believed correct, written down. |
| `NUMERICALLY VERIFIED` | Confirmed by Monte-Carlo simulation at the stated precision. |
| `FORMALLY PROVED` | A written proof exists that a referee could check. **Nothing in this package currently carries this tag.** |
| `CONJECTURED` | A plausible explanation consistent with observed data; no derivation. |
| `PROPOSED` | A candidate construction with neither derivation nor evidence yet. |
| `IMPLEMENTED AND MEASURED` | Code exists and produced the stated numbers. Says nothing about correctness of the underlying claim. |
| `PROOF PENDING` | Derived and numerically verified, but the formal write-up does not yet exist. |
| `OPEN` | Not resolved. |

**Register of every named result:**

| Result | Tag |
|---|---|
| §4.2 classical interval | `TEXTBOOK` |
| §4.3 PPI estimator and variance | `TEXTBOOK` |
| §4.4 PPI++ estimator, λ* | `TEXTBOOK` |
| §4.5 PPI++ variance/interval as studied | `IMPLEMENTED AND MEASURED` |
| §4.6 leverage term identification | `DERIVED`, `NUMERICALLY VERIFIED` |
| §4.7 algebraic identity θ̂_b = OLS fitted value | `DERIVED` |
| §4.7 exact `t_{n−2}` pivot at N → ∞ | `DERIVED`, `NUMERICALLY VERIFIED`, `PROOF PENDING` |
| §4.8.1 `CI_LC` at N → ∞ | `NUMERICALLY VERIFIED`, `PROOF PENDING` |
| §4.8.2 `CI_LC` at finite N | `IMPLEMENTED AND MEASURED`, known inexact |
| §4.8.3 (V_full) | `DERIVED`, `NUMERICALLY VERIFIED` (E4, leading terms; the `σ²σ_f²/(N S_ff)` term algebraically only) |
| §4.8.3 (V_full) = exact conditional variance, via the E4 decomposition | `DERIVED`, `PROOF PENDING` |
| §4.8.3 exact decomposition `θ̂−θ = ε̄ + (b−β)(μ_f−f̄_n) + b(f̄_N−μ_f)` | `DERIVED`, verified to 1e-12 |
| §4.8.3 double-counting diagnosis | `DERIVED` but **refuted as the cause** of the over-coverage (E4) |
| §4.8.3 effective-df explanation | `DERIVED`, `NUMERICALLY VERIFIED` (E4: predicts coverage to 0.003 over 81 cells, no free parameters) |
| §4.8.4 R1 | `IMPLEMENTED AND MEASURED`; **does not repair** (E4) |
| §4.8.4 R2 | `IMPLEMENTED AND MEASURED`; meets Q1's ±0.004 at every cell but is **not conservative** (E4) |
| §4.8.4 R3 | `IMPLEMENTED AND MEASURED`; **infeasible as specified** (E4) |
| §4.8.4 R4 | `PROVED` for the idealised statistic (Theorem C); `OPEN` for the implemented one; 81/81 cells at or above nominal |
| §4.8.3 exact conditional pivot (Theorem A′) | `PROVED`; the distributional fact is `TEXTBOOK`, the **identification** is what is new (E4d) |
| §4.8.3 exact marginal law | `DERIVED` as a 5-fold integral; no standard family; closed form `OPEN` |
| §4.8.3 Theorem B (idealised law + exact coverage integral) | `PROVED` for the idealised statistic — but **`TEXTBOOK`**: it is the Behrens–Fisher statistic with one variance known (E4d) |
| §4.8.4 R4 / Theorem C conservativeness | **`TEXTBOOK`** — Mickey & Brown (1966) at `ν₂ = ∞`; our proof is an exposition, novelty withdrawn (E4d, `theory_finite_N.md` §III.2) |
| §4.8.4 R5 (exact-quantile rule, new) | `IMPLEMENTED AND MEASURED`; no better than R2 |
| §4.8.3 idealisation error `Δ = coverage_act − coverage_id` | rate `O(1/N)` `DERIVED` + `NUMERICALLY VERIFIED`; bound constant `CONJECTURED`; sign **not** definite (counterexample at n=5, ρ=0.9) — see `docs/theory_finite_N.md` Part II |
| Conservativeness of the **practical** interval (R4) | `OPEN`; supported at 81 + 48 cells, min coverage 0.9488 |
| §4.9 shrunk λ̃ | `PROPOSED`; measured to fail as a validity fix |
| Validity–efficiency frontier (a lower bound on the width any valid interval must have) | `OPEN` |
| M-estimation extension | `OPEN` |
