# 06 — Baselines and Comparisons

Ten methods. Every experiment in `07` runs **all applicable baselines on the same random draws**
(common random numbers). Baseline identifiers `B1`–`B10` are fixed and must be used verbatim in code,
column headers, CSV files, figure legends and prose.

> **The proposed method is not assumed to win.** B7 and B8 in particular are serious competitors,
> and B8 (BCa) is a pre-registered kill criterion (K3 in `10`). Report them fairly. If a baseline
> beats LC-PPI, that is the result.

---

## Summary table

| ID | Name | Point estimate | Variance / interval mechanism | Quantile | Role |
|---|---|---|---|---|---|
| B1 | Classical *t* | `Ȳ` | `s_Y²/n` | `t_{n−1}` | validity reference / harness check |
| B2 | PPI | `Ȳ − (f̄_n − f̄_N)` | asymptotic | `z` | original method |
| B3 | **PPI++ (as shipped)** | `Ȳ − λ̂(f̄_n − f̄_N)` | asymptotic, no leverage | `z` | **the target of the critique** |
| B4 | PPI++ with *t* quantile | as B3 | as B3 | `t_{n−2}` | isolates the quantile effect |
| B5 | Cross-fitted PPI++ | fold-split λ̂ | as B3 | `t_{n−2}` | the fix recommended by the paper documenting the failure |
| B6 | Hard-threshold PPI++ | λ̂ or 0 by significance | as B3 | `t_{n−2}` | the crudest plausible fix |
| B7 | Percentile bootstrap | `Ȳ − λ̂(f̄_n − f̄_N)` | resampling | empirical | model-free competitor |
| B8 | **BCa bootstrap** | as B7 | resampling + bias/accel. | empirical | **strongest untested competitor; kill criterion K3** |
| B9 | **LC-PPI (proposed)** | `Ȳ + b(f̄_N − f̄_n)` | `s²(1/n + leverage) + b²σ̂_f²/N` | `t_{n−2}` | the proposal |
| B10 | LC-PPI + shrunk λ̃ | uses `λ̃` | as B9 with `λ̃` | `t_{n−2}` | efficiency variant |

---

## B1 — Classical *t* interval

**Formula**

```
θ̂ = Ȳ                     s_Y² = (1/(n−1)) Σ (Y_i − Ȳ)²
CI = Ȳ ± t_{n−1,1−α/2} · s_Y/√n
```

**Inputs:** `Y` only. Ignores `f` and the unlabelled pool entirely.
**Output:** point estimate, interval.
**Purpose:** (i) the validity reference — under a normal model it is *exact*, so its measured
coverage tells you whether the simulation harness itself is correct; (ii) the efficiency reference —
all interval widths are reported relative to B1.
**Expected behaviour:** coverage indistinguishable from nominal at every n and ρ. Width ratio 1.000
by definition.
**Why included:** if B1 does not come out at 0.950 ± 3 Monte-Carlo standard errors, **stop and debug
the harness**; nothing else in the study is interpretable.
**Literature:** `TEXTBOOK`.

**Measured (preliminary, N = 10⁶, 200k reps):** 0.9496 / 0.9490 / 0.9511 at n = 10, ρ = 0/0.5/0.9.

---

## B2 — PPI

**Formula**

```
θ̂ = f̄_N + (1/n) Σ (Y_i − f_i)  =  Ȳ − (f̄_n − f̄_N)
V̂ = V̂ar(Y − f)/n + V̂ar(f)/N
CI = θ̂ ± z_{1−α/2} √V̂
```

**Purpose:** shows the failure mode PPI++ was invented to fix — when the predictor is poor,
`Var(Y − f) > Var(Y)` and the interval is *wider* than B1.
**Expected behaviour:** width ratio > 1 at low ρ, < 1 at high ρ; coverage closer to nominal than B3
at small n (because it estimates no tuning coefficient, so it carries no leverage penalty) but the
interval is inefficient.
**Why included:** it is the only PPI-family method with **no estimated coefficient**, so it isolates
the effect of estimating λ. If B2 covers well and B3 does not, that is direct evidence for the
project's diagnosis.
**Literature:** Angelopoulos et al., *Science* 2023 (see `03`).

> **Note:** B2 was **not** measured in the preliminary study. Its behaviour above is a prediction,
> not a result.

---

## B3 — PPI++ as shipped

**Formula** — see `04` §4.4–4.5.

```
λ̂ = Ĉov(Y,f) / [(1 + n/N) V̂ar(f)]
θ̂ = Ȳ − λ̂ (f̄_n − f̄_N)
V̂ = V̂ar(Y − λ̂ f)/n + λ̂² V̂ar(f)/N
CI = θ̂ ± z_{1−α/2} √V̂
```

**Purpose:** the object of the critique. Every claim in this project is relative to B3.
**Expected behaviour:** materially below nominal coverage at small n, approximately invariant in ρ.
**Why included:** it is the method practitioners actually run.
**Literature:** Angelopoulos, Duchi, Zrnic, arXiv:2311.01453.

**Measured (preliminary, N = 10⁶, 200k reps):**

| n | ρ = 0.0 | ρ = 0.5 | ρ = 0.9 |
|---|---|---|---|
| 10 | 0.8784 | 0.8791 | 0.8787 |
| 20 | 0.9194 | 0.9198 | 0.9196 |
| 50 | 0.9393 | 0.9384 | 0.9394 |

> **OPEN — TO BE VERIFIED.** Whether the shipped `ppi-py` code path uses exactly this variance and a
> normal quantile. Read the source before publishing any claim about "PPI++ as shipped". If it
> differs, rename this baseline to "PPI++ (asymptotic Wald, as specified in the paper)" and add the
> true shipped variant as B3b.

---

## B4 — PPI++ with a Student-*t* quantile

**Formula:** identical to B3 except `t_{n−2,1−α/2}` replaces `z_{1−α/2}`.
**Purpose:** decomposition. Isolates how much of B3's deficit is merely the wrong quantile versus
the missing leverage term.
**Expected behaviour:** partial recovery only.
**Why included:** it is the first thing a referee will ask about, and it must be shown not to
suffice.

**Measured (preliminary, cover.py, N = 10⁵, 60k reps), n = 10:** 0.9241 (ρ=0), 0.9247 (ρ=0.2),
0.9234 (ρ=0.5), 0.9240 (ρ=0.8), 0.9249 (ρ=0.95).
Recovery: from 0.878 to ≈ 0.924 against a 0.950 target ≈ **64% of the deficit**.

---

## B5 — Cross-fitted PPI++

**Formula.** Split the gold sample into folds A and B of sizes ⌊n/2⌋ and ⌈n/2⌉. Estimate `λ̂_A` on
fold A and apply it to form the estimate on fold B, and vice versa; average the two half-estimates.

```
λ̂_A = S_fY(A)/S_ff(A),  λ̂_B = S_fY(B)/S_ff(B)
θ̂_B = mean_{i∈B}(Y_i − λ̂_A f_i) + λ̂_A f̄_N
θ̂_A = mean_{i∈A}(Y_i − λ̂_B f_i) + λ̂_B f̄_N
θ̂  = ½(θ̂_A + θ̂_B)
```
Variance as in B3, computed on the combined sample; quantile `t_{n−2}`.

**Purpose:** this is the variant analysed — and in effect recommended — by the paper that documents
the finite-sample failure (`03`, No Free Lunch). It removes the bias from re-using labels but pays
in effective sample size.
**Expected behaviour:** bias removed; **coverage still short of nominal**; point-estimate MSE
*worse* than B3 at small n.
**Why included:** it is the incumbent fix. Showing that the incumbent fix does not restore coverage
is a load-bearing part of the argument.
**Literature:** Mani, Xu, Lipton, Oberst, arXiv:2505.20178.

**Measured (preliminary), n = 10:** coverage 0.9254 / 0.9261 / 0.9245 / 0.9248 / 0.9258 at
ρ = 0 / 0.2 / 0.5 / 0.8 / 0.95 — i.e. essentially indistinguishable from B4.
**MSE (preliminary, probe.py, ratio to classical), n = 10:** 1.4792 at ρ = 0, 1.4826 at ρ = 0.1,
1.1170 at ρ = 0.5, 0.2766 at ρ = 0.9. Compare B3 at the same cells: 1.1416, 1.1341, 0.8523, 0.2137.
**Cross-fitting is substantially worse in MSE at small n.**

> **DISCREPANCY FOUND 2026-08-19 — the formula above and the measured numbers below are two
> different estimators.** The formula above (average the two *fold-level estimates*) is what
> `probe.py` computes, and it is the source of Table A5's B5 column. `cover.py`, which produced
> **Table A4's** B5 column, instead averages the two *fold λ̂'s* and applies the average to the
> whole gold sample. They are not equivalent: at n = 10, N = 10⁵, R = 60,000 the fold-split
> estimator covers **0.9092** and the averaged-λ estimator covers **0.9243** (the latter matching
> Table A4's 0.9254 within tolerance). Both are registered — `B5` is the formula above, `B5-avglam`
> is `cover.py`'s — so each measured number has an exact counterpart. `B5-kfold` (K = 5) is the
> third scheme, covering 0.9173. **None of the three reaches 0.945**, so H2 survives the difference;
> but any claim about "cross-fitting" must name the scheme. Logged in `13`.

> **Implementation note.** The preliminary code's fold construction and averaging is one of several
> reasonable cross-fitting schemes. A different scheme (e.g. K-fold with K > 2, or averaging λ̂
> rather than averaging estimates) may perform better. **OPEN — the implementation should offer at
> least the 2-fold scheme above and one K-fold alternative, and report both**, so that the claim
> "cross-fitting does not fix it" is not an artefact of one particular split.

---

## B6 — Hard-threshold PPI++

**Formula**

```
t² = (b / se(b))²,   se(b) = s/√S_ff
λ_thr = λ̂  if t² > 4   (i.e. |t| > 2)  else  0
```
then B3's variance and a `t_{n−2}` quantile with `λ_thr` in place of `λ̂`.

**Purpose:** the crudest plausible repair — "only use the model if its slope is significant".
**Expected behaviour:** helps at ρ = 0 (where it usually selects λ = 0 and reduces to B1), but is
**discontinuous** in the data and therefore behaves erratically in the transition region.
**Why included:** to demonstrate that a selection rule is not a substitute for a correct variance.

**Measured (preliminary), n = 10:** 0.9409 (ρ=0) → 0.9367 (0.2) → 0.9237 (0.5) → **0.9194** (0.8)
→ 0.9248 (0.95). **Non-monotone and worst where the predictor is good** — the opposite of what a
useful rule should do.

**Threshold value:** `|t| > 2` was used in the preliminary study. **OPEN — no principled
justification; the implementation should expose the threshold as a parameter and Experiment D
should sweep at least {1.5, 2, 3}** so that B6's failure is not an artefact of one arbitrary cut.

> **Swept 2026-08-19** (n = 10, N = 10⁵, R = 60,000), registered as `B6-t1.5` / `B6` / `B6-t3`.
> Every threshold falls below 0.945 somewhere, and the non-monotonicity **grows** with the
> threshold — exactly as a selection-rule explanation predicts, since a higher bar means more
> selection and a sharper discontinuity:
>
> | `t_crit` | ρ=0 | ρ=0.2 | ρ=0.5 | ρ=0.8 | ρ=0.95 | spread |
> |---|---|---|---|---|---|---|
> | 1.5 | 0.9338 | 0.9306 | 0.9231 | 0.9203 | 0.9238 | 0.0135 |
> | 2 | 0.9403 | 0.9363 | 0.9247 | 0.9178 | 0.9238 | 0.0225 |
> | 3 | 0.9489 | 0.9459 | 0.9325 | **0.9128** | 0.9231 | 0.0361 |
>
> `t_crit = 3` is nearly nominal at ρ = 0 (it almost always selects λ = 0 and reduces to the
> classical mean with a `t_{n−2}` quantile) and the **worst of all** at ρ = 0.8. That is the
> sharpest available demonstration that a selection rule is not a substitute for a correct
> variance.

---

## B7 — Percentile bootstrap

**Formula.** Resample the n gold pairs `(Y_i, f_i)` with replacement `B` times. For each resample
recompute λ̂ and `θ̂*`. The interval is the `[α/2, 1−α/2]` empirical quantiles of the `θ̂*`.

- The unlabelled pool is **not** resampled by default (it is large; its contribution is `O(1/N)`).
  **OPEN — decide and document:** whether to also resample `f_unlabeled` (or add its analytic
  variance) when `N/n` is small. Experiment E should test both.
- `B = 2000` default; `B = 9999` for the final tables.

**Purpose / why included:** the bootstrap is the standard model-free answer to "the plug-in variance
is wrong because a parameter was estimated". It automatically accounts for the variability of λ̂ and
requires no derivation.
**Expected behaviour:** better than B3, but the percentile bootstrap is known to be first-order
accurate only and to under-cover at very small n.

> **E3 RESULT (2026-08-19, `results/B_bootstrap.csv`, N = 10⁶, R = 20,000, B = 2000, MC s.e.
> 0.0015).** The expectation is confirmed, and the under-coverage is far larger than "very small n"
> suggests. Coverage at n = 10: **0.8991 – 0.9016** across ρ ∈ {0, 0.3, 0.5, 0.7, 0.9}; at n = 20:
> 0.9237 – 0.9286; at n = 50: 0.9382 – 0.9425. Better than B3 (0.878) at every cell, and short of
> nominal at every cell. At `B = 9999` (`results/B_final.csv`) the n = 10 figures are unchanged
> (0.8998 – 0.9056), so the shortfall is **not** bootstrap Monte-Carlo error.
**Literature:** Efron; Efron & Tibshirani (1993). `CLASSICAL`.

> ~~**Not yet run.** No preliminary result exists for B7.~~ Run in E3, 2026-08-19.

---

## B8 — BCa bootstrap

**Formula.** As B7, with the bias-correction constant `ẑ₀` from the proportion of bootstrap
replicates below the observed estimate, and the acceleration constant `â` from a jackknife over the
n gold pairs; adjusted percentiles per the standard BCa formulas.

**Purpose:** BCa is second-order accurate and is the strongest general-purpose competitor.
**Expected behaviour:** unknown. It may well attain nominal coverage at n = 10.

> **E3 RESULT (2026-08-19) — it does not, and K3 does not fire.** At n = 10, N = 10⁶, R = 20,000:
> coverage **0.8983 – 0.9026** at `B = 2000` and **0.9011 – 0.9038** at `B = 9999`. The K3 trigger
> requires ≥ 0.945 at every ρ; the observed shortfall is ≈ 0.045, about **30 Monte-Carlo standard
> errors**. BCa is statistically **indistinguishable from the percentile bootstrap** in this
> problem (differences within ± 0.003, i.e. ~2 MC s.e., with no consistent sign), which is what one
> should expect: the PPI++ estimator is very nearly unbiased and symmetric here, so `ẑ₀ ≈ 0` and
> the acceleration has almost nothing to correct. Second-order accuracy does not help when the
> first-order problem is that the *studentising quantity itself* is estimated from the same n
> points. Interval widths are **narrower** than B9's (0.933 vs 1.081 at ρ = 0), which is the
> under-coverage expressed in different units rather than an efficiency gain. Cost: ≈ 10⁴ × B9 at
> `B = 2000` and ≈ 10⁵ × at `B = 9999`.
**Why included:** **this is a pre-registered kill criterion.** If B8 covers at ≥ 0.945 across all ρ
at n = 10 *and* offers no disadvantage, the project's headline changes (see K3 in `10`). Note that
even in that case LC-PPI retains three arguments — it is `O(n+N)` rather than `O(B(n+N))`, it is
deterministic, and it comes with an exactness proof rather than an asymptotic-order argument — but
these are weaker claims and must be stated as such.
**Literature:** Efron & Tibshirani (1993). `CLASSICAL`.

> ~~**Not yet run.** This is the single highest-priority missing experiment.~~ Run in E3,
> 2026-08-19; see the E3 result above and `10` K3.

---

## B9 — LC-PPI (proposed)

**Formula** — `04` §4.8.2, `05` Algorithm.

```
θ̂ = Ȳ + b (f̄_N − f̄_n)
V̂ = s² [1/n + (f̄_N − f̄_n)²/S_ff] + b² σ̂_f²/N
CI = θ̂ ± t_{n−2,1−α/2} √V̂
```

**Expected behaviour:** exactly nominal coverage under M1 as `N → ∞`; conservative at finite N.
**Why included:** it is the proposal.

**Measured (preliminary, N = 10⁶, 200k reps):**

| n | ρ | coverage | width ÷ B1 |
|---|---|---|---|
| 10 | 0.0 | 0.9499 | 1.081 |
| 10 | 0.5 | 0.9492 | 0.936 |
| 10 | 0.9 | 0.9496 | 0.471 |
| 20 | 0.0 | 0.9492 | 1.031 |
| 20 | 0.5 | 0.9497 | 0.893 |
| 20 | 0.9 | 0.9497 | 0.449 |
| 50 | 0.0 | 0.9501 | 1.011 |
| 50 | 0.5 | 0.9494 | 0.876 |
| 50 | 0.9 | 0.9500 | 0.441 |

**Read the width column honestly.** At ρ = 0, n = 10 a *valid* interval is **8.1% wider than
classical** — the price of using a useless predictor honestly. B3's 0.814 width ratio in the same
cell is not an efficiency gain; it is the under-coverage expressed in different units.

---

## B10 — LC-PPI with shrunk λ̃

**Formula.** As B9 but with `λ̃ = λ̂(1 − c/t²)₊` in place of `b`, and correspondingly recomputed
residuals, leverage and variance.

**Purpose:** efficiency, not validity. The question is whether shrinking the coefficient narrows the
interval at low ρ without breaking coverage.
**Expected behaviour:** unknown. Note that shrinkage changes the residual variance and the leverage
term simultaneously, and the exactness argument of `04` §4.7 **does not carry over** to a shrunken
coefficient. B10 must therefore be validated purely empirically.
**Why included:** it was proposed during selection and should not be silently dropped; but see the
demotion note in `04` §4.9.

**Measured (preliminary — as a *validity* fix, applied on top of the B3 variance, not on top of
B9):** at n = 10 with c = 3, coverage 0.9469 / 0.9441 / 0.9332 / 0.9211 / 0.9230 at
ρ = 0 / 0.2 / 0.5 / 0.8 / 0.95 — **non-uniform; it failed.** With c = 1: 0.9373 / 0.9348 / 0.9255 /
0.9217 / 0.9245.

> **Important:** the preliminary λ̃ results were obtained **without** the leverage correction. B10 as
> defined here (shrinkage *plus* leverage correction) has never been run.

---

## Reporting requirements common to all baselines

For every (method, cell) the following must be recorded:

1. empirical coverage, with its Monte-Carlo standard error `√(p(1−p)/R)`;
2. mean interval width and median interval width;
3. width ratio to B1;
4. MSE of the point estimate, and its ratio to B1;
5. fraction of the oracle efficiency gain retained, where the oracle uses the population λ*;
6. the count of degenerate/failed constructions (e.g. `S_ff = 0`, bootstrap non-convergence).

Item 6 must never be silently dropped from the denominator. If a method fails to produce an interval
in some replicates, report the failure rate and treat those replicates as non-covering in a clearly
labelled secondary column.
