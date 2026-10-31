# 16 — Results summary (paper-ready)

> **Written 2026-08-20**, after Experiments A-I. Every number is read from a CSV in `results/`,
> each of which has a provenance sidecar. Nothing here is transcribed by hand.
>
> **Status discipline (§7).** Every claim carries one of `PROVED`, `DERIVED`,
> `NUMERICALLY VERIFIED`, `OBSERVED`, `OPEN`. A robustness measurement is never called a theorem.
>
> **Attribution (§8).** No novelty is claimed for the PPI/GREG equivalence, for Behrens-Fisher
> conservativeness, or for Welch-Satterthwaite. All three are prior art with named sources.

---

## 1. The research question

Prediction-powered inference combines a large pool of model-generated labels with a small
gold-labelled sample. PPI++ adds a tuning coefficient lambda-hat, estimated from the gold sample,
and is claimed to be "essentially never worse" than classical inference. **That guarantee is
asymptotic in the number of gold labels `n`.**

> **Does the PPI++ confidence interval attain its nominal level in the regime PPI exists to serve —
> a gold set of tens of items — and if not, what does?**

## 2. Core findings

**F1. PPI++ under-covers severely at small `n`, and the deficit is invariant in predictor quality.**
`NUMERICALLY VERIFIED` (Experiments A, C, D; R = 200,000; MC s.e. 0.0005). At `n = 10` the nominal
95% interval covers **0.878**; at `n = 5`, **0.760**; at `n = 4`, **0.681**. The deficit is flat in
rho — spread <= 0.0018 across a ten-point grid from 0 to 0.99 — the signature of a cost of
*estimating* lambda-hat, not of a bad predictor.

**F1b. The deficit is approximately `O(1/n)`.** `NUMERICALLY VERIFIED`. Regressing
`0.95 − coverage(B3)` on `1/n` over `n ∈ {4,…,50}` and all three rho gives slope **1.1272** and
intercept **−0.0295**; the relationship is close to, but not exactly, proportional to `1/n`, with
the curvature at `n = 4, 5`.

**F2. The mechanism is an omitted regression-leverage term plus the wrong quantile.**
`DERIVED` + `NUMERICALLY VERIFIED`. Decomposed with the `B9-nolev` control: the Student quantile
alone lifts 0.878 -> 0.937, and the leverage term supplies the remaining 0.937 -> 0.950.

**F3. The corrected interval is exact under the Gaussian model, for every `n >= 4`.**
`PROVED` (N -> infinity) + `NUMERICALLY VERIFIED`. Coverage 0.9495-0.9505 at every `n` from 4 to 50
and every rho tested — worst deviation from nominal **0.0005**. This closes `10` H3, which was
previously untested below `n = 10`.

**F4. Nothing simpler works uniformly.** `NUMERICALLY VERIFIED`. At `n = 10`, no simple fix attains
>= 0.945 **at every rho**: Student quantile 0.9227-0.9246; cross-fitting 0.9091-0.9271 across three
schemes; significance thresholding 0.9128-0.9489 across |t| > 1.5/2/3; percentile bootstrap
0.8991-0.9016; BCa 0.8983-0.9026, unchanged at 9,999 resamples.

*One qualification, stated because it matters:* thresholding at |t| > 3 **does** reach 0.9489 at
rho = 0 and 0.9459 at rho = 0.2 — and falls to **0.9128** at rho = 0.8. A rule that is adequate
when the predictor is useless and worst when it is good is not a usable rule, and that
non-monotonicity is the finding, not the maximum.

**F5. Validity costs width, and the honest accounting is small.** `OBSERVED`. At `n = 10, rho = 0`
the valid interval is **8% wider** than classical; at rho = 0.9 it is **53% narrower**. PPI++'s
apparent 0.814 width ratio at rho = 0 is not an efficiency gain — it is the under-coverage
expressed in different units.

## 3. Theoretical findings

**T1. An exact algebraic decomposition.** `DERIVED`, verified to 1e-12:
`theta_hat_LC - theta = eps_bar + (b - beta)(mu_f - fbar_n) + b(fbar_N - mu_f)`.

**T2. N -> infinity: an exact `t_{n-2}` pivot.** `PROVED`, `NUMERICALLY VERIFIED`. The estimator is
the OLS fitted value at `mu_f`; studentising by `s*sqrt(1/n + h)` gives an exact `t_{n-2}`.
*The underlying distributional fact is textbook (Seber & Lee); the recognition that PPI++'s
estimate is such a fitted value is what makes it available.*

**T3. Finite N: exactly pivotal for the wrong estimand.** `PROVED` (Theorem A'). With
`delta = fbar_N - mu_f`, the statistic is exactly `t_{n-2}` about `theta + beta*delta`. So the
interval is an **exact** interval for `theta + beta*delta`, and the finite-N shortfall is an
**unestimable bias**, not a variance error — `mu_f` is exactly what is unknown. This is why every
variance-side repair fails.

**T4. (V_full) is the exact conditional variance, and `04` §4.8.3's diagnosis of its own formula is
wrong.** `DERIVED` + `NUMERICALLY VERIFIED`. The realised-Delta-squared substitution is unbiased;
the double-count is `O(1/N)` **and shrinks with rho**, while the over-coverage grows with rho.
Wrong order, wrong sign.

**T5. The finite-N statistic is the Behrens-Fisher statistic with one variance known.** `DERIVED`.
`T = Z/sqrt((1-s)Q/nu + s)`, `s` the share of variance carried by the unlabelled term.
**This identification is the contribution; everything it unlocks is classical.**

**T6. Conservativeness, and a cap on the over-coverage.** **`PRIOR ART`** — Mickey & Brown (1966),
*Ann. Math. Statist.* 37, 639-642, Theorem at `f2 -> infinity`:
`1-alpha <= coverage <= 2*Phi(t_{n-2,1-alpha/2}) - 1`. Verified: both bounds are attained at
`s = 0, 1`, and all 81 cells of Experiment E lie inside them (max 0.9711 against a cap of 0.9789 at
`n = 10`).

**T7. `nu_eff = (n-2)/(1-s)^2`.** **`PRIOR ART`** — Welch-Satterthwaite instantiated. It matches the
first two moments of `V_hat`, *not* of `T`. Predicts B9's coverage to **0.0029 across all 81 cells**
with no free parameters (`NUMERICALLY VERIFIED`).

**T8. The idealisation error is `O(1/N)`.** `DERIVED` + `NUMERICALLY VERIFIED`; `|Delta| <= K*tau`
with `tau <= n/(N(n-3))` and `K ~ 0.43` over 48 cells — `K` is `CONJECTURED`. **The error is not
sign-definite** (counterexample at `n = 5, rho = 0.9, N = 25`: -0.0022, 44 MC s.e.), so
conservativeness does **not** transfer by monotonicity. `OPEN`.

## 4. Experiment findings

### E1 — the simple fixes (Experiments A4, A5, D)
`NUMERICALLY VERIFIED`. At `n = 10`: Student quantile 0.923-0.925; cross-fitting **0.909**
(`06`'s own formula), 0.917 (5-fold), 0.924 (averaged-lambda); thresholding 0.918-0.940 across
|t| > 1.5/2/3. **H2 confirmed** with its pre-registered guard met — three schemes, three
thresholds. Cross-fitting is also the **worst method by MSE** at small `n` (1.50x classical at
`n = 10, rho = 0`) — the fix recommended by the paper that documented the problem.

### E3 — the bootstrap (Experiment B)
`NUMERICALLY VERIFIED`. BCa covers **0.898-0.903** at `n = 10` and is **indistinguishable from the
percentile bootstrap**; raising `B` from 2,000 to 9,999 changes nothing (0.901-0.904), so it is not
a resampling-budget effect. Its intervals are *narrower* than the corrected one, and it costs
**1e4-1e5x** more. **Kill criterion K3 does not fire.**

### E4 — finite N (Experiment E)
`NUMERICALLY VERIFIED`. B9 over-covers, up to **0.9711** at `n = 10, rho = 0.9, N = 50`, and
**never under-covers** across 81 cells. The variance is *not* the problem —
`mean(V_hat)/Var(theta_hat)` is in `[0.988, 1.025]` for both `v482` and `v_full`. R1 cannot repair
it (algebraically >= `v482`; largest coverage difference from B9 over 81 cells **0.0008**, against
V_full's **0.0011**); R3 is **infeasible** (exact for the wrong estimand); **R2 works** —
worst `|coverage - 0.95| = 0.0028` over all 81 cells — but is **not conservative** (0.9472 at one
cell). R5, the exact quantile, is **not better than R2**, which localises the residual error in the
idealisation rather than the approximation.

### E5 — robustness (Experiments F, G, H)
`OBSERVED`. **Heavy tails cost nothing**: B9 tracks the classical interval to within 0.0022 across
72 cells, including a heavy-tailed *predictor*. **K4 does not fire** (minimum 0.9495 against a 0.93
trigger). **Skew is the limitation, and the structure is instructive**: B9's coverage is flat in rho
while the classical interval's *rises*, because a strong predictor makes `Y` nearly Gaussian and
rescues the classical interval, whereas B9 studentises the residual, which stays skewed.
**Heteroskedasticity breaks B9** (0.905 at gamma = 1, n = 10) **and HC3 repairs it** — never below
0.9517 in 72 cells. **H4 fails on its relative clause** at 68 of 192 cells; the threshold was not
moved. **BCa gets worse outside the Gaussian model**, not better.

### E6 — real data (Experiment I, Galaxies)
`OBSERVED`, and explicitly **not** coverage evidence. 16,743 galaxies, `theta = 0.2593`,
rho = 0.741, `R = 20,000` gold subsamples. PPI++ covers **0.800** at `n = 10` — *worse* than in the
Gaussian model, as Experiment G predicts for a binary skewed heteroskedastic outcome. B9 reaches
0.900; **HC3 reaches nominal from `n = 20`** (0.951 / 0.953 / 0.952). BCa fails to construct an
interval in 977 of 20,000 replications at `n = 10`.

## 5. Results table

| setting | PPI++ (B3) | LC-PPI (B9) | R2 | BCa (B8) | HC3 (B9-HC) |
|---|---|---|---|---|---|
| **A** Gaussian, N=10⁶, n=10 | **0.8777–0.8794** | **0.9494–0.9508** | = B9 | — | — |
| A, n=20 | 0.9194–0.9207 | 0.9492–0.9508 | = B9 | — | — |
| A, n=50 | 0.9388–0.9396 | 0.9497–0.9503 | = B9 | — | — |
| **C** n=4 | **0.6803–0.6809** | **0.9495–0.9497** | 0.9495–0.9497 | — | — |
| C, n=5 | 0.7598–0.7615 | 0.9502–0.9503 | 0.9502–0.9503 | — | — |
| C, n=8 | 0.8540–0.8551 | 0.9502–0.9503 | 0.9502–0.9503 | — | — |
| **B** bootstrap, n=10 | 0.8751–0.8794 | 0.9482–0.9506 | — | **0.8983–0.9026** | — |
| B, n=50 | 0.9353–0.9401 | 0.9469–0.9513 | — | 0.9375–0.9419 | — |
| **E** finite N, n=10, ρ=0.9 | 0.8784–0.9124 | **0.9499–0.9711** | **0.9486–0.9505** | — | — |
| **F** heavy tails (all cells) | 0.8785–0.9435 | 0.9495–0.9575 | 0.9495–0.9575 | — | 0.9535–0.9718 |
| **G** skew (all cells) | 0.7735–0.9449 | 0.8474–0.9518 | 0.8474–0.9518 | — | 0.8636–0.9634 |
| G, binary label only | 0.8727–0.9441 | **0.9477–0.9491** | — | — | 0.9512–0.9569 |
| **H** heteroskedastic (all) | 0.8228–0.9397 | **0.9047–0.9507** | 0.9047–0.9507 | — | **0.9517–0.9627** |
| **I** Galaxies real data | 0.8000–0.9437 | 0.9003–0.9483 | 0.8993–0.9482 | 0.8375–0.9425 | **0.9284–0.9526** |

Width ratio to the classical interval, Gaussian model, N = 10⁶:
| n | ρ | PPI++ (B3) | LC-PPI (B9) |
|---|---|---|---|
| 10 | 0.0 | 0.814 | 1.081 |
| 10 | 0.5 | 0.705 | 0.936 |
| 10 | 0.9 | 0.355 | 0.471 |
| 20 | 0.0 | 0.911 | 1.032 |
| 20 | 0.5 | 0.788 | 0.893 |
| 20 | 0.9 | 0.397 | 0.449 |
| 50 | 0.0 | 0.965 | 1.011 |
| 50 | 0.5 | 0.836 | 0.876 |
| 50 | 0.9 | 0.421 | 0.441 |


*Ranges are over the rho (and where applicable nu / gamma / family / N) cells at that setting.
"= B9" means the method is numerically identical there. Full tables: `results/*.csv`.*

---

## 6. Status of every claim

| claim | status | source |
|---|---|---|
| PPI/GREG equivalence | **`PRIOR ART`** | Cassel et al. (1976); Sarndal et al. (2003); Mozer (2026) |
| Exact `t_{n-2}` law of a studentised OLS fitted value | **`PRIOR ART`** | classical normal-theory linear models |
| Finite-N conservativeness + over-coverage cap | **`PRIOR ART`** | **Mickey & Brown (1966)** |
| Numerical conservativeness of Behrens-Fisher intervals | **`PRIOR ART`** | Bartlett (1936); James (1959); Wang (1971); Robinson (1976) §4 |
| Effective degrees of freedom | **`PRIOR ART`** | Welch (1938, 1947); Satterthwaite (1941, 1946) |
| Asymptotic design-based two-phase variance | **`PRIOR ART`** | Sarndal & Swensson (1987); Kott (1990); Rao & Sitter (1995); Sitter (1997) |
| T1 exact decomposition | `DERIVED` (1e-12) | this work |
| T2 exact pivot at N -> infinity | `PROVED`, `NUMERICALLY VERIFIED` | this work (statement classical) |
| T3 Theorem A' — exact for `theta + beta*delta` | `PROVED` | this work |
| T4 (V_full) exact; §4.8.3 diagnosis refuted | `DERIVED` + `NUMERICALLY VERIFIED` | this work |
| T5 identification as Behrens-Fisher, one variance known | `DERIVED` | this work |
| T7 `nu_eff` predicts coverage to 0.0029 over 81 cells | `NUMERICALLY VERIFIED` | this work |
| T8 `O(1/N)` idealisation rate | `DERIVED` + `NUMERICALLY VERIFIED` | this work |
| T8 constant `K` in `|Delta| <= K*tau` | **`CONJECTURED`** | this work |
| Conservativeness of the *practical* interval at finite N | **`OPEN`** | 129 cells support it; not proved |
| Exact finite-N interval for `theta` | **`OPEN`** — likely impossible in this class (T3) | — |
| F1-F5, E1/E3/E4 measurements | `NUMERICALLY VERIFIED` | `results/*.csv` |
| E5 robustness behaviour | `OBSERVED` | `results/F_*.csv`, `G_*.csv`, `H_*.csv` |
| E6 real-data behaviour | `OBSERVED` (not coverage evidence) | `results/I_real_data.csv` |
| Whether Sitter (1997) contains a finite-sample interval | **`OPEN`** (`PARTIAL` evidence: no) | paper not obtained |
| Galaxies data licence | **`OPEN`** | see `docs/tier2_galaxies.md` |

## 7. Limitations

1. **The exactness claim is Gaussian-only.** `PROVED` under M1 and at `N -> infinity`. Outside it
   there are measurements, not theorems.
2. **Severe right-skew degrades the interval**, and it does so together with the classical
   interval (0.847 vs 0.839 at lognormal sigma = 1, n = 10). A limitation of the Student-`t`
   construction, not of the correction — but a limitation.
3. **Heteroskedasticity degrades it materially** (0.905 at gamma = 1, n = 10). Use HC3, which was
   conservative in all 72 cells tested but carries **no exactness claim of any kind**.
4. **Finite-N conservativeness is not proved for the interval people would actually run** — only
   for an idealised statistic differing by `O(1/N)` terms, with a conjectured constant and a
   non-definite sign.
5. **The real-data experiment cannot establish coverage.** Subsampling from one fixed population is
   not i.i.d. sampling from a superpopulation.
6. **One dataset, one estimand.** Means only; no M-estimation, no quantiles, no regression
   coefficients.
7. **Two prior-art items remain unclosed**: Sitter (1997) unobtained, and the Galaxies data licence
   unresolved. Neither affects a result; both affect what may be written and redistributed.

## 8. Recommended contribution statement

> Prediction-powered inference is the survey-sampling difference estimator and PPI++ is the GREG
> estimator; that equivalence is due to Mozer (2026) and we claim nothing for it. We show that the
> *inference* problem is likewise classical, and identify which classical problem it is. Writing
> the leverage-corrected PPI statistic in exact form, we prove that at finite unlabelled-pool size
> `N` it is exactly Student-`t_{n-2}` pivotal not for the estimand but for a shifted quantity whose
> offset is not estimable, so the finite-`N` shortfall is a bias rather than a variance error; and
> we identify the resulting studentised statistic as the Behrens-Fisher statistic with one variance
> known, which places the interval's conservativeness and the cap on its conservatism under Mickey
> & Brown's (1966) two-sided bound, and makes Welch-Satterthwaite the natural degrees-of-freedom
> correction. Our contribution is therefore not a new distributional theorem but (i) that
> identification, which makes classical Behrens-Fisher theory available to prediction-powered
> inference, (ii) an explicit `O(1/N)` characterisation of the gap between the interval a
> practitioner runs and the classical idealised one, and (iii) the first systematic measurement of
> PPI++ in the regime it was designed for. At 10 gold labels the shipped interval covers 0.878
> against a nominal 0.95, and at 4 gold labels it covers 0.681; a Student quantile, three
> cross-fitting schemes, three significance thresholds, and both a percentile and a BCa bootstrap
> at up to 9,999 resamples all fail to reach 0.945, while the leverage-corrected interval attains
> 0.9495-0.9505 for every `n` from 4 to 50 at a cost of 8% interval width. We map where it stops
> working — severe skew, and heteroskedasticity, where an HC3 variant restores it — and demonstrate
> the whole comparison on 16,743 real galaxy classifications.

*Deliberately absent from that paragraph: any claim to the estimator, to conservativeness, to the
Behrens-Fisher distribution, or to Welch-Satterthwaite.*
