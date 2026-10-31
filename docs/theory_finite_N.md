# Finite-N theory for LC-PPI

> ## ATTRIBUTION — read before citing anything in this document
>
> Three things in here are **prior art and are not claimed**:
> * the **estimator** — PPI is the difference estimator of Cassel et al. (1976), PPI++ is the GREG
>   estimator of Särndal et al. (2003); recognised for PPI by **Mozer (2026)**, arXiv:2603.19160;
> * **Theorem C (conservativeness)** — this is **Mickey & Brown (1966)**, *Ann. Math. Statist.*
>   37(3), 639–642, at `f₂ → ∞`. **Verified against the paper itself** (E6b): their Theorem is
>   stronger than ours in both directions — see §V.1. Theorem B's distributional form is the
>   Behrens–Fisher statistic with one variance known, equally classical;
> * the **effective degrees of freedom** `ν_eff = (n−2)/(1−s)²` — Welch–Satterthwaite instantiated
>   (Satterthwaite 1941, 1946; Welch 1938, 1947).
>
> "Theorem B" and "Theorem C" are **labels for results restated here**, not claims of priority. See
> Part III for the full matrix and §III.5 for what is actually new.

> **Written 2026-08-20, after Experiment E (E4).** This document derives the finite-N behaviour of
> the leverage-corrected interval. It is written to the status discipline of `04` §4.12: every
> statement carries a tag, and nothing is tagged `PROVED` unless a referee could check the argument
> written here. `04` remains the mathematical source of truth; where this document derives
> something new, `04` §4.8.3–4.8.4 has been updated to point here.
>
> **Every claim below is accompanied by an executable check** in `tests/test_finite_n.py`.

---

## 1. The model  (assumptions M1 of `04` §4.10, restated, nothing added)

* `f_1..f_n ~ iid N(μ_f, σ_f²)`, and `Y_i = α + β f_i + ε_i` with `ε_i ~ iid N(0, σ²)`
  independent of `f`. Equivalently `(Y_i, f_i)` iid bivariate normal (A1), `σ_f² > 0` (A3),
  homoskedastic (A7).
* `f^u_1..f^u_N ~ iid N(μ_f, σ_f²)`, **independent of the labelled sample** (A2).
* `θ = E[Y] = α + β μ_f` (A6). `n ≥ 4` (A4). `f` is a fixed function (A5).
* Statistics: `f̄_n`, `Ȳ`, `S_ff = Σ(f_i − f̄_n)²`, `S_fY`, `b = S_fY/S_ff`,
  `s² = RSS/(n−2)`, `f̄_N = mean of the pool`, and
  `θ̂_LC = Ȳ + b(f̄_N − f̄_n)`.
* Two derived quantities used throughout:
  `δ = f̄_N − μ_f ~ N(0, σ_f²/N)` and `Δ = f̄_N − f̄_n = δ + (μ_f − f̄_n)`.

Nothing here is stronger than M1. `σ̂_f²` is treated as known below; E4 measured the difference
between estimating it and knowing it as undetectable (largest coverage difference 0.0047 ≈ 2.2
combined Monte-Carlo standard errors), so this is a simplification of the *analysis*, not of the
model, and it is flagged where it is used.

---

## 2. The exact decomposition   `DERIVED`, verified to 1e-12

```
θ̂_LC − θ  =  ε̄  +  (b − β)(μ_f − f̄_n)  +  b·(f̄_N − μ_f)                            (D)
```

*Proof.* `θ̂_LC − θ = (Ȳ − α − β f̄_n) + (b − β)(f̄_N − f̄_n) + β(f̄_N − μ_f)` by adding and
subtracting `β f̄_n` and `βf̄_N`; the first bracket is `ε̄`. Writing `f̄_N − f̄_n = (μ_f − f̄_n) + δ`
and collecting the two `δ` terms, `(b−β)δ + βδ = bδ`. ∎

**Why (D) rather than `04` §4.8.3's form.** §4.8.3 writes the middle term as `(b − β)Δ`, whose
multiplier `Δ` is *random and correlated with `δ`*. In (D) the middle multiplier `μ_f − f̄_n` is
fixed given the labelled design, and every bit of unlabelled randomness sits in the single term
`b·δ`. That regrouping is what makes the rest of this document possible.

**Independence structure, conditional on the labelled design `F = (f_1..f_n)`:**

| quantity | law given `F` | independent of |
|---|---|---|
| `ε̄` | `N(0, σ²/n)` | `b − β`, `s²`, `δ` |
| `b − β` | `N(0, σ²/S_ff)` | `ε̄`, `s²`, `δ` |
| `(n−2)s²/σ²` | `χ²_{n−2}` | `ε̄`, `b`, `δ` |
| `δ` | `N(0, σ_f²/N)` | everything labelled (A2) |

`ε̄ ⟂ (b−β)` because `Cov(ε̄, b−β | F) = (σ²/S_ff)·Σ(f_i − f̄_n)/n = 0` and both are jointly normal;
`s² ⟂ (ε̄, b)` is the standard normal-theory orthogonality of residuals and fitted coefficients.

---

## 3. The exact conditional variance   `DERIVED`  — and (V_full) is correct

Taking variances in (D) given `F`, the three terms being uncorrelated and `E[b] = β`:

```
Var(θ̂_LC | F)  =  σ²/n  +  σ²(μ_f − f̄_n)²/S_ff  +  E[b²]·σ_f²/N
               =  σ²/n  +  σ²(μ_f − f̄_n)²/S_ff  +  (β² + σ²/S_ff)·σ_f²/N              (V*)
```

**(V*) is algebraically identical to (V_full) of `04` §4.8.3** (checked to 1e-15). So §4.8.3's
derivation is right, and its `σ²σ_f²/(N·S_ff)` term is not an artefact: it is the cost of
multiplying an *estimated* slope by an *estimated* gap.

Two consequences, both of which contradict the diagnosis §4.8.3 attaches to its own formula:

* **The realised-`Δ²` substitution is unbiased.** `E[Δ² | F] = (μ_f − f̄_n)² + σ_f²/N` exactly, so
  the §4.8.2 leverage term has expectation equal to (V*)'s.
* **The double count is real but cannot be the cause of the over-coverage.** Using `b²` where
  (V*) has `β²` inflates the variance by exactly `σ²σ_f²/(N·S_ff)` — `O(1/N)` relative, under 1 %
  at `N = 200, n = 10` — and it carries a factor `σ² = 1 − ρ²`, so it *shrinks* as ρ grows while
  the observed over-coverage *grows* sharply with ρ. Wrong size, wrong sign in ρ.

Measured (E4, 81 cells, `R = 200,000`): `mean(V̂)/Var(θ̂)` ∈ `[0.988, 1.025]` for both `v_full` and
`v482`. **The variance expression was never the problem.**

---

## 4. Theorem A′ — the exact conditional pivot   `PROVED`   (Proposition 1 of the manuscript)

> **Theorem A′.** Under M1, for every `n ≥ 4` and every finite `N`,
> ```
> G  =  ( θ̂_LC − θ − βδ ) / ( s·√(1/n + Δ²/S_ff) )   ~   t_{n−2}
> ```
> **exactly**, conditionally on `(F, f̄_N)` and therefore marginally; and `G` is independent of
> `f̄_N`.

*Proof.* By (D), `θ̂_LC − θ − βδ = ε̄ + (b − β)Δ`. Condition on `(F, δ)`; then `Δ` is fixed, and by
§2 the sum is `N(0, σ²(1/n + Δ²/S_ff))`, while `(n−2)s²/σ² ~ χ²_{n−2}` independently of it. Their
studentised ratio is `Z/√(χ²_{n−2}/(n−2)) ~ t_{n−2}`, whose law does not depend on the conditioning
values; hence `G ⟂ (F, f̄_N)` and the marginal law is the same. ∎

**This is the only exact distributional statement available at finite N, and it is exact for the
estimator as implemented — no idealisation.** Its content is precisely the negative result: the
§4.8.2 interval is an *exact* `1−α` interval for `θ + βδ`, not for `θ`. The gap between those two
estimands is the whole finite-N problem, and `βδ` is not estimable because `μ_f` is unknown. This
is why `04` §4.8.4's R3 ("correct the conditional bias") is not under-developed but **blocked**;
E4 measured its consequence, 0.6225 coverage at `n = 50, ρ = 0.9, N = 50`.

*Check:* `test_conditional_pivot_is_exactly_t_n_minus_2` (KS against `t_{n−2}` marginally and in
both tails of `f̄_N`, plus a zero-correlation check).

---

## 5. The exact marginal law is not a standard family   `DERIVED`

Write, given `F`, `ψ = σ/√S_ff`, `κ = ψσ_f/(σ√N)`, `u = (μ_f − f̄_n)/√S_ff`, and let
`Z_0, Z_1, W ~ N(0,1)` and `Q ~ χ²_{n−2}` be independent. Then (D) becomes

```
θ̂_LC − θ  =  σ[ Z_0/√n + (u + κW)·Z_1 ]  +  (βσ_f/√N)·W ,      s² = σ²Q/(n−2)          (E)
```

and the studentised statistic `T = (θ̂_LC − θ)/√V̂` with
`V̂ = s²(1/n + Δ²/S_ff) + b²σ_f²/N` is a ratio in `(Z_0, Z_1, W, Q)` in which

* the numerator contains the **product** `Z_1·W` of two independent normals, so it is not normal;
* `V̂` is **not independent of the numerator** — both involve `Z_1` (through `b`) and `W`
  (through `Δ`).

Marginalising over `F` adds two more variables (`u` is a scaled `t_{n−1}`, `S_ff` a `χ²_{n−1}`).

> **The exact law of `T` therefore does not reduce to `t`, to a scaled `t`, to a variance-gamma, or
> to any other standard family.** It is a five-fold integral. `04` §4.8.3's suspicion of a
> product-normal component is confirmed by the `κ Z_1 W` term in (E). No closed form is claimed.

---

## 6. Theorem B — the exact law under an explicit idealisation   `PROVED (for the idealised statistic)`

Two `O(1/N)` simplifications make the problem exactly solvable. **State them, because the theorems
below are about the idealised statistic and not about the implemented one.**

* **(I1)** drop the product term, i.e. set `κ = 0` in (E). Equivalently, ignore
  `σ²σ_f²/(N·S_ff)` — the term quantified in §3 as under 1 % at `N = 200, n = 10`.
* **(I2)** replace the realised `Δ²` by `E[Δ² | F]` and `b²` by `β²` in `V̂`, which makes `V̂`
  independent of the numerator.

Under (I1)+(I2), write `V = V_1 + V_2` with `V_1 = σ²(1/n + h)` (`h` the leverage) and
`V_2 = β²σ_f²/N`, and let

```
s  =  V_2 / (V_1 + V_2)                                                                 (S)
```

be **the share of the total variance carried by the component whose estimate carries no
estimation error**. Then the numerator is exactly `N(0, V)`, `V̂ = V_1·Q/(n−2) + V_2` with
`Q ~ χ²_{n−2}` independent of it, and

> **Theorem B.** `T = Z / √( (1−s)·Q/ν + s )` with `ν = n−2`, `Z ~ N(0,1)`, `Q ~ χ²_ν`, `Z ⟂ Q`.
> Equivalently `T` is a **normal variance mixture** with mixing variable `(1−s)Q/ν + s`. Its exact
> coverage is the one-dimensional integral
> ```
> P(|T| ≤ q)  =  E_Q[ 2Φ( q·√((1−s)Q/ν + s) ) − 1 ]                                    (C)
> ```

*Proof.* `T = √V·Z / √(V_1 Q/ν + V_2)`; divide numerator and denominator by `√V`. ∎

**Identification.** At `s = 0` this is exactly `t_ν`; at `s = 1` it is exactly `N(0,1)`. In between
it is the law of `√(1−s)·t_ν + √s·Z′` for an independent `Z′` — i.e. the **Behrens–Fisher
distribution with one degree of freedom infinite**. The finite-N PPI problem is, under (I1)+(I2), a
Behrens–Fisher problem: a linear combination of an estimated-variance component and a
known-variance component.

*Check:* `test_idealised_law_and_its_exact_coverage_integral` (integral vs Monte Carlo, < 0.003).

---

## 7. Moments, and what Satterthwaite actually matches   `DERIVED`

From Theorem B, for `ν > 2` and `ν > 4` respectively,

```
E[T]    = 0
E[T²]   = E_Q[ ((1−s)Q/ν + s)^{-1} ]
E[T⁴]   = 3·E_Q[ ((1−s)Q/ν + s)^{-2} ]
kurt(T) = E[T⁴]/E[T²]²
```

At `ν = 8`, `s = 0.45`: `Var(T) = 1.0716`, `kurt(T) = 3.1995`. The Satterthwaite surrogate
`t_{ν_eff}` with `ν_eff = 26.45` has `Var = 1.0818` and `kurt = 3.2673`.

> **Satterthwaite matches neither the variance nor the kurtosis of `T`.** What it matches is the
> **first two moments of the variance estimator `V̂`**: `E[V̂] = V` by construction, and equating
> `Var(V̂) = 2V_1²/ν` to the variance `2V²/ν_eff` of a scaled `χ²_{ν_eff}` gives
> ```
> ν_eff  =  V² / (V_1²/ν)  =  ν / (1 − s)²                                             (N)
> ```
> **This is where `ν_eff = (n−2)/(1−s)²` comes from.** It is the textbook Welch–Satterthwaite
> formula for a two-component variance with `ν_1 = n−2` and `ν_2 = ∞`, instantiated — not a fitted
> constant, and not a property of `T`. Having matched `V̂`, the step to `T ≈ t_{ν_eff}`
> *additionally* assumes the numerator is normal and independent of `V̂`, which (I1) and (I2) are
> exactly what buy.

`s` in (N) is the population quantity (S). The implementation uses the plug-in
`ŝ = (b²σ̂_f²/N)/V̂`, which is what `E4`'s `unlabeled_share_mean` metric reports.

---

## 8. Theorem C — conservativeness   `PROVED (for the idealised statistic)`

> **Theorem C.** In the setting of Theorem B, the coverage of the `t_{n−2}` interval,
> `C(s) = P(|T| ≤ t_{ν,1−α/2})`, is **non-decreasing in `s` on `[0,1]`**, with `C(0) = 1−α`.
> Hence `C(s) ≥ 1−α` for all `s`: **the §4.8.2 interval never under-covers.**

*Proof.* Put `x = Q/ν` (so `E[x] = 1`), `g(x) = (1−s)x + s`, `q = t_{ν,1−α/2}`. From (C),
```
C(s) = E[ 2Φ(q√g(x)) − 1 ] ,     dC/ds = q · E[ (1 − x) · ψ(g(x)) ],
ψ(g) = φ(q√g)/√g = (2π)^{-1/2} e^{-q²g/2} / √g .
```
`ψ` is strictly decreasing on `g > 0`, and `g` is increasing in `x`, so `ψ∘g` is decreasing in `x`;
`1 − x` is also decreasing in `x`. Two decreasing functions of the same random variable are
non-negatively correlated (Chebyshev's association inequality), so
`E[(1−x)ψ(g(x))] ≥ E[1−x]·E[ψ(g(x))] = 0`. Hence `dC/ds ≥ 0`. At `s = 0`, `T ~ t_ν` exactly and
`C(0) = 1−α`. ∎

This is the proof of R4 (`04` §4.8.4) **for the idealised statistic**. The implemented statistic
differs by the `O(1/N)` terms (I1)+(I2) drop; over the 81 cells of Experiment E it was at or above
nominal in every one, minimum 0.9488 at the `N → ∞` end where the theory says exactly 0.95.

*Check:* `test_coverage_is_monotone_increasing_in_the_unlabelled_share`.

---

## 9. R2, and the exact-quantile rule R5   `NUMERICALLY VERIFIED`

Theorem B makes the exact quantile computable: solve (C) for `q` at level `1−α`. Because
`q ∈ [z_{1−α/2}, t_{ν,1−α/2}]` it can be reported as an effective `t` degrees of freedom `ν* ≥ ν`.
That is **R5**, implemented as `var_lc_ppi(mode="r5")` and registered as `B9-R5`. R5 removes the
Satterthwaite approximation error entirely; R2 does not.

**Measured over the 81 cells of Experiment E:**

| | worst \|coverage − 0.95\| | RMSE | cells > 3 MC s.e. below nominal | mean width ÷ B9 |
|---|---|---|---|---|
| B9 (`v482`) | 0.0211 | 0.0043 | 0 | 1.0000 |
| B9-R1 | 0.0213 | 0.0044 | 0 | 1.0004 |
| **B9-R2** (Satterthwaite) | **0.0028** | **0.0008** | 1 | 0.9917 |
| **B9-R5** (exact quantile) | 0.0028 | 0.0009 | **5** | 0.9911 |

> **R5 is not better than R2, and that is the informative result.** R5 is exact *for the idealised
> statistic*. Since removing the quantile approximation does not improve coverage, the residual
> ≈ 0.002–0.003 deviation is **not** quantile-approximation error: it is the cost of the
> idealisations (I1)+(I2) themselves — the dependence between `V̂` and the numerator, and the
> product term. R5 sits systematically *below* R2 (e.g. 0.9490 vs 0.9498 at `n=10, ρ=0.9, N=50`),
> so the idealised law slightly understates the true spread and Satterthwaite's error happens to
> compensate. **R2's accuracy is therefore partly fortuitous**, and it should not be presented as
> the more principled rule; R5 is the more principled rule and is marginally worse.

---

## 10. Status table

| Claim | Status |
|---|---|
| LC-PPI variance formula `s²(1/n+h) + b²σ̂_f²/N` (`04` §4.8.2) | `IMPLEMENTED AND MEASURED`; unbiased for (V*) to within Monte-Carlo error over 81 cells |
| Exact decomposition (D) | `PROVED` (§2), verified to 1e-12 |
| (V_full) = exact conditional variance (V*) | `PROVED` (§3), verified to 1e-15 algebraically and `[0.988, 1.025]` numerically |
| `04` §4.8.3's double-counting diagnosis as the *cause* | **`REFUTED`** (§3): wrong order in `N`, wrong sign in ρ |
| Exact marginal distribution of `T` | `DERIVED` as a five-fold integral (§5); **no standard family**, `OPEN` in closed form |
| Theorem A′ — exact conditional `t_{n−2}` pivot for `θ + βδ` | `PROVED` (§4), exact for the implemented estimator |
| Theorem B — idealised law and exact coverage integral | `PROVED` for the idealised statistic (§6) |
| Effective-df formula `ν_eff = (n−2)/(1−s)²` | `DERIVED` (§7) as Welch–Satterthwaite instantiated; **not** a moment match for `T` |
| Coverage prediction from `ν_eff` | `NUMERICALLY VERIFIED`: 0.0029 max error over 81 cells, no free parameters |
| Theorem C — conservativeness of §4.8.2 | `PROVED` for the idealised statistic (§8); `OPEN` for the implemented one |
| R2 validity | **moment-matched approximation**; not exact, not conservative (0.9472 at `n=50, ρ=0.9, N=50`) |
| R5 validity | exact under (I1)+(I2); **not** better than R2 in practice (§9) |
| Finite-N behaviour of `v482` | `NUMERICALLY VERIFIED` conservative over 81 cells; `PROVED` only under (I1)+(I2) |

---

## 11. What is left to prove

1. **Remove (I1)+(I2) from Theorem C.** The obstruction is the dependence of `V̂` on the numerator
   through `b` and `Δ`. A coupling or conditioning argument may work; nothing here attempts it.
2. **A closed form, or a sharp bound, for the exact law of §5.** The product term `Z_1 W` puts it
   in normal-product / generalized-hyperbolic territory; a bound on the total-variation distance to
   the Theorem B law would convert Theorem C into a statement about the implemented interval with
   an explicit `O(1/N)` remainder. **This is the single most valuable open item.**
3. **An exactness result, if one exists at all.** Theorem A′ says the obstruction is the unestimable
   bias `βδ`, so an exact interval for `θ` would need either `μ_f` or a fundamentally different
   construction. A negative result — that no exact interval exists in this class — would be just as
   publishable.


---
---

# PART II — Bounding the idealisation error   (E4c, 2026-08-20)

Part I proved things about an *idealised* statistic. Part II asks how far the statistic a
practitioner actually computes is from it. Every claim carries a status tag; the numbers come from
`ppi_finite.theory.ablation_coverages`, which evaluates all four statistics on **common random
numbers** so their differences are attributable to the idealisation removed and not to noise.

## II.1 The practical finite-N statistic   `DEFINITION`

Conditional on the labelled design `F`, with `u = (μ_f − f̄_n)/√S_ff`, `κ = σ_f/(√N·√S_ff)`,
`σ² = Var(ε)`, and `Z₀, Z₁, W ~ N(0,1)`, `Q ~ χ²_{n−2}` mutually independent:

```
X_act  =  σ[ Z₀/√n + (u + κW)·Z₁ ]  +  (βσ_f/√N)·W                        numerator
V̂_act  =  s²( 1/n + (u + κW)² )  +  b²·σ̂_f²/N ,   s² = σ²Q/(n−2),  b = β + σZ₁/√S_ff
T_act  =  X_act / √V̂_act                                                   what B9 computes
```

`Δ/√S_ff = u + κW` and `δ = σ_f W/√N`, so nothing is hidden: `W` is the unlabelled draw and it
appears in **both** numerator and denominator.

## II.2 The idealised statistic   `DEFINITION`

```
X_id  =  σ[ Z₀/√n + u·Z₁ ]  +  (βσ_f/√N)·W                                 exactly N(0, v₁+v₂)
V̂_id  =  s²( 1/n + u² )  +  β²σ_f²/N                                       ⟂ X_id
T_id  =  X_id / √V̂_id   =   Z / √( (1−s)Q/ν + s )                          Theorem B
```
with `v₁ = σ²(1/n + h)`, `h = u²`, `v₂ = β²σ_f²/N`, `s = v₂/(v₁+v₂)`, `ν = n−2`.

**Exactly three terms differ**, and each is one idealisation:

| # | present in `T_act` | absent from `T_id` | what it is |
|---|---|---|---|
| P | `σκW·Z₁` in the numerator | — | the **product term**: estimated slope × estimated gap |
| D | `(u+κW)²` in `V̂` | `u²` | the denominator's dependence on `W`, which is also in the numerator |
| B | `b²σ̂_f²/N` in `V̂` | `β²σ_f²/N` | the denominator's dependence on `Z₁`, which is also in the numerator |

## II.3 Every idealisation, with its order in `N`   `DERIVED`

Write `v₃ = σ²σ_f²/(N·S_ff)` — the variance of the product term — and `V = v₁+v₂+v₃` (the exact
conditional variance (V\*) of Part I §3).

| idealisation | random quantity removed | conditional mean | conditional s.d. | order | sign known? |
|---|---|---|---|---|---|
| **P** | `σκW Z₁` | `0` | `√v₃` | `v₃/V = O(1/N)` in variance | removes positive variance ⇒ `X_id` less spread |
| **D** | `s²(2uκW + κ²W²)` | `s²κ² = O(1/N)` | `2s²\|u\|κ = O(N^{−1/2})` | mean `O(1/N)`, fluctuation `O(N^{−1/2})` | mean **positive** |
| **B** | `(2βσZ₁/√S_ff + σ²Z₁²/S_ff)·σ̂_f²/N` | `v₃ = O(1/N)` | `2\|β\|σσ_f²/(√S_ff N) = O(1/N)` | `O(1/N)` | mean **positive** |
| **σ̂_f²** | `(σ̂_f² − σ_f²)·(b²/N)` | `0` | `O(N^{−3/2})` | negligible beside `v₃` | — |

**Why the `O(N^{−1/2})` fluctuation in D does not make the error `O(N^{−1/2})`.** Coverage is a
smooth functional of the perturbations, and the leading fluctuation `2uκW` is **odd in `W`**, so
its first-order contribution has zero conditional mean. Expanding the coverage functional to second
order, the surviving contributions are `E[perturbation] = O(1/N)` and
`E[perturbation²] = 4u²κ² + O(κ⁴) = O(1/N)`. Hence

> **Proposition II.1 (rate).** `Δ(n, ρ, N) := coverage_act − coverage_id = O(1/N)` as `N → ∞`,
> for fixed `n` and `ρ`, with remainder `O(N^{−3/2})`.
> Status: **`DERIVED`** (a formal second-order expansion; the remainder is not rigorously
> controlled) + **`NUMERICALLY VERIFIED`** (§II.7).

## II.4 The product term `b·(f̄_N − μ_f)`   `DERIVED`

Its contribution is the single term `σκW Z₁`, a **product of two independent centred normals** —
i.e. a variance-gamma variate with variance `v₃`. Two exact facts:

* `E[X_act³ | F] = 6·(βσ_f/√N)·(σu)·(σκ)`, which is **non-zero**: the product term makes the exact
  numerator *skewed given the design*. But `u` is symmetric about 0 over draws of `F`, so the
  marginal skewness vanishes. Coverage, a marginal quantity, is therefore hit only at second order.
* Its relative size is `v₃/V ≤ v₃/v₁ = σ_f²/(N S_ff(1/n + h)) ≤ n σ_f²/(N S_ff)`, whence
  `τ := E[v₃/V] ≤ n/(N(n−3))` for `n ≥ 4` (using `E[σ_f²/S_ff] = 1/(n−3)`).

> The product term contributes at rate **`C/N`**, not `C/√N`. Status: `DERIVED`.

## II.5 The variance-estimation term   `DERIVED`, three effects separated

`04` §4.8.3 treated "the variance estimate is wrong" as one thing. It is three, with different
orders and different signs:

1. **Bias.** `E[V̂_act | F] − V = v₃` exactly: using `b²` where the exact variance has `β²`
   over-states by `σ²σ_f²/(N S_ff)`. `O(1/N)`, positive, and `∝ σ² = 1−ρ²`, so it **shrinks** with
   ρ. (This is why it cannot explain an over-coverage that grows with ρ — Part I §3.)
2. **Fluctuation.** `V̂_act` varies about its mean through `Q` (`O(1)` — this is the whole point of
   the `t` quantile), through `W` (`O(N^{−1/2})` relative) and through `Z₁` (`O(1/N)` relative).
3. **Dependence with the numerator.** `V̂_act` shares `W` and `Z₁` with `X_act`; `V̂_id` shares
   nothing with `X_id`. This is the effect that cannot be removed by any change to the *formula* —
   it is structural — and it is the one the idealisation buys away.

Measured attribution at `n = 10, ρ = 0.9, N = 25`, `R = 4·10⁶`, common random numbers:

| statistic | coverage | step | effect |
|---|---|---|---|
| `A0` practical | 0.97703 | — | — |
| `A1` drop product term (P) | 0.97311 | A0→A1 | **−0.0039** |
| `A2` also remove `W`-dependence (D) | 0.97221 | A1→A2 | −0.0009 |
| `A3` also `b²→β²` (B) = idealised | 0.97522 | A2→A3 | **+0.0030** |
| | | **net** | **+0.0018** |

> **The three effects partly cancel.** A bound built by summing their absolute sizes would be
> ≈ 0.0078 here, four times the truth. Any useful bound must exploit the cancellation, which is
> why a term-by-term triangle inequality is not the right tool.

## II.6 The unlabelled-pool variance `Var(f̄_N) = σ_f²/N`   `DERIVED`

It enters in four distinct places, which is the reason the finite-N problem is not a single
correction:

* **numerator, linearly:** `(βσ_f/√N)W` — the unestimable bias `βδ` of Theorem A′; variance `v₂`;
* **numerator, multiplicatively:** `σκW Z₁` — the product term; variance `v₃`;
* **denominator, through the leverage:** `(u + κW)²` replaces `u²`, adding `κ²` in mean and
  `O(N^{−1/2})` in fluctuation;
* **denominator, additively:** `b²σ̂_f²/N`, whose mean exceeds `β²σ_f²/N` by exactly `v₃`.

On coverage: the first raises the target from `θ` to `θ + βδ` (Theorem A′) and is what makes `s > 0`
— the *whole* first-order effect, quantified exactly by Theorem B; the other three are the
`O(1/N)` idealisation error bounded here.

## II.7 Numerical validation   `NUMERICALLY VERIFIED`

`R = 3·10⁶` per cell, paired ablations. `τ = E[v₃/V]`. Full 48-cell table in `13`; extract:

| n | N | ρ | actual | idealised | \|Δ\| | τ | \|Δ\|/τ |
|---|---|---|---:|---:|---:|---:|---:|
| 5 | 25 | 0.00 | 0.96512 | 0.94994 | 0.01518 | 0.05888 | 0.258 |
| 5 | 25 | 0.90 | 0.98965 | 0.99194 | 0.00229 | 0.03702 | 0.062 |
| 10 | 25 | 0.00 | 0.95787 | 0.95002 | 0.00785 | 0.04701 | 0.167 |
| 10 | 100 | 0.90 | 0.96639 | 0.96577 | 0.00062 | 0.00895 | 0.069 |
| 10 | 1000 | 0.50 | 0.95047 | 0.95026 | 0.00022 | 0.00125 | 0.174 |
| 50 | 25 | 0.99 | 0.95581 | 0.95563 | 0.00018 | 0.00043 | 0.432 |
| 50 | 1000 | 0.90 | 0.95177 | 0.95165 | 0.00012 | 0.00086 | 0.142 |

**Rate.** `N·Δ` converges: at `n = 10, ρ = 0` it is 0.193 / 0.223 / 0.236 / 0.241 / 0.256 for
`N = 25…400` and 0.254 / 0.241 / 0.242 at `N = 400 / 1600 / 6400`. At `n = 10, ρ = 0.9` it rises
before settling (0.045 → 0.235) — a crossover, not a violation: while `v₂` dominates, `s → 1`, the
statistic is nearly normal and coverage is insensitive to the perturbations.

**Bound.** `max |Δ|/τ = 0.432` over 48 cells spanning `n ∈ {5,10,20,50}`, `ρ ∈ {0,0.5,0.9,0.99}`,
`N ∈ {25,100,1000}` — including `N < n`.

> **Conjecture II.2.** `|Δ| ≤ K·τ` with `K = 1/2`, `τ = E[v₃/V] ≤ n/(N(n−3))`.
> Status: **`CONJECTURED`**, supported at 48/48 cells with `K = 0.432` attained. The rate is
> `DERIVED` (Prop. II.1); the constant is **not** proved and 48 cells are not a proof.

## II.8 The sign is not uniform — a counterexample   `NUMERICALLY VERIFIED`

The tempting route to a theorem is `coverage_act ≥ coverage_id ≥ 1−α`. **It is false.**

| n | ρ | N | actual | idealised | Δ | z |
|---|---|---|---:|---:|---:|---:|
| 5 | 0.90 | 5 | 0.99630 | 0.99770 | **−0.00140** | −37.7 |
| 5 | 0.90 | 25 | 0.98974 | 0.99198 | **−0.00224** | −44.4 |
| 5 | 0.90 | 100 | 0.97670 | 0.97693 | −0.00023 | −4.2 |
| 5 | 0.99 | 100 | 0.99647 | 0.99679 | −0.00033 | −14.8 |

`Δ < 0` at 4 of 48 cells, at 38 and 44 standard errors — not noise. **The sign argument is dead.**

What survives is weaker but still useful: `Δ < 0` occurred **only** where `coverage_id` already
exceeds nominal by a wide margin (0.977–0.998), i.e. where `s` is large. Where `coverage_id ≈ 1−α`
(that is, `s ≈ 0`), `Δ` was positive at every cell tested. Formalising "the error is negative only
where the margin is large" is exactly what a transfer theorem would need, and it is **`OPEN`**.

**Meanwhile the practical interval never under-covered:** `min coverage_act = 0.95010` over these
48 adversarial cells (`n` down to 5, `N` down to 5, `ρ` up to 0.99), and 0.9488 over Experiment E's
81 cells, at the `N → ∞` end where the theory says exactly `1−α`.

## II.9 Main proposition   `DERIVED` + `NUMERICALLY VERIFIED`; not `PROVED`

> **Proposition II.3 (finite-N transfer).** Under M1, for `n ≥ 4`,
> ```
> coverage_act(n, ρ, N)  =  coverage_id(n, ρ, N)  +  Δ(n, ρ, N),
> coverage_id  =  E_F[ C(s(F)) ] ≥ 1−α        (Theorem C, PROVED)
> |Δ|  =  O(1/N),  and empirically  |Δ| ≤ K·τ  with  τ ≤ n/(N(n−3)),  K ≈ 0.43.
> ```
> Consequently `coverage_act ≥ 1−α − K·τ`: **conservative up to an explicit `O(1/N)` term.**

This is **Outcome B/C** of the brief, not Outcome A. It is *not* a proof that the practical
interval is conservative, because `K` is not proved and `Δ` is not sign-definite. It **is** a
proof of the rate structure plus a quantified, falsifiable bound.

## II.10 Assumptions, corollaries, limitations

**Assumptions.** M1 throughout (`04` §4.10) — bivariate normality, homoskedasticity, independent
pool, fixed predictor, `n ≥ 4`. `n ≥ 4` is needed twice: for `s²`, and for `E[σ_f²/S_ff] = 1/(n−3)`
to be finite, which is why the bound degrades as `n ↓ 4`.

**Corollaries.**
* `N/n ≳ 1000` ⇒ `τ ≲ 10^{−3}/n` ⇒ `|Δ| ≲ 2·10^{−4}`: the idealisation is irrelevant in the regime
  PPI is actually used in, and Theorem C effectively governs.
* The `O(1/N)` rate holds **uniformly in ρ**; ρ enters the constant, not the rate.
* Because `Δ` is `O(1/N)` while the *first-order* finite-N effect (`s > 0`, Theorem B) is `O(n/N)`
  in the variance share, the idealisation error is smaller than the effect it approximates by a
  factor `O(1/n)` — the reason Theorem B is useful at all.

**Limitations, stated plainly.**
1. `K` is a numerical maximum over 48 cells, not a proved constant. `OPEN`.
2. Prop. II.1's remainder is not rigorously controlled — no explicit third-derivative bound on the
   coverage functional is given. `OPEN`.
3. `Δ` is not sign-definite (§II.8), so Theorem C does **not** transfer by monotonicity. `OPEN`.
4. Everything is under M1. Nothing here survives heavy tails or skew automatically; that is E5.
5. `σ̂_f²` is treated as known in the ablations. Justified by measurement (E4: ≤ 0.0047, ≈ 2.2 MC
   s.e.) and by its `O(N^{−3/2})` order, but it is an assumption of the *analysis*.

## II.11 Prior art for Part II   `PARTIAL` / `UNVERIFIED`

Focused checks only; see `docs/prior_art_findings.md` §4c for the full record.

* **Theorem C may be classical.** Search surfaced Robinson, *Properties of Student's t and of the
  Behrens–Fisher Solution to the Two Means Problem*, **Annals of Statistics 4(5), 1976**, reported
  as establishing that "the coverage probabilities for the Behrens–Fisher intervals are always
  larger than the nominal significance level ... making the Behrens–Fisher and Student's t
  procedures conservative". If that is what it proves, **Theorem C is a special case** (`ν₂ = ∞`)
  and must be cited, not claimed. Status: `UNVERIFIED` — **retrieve and read before any write-up.**
* **Two-phase difference/regression estimators** with an estimated auxiliary mean are studied, but
  the results located are MSE-based and "up to first order of approximation" — asymptotic, not
  finite-sample-exact or conservative. Status: `PARTIAL`.
* **What would remain new**, if the above hold: the *identification* of finite-N LC-PPI as a
  Behrens–Fisher problem with one degree of freedom infinite; Theorem A′; and the `O(1/N)`
  idealisation bound of Part II. All three are about the PPI construction, not about Behrens–Fisher
  in the abstract.


---
---

# PART III — Prior-art closure   (E4d, 2026-08-20)

Three sources read in full or as fully as access allowed. **The conclusion is negative for our
headline theorems and it is recorded without softening.**

## III.1 What each source establishes

### Robinson (1976), *Ann. Statist.* 4(5), 963–971 — read in full (9 pages, scanned)

* **Theorems 1 and 2** are about **conditional inference**: there are no negatively biased relevant
  selections, hence no negatively biased relevant subsets in the sense of Buehler (1959), for
  Student's *t* and for the Behrens–Fisher solution. This is a betting/relevant-subsets property,
  **not** a coverage statement.
* **§4 "The coverage properties of Behrens–Fisher intervals"** reviews the coverage question and
  reports it as **numerical**: Bartlett (1936) first showed the regions are not exact; type-I
  errors were computed by James (1959), Mehta & Srinivasan (1970), Wang (1971); Robinson tabulates
  them himself for `n₁, n₂ ∈ {2,…,100, ∞}` and variance ratios from `10⁻⁴` to `10³`, concluding
  "In all cases the probability of type I error was found to be less than the nominal value … to
  infer **with reasonable certainty** that the type I error rate is always less than the nominal
  significance level for Behrens–Fisher tests." **That is a tabulation, not a proof.**
* **§5** proposes "conservative" as a technical term meaning (i) coverage ≥ nominal **and** (ii) no
  negatively biased relevant selections.

**Verdict: Robinson does not prove our Theorem C.** But his grid explicitly includes `n₂ = ∞`, the
known-variance case, so the *phenomenon* was already documented numerically for our exact setting.

### Mickey & Brown (1966), *Ann. Math. Statist.* 37, 639–642 — **this is the source that matters**

Located via Kim & Cohen's review of the Behrens–Fisher problem, which states the result and
attributes it:

> "a simple way to solve this problem, however, is to use a proxy `t[min(ν₁, ν₂)]` … Restated in
> terms of a confidence interval, we have `x̄₁ − x̄₂ ± t_α[min(ν₁,ν₂)]·√(s₁²/n₁ + s₂²/n₂)`.
> **Note, in this case … the actual confidence is greater than 100(1−α)% (Mickey & Brown, 1966).**"

Status: `PARTIAL` — the statement and attribution are read; the Mickey & Brown paper itself has
**not** been retrieved.

### Mozer (2026), arXiv:2603.19160 — read in full (18 pages, text extracted)

Establishes the algebraic equivalences: PPI's mean estimator = the **difference estimator** of
Cassel, Särndal & Wretman (1976); PPI++ = the **GREG** estimator of Särndal et al. (2003). Discusses
design-based vs superpopulation inference, the role of the unlabelled pool, differential prediction
error and the SATE/PATE distinction, optimal allocation, and calibration.

**What it does not contain, verified by exhaustive term search of the full text:** the strings
*finite-sample*, *finite sample*, *degrees of freedom*, *Student*, *small sample*, *Satterthwaite*,
*normality*, *conservative* — **zero occurrences of each**. "asymptot" appears twice, both times
asserting asymptotic equivalence or design-consistency. The regime it names is "n in the hundreds,
N in the thousands or tens of thousands".

It *does* identify the right frame for our problem without solving it (§3.3): PPI "does not account
for the additional uncertainty from the unlabeled pool being a sample rather than a census. The
survey sampling framework handles this using variance estimators that incorporate both phases of
sampling … a two-phase design".

### Sitter (1997), *JASA* 92(438), 780–787, and the two-phase literature — `PARTIAL`

Sitter (1997) itself is paywalled at JSTOR and Taylor & Francis and was **not retrieved**. It is
characterised here from (a) the publisher's description — variance estimators for the two-phase
regression estimator "and their use in constructing confidence intervals, including design-based
linearization variance estimators" — and (b) a 1998 *Statistica Sinica* paper on replication
variance estimation for two-phase sampling that reviews it, read in full: "A jackknife procedure has
been suggested for the ratio estimator by Rao and Sitter (1995) and for the regression estimator by
Sitter (1997). These procedures involve creating jackknife replicates using the entire first-phase
sample. Särndal and Swensson (1987) discuss variance estimation for regression estimation in the
general two-phase situation. Kott (1990, 1995) discussed variance estimation …"

That review's own analysis is a design-based expansion with `O_p(n^{−3/2})` remainders. Term search
of it: *Student* 0, *degrees of freedom* 0, *Satterthwaite* 0, *coverage* 0, *confidence interval* 0.

**Verdict, stated with its uncertainty:** the two-phase literature owns the **estimator** and its
**asymptotic, design-based variance**, including the case where the auxiliary mean comes from a
first-phase sample — which is structurally our finite-N setting. Nothing located in it is an exact
finite-sample distribution, a Student-*t*/Satterthwaite interval, or a conservativeness theorem.
**This conclusion is `PARTIAL` and would be overturned by reading Sitter (1997) directly.**

## III.2 The exact mathematical correspondence for Theorem C

Two-sample Behrens–Fisher with the **second variance known** (`ν₂ = ∞`):

```
D = x̄₁ − x̄₂ − δ  ~  N(0, V₁ + V₂),     V₁ = σ₁²/n₁,  V₂ = σ₂²/n₂ known
V̂  = s₁²/n₁ + σ₂²/n₂ = V₁·Q/ν₁ + V₂,     Q ~ χ²_{ν₁} ⟂ D
T  = D/√V̂  =  Z / √( (1−s)Q/ν₁ + s ),   s = V₂/(V₁+V₂)
```

This is **identical**, symbol for symbol, to Theorem B's statistic with `ν₁ = n−2`. And
`min(ν₁, ν₂) = min(n−2, ∞) = n−2`, so the classical `t_{min(ν₁,ν₂)}` rule **is** our `t_{n−2}` rule,
and "the actual confidence is greater than `100(1−α)%`" **is** our Theorem C.

Verified numerically in `tests/test_finite_n.py::test_theorem_C_statistic_is_the_behrens_fisher_statistic_with_one_variance_known`:
simulating the two-sample construction directly reproduces our exact coverage integral to 2·10⁻⁴ at
`s = 0, 0.333, 0.800`.

> **Theorem C is therefore not new. It is Mickey & Brown (1966) at `ν₂ = ∞`. Every novelty claim
> around it is withdrawn.** What remains is a short self-contained proof of the special case (Part I
> §8), offered as an exposition, not as a contribution.

## III.3 Prior-art matrix

| Result | Existing equivalent? | Closest source | Equivalence | What remains new |
|---|---|---|---|---|
| **Theorem A′** (exact `t_{n−2}` pivot for `θ + βδ`) | **Partly** | Classical normal-theory result that a studentised OLS fitted value at a **fixed** design point is exactly `t_{n−2}` (Seber & Lee); estimator identification in Mozer (2026) | The *distributional fact* is textbook. Applying it **conditionally on the random `f̄_N`** and reading off the consequence is not stated in either source | The **identification**: that finite-N LC-PPI is exactly pivotal for a *shifted* estimand, so the shortfall is an **unestimable bias**, not a variance error. PPI-specific |
| **Theorem B** (idealised law = normal variance mixture / BF form) | **Yes** | Behrens–Fisher literature; the exact coverage integral is how James (1959), Wang (1971) and Robinson (1976) §4 compute type-I error | **Exact.** The statistic is the BF statistic with one variance known | Nothing distributional. Only the *recognition* that finite-N PPI lands there |
| **Theorem C** (conservativeness) | **Yes** | **Mickey & Brown (1966)**, *Ann. Math. Statist.* 37, 639–642, via Kim & Cohen's review | **Exact**, at `ν₂ = ∞` | **Nothing.** Claim withdrawn. Our proof is an exposition of a special case |
| **Proposition II.1** (`Δ = O(1/N)`) | **None located** | Two-phase variance literature (Sitter 1997; Kott; Rao & Sitter 1995) gives asymptotic *variance* estimators, not coverage-error expansions | No equivalent found; `PARTIAL` because Sitter was not retrieved | The rate result itself, and the fact that the `O(N^{−1/2})` fluctuation cancels |
| **Proposition II.3** (`coverage ≥ 1−α − Kτ`) | **None located** | as above | No equivalent found | The explicit finite-`N` coverage-error bound — but `K` is `CONJECTURED`, so this is a candidate, not a theorem |

## III.4 The five kinds of prior art, kept separate

| kind | status |
|---|---|
| **Estimator** | **Fully prior art.** Cassel–Särndal–Wretman (1976) difference estimator; Särndal et al. (2003) GREG; Mozer (2026) for the PPI equivalence. We claim nothing. |
| **Variance** | **Prior art in the asymptotic, design-based sense** (two-phase literature, incl. the estimated auxiliary mean). Our exact conditional variance (V\*) under a normal superpopulation was not located there, but the search is `PARTIAL`. |
| **Distributional** | **Prior art.** The finite-N statistic is the Behrens–Fisher statistic with one variance known; its law and coverage integral are classical. |
| **Confidence interval** | **Prior art for the conservativeness** (Mickey & Brown 1966). The `t_{min(ν)}` rule is textbook. |
| **PPI-specific** | **Not prior art.** No PPI paper located contains a finite-sample interval, a degrees-of-freedom correction, or the small-`n` measurements. Mozer explicitly does not. |

## III.5 What is genuinely new

Stated narrowly, and only what the literature supports:

1. **The identification.** That PPI++/LC-PPI inference at finite `N` is *exactly* the Behrens–Fisher
   problem with one variance known — and therefore that a century of BF theory applies to it
   off the shelf. Neither the PPI literature (Mozer 2026 verified) nor the BF literature makes this
   connection. This is Outcome **A**: a new PPI-specific interpretation of classical theorems.
2. **Theorem A′ as a diagnosis.** The finite-N shortfall is an unestimable *bias*
   `β(f̄_N − μ_f)`, not a variance error — which explains why every variance-side repair (R1, `v_full`)
   fails and closes `04` §4.8.4's R3 as infeasible.
3. **Propositions II.1 / II.3** — the `O(1/N)` idealisation-error rate and the explicit coverage-error
   bound linking the *practical* PPI interval to the classical BF result. No equivalent located.
   Outcome **C**, but `CONJECTURED` in its constant.
4. **The empirical body**, which is new in the PPI literature and is not in dispute: PPI++ covers
   ≈ 0.878 at `n = 10`; neither the *t* quantile, nor three cross-fitting schemes, nor three
   thresholds, nor the percentile bootstrap, nor BCa reaches 0.945; the leverage-corrected interval
   reaches 0.950; and the finite-N behaviour is mapped over 81 cells.

## III.6 What is inherited

* The estimator: Cassel, Särndal & Wretman (1976); Särndal, Swensson & Wretman (2003); recognised
  for PPI by Mozer (2026), arXiv:2603.19160.
* The exact `t_{n−2}` law of a studentised OLS fitted value: classical normal-theory linear models.
* The Behrens–Fisher distributional form and its exact coverage integral: Behrens (1929), Fisher
  (1935), James (1959), Wang (1971), Robinson (1976) §4.
* Conservativeness of the `t_{min(ν₁,ν₂)}` interval: **Mickey & Brown (1966)**.
* Welch–Satterthwaite effective degrees of freedom: Satterthwaite (1941, 1946), Welch (1938, 1947).
* Asymptotic design-based variance estimation with an estimated auxiliary mean: Särndal & Swensson
  (1987), Kott (1990, 1995), Rao & Sitter (1995), Sitter (1997).

## III.7 Revised contribution statement   (paper-ready)

> Prediction-powered inference is the survey-sampling difference estimator, and PPI++ is the GREG
> estimator (Mozer, 2026). We show that its *inference* problem is likewise classical, and identify
> which classical problem it is. Writing the leverage-corrected PPI statistic in its exact form, we
> prove that at finite unlabelled-pool size `N` it is exactly Student-`t_{n−2}` pivotal not for the
> estimand `θ` but for the shifted quantity `θ + β(f̄_N − μ_f)`, whose offset is not estimable
> because `μ_f` is unknown. The resulting studentised statistic is precisely the Behrens–Fisher
> statistic with one variance known, so the interval's conservativeness is Mickey & Brown's (1966)
> `t_{min(ν₁,ν₂)}` bound at `ν₂ = ∞`, and the natural degrees-of-freedom repair is
> Welch–Satterthwaite instantiated. Our contribution is therefore not a new distributional theorem
> but (i) this identification, which makes classical Behrens–Fisher theory available to PPI, (ii) an
> explicit `O(1/N)` bound on the gap between the practical PPI interval and the classical idealised
> one, and (iii) the first systematic measurement of PPI++'s finite-sample coverage in the regime it
> was designed for: at `n = 10` gold labels the shipped interval covers 0.878 against a nominal
> 0.95, and neither a Student quantile, nor cross-fitting in three schemes, nor significance
> thresholding at three levels, nor a percentile or BCa bootstrap at up to 9,999 resamples reaches
> 0.945, while the leverage-corrected interval reaches 0.950 at a cost of 8 % interval width.

## III.8 Limitations

1. **Everything is under M1** — bivariate normality, homoskedasticity, independent pool, fixed
   predictor. Nothing above survives heavy tails or skew automatically. That is E5.
2. **Mickey & Brown (1966) has not been retrieved**, only its statement in a review. Retrieve before
   submission; if it does not cover `ν₂ = ∞`, Theorem C's status changes back.
3. **Sitter (1997) has not been retrieved.** The claim that the two-phase literature contains no
   finite-sample interval result is `PARTIAL` and is the most likely place for a further hit.
4. **Proposition II.3's constant `K` is conjectured**, and `Δ` is not sign-definite (Part II §II.8),
   so the transfer of conservativeness to the practical interval is supported by 129 simulated
   cells, not proved.
5. The identification in III.5(1) is a *framing* contribution. Its value depends on the reader
   agreeing that the connection is non-obvious; that is a judgement, not a theorem.


---
---

# PART IV — Outside the Gaussian model   (E5, 2026-08-20)

**Nothing in this part is a theorem, and nothing in Parts I–III extends to it.** Parts I–III assume
M1. Part IV reports what was *measured* when M1 fails. The distinction is kept in the language:
Parts I–III say "proved" and "derived"; Part IV says "observed".

## IV.1 Theory-supported region vs empirical robustness region

| region | model | status of the interval |
|---|---|---|
| **Theory-supported** | M1: bivariate normal, homoskedastic, independent pool, fixed predictor, `n ≥ 4` | Exactly `t_{n−2}`-pivotal for `θ + βδ` (Theorem A′, `PROVED`); conservative for `θ` under the `O(1/N)` idealisation (Theorem C, `PROVED`, and classical — Mickey & Brown 1966) |
| **Empirical robustness** | heavy tails, skew, binary labels, heteroskedasticity | **Measurements only.** 192 cells at `R = 200,000`. No exactness, no conservativeness, no bound |

## IV.2 What was observed   `NUMERICAL RESULTS`

**Heavy tails (F, 72 cells).** Nothing breaks. B9's minimum coverage is **0.9495** with heavy
residuals and **0.9494** with a heavy predictor — the variant that stresses the leverage term most,
since leverage depends directly on the tails of `f`. B9 tracks B1 to within **0.0022** at every
cell. Both drift slightly *upward* as `ν` falls (0.9594 / 0.9572 for B1 / B9 at `ν = 3, n = 10`):
the `t` interval is conservative under heavy tails, and both methods inherit that equally.

**Skew (G, 84 cells).** The one structural finding of E5, and it is not what the pre-registration
expected. B9's coverage under skew is **almost exactly constant in ρ** — mean 0.9286 / 0.9283 /
0.9283 at ρ = 0 / 0.5 / 0.9 — while **B1's rises**, 0.9282 / 0.9431 / 0.9508. The mechanism:
`Y = ρf + √(1−ρ²)ε`, so a strong predictor makes `Y` itself nearly Gaussian and rescues the
classical interval, whereas the corrected interval studentises the **residual**, which stays skewed
at every ρ. The correction cannot benefit from predictor quality here, because predictor quality is
precisely what it has already removed.

Severity matters more than ρ: `|coverage − 0.95|` averaged over the 84 cells is B1 0.0122,
**B9 0.0217**, B9-HC 0.0168, B3 0.0509. Worst cell: lognormal `σ = 1`, `n = 10`, where B9 is 0.847
— but B1 at ρ = 0 there is **0.839**, so the failure is largely the `t` interval's, not the
correction's. Tail miscoverage is badly asymmetric for everything (0.003 left / 0.149 right for B9),
and B9's is **no worse than B1's** at any cell, which is the failure condition `07` set.

**The binary case is B9's best result anywhere in E5.** `|coverage − 0.95|` averaged over its 12
cells: **B9 0.0019**, B1 0.0148, B3 0.0312. This is the case `08` §T1.3 calls "the most practically
important synthetic case, because in LLM-annotation work the gold label is usually a binary human
judgement". At `n = 10` B1 *over*-covers at 0.978 and B3 under-covers at 0.869; B9 sits at 0.950.

**Heteroskedasticity (H, 72 cells).** The genuine failure mode. Under `σ(f) = exp(γf)` B9 falls to
**0.9047** at `γ = 1, n = 10`, and is below B1 by up to 0.068. This is expected from the theory:
`s²` is a single pooled residual variance, so when `Var(Y | f)` depends on `f` the leverage term is
misweighted — assumption A7 doing exactly what A7 was flagged for.

**HC3 repairs it.** `B9-HC` never falls below **0.9517** across all 72 heteroskedastic cells and is
below 0.94 in none of them. The price is conservatism under homoskedasticity (0.9584 against B9's
0.9503 at `n = 20`) and the loss of every exactness claim: HC3 is asymptotic, `07` says so, and this
repository flags it as `asymptotic_only` in the method's own output. HC3 does **not** repair skew
(0.864 at lognormal severe).

**The bootstrap gets worse, not better.** `07` predicted BCa would shine under skew — "BCa is
designed for exactly this". Measured: under heavy tails B8 covers **0.876–0.900** at `n = 10`,
*worse* than its 0.898–0.903 under the Gaussian model and worse as tails thicken; under skew,
**0.829–0.959**, beating B9 in 1 of 9 cells. K3's outstanding second clause is settled and K3 is
closed.

## IV.3 Scope statement for the paper   `ROBUSTNESS OBSERVATION`

> The correction is exact under the Gaussian working model and empirically indistinguishable from
> the classical interval under heavy tails, including heavy-tailed predictors. Under right-skewed
> residuals it inherits the classical interval's degradation and does not benefit from a strong
> predictor the way the classical interval does, because it studentises the residual rather than the
> outcome; both are materially below nominal for severe skew at small `n`, and this is a limitation
> of the Student-`t` construction, not of the leverage correction. Under multiplicative
> heteroskedasticity it degrades materially and should be replaced by the HC3 variant, which was
> conservative in every cell tested. For binary gold labels — the dominant case in LLM-annotation
> work — it was the most accurate of the three methods tested at every sample size.

## IV.4 Open after E5

1. **Why is B9's skew coverage flat in ρ?** The mechanism above is a plausible explanation supported
   by the numbers; it is `CONJECTURED`, not derived. A residual-skewness correction (a Cornish–Fisher
   or `BCa`-style adjustment applied to the *residual* rather than the estimator) is the obvious
   candidate repair and was **not** tried.
2. **Is B9-HC conservative under heteroskedasticity, or was that 72 cells of luck?** `OPEN`; no
   theory offered and none available — HC3 is asymptotic.
3. **The `t` interval's own skew failure.** B1 at 0.839 is not this project's problem, but a paper
   claiming small-`n` validity must say where the whole `t` family stops working.


---
---

# PART V — Prior-art closure, verified against the primary sources   (E6b, 2026-08-20)

Part III reached its conclusions from a review article. Part V reads the sources.

## V.1 Mickey & Brown (1966) — **read in full**, and it is stronger than our Theorem C

M. Ray Mickey and Morton B. Brown, *Bounds on the Distribution Functions of the Behrens-Fisher
Statistic*, **Annals of Mathematical Statistics 37(3), 1966, 639–642** (received 21 October 1965,
revised 7 January 1966). Retrieved from Project Euclid as a scanned PDF and read page by page.

Their statement, quoted verbatim from §2:

> **THEOREM.** *Let X be normally distributed with zero mean and unit variance, and let f₁W₁ and
> f₂W₂ be distributed as chi-square variates with f₁ and f₂ degrees of freedom (df) respectively,
> such that X, W₁, and W₂ are mutually independently distributed. Then for all γ in the interval
> 0 ≦ γ ≦ 1,*
>
> (1)   `P{|T₁| < v} ≦ P{|V_γ| < v} ≦ P{|T₂| < v}`
>
> *where*  `V_γ = X/(γW₁ + (1 − γ)W₂)^½`

and, from the following page:

> *and T₁ and T₂ are Student-t variables with* `min(f₁, f₂)` *and* `(f₁ + f₂)` *degrees of freedom
> respectively.*

Their §1 states the purpose plainly: "The purpose of this note is to supply an **analytical proof**
of the above proposition." The proof shows `(∂²/∂γ²) P{|V_γ| < v} < 0` — the coverage is **concave
in γ** — so its minimum over `γ ∈ [0,1]` is attained at an endpoint, where `V_γ` is exactly `t_{f₂}`
or `t_{f₁}`.

### The correspondence, exactly

Put `γ = 1 − s`, `W₁ = Q/ν` (so `f₁ = ν = n − 2`), and let `f₂ → ∞`, whence `W₂ → 1` and
`t_{f₂} → N(0,1)`:

```
V_γ  =  X / ((1−s)·Q/ν + s·W₂)^½   ⟶   Z / √((1−s)Q/ν + s)   =   our T   (Theorem B)
min(f₁, f₂) = min(ν, ∞) = ν = n−2                             =   our quantile
```

Evaluating (1) at `v = t_{n−2, 1−α/2}` gives, on the left, `P{|t_{n−2}| < v} = 1−α ≤ coverage` —
**which is exactly our Theorem C**. It is the `f₂ → ∞` limiting case of their Theorem, obtained by
continuity, and their proof covers it.

> **Theorem C is withdrawn as a contribution.** It is Mickey & Brown (1966). Our §8 proof is an
> independent short argument for one limiting case and is retained only as exposition.

### Their result is *stronger*: it also caps the over-coverage

The right-hand inequality, which we did not have, gives at `f₂ → ∞`

```
coverage  ≤  P{|t_{f₁+f₂}| < v}  ⟶  P{|Z| < t_{n−2,1−α/2}}  =  2Φ(t_{n−2,1−α/2}) − 1
```

so the finite-`N` over-coverage is **bounded above by a quantity depending only on `n`**:
0.99854 at `n = 5`, **0.97889 at `n = 10`**, 0.96435 at `n = 20`, 0.95564 at `n = 50`.

**Checked, not assumed** (`tests/test_finite_n.py`): the exact coverage integral attains both
bounds at `s = 0` and `s = 1` for `ν ∈ {3, 8, 18, 48}` — as the concavity proof predicts — and all
**81 cells** of Experiment E lie inside `[0.95, 2Φ(t_{n−2,.975})−1]`, the observed maximum being
0.9711 at `n = 10` against a cap of 0.97889.

**This is a better result than ours and we adopt it, with citation.** The honest framing of the
finite-`N` behaviour is therefore: *the interval is conservative, and its conservatism is bounded
by a classical two-sided inequality* — neither half of which is new.

## V.2 Sitter (1997) — **not obtained**; characterised from two independent reviews

R. R. Sitter, *Variance Estimation for the Regression Estimator in Two-Phase Sampling*, **JASA
92(438), 1997, 780–787** (bibliographic details confirmed via Crossref, DOI
`10.1080/01621459.1997.10474031`; author affiliation Simon Fraser University).

**The paper is paywalled and was NOT read.** A final retrieval attempt (2026-08-20) settled this:
**OpenAlex reports the work as `closed` with no open-access location**, so no lawful open copy
exists — not an author preprint, not an institutional repository, not a technical-report version.
Crossref confirms the bibliographic record but carries no abstract.

What OpenAlex *does* supply is a summary of the abstract's content, and it is worth recording
because it is consistent with everything below: the paper compares **design-based linearization,
jackknife and bootstrap** variance estimators "for constructing confidence intervals", and evaluates
them **by simulation** under both unconditional and conditional sampling, finding the linearized
jackknife best unconditionally and the jackknife better conditionally. **That is a paraphrase
returned by an indexing service, not the author's abstract, and it is treated as such.** It
mentions no Student-`t` construction, no degrees of freedom, no Satterthwaite correction and no
finite-sample theorem.

The paper is therefore characterised from two independent papers that review it in depth and *were*
read in full:

1. *Replication Variance Estimation for Two-Phase Samples*, **Statistica Sinica 8 (1998),
   1153–1164**: "A jackknife procedure has been suggested for the ratio estimator by Rao and Sitter
   (1995) and for the regression estimator by **Sitter (1997)**. These procedures involve creating
   jackknife replicates using the entire first-phase sample." Its own analysis is a design-based
   expansion with `O_p(n^{−3/2})` remainders. Term counts: *Student* 0, *degrees of freedom* 0,
   *Satterthwaite* 0, *coverage* 0.
2. *Variance Estimation by Jackknife Method Under Two-Phase Complex Survey Design* (Statistics
   Sweden / JOS): reviews Rao & Sitter (1995), Särndal (1980) GREG, Brewer (1979), the QR-class.
   Its intervals are `t_G ∓ t_{α/2}√v` where `t_{α/2}` is described as **"the normal score"**, and
   its coverage assessment is an empirical "Actual Coverage Percentage" over simulated samples,
   justified by checking that skewness and kurtosis are "not far from (0, 3)". Term counts:
   *Student* 0, *degrees of freedom* 0, *Satterthwaite* 0.

**Assessment, with its uncertainty stated.** The abstract-level evidence and two independent
reviews of this literature agree: they use
**normal-quantile Wald intervals with empirically checked coverage**, and neither contains a
Student-`t` construction, an effective-degrees-of-freedom correction, or a finite-sample
conservativeness result. The two-phase literature owns the **estimator** and its **asymptotic,
design-based variance**, including the case where the auxiliary mean comes from a first-phase
sample — which is structurally our finite-`N` setting. Status: **`PARTIAL`**. This would be
overturned by reading Sitter (1997) directly, and that should be done before submission.

## V.3 Prior art vs our contribution — the final line

**Prior art. Claim nothing here.**

| object | source |
|---|---|
| the estimator (`Ȳ + b(f̄_N − f̄_n)`) | difference estimator, Cassel, Särndal & Wretman (1976); GREG, Särndal et al. (2003) |
| its equivalence to PPI / PPI++ | **Mozer (2026)**, arXiv:2603.19160 |
| its asymptotic, design-based variance with an estimated auxiliary mean | two-phase literature: Särndal & Swensson (1987); Kott (1990, 1995); Rao & Sitter (1995); **Sitter (1997)** |
| the exact `t_{n−2}` law of a studentised OLS fitted value at a fixed point | classical normal-theory linear models (Seber & Lee) |
| the finite-`N` statistic's distributional form | Behrens–Fisher with one variance known; Behrens (1929), Fisher (1935) |
| **conservativeness, and the cap on over-coverage** | **Mickey & Brown (1966)**, Thm at `f₂ → ∞` |
| numerical conservativeness of Behrens–Fisher intervals | Bartlett (1936); James (1959); Wang (1971); Robinson (1976) §4 |
| effective degrees of freedom `ν_eff = (n−2)/(1−s)²` | Welch (1938, 1947); Satterthwaite (1941, 1946) |

**Ours, and deliberately narrow.**

1. **The identification** that PPI/PPI++ inference at finite `N` *is* the Behrens–Fisher problem
   with one variance known — which is what makes every row of the table above applicable to PPI.
   Absent from the PPI literature (Mozer 2026 verified by full-text term search: *finite-sample*,
   *degrees of freedom*, *Student*, *Satterthwaite*, *conservative* — zero occurrences each) and
   absent from the Behrens–Fisher literature, which has no PPI.
2. **Theorem A′ as a diagnosis** — exactly `t_{n−2}`-pivotal for `θ + β(f̄_N − μ_f)`, so the
   finite-`N` shortfall is an **unestimable bias**, not a variance error. This is what explains why
   every variance-side repair fails and why `04` §4.8.4's R3 is infeasible rather than
   under-developed. The underlying distributional fact is textbook; the diagnosis is not.
3. **The `O(1/N)` idealisation bound** (Part II) connecting the *practical* PPI interval to the
   classical idealised one. No equivalent located. `K` remains conjectured.
4. **The empirical body** — Experiments A–I. New in the PPI literature and not in dispute.

---

# PART VI — Gate 1: a rigorous finite-`N` coverage bound   (2026-08-24, branch `gate1-finite-n-theory`)

Part II derived the rate `|Δ| = O(1/N)` by a formal second-order expansion whose remainder was
never controlled, and paired it with an **empirically fitted** constant (`K ≈ 0.43`, Conjecture
II.2). Gate 1 asked whether either can be made rigorous. **Both can**, by a route that does not
resemble the Part II expansion — and along the way one claim in Part II turns out to be false for
the statistic actually deployed. Nothing in Parts I–V is edited; §VI.8 states the correction
explicitly, old claim beside new.

## VI.1 The exact practical statistic   `DEFINITION` (unchanged from §II.1)

Conditional on the labelled design `F`, with `u = (μ_f − f̄_n)/√S_ff`, `κ = σ_f/(√N·√S_ff)`,
`c = βσ_f/√N`, `ν = n−2`, and the primitives

```
Z₀, Z₁, W ~ N(0,1),   Q ~ χ²_ν,   σ̂_f² = σ_f²·χ²_{N−1}/(N−1) ,   all mutually independent
```

(`W` is the standardised pool fluctuation, `δ = σ_f W/√N`; `σ̂_f² ⟂ W` because a normal sample mean
and sample variance are independent, and `σ̂_f² ⟂ F` by A2):

```
X_act = σ[ Z₀/√n + (u+κW)·Z₁ ] + cW                                            numerator
V̂_act = s²·g_act + b²·σ̂_f²/N ,  s² = σ²Q/ν,  b = β + σZ₁/√S_ff,  g_act = 1/n + (u+κW)²
T_act = X_act/√V̂_act ,   interval covers iff |T_act| ≤ t := t_{ν,1−α/2}
```

This is `mode="v482"` of `variance.var_lc_ppi` — the deployed B9 — with `σ̂_f²` **estimated**,
which is what `unlabeled_mode="explicit"` does. That last point is where §II.8 went wrong.

## VI.2 The three statistics and the object of study   `DEFINITION`

```
T_act : V̂ = s²·g_act + b²σ̂_f²/N        the deployed statistic
T_mid : V̂ = s²·g_act + β²σ_f²/N        term B removed  (b² → β², σ̂_f² → σ_f²)
T_id  : V̂ = s²·g_id  + β²σ_f²/N        terms P and D also removed = Theorem B's statistic
```

with `g_id = 1/n + u²`. The object of study is

```
Δ_cov(n, ρ, α, N)  =  coverage_act − coverage_id  =  Δ_B + Δ_PD ,
Δ_B  = coverage_act − coverage_mid ,      Δ_PD = coverage_mid − coverage_id .
```

The split is not cosmetic: `V̂_mid` and `V̂_id` are free of `Z₁` while `V̂_act` is not, and each half
is handled by a different exact reduction below.

## VI.3 Two exact conditional reductions   `PROVED`

Write `ψ(x, y) := P(|Z + y| ≤ x) = Φ(x−y) − Φ(−x−y)` for `Z ~ N(0,1)`. Then

```
|ψ_x| ≤ 2φ(0) = 0.79788… ,      |ψ_y| ≤ φ(0) = 0.39894…      uniformly in (x, y).      (ψ-bnd)
```

> **Lemma VI.1 (reduction 1).** For any statistic whose `V̂` is measurable w.r.t.
> `(F, Q, W, Z₁, σ̂_f²)` — in particular `T_act` and `T_mid` —
> ```
> P(|T| ≤ t | F, Q, W, Z₁, σ̂_f²)  =  ψ( t√(n V̂)/σ , √n·m/σ ) ,     m = σ(u+κW)Z₁ + cW.
> ```
> *Proof.* Given the conditioning, `X = σZ₀/√n + m` with `Z₀ ~ N(0,1)` and `V̂` fixed, so
> `P(|X| ≤ t√V̂) = P(|Z₀ + √n m/σ| ≤ t√(nV̂)/σ)`. ∎

> **Lemma VI.2 (reduction 2).** For a statistic whose `V̂` is free of `Z₁` — `T_mid` and `T_id`,
> not `T_act` — with `λ := c/σ = |β|σ_f/(σ√N)` and `g` the corresponding leverage,
> ```
> P(|T| ≤ t | F, Q, W)  =  ψ( t√(Q/ν + λ²/g) , λW/√g ) .
> ```
> *Proof.* Condition on `(F, Q, W)`. Then `a := u + κW` is a **fixed real number**, and
> ```
> X = σ(Z₀/√n + a·Z₁) + cW
> ```
> with `Z₀, Z₁` independent `N(0,1)`. A linear combination of independent centred normals is
> normal with the summed variances, so
> ```
> X | (F, Q, W)  ~  N( cW , σ²(1/n + a²) )  =  N( cW , σ²g ) ,       g = g_act = 1/n + a².
> ```
> `V̂ = s²g + c²` is fixed under the same conditioning (this is exactly where freedom from `Z₁` is
> used). Standardising the event `|X| ≤ t√V̂` by the conditional standard deviation `σ√g`,
> ```
> P(|X| ≤ t√(s²g + c²) | F, Q, W)  =  ψ( t√(s²g+c²)/(σ√g) , cW/(σ√g) ) ,
> ```
> and substituting `s²/σ² = Q/ν` and `c²/σ² = λ²` gives the stated form. ∎

**`σ` cancels identically in Lemma VI.2** — verified symbolically — and that cancellation is the
whole reason the bound below is `O(1/N)` rather than `O(N^{−1/2})`.

## VI.4 The perturbations, in the coordinates that matter   `DERIVED`

Part II measured the perturbations on `T` itself, where `T_act − T_id` is `O(N^{−1/2})` with mean
≈ 0; recovering `O(1/N)` from that then needs a second-order expansion with an uncontrolled
remainder. In the `(x, y)` coordinates of Lemmas VI.1–VI.2 **every perturbation is already
`O(1/N)` or smaller**, so a first-order mean-value step suffices and no Stein, anti-concentration,
Berry–Esseen or coupling argument is needed anywhere.

| | perturbation | exact expression | order |
|---|---|---|---|
| **B** | `δx_B` | `t√n(√V̂_act − √V̂_mid)/σ`, with `V̂_act − V̂_mid = (b²σ̂_f² − β²σ_f²)/N` | `O(1/N)` |
| **B** | `δy_B` | `0` — the numerators are identical | exactly 0 |
| **P,D** | `δy_PD` | `λW(g_act^{−1/2} − g_id^{−1/2})` | `O(λκ) = O(1/N)` |
| **P,D** | `δx_PD` | `t[√(Q/ν+λ²/g_act) − √(Q/ν+λ²/g_id)]` | `O(λ²κ) = O(N^{−3/2})` |

with `g_act − g_id = 2uκW + κ²W²` (symbolically verified). Both `λ` and `κ` are `O(N^{−1/2})`, so
the products above are the stated orders. The `O(N^{−1/2})` fluctuation Part II worried about has
not been cancelled by a symmetry argument — it never appears in these coordinates.

## VI.5 Theorem G1   `PROVED`   (Theorem 1 of the manuscript)

> **Theorem G1 (finite-`N` coverage error).** Under M1 with `n ≥ 4`, `ρ² < 1` and `N > 1`,
> ```
> |coverage_act − coverage_id|  ≤  B(n, ρ, α, N)  =  B_B + B_PD                          (G1)
> ```
> where, with `G := E[ (√(Q/(νn)) + √(Q/(νn)+λ²))^{−1} ]`, `m_abs := E[κ|u|/g_id]`,
> `m_sq := E[κ²/g_id]`, and all expectations over the stated primitives,
> ```
> B_B  = 2φ(0)·(t√n /(σ²N))·G·[ σ_f²(2|β|σ√(2/π)·E[S_ff^{−1/2}] + σ²E[S_ff^{−1}])
>                                + β²σ_f²√(2/(N−1)) ]
> B_PD = 2φ(0)·(tλ²n√ν/2)·E[Q^{−1/2}]·(2√(2/π)·m_abs + m_sq)  +  φ(0)·λ√n·(2 m_abs + 2√(2/π)·m_sq)
> ```
> **`B` is the object that is proved:** an explicit, non-asymptotic, exactly computable function of
> `(n, ρ, α, N)`. No limit is taken anywhere in its derivation.
>
> Moreover, **for fixed `(n, ρ, α)` as `N → ∞`**,
> ```
> B  =  C(n,ρ,α)/N  +  O(N^{−3/2}) ,        C(n,ρ,α) := lim_{N→∞} N·B  <  ∞ .            (G1a)
> ```

> **(G1a) is asymptotic and must not be read as a finite-`N` bound.** `N·B` is *decreasing* in `N`
> at every `ρ > 0` (at `n = 10, ρ = 0.5`: 6.76 → 6.04 → 5.53 → 5.29 for `N = 25, 10², 10³, 10⁷`),
> so `C` is an **infimum** and
> ```
> B(n,ρ,α,N)  >  C(n,ρ,α)/N        at every finite N.
> ```
> **A universal bound of the form `|Δ| ≤ C(n,ρ,α)/N` is therefore NOT proved, and is false as
> stated.** Quote (G1) with `B`; quote (G1a) only with its `N → ∞` scope attached.

*Proof.* Split `Δ = Δ_B + Δ_PD`.

*Term B.* Apply Lemma VI.1 to `T_act` and `T_mid`; the two share `(F,Q,W,Z₁,σ̂_f²)` and their
numerators, so `δy = 0` and the mean-value theorem with (ψ-bnd) gives `|Δ_B| ≤ 2φ(0)·E|δx_B|`.
Since `√V̂_act ≥ s√g_act ≥ s/√n = σ√(Q/(νn))` and `√V̂_mid ≥ √(s²/n + c²) = σ√(Q/(νn)+λ²)`,
```
|δx_B| = t√n·|V̂_act−V̂_mid| / (σ(√V̂_act+√V̂_mid))
       ≤ t√n·|b²σ̂_f² − β²σ_f²| / (σ²N·(√(Q/(νn)) + √(Q/(νn)+λ²))) .
```
`Q` is independent of `(Z₁, S_ff, σ̂_f²)`, so the expectation factorises into `G` and
`E|b²σ̂_f² − β²σ_f²|`; the latter is bounded by the triangle inequality using
`b² − β² = 2βσZ₁/√S_ff + σ²Z₁²/S_ff`, `E|Z₁| = √(2/π)`, `E[Z₁²] = 1`,
`E[S_ff^{−1}] = 1/(σ_f²(n−3))` (finite for `n ≥ 4`), and `E|σ̂_f² − σ_f²| ≤ σ_f²√(2/(N−1))`.

*Terms P and D.* Apply Lemma VI.2 to `T_mid` and `T_id`; the mean-value theorem gives
`|Δ_PD| ≤ 2φ(0)E|δx_PD| + φ(0)E|δy_PD|`. Using `g ≥ 1/n` throughout,
```
|δy_PD| ≤ λ|W|·|g_act−g_id| / (√g_act√g_id(√g_act+√g_id)) ≤ λ|W|(2|u|κ|W| + κ²W²)·√n/g_id ,
|δx_PD| ≤ tλ²·|g_act−g_id|/(g_act g_id) / (2√(Q/ν))       ≤ tλ²(2|u|κ|W| + κ²W²)·n/(2 g_id √(Q/ν)) .
```
Taking expectations, with `W ⟂ (F,Q)` and `E[W²] = 1`, `E|W|³ = 2√(2/π)`, and collecting the
design-only expectations into `m_abs` and `m_sq`, gives `B_PD`. ∎

**Status of each statement, kept separate.**

| statement | status |
|---|---|
| `\|Δ\| ≤ B(n,ρ,α,N)`, `B` explicit and computable at any finite `N` | **`PROVED`** — non-asymptotic |
| `B = C/N + O(N^{−3/2})` for **fixed** `(n,ρ,α)` as `N → ∞`, `C < ∞` | **`ASYMPTOTIC`** — a property of `B`, not a finite-`N` bound |
| `\|Δ\| ≤ C(n,ρ,α)/N` for all `N` | **NOT PROVED, and false**: `N·B ↓ C`, so `B > C/N` at every finite `N` |
| a constant uniform in `ρ` on `\|ρ\| < 1` | **NOT PROVED** — the proof divides by `σ` twice, so `C → ∞` as `ρ² → 1` |
| `coverage_act ≥ 1−α` (global conservativeness) | **DISPROVED** — counterexample, §VI.8 |

**What is and is not proved.** The inequality (G1) holds for every `(n ≥ 4, ρ² < 1, α, N > 1)` — it
is not asymptotic and has no unspecified constant. `C(n,ρ,α)` is finite and computed by
`ppi_finite.theory_gate1.coverage_bound`; `G`, `m_abs`, `m_sq` are expectations of explicit
functions of `(Q)` and `(f̄_n, S_ff)` and are evaluated by **Gauss–Legendre quadrature on the
quantile scale**, not by simulation. Calling those three numbers "computed to quadrature accuracy"
rather than "proved in closed form" is the honest description.

**The `O(1/N)` rate does not depend on the quadrature.** Each of the three admits a closed-form
dominating bound — `G ≤ √(νn)·E[Q^{−1/2}]/2` (drop `λ² ≥ 0`), `m_abs ≤ (√n/2)·E[κ]` (AM–GM on
`|u|/g_id`), `m_sq ≤ n·E[κ²] = n/(N(n−3))` (`1/g_id ≤ n`) — each verified to dominate the quadrature
value. Substituting them gives a fully analytic, larger `B` with the same `1/N` rate. The quadrature
is a **sharpening only** (≈2× at `n = 10`, ≈4× at `n = 50`), never load-bearing for the rate.

`C(n,ρ,α)` at `α = 0.05`, `lim_{N→∞} N·B`:

| `n` | `ρ=0` | `ρ=0.5` | `ρ=0.9` | `ρ=0.99` |
|---|---:|---:|---:|---:|
| 4 | 12.16 | 21.59 | 45.90 | 126.90 |
| 5 | 4.39 | 9.91 | 24.17 | 71.82 |
| 10 | 1.46 | 5.29 | 15.19 | 48.35 |
| 20 | 1.03 | 5.27 | 16.24 | 53.06 |
| 50 | 0.87 | 6.70 | 21.78 | 72.65 |

**`C` is not uniform in `ρ`.** The proof divides by `σ` twice, so `C` grows without bound as
`ρ² → 1`; it is finite on every compact subset of `|ρ| < 1`. That is a limitation of the technique,
not evidence about the truth: measured `|Δ|` is *smaller* at high `ρ`, so the bound is loosest
exactly where it is largest.

## VI.6 Corollary G2 — at `ρ = 0` the product and leverage idealisations vanish   `PROVED`

> **Corollary G2.** If `β = 0` then `λ = 0`, so by Lemma VI.2 `(x, y)` coincide **pointwise** for
> `T_mid` and `T_id`. Hence `Δ_PD = 0` exactly, and `T_mid ~ t_ν` exactly, for every `n ≥ 4`,
> every `N`, and every realisation. The entire finite-`N` error at `ρ = 0` is term B.

Measured: `coverage_mid = coverage_id = 0.95000` to five decimals at `n ∈ {5,10,20,50}`,
`N ∈ {25,100,1000}`, and `B_PD = 0` identically. This is a *sharper* statement than anything in
Part II, which treated P and D as `O(1/N)` contributions at all `ρ`.

## VI.7 Conservativeness   `DISPROVED for the deployed interval`

Theorem C (= Mickey & Brown 1966) gives `coverage_id ≥ 1−α`, so Theorem G1 yields

> **Corollary G3.** `coverage_act ≥ 1 − α − B(n,ρ,α,N)`, with `B` explicit.

That is the strongest one-sided statement available, and the slack is **necessary**: §VI.8 exhibits
cells where `coverage_act < 1−α`. So `coverage_act ≥ 1−α` is **false**, not merely unproved.
Corollary G3 replaces Proposition II.3, whose `K·τ` term rested on a fitted constant.

## VI.8 Counterexamples, and a correction to §II.8   `NUMERICALLY VERIFIED`

**(a) `Δ` is not sign-definite** — confirmed, as §II.8 already reported. Measured
`Δ = −0.00364` at `n=5, ρ=0.9, N=25` and `−0.01090` at `n=50, ρ=0.99, N=25`
(`R = 2·10⁶`, Rao–Blackwellised, MC s.e. ≪ 10⁻⁴).

**(b) NEW — the deployed interval under-covers.** §II.8 states:

> *old claim (§II.8):* "**Meanwhile the practical interval never under-covered:**
> `min coverage_act = 0.95010` over these 48 adversarial cells."

> *proposed new claim:* **The deployed interval does under-cover.** Over the same
> `n ∈ {5,10,20,50} × ρ ∈ {0,0.5,0.9,0.99} × N ∈ {25,100,1000}` grid,
> `min coverage_act = 0.94480` at `n = 50, ρ = 0.99, N = 25` (`R = 2·10⁶`).

Three independent runs agree on the sign and size at that cell: 0.94480 (grid, `R = 2·10⁶`),
0.94437 (package simulator, `R = 6·10⁵`), 0.94473 (ablation below, `R = 3·10⁶`).

*Mathematical evidence.* `theory.ablation_coverages`, which produced the §II.8 number, sets
`sigma2_f = 1.0` as a **known constant** — it never draws `σ̂_f²`. §1 of this document flags that
simplification, but §II.8's sentence was then read as a statement about B9 as deployed. It is not.
With `σ̂_f²` estimated from the pool, an **independent** check using the package's own simulator
(`dgp_gaussian(unlabeled_mode="explicit")` → `registry.get("B9")`, `R = 6·10⁵`) gives

| `n` | `ρ` | `N` | coverage | s.e. | `z` vs 0.95 |
|---|---|---|---:|---:|---:|
| 50 | 0.90 | 25 | 0.94800 | 0.00029 | **−7.0** |
| 50 | 0.99 | 25 | 0.94437 | 0.00030 | **−19.0** |
| 20 | 0.99 | 25 | 0.95487 | 0.00027 | +18.2 |
| 50 | 0.90 | 100 | 0.95503 | 0.00027 | +18.8 |

*Mechanism.* Toggling one idealisation at a time at `n=50, ρ=0.99, N=25`:

| statistic | coverage |
|---|---:|
| deployed (`b²`, `σ̂_f²`) | 0.94473 |
| `σ_f²` known (`b²`, `σ_f²`) | 0.95579 |
| `b → β`, `σ̂_f²` estimated | 0.94467 |
| both idealised (`= T_mid`) | 0.95571 |

The `b² → β²` substitution is irrelevant here; **estimating `σ_f²` is the whole effect.** §II.3's
table records the `σ̂_f²` row as "conditional s.d. `O(N^{−3/2})` … negligible beside `v₃`". The
absolute order is right; the comparison is not. What governs coverage is the size **relative to
`V`**, and that is `s·√(2/(N−1)) = O(N^{−1/2})·s`, negligible only while `s` is small. **This is a
fourth idealisation, not one of the three of §II.2, and it is the only one that can push coverage
below `1−α`.** The limiting regime below says exactly when.

### VI.8.1 The strong-predictor boundary   `DERIVED` + `NUMERICALLY VERIFIED`

> **Result G8 (limiting strong-predictor regime).** Under M1, fix `n ≥ 4` and `N > 1` and let
> `ρ → 1`. Then `σ² = (1−ρ²)Var(Y) → 0`, so `s²g_act → 0` and `X_act → cW`, whence
> ```
> T_act  =  X_act/√V̂_act  ⟶  (βσ_f W/√N) / √(β²σ̂_f²/N)  =  W / √(χ²_{N−1}/(N−1))  ~  t_{N−1} .
> ```
> **The statistic degenerates to `t_{N−1}`, while the interval uses the `t_{n−2}` quantile.** Hence
> ```
> coverage_act  ⟶  P( |t_{N−1}| ≤ t_{n−2, 1−α/2} )
> ```
> which is `< 1−α` **iff `N − 1 < n − 2`, i.e. iff `N < n − 1`**, and `> 1−α` otherwise.

*Why the two degrees of freedom differ.* `σ̂_f²` is the pool sample variance, carrying `N−1` degrees
of freedom; `s²` carries `n−2`. As `s → 1` the denominator is dominated by `b²σ̂_f²/N`, so the
statistic inherits the **pool's** degrees of freedom while the quantile still supplies the **gold
sample's**. When the pool is the smaller of the two, the quantile is too short.

*Numerical check* (measured at `ρ = 0.99`, the largest `ρ` on the grid, against the `ρ → 1` limit):

| `n` | `N` | `n−2` | `N−1` | predicted limit | measured at `ρ = 0.99` |
|---|---|---:|---:|---:|---:|
| 50 | 25 | 48 | 24 | 0.94427 | 0.94480 |
| 20 | 25 | 18 | 24 | 0.95367 | 0.95487 |
| 10 | 25 | 8 | 24 | 0.96994 | 0.97208 |
| 5 | 25 | 3 | 24 | 0.99599 | 0.99637 |

The limit predicts every measured value to `2·10⁻³`, at `ρ = 0.99` rather than in the limit itself.

**Scope — this is a mechanism, not a general theorem.** G8 is a limiting statement *inside model
M1* (bivariate normal, homoskedastic, independent pool, `σ̂_f²` the pool sample variance). It says
nothing about arbitrary distributions: outside M1 neither `χ²_{N−1}` nor `t_{n−2}` is the right
reference law, and the `N < n − 1` threshold has no reason to survive. It is offered as the
explanation of the §VI.8 counterexample and as a description of where to expect under-coverage
under M1 — **not** as a universal theorem about the deployed interval.

*Consistency with Theorem G1.* G8 sharpens the mechanism claim G5 of the claim audit; it collides with no earlier label. None of this contradicts (G1): at `n=50, ρ=0.99, N=25` the bound is
`B = 1` (vacuous but valid) and the measured `|Δ| = 0.0109` respects it. G1 is informative precisely
where under-coverage does not occur, and silent where it does.

Theorem G1 is *not* contradicted: its bound at that cell is 1.0 (vacuous but valid), and the
measured `|Δ| = 0.01090` respects it.

*Scope of the correction.* This does not touch any experiment in the manuscript. Experiment E's
grid is `N ∈ {50,…,5000} × ρ ∈ {0,0.5,0.9}` and never reaches `N < n` at `ρ = 0.99`; every reported
Experiment E number stands. What changes is a sentence in **this** document and, if Gate 1 is
accepted, the scope qualifier on the manuscript's §7.4 statement — see §VI.11.

## VI.9 Numerical validation   `NUMERICALLY VERIFIED`

`Δ` measured on the full `4 × 4 × 3` grid, `R = 2·10⁶` per cell, common random numbers, using the
Rao–Blackwellised reductions (Lemma VI.1 for `T_act`, Lemma VI.2 for `T_id`):

* **48 cells, 0 violations** of Theorem G1.
* `max |Δ|/B = 0.170` (at `n=10, ρ=0, N=1000`): the bound is about **6× loose** at its tightest.
* The bound is informative (`B < 0.05`) at **18/48** cells and vacuous (`B = 1`) at 10, all of
  which have `ρ ≥ 0.9` together with `N ≤ 100`.
* `B < 0.01` needs `N ≥ 155` at `n=10, ρ=0`; `N ≥ 666` at `ρ=0.5`; `N ≥ 1987` at `ρ=0.9`.

Symbolically verified with `sympy`: decomposition (D); the `σ`-cancellation in Lemma VI.2 in both
coordinates; `β = 0 ⇒ (x,y)_act ≡ (x,y)_id`; and both variance differences of §VI.4.

Tests: `tests/test_theory_gate1.py` — 8 fast, 11 `slow`.

## VI.10 Focused prior art   `PARTIAL`

Searched only the mathematically adjacent questions, per the Gate-1 brief. No directly applicable
result was found for: an explicit finite-sample coverage-error bound for a Behrens–Fisher-type
statistic perturbed by a finite auxiliary pool; an error bound for the Welch–Satterthwaite
approximation; or a finite-sample (as opposed to asymptotic) coverage bound for a two-phase
regression estimator. The higher-order-accuracy literature (Edgeworth expansions giving `O(1/n)`
coverage error for studentised statistics) concerns the limit in the *primary* sample size and does
not cover a perturbation in a second sample size that enters numerator and denominator jointly.

**No novelty is claimed for the technique.** Conditioning to obtain an exact conditional coverage
function and then applying a mean-value bound with the uniform derivative bounds (ψ-bnd) is
elementary and standard. What is specific here is the *choice* of conditioning that makes every
perturbation `O(1/N)`.

Status `PARTIAL`: this was three targeted web searches plus one rate-limited Semantic Scholar
query, not a systematic review. It should be redone before any claim of novelty is published.

## VI.11 Classification   (Gate-1 step 12)

**B — PROVED DIFFERENT EXPLICIT RATE**, together with **E — COUNTEREXAMPLE TO THE CURRENT
CONJECTURE**.

* Outcome **A** is *almost* right and deliberately not selected: `O(1/N)` **is** proved, but not
  with a constant uniform in `ρ`, and the brief's form `C(n,ρ,α)/N` is delivered only as
  `C(n,ρ,α)/N + O(N^{−3/2})` with `C` blowing up as `ρ² → 1`.
* **E** applies to a claim in §II.8 of *this* document, not to anything in the manuscript.

## VI.12 What remains open   `OPEN`

1. A constant uniform in `ρ` on `|ρ| < 1`, or a proof that none exists.
2. Closing the ~6× looseness. The bound sums absolute values and so cannot exploit the cancellation
   between the three idealisations documented in §II.5.
3. The exact leading coefficient `lim N·Δ` — the bound gives its magnitude, not its value or sign.
4. Whether `coverage_act ≥ 1−α` holds on a *restricted* region (`s` bounded away from 1, or
   `N ≥ n`). Every under-covering cell found has `N < n` and `ρ ≥ 0.9`; that is a description of
   four cells, not a theorem.
5. A quantile that accounts for the `χ²_{N−1}` randomness in `σ̂_f²` — the Welch–Satterthwaite
   variant `r2` already blends two degrees of freedom, but was never given the pool's. Whether that
   repairs the under-coverage of §VI.8 is untested and is **not** part of Gate 1.

---

# PART VII — Gate 2: validating the bound and mapping the boundary   (2026-08-24, branch `gate2-boundary`)

Gate 1 proved Theorem G1 and identified Result G8 (`ρ → 1` ⇒ `T_act → t_{N−1}`, under-coverage iff
`N < n−1`). Gate 2 validates both on a preregistered grid, maps the boundary, and asks whether an
`N`-aware quantile repairs the failure. Config: `configs/experiment_Gate2_boundary.toml`, written
before the run. Results: `results/gate2_{boundary,quantiles,summary}.csv`. Nothing under
`results/A`–`I` is touched and no registered estimator is modified.

## VII.1 Bound validation   `NUMERICALLY VERIFIED`

417 distinct cells (`grid_A` 225 + `grid_C` 264, overlap run once), `R = 400,000` each,
Rao-Blackwellised, conservative MC s.e. `0.00079`.

| | |
|---|---|
| bound violations | **0 / 417** |
| max `\|Δ_cov\| / B` | 0.2483 |
| min slack `B − \|Δ_cov\|` | 0.000731 |
| cells with `B < 0.05` (informative) | 40 |
| cells with `B ≥ 1` (vacuous) | 266 |

The bound is never attained: the tightest cell still carries a factor 4 of slack. The grid was
chosen to stress the theorem (`N` down to 2, `ρ` up to 0.99, `N ≪ n`), which is why two thirds of
the cells are vacuous — that is a property of the grid, not new information about the bound.
**`B` was never used as a threshold fitted to simulation; it is evaluated from Theorem G1 alone.**

## VII.2 The boundary, and the `ρ → 1` limit   `NUMERICALLY VERIFIED`

Result G8 predicts `coverage → P(|t_{N−1}| ≤ t_{n−2,1−α/2})`, crossing `1−α` exactly at `N = n−1`.
Measured at `ρ = 0.999`, `n = 50`, against the prediction — **the prediction is computed from the
limit, never fitted**:

| `N` | 3 | 9 | 20 | 40 | 48 | 49 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| measured `ρ=0.999` | 0.81850 | 0.92066 | 0.94110 | 0.94865 | 0.94952 | 0.95021 | 0.94984 | 0.95264 |
| limit prediction | 0.81794 | 0.92080 | 0.94123 | 0.94869 | 0.94988 | 0.95000 | 0.95012 | 0.95292 |

Agreement to `≤ 4·10⁻⁴` across three decades of `N`, and coverage approaches the limit **from
above** monotonically in `ρ` (`figures/Gate2_limit_rho.png`). At `N = n−1` the limit is exactly
`0.95` — the crossing is a point, not an interval, and the deficit vanishes continuously there.

*A caveat on detecting the boundary empirically.* Testing "coverage `< 0.95`" at 3 MC s.e. locates
the largest under-covering `N` at 8 for `n = 10` (`= n−2`, exact) but only at 30 for `n = 50`,
because near the boundary the true deficit (e.g. `0.0013` at `n=50, N=40`) is smaller than
`3 s.e. = 0.0024`. **The boundary is a statement about the sign, and the magnitude → 0 as
`N ↑ n−1`;** simulation cannot resolve the last stretch, which is why the limit prediction rather
than a fitted crossing point is what identifies it.

## VII.3 Candidate quantile rules   `NUMERICALLY VERIFIED`

All six evaluated on the same draws (`grid_A`, 225 cells, under-coverage counted at 3 MC s.e.):

| rule | min coverage | max coverage | under-covering cells | mean width ÷ B9 |
|---|---:|---:|---:|---:|
| `B9` `t_{n−2}` (deployed) | 0.70808 | 0.99940 | 44 | 1.000 |
| `C1` `t_{N−1}` | 0.81262 | 1.00000 | 90 | 1.391 |
| **`C2` `t_{min(n−2, N−1)}`** | **0.94999** | 1.00000 | **0** | 1.541 |
| `C3` Welch–Satterthwaite, both df | 0.87055 | 0.97592 | 77 | 1.221 |
| `C4` exact two-df quantile | 0.83456 | 0.97310 | 86 | 1.106 |
| `R2` W–S with pool df `= ∞` (deployed variant) | 0.70163 | 0.96932 | 130 | 0.878 |

Against the two limits Part F requires:

| | `N ≫ n` (`n=10, N=1000, ρ=0.5`) | `ρ→1, N<n` (`n=50, N=25, ρ=0.99`) |
|---|---:|---:|
| `B9` | 0.95045 ✓ | 0.94527 ✗ |
| `C1` | 0.91520 ✗ | 0.95091 ✓ |
| `C2` | 0.95045 ✓ | 0.95091 ✓ |
| `C3` | 0.94993 ✓ | 0.95060 ✓ |
| `C4` | 0.94979 ✓ | 0.95057 ✓ |
| `R2` | 0.94993 ✓ | 0.93936 ✗ |

**`C3` and `C4` fail Part F's fourth requirement.** In the region `N ≥ n−1`, where `B9` has **zero**
under-covering cells, `C3` has 43 and `C4` has 51 (worst deficits 0.027 and 0.040). Both narrow the
interval where `B9` was conservative, and the moment-matching error then pushes it below nominal.

**`C4` is worse than `C3`, and that is diagnostic.** `C4` is the *exact* quantile of the two-df
idealisation; `C3` is Welch–Satterthwaite's approximation to it. The exact rule performing worse
locates the residual error in the **idealisation** — the numerator/denominator dependence Gate 1
quantified — not in Satterthwaite's approximation. This reproduces, for the two-estimated-variance
case, exactly what `r5`-vs-`r2` showed for the one-known-variance case (manuscript §7.4).

## VII.4 Result G9 — the `t_min` rule is Mickey & Brown, correctly instantiated   `PRIOR ART`

> **Result G9.** Condition on the labelled design `F`. The idealised statistic with `σ_f²`
> **estimated** is
> ```
> T_id2  =  Z / √( γW₁ + (1−γ)W₂ ) ,    γ = v₁/(v₁+v₂),
> ν₁W₁ = (n−2)s²/σ² ~ χ²_{n−2} ,        ν₂W₂ = (N−1)σ̂_f²/σ_f² ~ χ²_{N−1} ,
> ```
> with `Z, W₁, W₂` mutually independent. **This is exactly Mickey & Brown's `V_γ`** with
> `f₁ = n−2`, `f₂ = N−1`. Their theorem's lower bound therefore gives, for every `γ ∈ [0,1]`,
> ```
> P( |T_id2| ≤ t_{min(n−2, N−1), 1−α/2} )  ≥  1 − α .
> ```

**Nothing here is new.** The deployed interval already *is* the `t_min` rule — instantiated with
`f₂ = ∞`, i.e. with `σ_f²` treated as known, giving `min(n−2, ∞) = n−2`. When `σ̂_f²` is estimated
from the pool the data supply `f₂ = N−1`, and the same classical bound returns
`t_{min(n−2, N−1)}`. **The under-coverage of §VI.8 is not a subtle finite-`N` effect: it is the
consequence of instantiating a classical inequality with `f₂ = ∞` when the data supply `f₂ = N−1`.**

*Numerical check of the correspondence* (`R = 3·10⁶`, direct simulation of `V_γ`): the `t_min`
interval covers `≥ 1−α` at all seven `(n, N, γ)` configurations tested, while `t_{n−2}` alone drops
to 0.9483, 0.9349 and 0.9146 exactly where `N−1 < n−2`.

*Transfer to the deployed statistic.* Exactly as for Theorem C, G9 is proved for the **idealised**
statistic; the deployed one differs by the Gate-1 perturbation, so what transfers is
`coverage_act ≥ 1−α − B(n,ρ,α,N)` (Corollary G3 with the `t_min` quantile), not `≥ 1−α`.
Empirically the gap never bites: **`C2` shows 0 significant under-coverage over all 417 cells**
(worst `0.94956`, i.e. `−0.00044`, against MC s.e. `0.00079`).

### Cost

| region | cells | `C2` vs `B9` |
|---|---:|---|
| `N ≥ n−1` | 266 | **bit-identical** — `max\|Δcoverage\| = 0`, width ratio `1.000000` |
| `N < n−1` | 151 | median width `×1.147`, mean `×1.96` (driven by `N = 2`) |

`C2` is a strict widening **only inside the region where `B9` provably fails**, and changes nothing
elsewhere. Worst `B9` cells repaired: `n=50, ρ=0.999, N=2`: `0.7073 → 0.9512`; `n=20, ρ=0.99, N=2`:
`0.7224 → 0.9703`.

**Because `C2 ≡ B9` whenever `N ≥ n−1`, every Experiment A–I result is unchanged by adopting it**
— Experiment E's grid satisfies `N ≥ n−1` throughout (`min[(N−1)−(n−2)] = 1`).

## VII.5 Prior art   `PARTIAL`

Focused search on finite-pool degrees-of-freedom corrections, Behrens–Fisher effective df with one
estimated and one known component, Welch-type rules with both components finite, finite-population
regression intervals with auxiliary-frame uncertainty, and `N`-dependent Student-`t` corrections.

* **The `t_min` conservativeness is Mickey & Brown (1966)** — already cited in the manuscript, which
  already states "the conservativeness of a `t_min` interval … are theirs. We claim neither."
  Gate 2 changes only *which degrees of freedom are supplied to it*.
* Welch (1947) / Satterthwaite (1946) with both components finite is the textbook form; `C3` is that
  formula instantiated, and it is **not** adopted.
* No result was found tying a finite auxiliary-sample size to the Student-`t` degrees of freedom of
  a two-phase or prediction-powered interval. The two-phase literature (§2.4) constructs intervals
  with a **normal** quantile, so the question does not arise there.

Status `PARTIAL`: two targeted web searches. **No novelty is claimed for anything in Part VII** —
the operative result is prior art applied with the right arguments.

## VII.6 Classification   (Gate-2 step J)

**A — `N`-AWARE QUANTILE THEORETICALLY JUSTIFIED AND EMPIRICALLY VALIDATED**, with the scope
stated exactly:

* *justified* by Mickey & Brown (1966), **not** by a new theorem of ours — the justification chain
  is exactly as strong as the manuscript's existing Theorem C, and no stronger;
* *validated* at 0/417 significant under-coverage, and free (`C2 ≡ B9`) wherever `N ≥ n−1`;
* what is **not** proved is `coverage_act ≥ 1−α` for the *deployed* statistic — only
  `≥ 1−α − B`, the Corollary G3 form. Global conservativeness remains **DISPROVED** for the
  `t_{n−2}` rule and **unproved** (though never violated) for the `t_min` rule.

Candidates `C3`/`C4` are classified separately as **E — failure**: they break coverage in a region
where the deployed rule is sound.

## VII.7 Open after Gate 2   `OPEN`

1. Prove `coverage_act ≥ 1−α` for the `t_min` rule on the *deployed* statistic, or find a
   counterexample. Gate 2 found none in 417 cells; that is not a proof.
2. The width cost at `N ≪ n` is large (`×6.3` at `N = 2`). Whether a rule exists that is
   conservative *and* narrower than `t_min` there is open — `C3`/`C4` are not it.
3. Redo the prior-art search properly before any publication claim.

---

# PART VIII — Gate 3: registering the finite-pool safeguard   (2026-08-24, branch `gate2-boundary`)

Gate 2 established that the deployed quantile is Mickey & Brown's `t_min` bound instantiated with
`f₂ = ∞`. Gate 3 registers the correct instantiation as a method and verifies that adopting it
changes nothing on any grid this paper reports.

## VIII.1 `B9-tmin`   `REGISTERED`

```
B9-tmin :  θ̂_LC ± t_{min(n−2, N−1), 1−α/2} · √V̂ ,     V̂ = s²(1/n + Δ²/S_ff) + b²σ̂_f²/N
```

Identical to `B9` in point estimate and variance; **only the quantile's degrees of freedom
differ**. Implemented as `mode="tmin"` in `variance.var_lc_ppi` (same `value`, `dof =
min(n−2, N−1)`) and registered as `B9-tmin`. `B9` is untouched.

**Attribution.** The `t_min` rule and its conservativeness are **Mickey & Brown (1966)**,
*Bounds on the Distribution Functions of the Behrens-Fisher Statistic*, Ann. Math. Statist. 37,
639–642. **No novelty is claimed for the rule.** What this repository contributes is:

1. the identification that the deployed LC-PPI interval *already is* that rule with `f₂ = ∞`,
   i.e. with `σ_f²` treated as known (§VII.4);
2. the demonstration that this under-covers once `N < n − 1`, with the mechanism (§VI.8, G8);
3. the verification that supplying `f₂ = N−1` repairs it **at zero cost on every reported grid**.

What is **not** claimed: global finite-`N` conservativeness of the deployed interval. Theorem G1,
(G1a) and Corollary G3 stand exactly as established in Part VI; `coverage_act ≥ 1−α` remains
**DISPROVED** for `t_{n−2}` and **unproved** (never violated in 417 cells) for `t_min`.

## VIII.2 Experiment E identity check   `NUMERICALLY VERIFIED`

`configs/experiment_E_finite_N_tmin.toml` mirrors `experiment_E_finite_N.toml` with `B9-tmin`
appended and **every locked input byte-identical**: `seed_entropy`, `R`, `alpha_levels`,
`unlabeled_mode`, `coef_mode`, the DGP block and the `(n, ρ, N)` sweep.

> **A seeding subtlety worth recording.** Cell seeds are derived from
> `crc32(experiment_id)` (`config.py:126`), so a *renamed* experiment reseeds every cell. The
> first attempt renamed the experiment to `E_finite_N_tmin` and consequently did **not**
> reproduce the original — differences of `2·10⁻³` in coverage, i.e. independent Monte-Carlo
> draws, not a defect. The config therefore keeps `experiment_id = "E_finite_N"` and protects the
> original by writing to a separate **directory**, `results/gate3/`.

| check | result |
|---|---|
| original Experiment E reproduces with `B9-tmin` added (10 methods × 54 cells × 18 metrics = 9,504) | **max \|diff\| = 0.000e+00**, 0 differing cells |
| `B9-tmin` vs `B9`, coverage | **0.000e+00**, 0/54 differing |
| `B9-tmin` vs `B9`, mean width | **0.000e+00**, 0/54 differing |
| `B9-tmin` vs `B9`, point estimate | **0.000e+00**, 0/54 differing |
| `B9-tmin` vs `B9`, all statistical metrics (918 comparisons) | **0.000e+00**, 0 differing |

Only `seconds_per_interval` differs (`≤ 6·10⁻⁶`), which is wall-clock timing, not a result.

The identity is structural, not luck: Experiment E's grid has `min[(N−1) − (n−2)] = 1`, so
`min(n−2, N−1) = n−2` at every cell. The first check also proves that *adding* the method
perturbed no other method's draws — C1 holds, as `cell_rng`'s docstring claims.

## VIII.3 Recommendation   `PRACTICE`

* **Standard deployed interval** — `t_{n−2}`. Correct whenever the pool is large relative to the
  gold sample, which is the regime PPI exists to serve.
* **Finite-pool safeguard** — `t_{min(n−2, N−1)}`. Use when `N` is finite, and **especially when
  `N < n − 1`**, where `t_{n−2}` is demonstrably anti-conservative.
* It **reduces to the standard interval automatically** when `N ≥ n − 1`, so it can be adopted as
  the default at no cost: there is no grid in this paper on which the two differ.

---

# PART IX — Gate 4: is the safeguard improvable?   (2026-08-25, branch `gate2-boundary`)

`B9-tmin` removes the demonstrated under-coverage but pays in width: over the 70 tested cells with
`N < n−1` its interval is on average **2.48×** the deployed one (median 1.38×, max **6.32×** at
`N = 2`). Gate 4 asks whether a narrower rule can keep the protection.

Config: `configs/experiment_Gate4_optimal_quantile.toml`. Results: `results/gate4_{quantiles,summary}.csv`.
Nothing under `results/A`–`I`, `gate2_*` or `gate3/` is touched.

## IX.1 The criteria, fixed before searching   `DEFINITION`

A successful rule `q` must: **(1)** reduce to `t_{n−2}` when `N ≥ n−1`; **(2)** reduce to `t_{N−1}`
in the `ρ → 1` limit when `N < n−1`; **(3)** not under-cover on the tested region (3 MC s.e.);
**(4)** never be wider than `t_min`; **(5)** preserve large-`N` behaviour.

## IX.2 The optimality of `t_min` among constant critical values   `DERIVED` — **not new**

Conditional on the design, the idealised statistic is Mickey & Brown's
`V_γ = Z/√(γW₁ + (1−γ)W₂)`, `ν₁W₁ ~ χ²_{ν₁}`, `ν₂W₂ ~ χ²_{ν₂}`, with `ν₁ = n−2`, `ν₂ = N−1` and
`γ = v₁/(v₁+v₂)` an **unknown** function of the parameters (§VII.4).

> **Proposition G12.** Let `q > 0` satisfy `P(|V_γ| ≤ q) ≥ 1−α` for **every** `γ ∈ [0,1]`. Then
> `q ≥ t_{min(ν₁,ν₂), 1−α/2}`.
>
> *Proof.* `V_1 = Z/√W₁ ~ t_{ν₁}` and `V_0 = Z/√W₂ ~ t_{ν₂}` exactly. The hypothesis at `γ = 1`
> and `γ = 0` gives `q ≥ t_{ν₁,1−α/2}` and `q ≥ t_{ν₂,1−α/2}`, so
> `q ≥ max(t_{ν₁,1−α/2}, t_{ν₂,1−α/2}) = t_{min(ν₁,ν₂),1−α/2}`, the last step because
> `t_{ν,1−α/2}` is strictly decreasing in `ν`. ∎

With Mickey & Brown's upper direction (`t_min` *is* valid for every `γ`), this makes `t_min`
**exactly the smallest uniformly valid constant critical value** — the lower bound is not merely a
bound, it is **attained**. Verified numerically: over `γ ∈ [0,1]` on a 101-point grid, at six
`(ν₁,ν₂)` pairs, `min_γ coverage = 0.95000` to `2·10⁻⁴` and the minimiser is always an endpoint.

**No novelty is claimed, and Gate 5 confirmed why.** G12 is **Theorem 1 of Wang & Liu (2026)**
(arXiv:2606.07847, primary source read): `t_{1−α/2,m} = inf{d_α : sup_ξ P_ξ(|Z₁(ξ)| > d_α) ≤ α}`
with `m = min(n₁,n₂)−1`. The underlying stochastic domination is **Hsu's (1938)**. G12 is recorded
because it answers Gate 4's Part D, not because it is new — see `docs/PRIOR_ART_GATE5.md`.

## IX.3 The one remaining opening, and why it closes   `NUMERICALLY VERIFIED`

G12 constrains **constant** `q` only. `γ` is partially observable through the realised variance
share, so a *data-dependent* rule is not excluded. The classical construction for exactly this
situation is **Berger & Boos (1994)**: eliminate the nuisance parameter over a confidence set for
it rather than over its whole range. Instantiated here (candidate `C5_bb`):

```
Ŝ = a 1−η confidence set for s = v₂/(v₁+v₂), from V̂₂/V̂₁ ~ (v₂/v₁)·F_{ν₂,ν₁}
q  = clip( max_{s ∈ Ŝ} q_exact(s ; α−η) ,  t_{n−2} ,  t_min )
```
valid because `P(miss) ≤ P(s ∉ Ŝ) + (α−η) = α`. Both clips are deliberate: the lower one enforces
criteria 1 and 5 (without it the rule narrows below the deployed interval where `N ≥ n−1`, which is
what sank `C3`/`C4` in Gate 2); the upper one enforces criterion 4.

**It buys essentially nothing where it matters.**

| grid | rule | cells | min coverage | under-covering | width ÷ B9 | width ÷ `t_min` |
|---|---|---:|---:|---:|---:|---:|
| primary (`ρ ≥ 0.9`) | `B9` | 108 | 0.70643 | **51** | 1.000 | 0.817 |
| primary | `B9-tmin` | 108 | 0.94987 | 0 | 1.708 | 1.0000 |
| primary | `C5_bb` | 108 | 0.94987 | 0 | 1.707 | **0.9999** |
| supplementary (`ρ ∈ {0,0.5}`) | `B9-tmin` | 24 | 0.95256 | 0 | 2.124 | 1.0000 |
| supplementary | `C5_bb` | 24 | 0.95256 | 0 | 1.985 | **0.9516** |

On the difficult grid the saving is **0.01 %**. Across all 70 cells with `N < n−1` it averages
1.68 %, exceeding 1 % at 12 cells and reaching 20 % only at `n = 50, ρ = 0, N = 3`.

**The obstruction is structural, not a defect of the construction.** `t_min` is wide exactly when
`ν₂ = N−1` is small — and that is exactly when the `F_{ν₂,ν₁}` interval for the variance share is
too wide to exclude the binding endpoint `γ = 0`. The regime that needs the improvement is the
regime that cannot supply the information for it. At `ρ ≥ 0.9`, `γ ≈ 0` is moreover the *truth*, so
no valid rule can narrow there at all.

## IX.4 Limit checks   `PROVED` (analytically, not by simulation)

* `N ≥ n−1` ⇒ `ν₂ ≥ ν₁` ⇒ `quantile_berger_boos` returns `t_{n−2}` **by construction**, with no
  data dependence whatsoever (criteria 1, 5). Tested exactly, `atol = 0`.
* `ρ → 1` with `N < n−1` ⇒ `γ → 0`, the confidence set contains 0, the upper clip binds, and
  `q = t_min = t_{N−1}` (criterion 2).
* `t_min` itself: `min(n−2, N−1) = n−2` for `N ≥ n−1`, `= N−1` otherwise — both limits exact.

## IX.5 Prior art   `PARTIAL`

* The `min(f₁,f₂)` lower bound is **Mickey & Brown (1966)** — read in full, already cited.
* **Closed in Gate 5.** Wang & Liu (2026), *Revisiting the Behrens–Fisher Problem: Validity-First
  Optimality*, arXiv:2606.07847 — full text read. Their **Theorem 1** is G12 exactly; their
  **Lemma 2** attributes the domination to **Hsu (1938)**, *Statistical Research Memoirs* 2, 1–24;
  their **Theorem 4** anticipates §IX.3's finding that adaptive rules only redistribute width.
  They call the rule the **Hsu–Scheffé conservative interval** and do not cite Mickey & Brown.
  Details, correspondence and safe wording: `docs/PRIOR_ART_GATE5.md`.
* Berger & Boos (1994) is the standard nuisance-elimination device; `C5_bb` is that device applied
  unchanged, and nothing is claimed for it.

Status after Gate 5: **CLOSED for G12** (primary source read). `PARTIAL` remains only for Hsu
(1938), which was not obtained and is attributed at second hand.

## IX.6 Classification   (Gate-4 Part H)

**C — `t_min` IS ESSENTIALLY OPTIMAL UNDER THE CURRENT ASSUMPTIONS.**

* Exactly optimal among constant critical values (G12), and the bound is attained.
* Data-dependent narrowing is *possible in principle* and was implemented with a classical, valid
  construction — but recovers 0.01 % on the difficult grid and 4.8 % at moderate `ρ`.
* Not **A**: nothing narrower is proved. Not **B**: the candidate is not empirically better where
  it matters. Not **D**: the optimality result is a real answer, and a small saving does exist away
  from the binding regime. Not **E**: no counterexample to `t_min` was found — 132 further cells,
  0 under-coverage.

## IX.7 Open after Gate 4   `OPEN`

1. Whether a rule exploiting information **beyond** the two degrees of freedom and the variance
   share — e.g. the leverage `h`, or `ρ̂` — can narrow validly. G12 does not forbid it; it only
   constrains constants.
2. The `×6.3` width at `N = 2` is intrinsic on this analysis: with one pool degree of freedom the
   interval must be a `t₁` interval. Whether the estimand should be re-scoped to exclude `N ≲ n`
   is a **methodological** question, not a quantile question.
3. `PARTIAL` prior art (§IX.5) must be closed before publication.
