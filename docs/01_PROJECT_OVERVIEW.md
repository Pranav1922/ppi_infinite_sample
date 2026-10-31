# 01 — Project Overview

## Project title

**Finite-Sample-Valid Prediction-Powered Inference: Exact Confidence Intervals for Scientific
Estimation from AI-Generated Labels**

## Domain

Statistical Machine Learning / Statistical Inference / Trustworthy AI-Assisted Measurement.

Secondary contact areas: LLM-as-annotator evaluation, computational social science, computational
biology, benchmark estimation, survey methodology.

## One-line summary

PPI++ confidence intervals appear to under-cover when the gold-labelled sample is small — the exact
regime PPI exists to serve — apparently because their variance expression omits a regression
leverage/extrapolation term induced by estimating the tuning coefficient from that same small
sample; this project derives, proves where possible, and empirically validates a corrected interval
that attains nominal coverage at finite sample size while retaining the efficiency benefit when the
predictor is good.

## Problem

A researcher wants to estimate a scientific quantity θ (e.g. a population mean, a prevalence, a
regression coefficient) over a large population. Obtaining gold labels is expensive — a human
annotator, a wet-lab assay, a clinician read. Obtaining model predictions is cheap — an LLM, a
classifier, a structure predictor.

The standard pattern is therefore:

- a **large** set of N items with model predictions `f` only, and
- a **small** set of n items with both a gold label `Y` and a prediction `f`.

Prediction-Powered Inference (PPI) is the standard method for turning this into a valid confidence
interval for θ. PPI++ refines it with a tuning coefficient λ that interpolates between "ignore the
model" (λ = 0, classical inference) and "use the model fully" (λ = 1), fitted from the gold labels.
PPI++ is presented as being essentially never worse than classical inference.

The practical difficulty: n is typically small — often 20 to 200, sometimes fewer — precisely
because human labelling is the bottleneck the method is meant to relieve. The guarantee is
asymptotic in n.

## Proposed contribution

Two coupled objects, both estimators. Neither is a controller, gate, wrapper, or fallback rule.

**(a) A leverage-corrected variance and interval.** In the limit N → ∞ (population prediction mean
μ_f known), the PPI++ point estimate coincides algebraically with the ordinary-least-squares
regression-imputation estimator: the fitted value of the regression of Y on f, evaluated at
f = μ_f. Under a bivariate-Gaussian working model that estimator has an *exact* finite-sample
pivot with a leverage term. The PPI++ variance expression retains only the `1/n` part and drops the
leverage part. Restoring it, with a `t_{n−2}` quantile, is the proposed correction.

**(b) A finite-N generalisation, and a shrunk tuning coefficient λ̃.** At finite N the unlabelled
mean f̄_N is itself random and enters multiplied by an estimated coefficient. The naive plug-in
extension of (a) is **not** exact — it over-covers. Deriving the correct finite-N pivot is the main
open theoretical work. Separately, a continuous positive-part shrinkage
λ̃ = λ̂ (1 − c/t²)₊ is retained as a secondary object, now aimed at interval *width* rather than
validity.

## Why it matters

Under-coverage is the dangerous direction of error. An interval that is too **wide** makes a
researcher under-claim; an interval that is too **narrow** makes them over-claim, and nothing in the
pipeline signals that it has happened. No exception is raised, no diagnostic fires, the number
simply looks better than it is.

PPI is being adopted in applied work through a reference implementation (`ppi-py`). If the
under-coverage is real and the correction is a closed-form drop-in, the cost of adoption is close to
zero and the benefit is a class of over-confident published intervals that stop being produced.

Equally: if the correction turns out to be already known in the survey-sampling literature, or a
bootstrap turns out to solve it just as well, that is a useful, publishable negative finding and the
project should say so rather than manufacture novelty. See `10_HYPOTHESES_AND_KILL_CRITERIA.md`.

## Current project status

### Already demonstrated

Produced by Monte-Carlo simulation during project selection. **Preliminary — must be independently
reproduced (see `09_REPRODUCTION_PROTOCOL.md`).**

- Under a bivariate-Gaussian working model with N = 10⁶ unlabelled items, the standard PPI++
  interval (plug-in λ̂, normal quantile) covered at **0.8784 / 0.8791 / 0.8787** at n = 10 for
  ρ = 0 / 0.5 / 0.9 against a nominal 0.95. The classical *t* interval covered at 0.9496 / 0.9490 /
  0.9511 in the same draws.
- The deficit shrank with n: ≈ 0.9194–0.9198 at n = 20, ≈ 0.9384–0.9394 at n = 50.
- Three simple repairs were measured and **none** restored nominal coverage uniformly:
  replacing the normal quantile by `t_{n−2}` (≈ 0.9234–0.9249 at n = 10); cross-fitting λ̂
  (≈ 0.9245–0.9261 at n = 10); and a hard significance threshold on λ̂ (0.9409 at ρ = 0 falling to
  0.9194 at ρ = 0.8 — non-monotone in predictor quality).
- Cross-fitting also *increased* point-estimate MSE substantially at small n (up to ≈ 1.48× the
  classical MSE at n = 10).
- The proposed leverage-corrected interval covered at **0.9492–0.9501** in every (n, ρ) cell
  tested at N = 10⁶, while remaining efficient (e.g. 0.471× the classical width at n = 10, ρ = 0.9).
- At N = 200 the same construction **over-covers**, and the over-coverage grows with ρ
  (0.9513 at ρ = 0 → 0.9527 at ρ = 0.5 → 0.9612 at ρ = 0.9, at n = 10).

### Strongly supported (analysis exists; not independently verified, not formally proved)

- **The algebraic identity.** With N → ∞ and μ_f known, the PPI++ point estimate
  θ̂ = Ȳ − λ̂(f̄_n − μ_f) with λ̂ equal to the OLS slope is exactly the OLS fitted value at f = μ_f.
  This is elementary algebra and is expected to hold, but it has not been written out formally.
- **The diagnosis.** The coverage deficit is attributable to the omitted leverage term
  (f̄_n − μ_f)² / S_ff. Support: restoring exactly this term moved coverage from 0.8784 to 0.9499 in
  the ρ = 0 cell, and the intermediate "no-leverage but *t*-quantile" control sat at 0.9369 —
  i.e. the *t* quantile accounts for part of the deficit and the leverage term for the remainder.
- **The finite-N diagnosis (conjecture).** The over-coverage at small N grows monotonically with ρ,
  and vanishes at ρ = 0. This is consistent with the estimator being the sum of a `t_{n−2}`-pivotal
  regression component and an approximately normal component β(f̄_N − μ_f) whose scale is
  proportional to |β| ∝ ρ; applying a `t_{n−2}` quantile to the combined standard error is then
  conservative. **OPEN / TO BE VERIFIED** — this explanation is a conjecture consistent with the
  observed pattern, not a derivation.

### Open

- A formal proof of exact finite-sample coverage in the Gaussian, N → ∞ case.
- The correct finite-N pivot (or a proof of conservativeness with a stated bound).
- Behaviour under heavy-tailed, skewed, and heteroskedastic errors — entirely untested.
- Comparison against percentile and BCa bootstrap intervals — **not yet run**, and the strongest
  remaining simple control.
- Whether the shrunk λ̃ is needed at all once the leverage correction is in place.
- Extension to M-estimation (logistic regression coefficients, quantiles, GLM contrasts).
- **Prior art:** whether the model-assisted survey-sampling / GREG literature already contains an
  equivalent finite-sample-exact interval. This is the single largest risk to the project's
  contribution and is scheduled first.
- Any real-data confirmation. Nothing in this package has been run on real AI-annotation data.

## Computational profile

- **Critical path: CPU only.** All core theory and simulation work is NumPy/SciPy on a laptop.
- Typical Monte-Carlo cell: 10⁵–2×10⁵ replications, vectorised, seconds to a couple of minutes.
- **No GPU is required at any stage.** If a Tier-3 real-data case study involves running an LLM
  annotator, use cached or pre-existing model outputs; generating them is not on the critical path.
- Memory: keep unlabelled data out of memory by drawing the unlabelled *mean* directly where the
  simulation permits (see `09_REPRODUCTION_PROTOCOL.md` §Random number generation).
