# 02 — Research Problem and Gap

## 2.1 Real-world problem

A recurring measurement pattern has emerged across empirical research since roughly 2023:

| Component | Size | Cost | Examples |
|---|---|---|---|
| Model-generated labels | large (N) | ≈ free | LLM classification of documents, predicted protein properties, model-read images, LLM-as-judge scores |
| Human/gold labels | small (n) | expensive | trained annotators, wet-lab assays, clinician reads, expert adjudication |

The scientific target is a population quantity θ — a mean, a prevalence, a treatment contrast, a
regression coefficient — defined with respect to the **gold** label, not the model's label. The
model is a cheap proxy, and it is biased in unknown ways.

Two naive strategies both fail:

1. **Use only the gold labels.** Valid, but wastes the N model predictions and produces a wide
   interval. With n = 30 this is often too wide to support any conclusion.
2. **Treat model predictions as if they were labels.** Produces a very tight interval around the
   *wrong* quantity. The bias does not shrink with N.

The gold set is small *by construction*: the entire reason to use a model is that human labelling is
the binding constraint. Any method for this setting must therefore behave correctly at small n, not
merely in the limit.

## 2.2 Existing solution — PPI

Prediction-Powered Inference (Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, *Science* 2023)
resolves the dilemma with a **rectifier**. In its simplest form, for estimating θ = E[Y]:

```
θ̂_PPI = (mean of f over the N unlabelled items)  +  (mean of (Y − f) over the n gold items)
```

The first term uses all N predictions; the second term is a bias correction ("rectifier") estimated
from the n gold pairs. The estimator is unbiased for θ regardless of how bad the model is, because
the rectifier removes whatever systematic error the model has, on average.

Its confidence interval is built from the asymptotic variance of the two components. The gain over
classical inference comes from the rectifier `Y − f` having smaller variance than `Y` when the model
is good.

**Failure mode of plain PPI:** when the model is *bad*, `Var(Y − f)` can exceed `Var(Y)`, so PPI's
interval is **wider** than the classical one. Using the model actively hurts.

## 2.3 PPI++

PPI++ (Angelopoulos, Duchi, Zrnic, 2023) fixes that failure mode by introducing a tuning
coefficient λ ∈ ℝ:

```
θ̂_λ = mean over gold of (Y − λ f)  +  λ × (mean of f over the N unlabelled items)
```

- λ = 0 recovers the classical estimator (gold labels only).
- λ = 1 recovers plain PPI.
- The variance-minimising λ* is a covariance ratio.

Because λ* is unknown, PPI++ substitutes a plug-in estimate λ̂ computed from the **same n gold
pairs**. The paper's headline claim is that the tuned method is essentially never worse than either
classical or plain PPI.

That claim is **asymptotic in n**. λ̂ is treated as if it were the fixed constant λ*, and the
reported variance is the variance one would have if λ were known.

## 2.4 Failure mode — small-n under-coverage

This is the phenomenon the project investigates.

**Documented in the literature.** Mani, Xu, Lipton and Oberst (*No Free Lunch: Non-Asymptotic
Analysis of Prediction-Powered Inference*, arXiv:2505.20178) show non-asymptotically that the PPI++
safety claim fails at finite n. In a Gaussian setting they derive a correlation threshold — PPI++
improves on classical inference only when the correlation between prediction and label exceeds
approximately `1/√(n−2)` — and they show that the single-sample version (λ̂ and the mean estimated
on the same data) is biased with an optimistic variance estimate. They analyse a cross-fit variant
that removes the bias at the cost of sample size. **They do not propose a corrected confidence
interval.**

**Measured during project selection.** A Monte-Carlo study under a bivariate-Gaussian working model
(unit variances, correlation ρ, N = 10⁶ unlabelled, 200,000 replications, nominal 95%) produced:

| n | ρ | Classical *t* | **PPI++ (as shipped)** | Leverage-corrected |
|---|---|---|---|---|
| 10 | 0.0 | 0.9496 | **0.8784** | 0.9499 |
| 10 | 0.5 | 0.9490 | **0.8791** | 0.9492 |
| 10 | 0.9 | 0.9511 | **0.8787** | 0.9496 |
| 20 | 0.0 | 0.9489 | 0.9194 | 0.9492 |
| 20 | 0.5 | 0.9502 | 0.9198 | 0.9497 |
| 50 | 0.5 | 0.9497 | 0.9384 | 0.9494 |
| 50 | 0.9 | 0.9501 | 0.9394 | 0.9500 |

> **Existing preliminary results — must be independently reproduced by the implementing agent.**
> They are not independently verified.

**Scope restrictions that must be respected when citing these numbers.** They were obtained under:
a bivariate-Gaussian joint distribution of (Y, f) with unit variances; estimation of a **mean**;
a very large unlabelled pool (N = 10⁶); homoskedastic, symmetric errors; and no model
misspecification. They say nothing yet about heavy tails, skew, heteroskedasticity, M-estimation,
small N, or real data. **Do not generalise them beyond the tested setting.**

Two structural features of the observed deficit are worth flagging because they discipline the
hypothesis:

1. **It is approximately invariant in ρ.** At n = 10 the coverage was ≈ 0.878 whether the predictor
   was useless (ρ = 0) or excellent (ρ = 0.9). Hence it is *not* a "bad predictor" problem. It
   behaves like a fixed cost of having estimated one extra parameter.
2. **It decays with n** roughly like a two-degrees-of-freedom effect: 0.878 → 0.919 → 0.939 → 0.946
   at n = 10, 20, 50, 100.

## 2.5 Research gap — stated precisely

> **Gap.** No confidence interval in the prediction-powered-inference literature is known to attain
> nominal coverage at finite gold-sample size. The PPI++ interval is justified asymptotically; the
> paper that documents its finite-sample failure characterises the failure but does not repair it;
> and the three obvious repairs (Student-*t* quantile, cross-fitting, thresholding the tuning
> coefficient) were measured here and each closes only part of the deficit, none uniformly.

The project's proposed repair, and the specific claims to be established or refuted:

- **G1.** Identify the omitted quantity as a regression **leverage / extrapolation** term arising
  because λ̂ is estimated from the same gold sample.
- **G2.** Prove exact finite-sample coverage of the corrected interval in the Gaussian, N → ∞ case.
- **G3.** Derive the correct finite-N pivot, or prove conservativeness with a stated bound.
- **G4.** Characterise the **validity–efficiency frontier**: how much wider a *valid* interval must
  be than the (invalid) PPI++ interval, and than the classical interval, as a function of n and ρ.
- **G5.** Establish the range of distributional assumptions under which the correction remains at
  least conservative.

**What would close the gap without us.** If the model-assisted survey-sampling literature (GREG,
difference estimator, g-weights) already provides this interval, G1–G2 collapse into a translation
exercise and the project must narrow to G3–G5. This check is scheduled first; see
`03_LITERATURE_AND_PRIOR_WORK.md` and `10_HYPOTHESES_AND_KILL_CRITERIA.md` (K1).

## 2.6 Why the gap matters

A confidence interval is a **contract**: "if you repeat this study, 95% of the intervals you build
this way will contain the truth." A method whose intervals cover 88% of the time breaks that
contract silently.

Concretely, at n = 10 and nominal 95%, an empirical coverage of 0.878 means the effective error rate
is 12.2% rather than 5% — roughly **2.4× the advertised false-positive rate**. In a literature where
many such studies are run, that is a systematic inflation of apparent findings.

Three aggravating features:

1. **Invisible.** Nothing fails. There is no diagnostic, no warning, no residual plot that flags it.
2. **Concentrated where the method is used.** The deficit is largest at small n, and small n is the
   entire motivation for prediction-powered inference.
3. **Adoption is under way.** The reference implementation is public and in applied use, so the fix
   is a drop-in replacement rather than a call for a new methodology.

Symmetrically, the honest accounting matters too: a *valid* interval is necessarily **wider** than
the invalid one. In the preliminary study the corrected interval was 1.081× the classical width at
n = 10, ρ = 0 — i.e. when the predictor is useless, honest use of the model costs about 8% extra
width relative to ignoring it. Quantifying that price is part of the contribution, not a defect to
be hidden.

## 2.7 Scope

**In scope.**

- Finite-sample validity of interval estimates in the PPI setting.
- The mean estimand first; M-estimation as an extension if the mean case resolves.
- Analytic corrections, resampling comparisons, and Monte-Carlo validation.
- Honest characterisation of what validity costs in width.

**Explicitly out of scope.**

- Improving the predictor `f`. The predictor is an input, taken as given, of arbitrary quality.
  Nothing in this project trains or fine-tunes a model to be more accurate.
- Deep learning of any kind on the critical path. If a model is needed to *play the role of* an AI
  annotator in an experiment, use the simplest available (logistic regression, TF-IDF + linear,
  gradient boosting, or cached third-party outputs).
- Reinforcement learning. Not applicable to this problem and must not be introduced.
- Active-learning / labelling-budget allocation. Adjacent and interesting, but a different project.
- Calibration of the predictor. Handled elsewhere in the literature (see `03`, Calibeating PPI) and
  orthogonal to finite-sample validity.

**One-sentence scope statement to keep in view:** *this is a project about getting the second decimal
place of a coverage probability right at n = 10, and about proving why.*
