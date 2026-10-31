# 03 — Literature and Prior Work

## Citation discipline (read before using anything below)

Every entry carries a **verification status**:

| Status | Meaning |
|---|---|
| `VERIFIED` | The record was retrieved directly during project selection; title, identifier and substantive content were read. |
| `PARTIAL` | Identifier and title retrieved; substantive content summarised from an abstract or an intermediary rather than the full text. |
| `UNVERIFIED` | Reported during the search but not directly retrieved. **Must be checked before citing in any write-up.** |
| `CLASSICAL` | Standard textbook/monograph reference. Bibliographic details are well known but the specific page/section relevant here has not been located. |

Do not upgrade a status without actually retrieving the source. Do not invent volume numbers, page
numbers, DOIs, or venue names. If a field is unknown, write `OPEN — bibliographic detail unverified`.

---

## PPI — the base method

**Prediction-Powered Inference**
Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, Tijana Zrnic.
*Science*, 2023. DOI `10.1126/science.adi6000`. Preprint arXiv:2301.09633.
Status: `VERIFIED` (identifier and abstract retrieved; DOI confirmed).

- **Main idea.** Combine a large unlabelled set carrying model predictions with a small gold set, via
  a *rectifier* that estimates and removes the model's systematic error. Yields valid confidence
  intervals for a range of estimands (means, quantiles, GLM coefficients) without assuming the model
  is good.
- **Assumptions.** Gold pairs i.i.d. from the population; unlabelled units i.i.d. from the same
  population; asymptotic normality of the relevant averages; the predictor is a fixed function
  (i.e. not fitted on the gold set being used for inference).
- **Limitations relevant here.** (i) The interval can be **wider** than classical inference when the
  predictor is poor — the motivation for PPI++. (ii) Validity is asymptotic in n.
- **Solves:** debiasing; using unlabelled data legitimately.
- **Does NOT solve:** finite-sample coverage; the "worse than classical" regime.

**Reference implementation:** `ppi-py`, documented at
<https://ppi-py.readthedocs.io/en/stable/ppi.html>. Status: `VERIFIED` (documentation page located).
Relevance: this is what applied users actually run, so any correction should be expressible as a
drop-in change to the same interface.
**OPEN — TO BE VERIFIED:** the exact variance expression and quantile (normal vs Student) used in
the shipped code path must be read from source before the project claims what "as shipped" does.
The preliminary study assumed a normal quantile with the asymptotic variance and no leverage term.

---

## PPI++ — the tuned method under investigation

**PPI++: Efficient Prediction-Powered Inference**
Anastasios N. Angelopoulos, John C. Duchi, Tijana Zrnic. 2023. arXiv:2311.01453.
Status: `VERIFIED` (identifier and abstract retrieved).

- **Main idea.** Introduce a power-tuning coefficient λ so that the estimator interpolates between
  classical inference (λ = 0) and PPI (λ = 1); choose λ to minimise asymptotic variance.
- **Tuning coefficient.** The variance-minimising value is a covariance ratio; the method substitutes
  a plug-in estimate computed from the gold sample. See `04_MATHEMATICAL_FORMULATION.md` §4.4 for the
  exact form used in this project.
- **Confidence interval.** Wald-type, built from the asymptotic variance of θ̂_λ treating λ as if it
  were the known optimum.
- **Stated claim.** That the tuned procedure is essentially never worse than either classical or
  prediction-powered inference. **This claim is the object of the project.** It is asymptotic.
- **Limitations relevant here.** The variance expression conditions on λ being known. Nothing in the
  interval accounts for λ̂ having been estimated on the same n points as the mean.
- **Solves:** the "worse than classical" failure of plain PPI, asymptotically.
- **Does NOT solve:** finite-sample coverage; the cost of estimating λ̂.

---

## No Free Lunch — the paper that documents the failure

**No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference**
Pranav Mani, Peng Xu, Zachary C. Lipton, Michael Oberst.
arXiv:2505.20178. v1 26 May 2025; v2 24 June 2026.
Venue: **OPEN — bibliographic detail unverified.** One search result during project selection
suggested ICML 2026; the arXiv landing page retrieved did **not** list a peer-reviewed venue. Treat
as a preprint until confirmed.
Status: `VERIFIED` for content of the retrieved sections; `PARTIAL` for the full proofs.

**What it found, and why it matters here:**

1. It analyses PPI/PPI++ **non-asymptotically** rather than asymptotically, which is what makes the
   small-n behaviour visible.
2. It establishes a **correlation threshold**: in a Gaussian setting, PPI++ beats classical inference
   only when the correlation between prediction and label is at least approximately `1/√(n−2)`.
   Below that, using the model is worse than ignoring it.
3. It identifies the mechanism explicitly. Quoted from the retrieved text:
   > "the covariance in particular needs to be estimated using labeled data: Notably, the true labels
   > Y are used for both (a) estimating the mean, and (b) estimating the quality of the
   > pseudo-labels."
4. It distinguishes **single-sample PPI++** (λ̂ and the mean from the same n points), which it shows
   to be biased with an **optimistic variance estimate**, from **cross-fit PPI++** (fold-split λ̂),
   which removes the bias but sacrifices effective sample size.

**Critically, what it does NOT do** (confirmed by direct retrieval of the discussion sections):

- It does **not** propose a corrected, shrunk, or finite-sample-safe tuning rule.
- It does **not** propose a corrected confidence interval.
- It analyses **mean estimation only** — no M-estimation, no logistic regression, no quantiles.
- It has no explicit future-work section.

This is the single most important reference for the project: it establishes the failure
independently (its authors have no stake in the fix) and leaves the repair undone.

---

## Adjacent literature

### Adaptive shrinkage

**Prediction-Powered Adaptive Shrinkage Estimation**
Sida Li, Nikolaos Ignatiadis. arXiv:2502.14166. Reported as ICML 2025.
Status: `PARTIAL` (identifier and PDF located; venue reported during search, not independently
confirmed → treat venue as `UNVERIFIED`).

- **Relevance.** It is the nearest thing to "shrinkage in PPI", so the project must distinguish
  itself from it clearly.
- **What it solves.** Shrinkage **across many parallel estimation problems** — an empirical-Bayes
  setting where information is borrowed between tasks.
- **What it does NOT solve.** The single-problem, small-n regime. Its shrinkage direction and its
  justification are cross-problem; it does not address the within-problem cost of estimating λ̂, and
  it does not give a finite-sample-valid interval for one estimand.

### PPI power analysis

**Power Analysis for Prediction-Powered Inference**
Yiqun T. Chen, Moran Guo, Shengyi Li (Johns Hopkins University). arXiv:2603.16041, 17 March 2026.
Software: R package `pppower`, <https://github.com/yiqunchen/pppower>.
Status: `VERIFIED` (content retrieved).

- **What it solves.** Closed-form power and sample-size formulas for PPI/PPI++ across one-sample
  means, two-sample comparisons, paired designs, 2×2 tables, and GLM regression contrasts. Rule of
  thumb: labelled-sample savings scale roughly with the R² between predictions and gold labels.
- **What it does NOT solve.** It **assumes the oracle λ\* is known**, and it does not address the
  small-n regime where PPI++ underperforms classical inference. It is therefore complementary, not
  competing: our correction changes the variance it would plug in.

### Diagnostics and review

**Demystifying Prediction Powered Inference**
Yilin Song (Columbia Mailman), Dan M. Kluger (MIT), Harsh Parikh (Yale, corresponding),
Tian Gu (Columbia, corresponding). arXiv:2601.20819.
Date: the arXiv identifier implies January 2026; an intermediary reported "January 2025".
**OPEN — bibliographic detail unverified; resolve the date from the arXiv record.**
Status: `PARTIAL`.

- **Relevance.** A diagnostic/review treatment of PPI. Reports that "double-dipping, i.e. reusing
  training data for inference, leads to anti-conservative confidence intervals and coverages"
  (Section 5.2). Note this refers to reusing **predictor-training** data, which is a *different*
  mechanism from the λ̂-estimation effect studied here — do not conflate the two.
- Section 3.2 is titled to include "Control variates, survey sampling approaches, and finite-sample
  limits", which makes this paper a useful entry point into the GREG prior-art check below.
- **What it does NOT solve.** It proposes no finite-sample correction.

### Calibration

**Calibeating PPI** (isotonic post-calibration of the predictor within PPI)
Reported as: van der Laan & van der Laan, arXiv:2604.21260, April 2026.
Status: `UNVERIFIED` — reported during the search but not directly retrieved. **Check before
citing.**

- **Reported relevance.** Addresses predictor **miscalibration** via isotonic recalibration of the
  score, with a first-order optimality claim.
- **Why it does not close our gap (if the report is accurate).** Miscalibration and finite-sample
  validity are different problems. Moreover isotonic regression fitted on a tiny gold set is itself
  high-variance, so this approach would plausibly *worsen* the small-n behaviour rather than fix it —
  a point worth testing empirically if time allows.

### Cross-prediction

**Cross-prediction-powered inference** — reported in the search results as an existing work
(a ResearchGate record was surfaced). Commonly attributed to Zrnic and Candès.
Status: `UNVERIFIED` — bibliographic details, venue and year were **not** confirmed. Do not cite
until retrieved.

- **Why it is on the list.** It concerns using cross-fitting to allow the predictor itself to be
  trained on the gold data. Relevant because our B5 baseline is a cross-fit variant, and because the
  variance accounting for cross-fitting is closely related.

### Model-assisted estimation, regression estimators, GREG, finite-population inference

**Model Assisted Survey Sampling** — Carl-Erik Särndal, Bengt Swensson, Jan Wretman.
Springer, 1992. Status: `CLASSICAL`.

**Sampling Techniques**, 3rd edition — William G. Cochran. Wiley, 1977. Status: `CLASSICAL`.
Relevant chapters: the **ratio estimator** and the **regression (difference) estimator**, including
its variance when the regression coefficient is estimated from the sample.

**On the Bias of the Generalized Regression Estimator in Survey Sampling** — *Acta Applicandae
Mathematicae*, Springer. Status: `UNVERIFIED` (a link was surfaced; authors and year not recorded).

**Software:** the R package `mase` implements a generalized regression estimator (`greg`).
Status: `PARTIAL` (documentation pages located).

**Why this cluster is central:** the PPI estimator is, structurally, a **difference/regression
estimator** — the classical survey-sampling device of adjusting a sample mean using an auxiliary
variable whose population mean is known. In that literature:

- the estimator `ȳ + b(X̄ − x̄)` with `b` an estimated regression coefficient is standard;
- its variance **including** the effect of estimating `b` has been studied since at least Cochran;
- "g-weights" in the GREG framework exist partly to improve finite-sample behaviour.

### Bootstrap confidence intervals

**Bootstrap methods** — Bradley Efron; and Efron & Robert Tibshirani, *An Introduction to the
Bootstrap*, Chapman & Hall, 1993, for the BCa (bias-corrected and accelerated) interval.
Status: `CLASSICAL`.

Relevance: the bootstrap is the **strongest simple control that has not yet been run**. If a BCa
interval attains nominal coverage at n = 10 across all ρ, the project's headline claim weakens from
"the only known fix" to "an exact, closed-form, and far cheaper fix". This is kill criterion K3
(`10_HYPOTHESES_AND_KILL_CRITERIA.md`).

### Non-asymptotic inference and exact normal-theory results

The exactness argument for the proposed correction rests on standard normal-theory linear-regression
distribution theory: the studentised fitted value of an OLS regression at a fixed design point has
an exact Student-*t* distribution with `n − 2` degrees of freedom. Status: `CLASSICAL` (any
graduate linear-models text; e.g. Seber & Lee, *Linear Regression Analysis*). The **novel** step, if
it is novel, is the recognition that the PPI++ estimator *is* such a fitted value and therefore
inherits an exact pivot.

---

## IMPORTANT PRIOR-ART WARNING

> **The GREG / model-assisted survey-sampling literature is a major unresolved prior-art check.**

The proposed correction restores a term of the form `(μ_f − f̄_n)² / S_ff` to the variance of an
estimator of the form `Ȳ + b(μ_f − f̄_n)`. That estimator is the **regression estimator** of classical
survey sampling, and the effect of estimating `b` on its variance has been studied there for
decades. It is entirely possible that:

- the exact finite-sample variance is already in Cochran or Särndal–Swensson–Wretman;
- the corresponding interval, or a g-weighted analogue, is already standard practice in that field;
- the PPI literature simply rediscovered the difference estimator without importing its
  finite-sample theory.

**Consequences for the project's claims.** Until this literature is mapped:

1. **Do not claim novelty of the correction itself.** The defensible framing at present is: *"the
   prediction-powered-inference literature ships an interval that under-covers at small n; the fix
   is available by recognising the estimator as a regression estimator, which the survey-sampling
   literature has long treated exactly."* That is a useful contribution — a translation plus a
   demonstration — but it is not the same claim as inventing the correction.
2. **The genuinely open pieces are less likely to be pre-empted**, because they are specific to the
   PPI setting rather than the survey setting: the **finite-N** case (in survey sampling the frame
   is usually a fixed finite population with the auxiliary mean known **exactly**, whereas in PPI
   the unlabelled pool is a *random sample* and `f̄_N` is itself an estimate), the validity–efficiency
   frontier, and the M-estimation extension.
3. **Schedule this check first.** See `10_HYPOTHESES_AND_KILL_CRITERIA.md` K1 and
   `12_CLAUDE_CODE_HANDOFF.md` Stage 0.

**Concrete search targets for the check** (to be executed and logged in `13_RESEARCH_LOG.md`):

- Cochran (1977), regression estimator: variance with estimated `b`; the small-sample bias term.
- Särndal, Swensson & Wretman (1992): GREG variance estimators, g-weights, and any finite-sample
  exactness results.
- Any survey-sampling result giving an **exact** (not asymptotic) interval for the regression
  estimator under a normal superpopulation model.
- The literature on **control variates** in Monte-Carlo simulation, where the same
  "estimated-coefficient" penalty is classical: the variance of a control-variate estimator with an
  estimated coefficient carries a known finite-sample inflation factor. This is arguably the closest
  prior art of all and **must** be checked. Search terms: "control variate", "estimated control
  coefficient", "loss factor", "(n−2)/(n−k−2)".

---

## Summary table: what is already solved, and by whom

| Question | Solved? | By |
|---|---|---|
| Debias model predictions using a small gold set | Yes | PPI (2023) |
| Avoid being worse than classical, asymptotically | Yes | PPI++ (2023) |
| Show that PPI++ fails at finite n | Yes | Mani, Xu, Lipton, Oberst (arXiv:2505.20178) |
| Sample-size planning for PPI | Yes, given oracle λ* | Chen, Guo, Li (arXiv:2603.16041) |
| Shrinkage across many PPI problems | Yes | Li & Ignatiadis (arXiv:2502.14166) |
| Fix predictor miscalibration inside PPI | Reported yes | Calibeating PPI (`UNVERIFIED`) |
| **A finite-sample-valid PPI confidence interval** | **Not found in the PPI literature** | — |
| The same, in survey-sampling language | **OPEN — unresolved prior-art check** | possibly Cochran / SSW / control-variate literature |
| Exact interval when the unlabelled mean is itself estimated (finite N) | **Open** | — |
