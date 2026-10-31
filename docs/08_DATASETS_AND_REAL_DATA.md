# 08 — Datasets and Real Data

Three tiers, deliberately ordered so that the **theoretical contribution never depends on a
download**. Tier 1 alone can carry Experiments A–H. Tier 2 makes the story concrete. Tier 3 is
confirmation.

---

## Tier 1 — Synthetic (the core; zero external dependencies)

Everything in Experiments A–H is generated in NumPy. No files, no downloads, no network.

### T1.1 — Bivariate Gaussian (primary DGP)

```python
# rng: numpy.random.Generator
f   = rng.standard_normal((R, n))                       # predictions on the labelled sample
eps = rng.standard_normal((R, n))
Y   = rho * f + np.sqrt(1 - rho**2) * eps               # gold labels
```

Properties: `Var(Y) = Var(f) = 1`, `Corr(Y, f) = rho`, `E[Y] = 0`, so the estimand is `θ = 0` and
coverage is simply the indicator that the interval contains 0.

**Unlabelled pool.** Two modes, and the choice matters:

- **Mean-only mode (memory-efficient, used in the preliminary study).** Only `f̄_N` and `σ_f²` enter
  the estimators, and under T1.1 `f̄_N ~ N(0, 1/N)` exactly. So draw
  `fbar_N = rng.standard_normal(R) / sqrt(N)` and set `σ̂_f² = 1`. This makes `N = 10⁶` free.
- **Explicit mode (required for B7/B8 if the bootstrap resamples the pool, and for any experiment
  where `σ̂_f²` must be *estimated*).** Materialise `f_unlabeled` of shape `(R, N)`. Only feasible
  for small `N`; chunk over replications.

> **Consistency requirement.** Mean-only mode silently treats `σ_f²` as known. At large N that is
> harmless; at `N = 50` it is not. **Experiment E must use explicit mode for `N ≤ 5000`** and must
> report whether the mode changes the answer. Log which mode each result used in the CSV.

### T1.2 — Heavy-tailed variants (Experiment F)

Residual `ε` drawn from a standardised Student-*t* with `ν ∈ {3, 5, 10, 30}`:
`eps = rng.standard_t(nu, size) / sqrt(nu/(nu-2))`. A second variant draws **`f`** from the same
heavy-tailed family, which stresses the leverage term directly.

### T1.3 — Skewed and discrete variants (Experiment G)

- centred/scaled `Exponential(1)`;
- centred/scaled `Gamma(k, 1)`, `k ∈ {1, 2, 5}`;
- lognormal, mild and severe;
- **binary gold label:** `Y ~ Bernoulli(sigmoid(a + b·f))`, with `f ~ N(0,1)` and `(a, b)` chosen to
  hit a target prevalence and a target correlation. This is the most practically important synthetic
  case, because in LLM-annotation work the gold label is usually a binary human judgement.

### T1.4 — Heteroskedastic variants (Experiment H)

`Y = ρf + √(1−ρ²)·σ(f)·ε/c` with `σ(f) = exp(γf)`, `γ ∈ {0, 0.25, 0.5, 1}`, and a two-group step
variant. `c` normalises `Var(Y) = 1` so that width comparisons stay interpretable.

### T1.5 — Misspecification variant (optional, valuable)

`Y = ρ·g(f) + noise` with `g` nonlinear (e.g. `g(f) = f + 0.3f²`). The linear-projection argument of
`04` §4.7 still makes `θ = α + βμ_f` exact, so the point estimator remains valid — this experiment
checks whether the *interval* survives when the conditional mean is not linear. Not required, but it
is the cheapest way to test a referee's most likely objection.

---

## Tier 2 — Semi-real (sklearn-bundled; no network required)

**Purpose.** Replace the synthetic `f` with an actual fitted model's predictions, on real feature
distributions, while keeping the "population" fully observed so that θ is known exactly.

**Construction recipe (identical for every dataset):**

1. Load the dataset; call the full set of rows the **population**. Compute `θ = mean(Y)` over it.
   This is the ground truth.
2. Split the population into a **predictor-fitting** part and an **inference** part.
   **Fit the model only on the predictor-fitting part.** This enforces assumption A5 (`04` §4.10) —
   the predictor must not be fitted on the data used for inference. Violating this is the single
   most common way to produce meaningless results here.
3. Generate `f` for every row of the inference part.
4. Subsample n rows as the gold set, a disjoint N rows as the unlabelled pool; run all baselines;
   check coverage against θ. Repeat `R` times.

**Deliberately weak predictors are wanted.** The interesting regime is `ρ ∈ [0.2, 0.9]`. Use a
small/underfit model (logistic regression on a subset of features, a depth-2 tree, TF-IDF + linear)
and tune the predictor's *badness* to sweep ρ. Do **not** use a strong model — a predictor with
ρ = 0.99 makes every method look fine.

**Candidate datasets, all bundled with scikit-learn (no download):**

| Dataset | Loader | Gold label `Y` | Predictor `f` | Why it is useful |
|---|---|---|---|---|
| California housing | `fetch_california_housing` | median house value (continuous) | ridge / small GBM on a feature subset | continuous, skewed target — stresses G |
| Diabetes | `load_diabetes` | disease progression (continuous) | ridge on a feature subset | very small (n ≈ 442) — realistic population size |
| Breast cancer | `load_breast_cancer` | malignancy (binary) | logistic on 3 features | binary target, the practically important case |
| Digits | `load_digits` | "is the digit a 3" (binary) | logistic on downsampled pixels | binary, controllable difficulty |
| Wine | `load_wine` | class indicator (binary) | logistic | tiny; stresses very small n |
| 20 newsgroups | `fetch_20newsgroups` | topic indicator (binary) | TF-IDF + linear | text, closest in spirit to LLM annotation |

> **Network caveat.** `fetch_california_housing` and `fetch_20newsgroups` download on first use;
> `load_*` are bundled and never touch the network. In the environment where the preliminary work
> was done, **external dataset downloads were blocked**, including `fetch_openml` and several
> `fetch_*` loaders. The implementation must therefore: try the fetcher, catch the failure, log it,
> and continue with the bundled loaders. **A blocked download is never a reason to stop; Tier 2 is
> not load-bearing.**

**Important limitation to state in any write-up.** Subsampling without replacement from a fixed
finite dataset is *not* i.i.d. sampling from a superpopulation. The finite-population correction is
real and, at `n/population` above a few percent, non-negligible. Either keep `n` small relative to
the population, or apply and document an FPC. Do not quietly ignore it.

---

> **E6 UPDATE (2026-08-20): the dataset actually used sits between the two tiers, and is
> documented as such.** `data/galaxies.npz` is a **fully observed population** of 16,743 items with
> a real gold label and a real model prediction for each. It has Tier 2's defining property — the
> population is fully observed, so `θ` is known exactly and containment can be checked — and Tier
> 3's provenance: the predictions are a genuine model's output, not one this repository fitted.
> None of the sklearn-bundled Tier 2 candidates below was used, and **no predictor was fitted**, so
> the Tier 2 recipe's steps 2–3 (fit on a held-out part) do not arise. Assumption A5 is therefore
> *inherited from the dataset's provenance rather than enforced*, which is recorded as a limitation
> in `docs/tier2_galaxies.md` §7. Full documentation, including the SHA-256 that pins the file:
> **`docs/tier2_galaxies.md`**.

---

## Tier 3 — Real AI-annotation data (confirmation only)

**Purpose.** Demonstrate the phenomenon where `f` is a genuine model annotation (e.g. an LLM
judgement) and `Y` is a genuine human label.

**This tier is explicitly not a dependency.** No theorem, no kill criterion, and no headline claim
rests on it. If nothing in this tier is reachable, the project proceeds and says so.

**Requirements a Tier-3 dataset must satisfy:**

- Both a human/gold label and a model prediction exist **for the same items**.
- The number of items is large enough that a "population" θ is well determined (ideally ≥ 5,000).
- The predictor was **not** fitted on the items used for inference, or the provenance is documented
  well enough to say so.
- Licensing permits research use, and redistribution status is recorded.

**Candidate sources, in order of preference:**

1. **The `ppi-py` companion datasets** used in the original PPI paper and its documentation. These
   are the natural comparison point because they are what the method's authors themselves used.
   Status: **`PARTIALLY RESOLVED 2026-08-20`** — `data/galaxies.npz` is structurally one of them and
   was used for E6. Its *contents* are now fully characterised (`docs/tier2_galaxies.md`); its
   *licence and retrieval provenance remain `OPEN`* and must be established before publication or
   redistribution.
2. **A HuggingFace text-classification dataset with human labels**, paired with predictions from a
   small open model run once and **cached to disk**. Generating the predictions is a one-time cost;
   after that all work is CPU post-processing of a cached column.
3. **An LLM-as-judge evaluation set** with human adjudication — e.g. a benchmark where a subset of
   model outputs has been human-scored. Status: **`RESOLVED 2026-09-20`** — MT-bench expert
   judgments with the GPT-4 judge (Zheng et al., 2023), CC-BY-4.0; Experiment J,
   `docs/tier3_mtbench.md`, Table 8 of the manuscript.

**Required documentation for whichever dataset is used** (write it into `docs/tier3_<name>.md`):

- data source and URL; retrieval date; licence
- what the human label is, and by whom it was produced
- what the model prediction is, which model, which prompt or checkpoint, which date
- the estimand (a mean of what, over which population)
- preprocessing: filtering, deduplication, handling of ties/abstentions/missing values
- the train/test separation used to satisfy A5
- population size, realised ρ, realised prevalence
- exact reproduction command

**Estimand for Tier 3.** Keep it simple: the population mean of the gold label (a prevalence, if the
label is binary). Do not attempt a regression coefficient here until the mean case is settled.

---

## Data hygiene rules (all tiers)

1. **Never let the gold labels of the inference sample influence `f`.** This is assumption A5, and
   violating it manufactures a spurious correlation that will make every method look good.
2. **Labelled and unlabelled samples must be disjoint.** Overlap induces a correlation the variance
   formulas do not model.
3. **Fix and record seeds** for every subsample. Reproduction is a stated deliverable.
4. **Record realised ρ** for every real-data configuration; it is the variable that governs
   everything and it is not controlled in Tiers 2–3.
5. **Do not filter items on the gold label.** Selecting, say, only items the model got right is a
   silent conditioning that invalidates the comparison.
6. **Store raw per-replication outcomes**, not just aggregates, so that tail behaviour and failure
   counts can be recomputed without re-running.
