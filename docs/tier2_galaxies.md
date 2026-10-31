# Galaxies — real-data population for Experiment I (E6)

> **Status of every number below: `OBSERVED` / `EVALUATION RESULT`.** Nothing here is a theoretical
> claim, and nothing here is evidence for the coverage theory — the synthetic experiments carry
> that (`13`, Experiments A–H). `07` Experiment I calls this a **confirmation** experiment and it
> is treated as one.

---

## 1. Dataset

| field | value |
|---|---|
| Name | `galaxies` |
| Local path | `data/galaxies.npz` |
| SHA-256 | `7be1d40a049f524fc51e150281541359a47c5eca80343dfac5b1c457fd7eae99` |
| Size on disk | 268,384 bytes |
| File mtime | 2023-09-04 |
| Format | NumPy `.npz`, two arrays |
| Distributed by | `ppi_py`, the reference implementation of PPI — <https://github.com/aangelopoulos/ppi_py>. `ppi_py/datasets/datasets.py` downloads each dataset from **Google Drive** via `gdown` under the exact filename `galaxies.npz`, which matches this file. |
| Upstream data | **Galaxy Zoo 2** — human morphology classifications of ~300,000 SDSS galaxy images. Portal: <https://data.galaxyzoo.org/>. Reference: **Willett et al. (2013), MNRAS 435, 2835**; for debiased classifications the portal "strongly advise[s]" **Hart et al. (2016)**. |
| Task | The PPI galaxy example estimates the **fraction of galaxies with spiral arms**, imputing labels with a computer-vision model to reduce the number of human annotations needed. |
| Retrieval date / provenance of this copy | Not recorded in the file; already present in `data/` when E6 began. File mtime 2023-09-04. |
| Licence — **code** | `ppi_py` is **MIT** licensed. **This covers the software, not the data.** |
| Licence — **data** | **No formal licence statement found**, at either end of the chain: `ppi_py/datasets/datasets.py` carries no per-dataset licence, citation or provenance metadata, and <https://data.galaxyzoo.org/> states no licence — it asks that the GZ2 papers be cited "if making any use of the GZ2 data". Status: **`OPEN`**. |

> **Provenance chain, established 2026-08-20 (E6b).**
> Galaxy Zoo 2 volunteer classifications of SDSS images (Willett et al. 2013) → a subset paired
> with computer-vision predictions → packaged as `galaxies.npz` and hosted on Google Drive by
> `ppi_py` → `data/galaxies.npz`, pinned by the SHA-256 above.
>
> **What is verified:** the distributor, the filename convention, the upstream dataset, the task,
> and the exact bytes in hand. **What is not:** the retrieval date of this particular copy, the
> `ppi_py` version that produced it, which model generated `Yhat`, and whether `Y = 1` denotes
> "has spiral arms" in this file specifically — the `.npz` carries no metadata of any kind, so the
> semantics rest on the `ppi_py` example notebook rather than on the file. Marked `PARTIAL`.
> **Nothing in E6's inference depends on the semantics**: the estimand is the population mean of
> `Y` whatever `Y` denotes.

### Licence and provenance — final status (2026-08-20)

| item | status |
|---|---|
| Distributor's **code** licence | **MIT** (`ppi_py`). Covers the software only. |
| Upstream **data** licence | **None stated.** <https://data.galaxyzoo.org/> publishes the GZ2 tables openly and asks that the papers be cited "if making any use of the GZ2 data", but states no licence and no redistribution terms. |
| Distributor's **data** metadata | **None.** `ppi_py/datasets/datasets.py` maps each dataset name to a Google Drive id and carries no licence, citation or provenance field. |
| **Verdict** | **`OPEN`, and treated as "do not redistribute".** Analysis and publication of *results* are unaffected; redistribution of the file or of derived row-level data is not authorised by anything located. Resolve with the Galaxy Zoo team before any data release. |

**Re-checked 2026-10-04 (faculty revision item 1), with no change.** The <https://data.galaxyzoo.org/>
GZ2 section still asks only that Willett et al. (2013) be cited and posts no licence or
redistribution terms for the classification tables. `ppi_py` is still MIT with no per-dataset terms
in `datasets.py` or the README. The portal links the GZ2 *images* to Zenodo record 3565489, which
is CC-BY-4.0, but `galaxies.npz` holds classification labels and model predictions, not images,
so that licence does not reach this file. No permission was sought. The manuscript (Limitation 9,
§12) now uses the faculty's wording: not redistributed, no claim to a right to do so.

### Redistribution — do not

`ppi_py`'s MIT licence covers its *code*. No licence attaches to the *data* at either end of the
chain. **This repository therefore does not redistribute `galaxies.npz`**: `data/` is git-ignored,
only the SHA-256 is committed, and anyone reproducing E6 must acquire the file themselves (§9.1).
Any publication using it must cite Willett et al. (2013) — and Hart et al. (2016) if debiased
classifications are involved — and should confirm the licence position with the Galaxy Zoo team
before redistributing anything derived from it.

### Acquisition

```python
# pip install ppi-py gdown
from ppi_py.datasets import load_dataset
load_dataset("./data/", "galaxies")          # writes ./data/galaxies.npz
```

Then verify you have the same bytes this repository was run against:

```
shasum -a 256 data/galaxies.npz
# expect 7be1d40a049f524fc51e150281541359a47c5eca80343dfac5b1c457fd7eae99
```

`tests/test_realdata.py` pins `M`, `θ`, `ρ` and `μ_f` independently, so a different file will fail
the suite rather than silently change the results.

## 2. Exact file structure

Read, not assumed (`realdata.load_population` raises rather than guessing):

| key | shape | dtype | contents |
|---|---|---|---|
| `Y` | `(16743,)` | `float64` | gold label; **binary**, values exactly `{0.0, 1.0}`; mean 0.2592725 |
| `Yhat` | `(16743,)` | `float64` | model prediction; continuous in `[1.01·10⁻⁴, 0.99999940]`; 16,718 distinct values; mean 0.2603170 |

**There is no `Yhat_unlabeled` key and no unlabelled arm.** The file is a *fully observed
population*: one gold label and one prediction for every item. The gold/unlabelled split is
therefore **constructed** by this repository, per `08` Tier 2 step 4 — see §4.

No NaNs, no infinities, no missing values; `Y` and `Yhat` are aligned row-wise by construction.

## 3. Population quantities

| quantity | value | role |
|---|---|---|
| `M` population size | 16,743 | — |
| **`θ = mean(Y)`** | **0.2592725318** | **EVALUATION ONLY** |
| `Corr(Y, Yhat)` | 0.74078 | reported, not controlled (`07`) |
| `μ_f = mean(Yhat)` | 0.2603170 | inference input (via the pool) |
| `σ_f² = Var(Yhat)` | 0.1102211 | inference input (via the pool) |
| Brier score `mean((Yhat − Y)²)` | 0.08672 | predictor-quality diagnostic |
| Accuracy at a 0.5 threshold | 0.8817 | predictor-quality diagnostic |

**Estimand.** The population mean of the gold label, `θ = E[Y]` — a prevalence, since `Y` is
binary. This is `08` Tier 3's prescribed estimand ("Keep it simple: the population mean of the gold
label") and matches the estimand of every synthetic experiment in the project.

## 4. Inference data vs evaluation-only information

| | what it is | may enter inference? |
|---|---|---|
| **Inference data** | the `n` gold rows' `(Y, Yhat)`; the pool's `f̄_N` and `σ̂_f²` | **yes** |
| **Evaluation-only** | `θ = 0.2592725`, the population mean of `Y` | **no** — carried on `PPIData.theta_true`, which `PPIView` strips, so no registered method can reach it (`15` §4.3) |
| **Evaluation-only** | the gold labels of the 16,733 pool rows | **no** — the protocol passes only `Yhat` summaries from the pool |

`tests/test_realdata.py` asserts that mutating `theta_true` leaves every method's output
bit-identical, the same structural guarantee the synthetic experiments carry.

## 5. Preprocessing and filtering

**None.** No rows dropped, no values transformed, no rescaling, no thresholding of `Yhat`. The file
is used exactly as it is on disk. This is deliberate: any filtering would be a choice that the
estimand would then depend on.

## 6. Subsampling protocol

Implemented in `realdata.subsample_protocol`; driven by `configs/experiment_I_real_data.toml`
through the ordinary runner, so it inherits the project's seed policy, common random numbers and
provenance sidecar unchanged.

1. The full 16,743 rows are the population `U`; `θ = mean(Y)` over `U`.
2. Draw `n` rows **without replacement** as the gold set — both `Y` and `Yhat` observed.
   Implemented by rejection on row-wise duplicates, which is uniform over `n`-subsets.
3. The remaining `M − n` rows are the unlabelled pool. **Only `Yhat` is used from them.**
   `f̄_N` and `σ̂_f²` are computed exactly from population totals minus gold-set totals, so the
   `(R, N)` pool array is never materialised.
4. All methods run on that one draw (common random numbers within a replication).
5. Repeat `R = 20,000` times, per `07` Experiment I.

`n ∈ {10, 20, 50, 100}` (`07` Experiment I). `N = M − n ∈ {16,733 … 16,643}`. Seed entropy
20,260,819; each cell's generator is derived from the sorted sweep indices exactly as in every
other experiment.

## 7. Assumptions, and where they are violated

| assumption (`04` §4.10) | status here |
|---|---|
| A1 bivariate normal | **violated by construction.** `Y` is binary and `Yhat` is a bounded skewed probability. Nothing Gaussian applies; this is the empirical-robustness region of `theory_finite_N.md` Part IV. |
| A2 pool independent of the gold sample | **violated.** The pool is the *complement* of the gold set in one finite population, so the two are negatively dependent, and `f̄_N` is nearly deterministic given `U`. |
| A5 predictor not fitted on the inference sample | **inherited, not enforced.** No model is fitted by this repository; the predictions ship with the file. Whether the network saw these galaxies in training is not recorded in the `.npz`. `OPEN`. |
| A7 homoskedasticity | **violated by construction.** `Y` binary ⇒ `Var(Y|f) = p(f)(1−p(f))` cannot be constant. Measured: residual variance varies **12.6×** across quintiles of `Yhat` (0.0158 → 0.1996). This is why `B9-HC` is included and it is a data-driven justification, not a default. |
| A6 estimand is a mean | satisfied. |

## 8. Limitations

1. **Subsampling from a fixed finite population is not i.i.d. sampling from a superpopulation**
   (`08` Tier 2, stated requirement). The containment frequencies in `results/I_real_data.csv` are
   *coverage under repeated subsampling from this one population*, not superpopulation coverage,
   and they are labelled that way on every figure.
2. **The finite-population correction is real and large for the pool.** `N/M = 99.7 %`, so the true
   variance of `f̄_N` is smaller than `σ_f²/N` by a factor `1 − N/M ≈ n/M ≈ 0.06–0.6 %`. Every
   method's unlabelled variance term is therefore over-stated by roughly `M/n`. The term is tiny in
   absolute size (the unlabelled share of the total variance is `4·10⁻⁴` at `n = 10` and
   `8·10⁻³` at `n = 100`), so nothing observable turns on it — but the finite-N machinery of E4 is
   **inapplicable as designed** here, and `B9-R2`'s adjustment is correspondingly inert.
3. **One dataset, one predictor, one estimand.** No claim generalises from it.
4. **The data licence is unresolved** (§1). The distributor's MIT licence covers code only and no
   data licence was found upstream, so this repository does not redistribute the file and any
   publication must settle the position with the Galaxy Zoo team. This blocks **redistribution**,
   not analysis. `OPEN`.
5. `θ` is known only because the population is fully observed. That is what makes the containment
   check possible and simultaneously what makes it not a real deployment.

## 9. Reproduction

### 9.1 Acquire the data

See §1 — the file is not distributed with this repository.

### 9.2 Run

```
make reproduce-I     # results/I_real_data.csv + .meta.json + I_real_data_raw/
make figures-I       # figures/I_*.png
```

The sidecar records the resolved config, the command line, the git commit, wall-clock seconds and
the Python/NumPy/SciPy/pandas versions. The dataset itself is **not** git-tracked (`data/` is
ignored); the SHA-256 in §1 is what pins it.
