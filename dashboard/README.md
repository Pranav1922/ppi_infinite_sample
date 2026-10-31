# Research dashboard

A **read-only presentation layer** over the saved results of the finite-sample PPI study.

```bash
streamlit run dashboard/app.py
```

Requires `streamlit` and `altair` in addition to the research package's own dependencies.

## What it is, and what it is not

This dashboard **reads** artefacts that the research pipeline already produced. It does not run
experiments, does not recompute saved statistics, and does not write anything. Specifically it
never modifies a result CSV, a config, a seed, a figure or a manuscript file. A test enforces
this: `tests/test_dashboard.py` hashes every file under `results/` and `data/` before and after
exercising the loaders and fails if any byte changes.

Two kinds of number appear, and they are always visually distinguished:

* **Saved research results** — read from `results/*.csv` exactly as generated. Each panel names the
  file it read at the bottom.
* **Live illustrations**, shown on a blue background and labelled *"Computed live for
  illustration"*. These exist because the saved experiments store *averages over Monte-Carlo
  replications*, not individual intervals — so a single lower/upper limit has to be computed to be
  displayed. They are not research results and enter no table in the paper.

## Pages

| page | reads |
|---|---|
| **Overview** | `results/C_sample_size.csv` |
| **Method comparison** | `results/{A_reproduction, C_sample_size, E_finite_N, I_real_data}.csv` |
| **Finite-`N` explorer** | `results/gate2_{boundary,quantiles}.csv` + the Theorem 1 bound, evaluated from its formula |
| **Coverage boundary** | `results/gate2_boundary.csv` |
| **Galaxies demonstration** | `results/I_real_data.csv`, `data/galaxies.npz` |

## Two things the dashboard is careful about

**Nominal is not empirical.** Every interval shown is *constructed* at a nominal 95 % level.
Whether it *attains* 95 % is what the research measures, and frequently it does not. Every page
carries this caveat, and coverage columns are always labelled as empirical measurements.

**The Galaxies page is not a coverage study.** The Galaxies data are a fully observed population,
so what is reported there is containment under repeated subsampling from one fixed finite
population — not superpopulation coverage. The column is labelled *containment fraction* for that
reason.

## Reproducibility

The footer shows two commits: the working tree's, and the commit recorded *inside* the results
files, which is the one that identifies the code that produced them. Paths are displayed
repo-relative; no absolute filesystem path is exposed.
