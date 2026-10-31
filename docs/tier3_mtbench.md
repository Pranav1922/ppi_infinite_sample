# MT-bench — second real-data population for Experiment J (LLM-annotation setting)

> **Status of every number below: `OBSERVED` / `EVALUATION RESULT`.** As for the Galaxies
> population (`docs/tier2_galaxies.md`), nothing here is evidence for the coverage theory of §4;
> the synthetic experiments carry that. Experiment J is a **confirmation** experiment in the
> Tier-3 sense of `08` — a genuine model annotation (`f`) against a genuine human label (`Y`).

---

## 1. Dataset

| field | value |
|---|---|
| Name | `mtbench` |
| Local path | `data/mtbench_gpt35.npz` (built by `experiments/prepare_mtbench.py`; not tracked in git, like `data/galaxies.npz`) |
| Source | `lmsys/mt_bench_human_judgments` on the Hugging Face Hub, revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25` (2023-07-20, the only release). Two parquet files, pinned by SHA-256 in `prepare_mtbench.py`: `human` (3,355 rows, `4877bc46…f5769`) and `gpt4_pair` (2,400 rows, `57068327…00e71`). |
| Retrieved | 2026-09-20 |
| Licence | **CC-BY-4.0** (dataset card). Research use and redistribution with attribution are permitted; we nonetheless do not commit the derived `.npz`, and rebuild it from the pinned source instead. |
| Reference | Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 Datasets and Benchmarks Track; arXiv:2306.05685. |
| Upstream task | Pairwise comparison of two chat models' answers to the 80 two-turn MT-bench questions. Six models: GPT-4, GPT-3.5-turbo, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B. |

## 2. What `Y` and `f` are

- **Population.** Every row of the `human` split is one expert's verdict on one comparison
  `(question, model pair, turn)`; the annotators are described by the dataset card as "mostly
  graduate students with expertise in the topic areas". The `gpt4_pair` split is GPT-4's verdict
  on the same 2,400 comparisons, judged in both presentation orders. Every human row matches
  exactly one GPT-4 row on `(question_id, model pair, turn)` — 3,355 of 3,355 join, no
  duplicates on the GPT-4 side.
- **Target model.** `gpt-3.5-turbo`, chosen by one rule fixed before any interval was computed:
  the model with the **most expert judgments** (1,478; the others have 996–1,106). Restricting to
  the comparisons involving it gives the population, `M = 1,478`.
- **Gold label `Y_i`.** `1` if the expert preferred GPT-3.5-turbo's answer, `0` if the other
  model's, `½` for a tie (both "tie" and "tie (bothbad)"). So `Y ∈ {0, ½, 1}`.
- **Prediction `f_i`.** The same score computed from GPT-4's verdict on the identical comparison.
  GPT-4's "tie (inconsistent)" — its two position-swapped verdicts disagree — is scored `½`. So
  `f ∈ {0, ½, 1}`; `f` is an LLM judge's annotation, produced by the dataset's authors, not
  fitted here on any gold label (assumption A5 holds by construction).
- **Estimand.** `θ = mean(Y)` over the population: **GPT-3.5-turbo's expert-judged win rate on
  MT-bench, ties scored ½.** Evaluation-only, as for Galaxies.
- **Ordering.** Each pair is canonicalised alphabetically before the join, and verdicts are
  re-expressed in that order, so the human and GPT-4 verdicts refer to the same answer.

## 3. Population quantities (pinned in `tests/test_realdata.py`)

| quantity | value |
|---|---|
| `M` | 1,478 |
| `θ` | 0.6518944519621109 |
| realised `ρ = corr(Y, f)` | 0.4886994008697362 |
| `μ_f = mean(f)` | 0.6204330175913396 |
| human–GPT-4 agreement (exact score match) | 0.626 |
| `Y` values | {0, ½, 1} — not binary; `Var(Y|f)` cannot be constant, so B9-HC is reported |

GPT-4 over-rates GPT-3.5-turbo's *losses* relative to the experts (`mean(f) = 0.620` against
`θ = 0.652`): the predictor is biased, which is the situation PPI's rectifier exists for.

## 4. Protocol

Identical to Experiment I (`configs/experiment_I_real_data.toml`): draw `n` rows without
replacement as the gold set, treat the remaining `M − n` as the unlabelled pool using only `f`,
run every method on that draw, repeat `R = 20,000` times; `n ∈ {10, 20, 50, 100}`,
`seed_entropy = 20260920`. Config: `configs/experiment_J_mtbench.toml`. Commands:

```
python3 experiments/prepare_mtbench.py     # downloads, verifies, builds data/mtbench_gpt35.npz
make reproduce-J                           # results/J_mtbench.csv (+ sidecar, raw)
make tables                                # Table 8 in docs/17_PAPER_TABLES.md
make figures-J                             # diagnostic figures J_*.png (not in the manuscript)
```

Run time: about 45 s on one CPU thread.

## 5. Results (Table 8 of the manuscript; `results/J_mtbench.csv`)

Containment of `θ` under repeated subsampling, nominal 0.95:

| `n` | B1 | B2 PPI | B3 PPI++ | B4 | B5 | B8 BCa | B9 LC-PPI | B9-HC |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.9304 | 0.9142 | 0.8652 | 0.9060 | 0.9018 | 0.8632 | 0.9341 | 0.9397 |
| 20 | 0.9425 | 0.9397 | 0.9124 | 0.9315 | 0.9256 | 0.9201 | 0.9447 | 0.9499 |
| 50 | 0.9496 | 0.9503 | 0.9432 | 0.9482 | 0.9467 | 0.9466 | 0.9529 | 0.9571 |
| 100 | 0.9537 | 0.9576 | 0.9526 | 0.9562 | 0.9541 | 0.9520 | 0.9571 | 0.9597 |

Mean width ÷ classical at `n = 10`: PPI++ 0.694, B9 0.928, B9-HC 1.063. Every method's mean
point estimate is 0.652 to three decimals at every `n`. B8 failed to construct an interval in 26
of 20,000 replications at `n = 10` (degenerate resamples); no other method failed anywhere.

## 6. Limitations, stated

1. **Containment, not coverage.** Subsampling without replacement from one fixed finite
   population; the same caveat as Galaxies, §9 of the manuscript.
2. **Sampling fraction.** `n/M` is 0.7 % at `n = 10` but **6.8 % at `n = 100`**, where every
   method over-contains (0.952–0.960): the finite-population correction is no longer negligible.
   The Galaxies population (16,743) never exceeds 0.6 %.
3. **Rows are judgments, not comparisons.** Several experts judged the same comparison, so the
   population contains repeated `(comparison, f)` pairs with possibly different `Y`. That is the
   population as defined; it is not i.i.d. from anything.
4. **One target model, one judge.** Six models and other judges (Claude, GPT-3.5) are in the
   upstream data; only GPT-3.5-turbo with the GPT-4 judge is evaluated, by the rule in §2.
5. **Ties scored ½** is a modelling choice; scoring ties as losses would change `θ` and `f`.
