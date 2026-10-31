# README — Research Package

**Project:** Finite-Sample-Valid Prediction-Powered Inference — Exact Confidence Intervals for
Scientific Estimation from AI-Generated Labels

**Package version:** 1.0 · **Prepared:** 19 August 2026 · **Status:** pre-implementation handoff

---

## Read this first

You are looking at a **research handoff package**, not a codebase. It contains the problem
statement, the mathematics, the preliminary evidence, and the implementation plan for a research
project that has already been selected, scoped, and given a preliminary empirical proof of concept.

If you are Claude Code (or any coding agent) opening this repository for the first time:

1. Read this README.
2. Read `01`–`14` **in order**. Do not skip `03` (prior art) or `10` (kill criteria).
3. Read `15_ARCHITECTURE_SPECIFICATION.md` — the architecture source of truth for the coding stage.
4. Then follow `12_CLAUDE_CODE_HANDOFF.md` operationally, using `15` §12 as the refined
   task-by-task sequence.

**Your first task is Stage 0 of `12`: the prior-art check. Write no code before it is done.**

---

## Sources of truth (in case of conflict, these win, in this order)

| Question | Authoritative document |
|---|---|
| What equation is correct? | `04_MATHEMATICAL_FORMULATION.md` |
| What numbers must reproduce, and to what tolerance? | `09_REPRODUCTION_PROTOCOL.md` |
| What code must exist, with what signatures? | `11_IMPLEMENTATION_SPEC.md` |
| How is the system laid out — modules, data flow, configs, storage, MVP? | `15_ARCHITECTURE_SPECIFICATION.md` |
| What do I do, in what order, and when am I done? | `12_CLAUDE_CODE_HANDOFF.md` |
| What is still unresolved? | `14_OPEN_QUESTIONS.md` |

If `12` appears to contradict `04`, **`04` wins** and the contradiction must be logged in
`13_RESEARCH_LOG.md` rather than silently resolved.

---

## Concise project summary

Prediction-Powered Inference (PPI) and its successor PPI++ let a researcher combine a **large set of
model-generated labels** with a **small set of human (gold) labels** to estimate a scientific
quantity with a tighter confidence interval than the gold labels alone would allow. PPI++ adds a
tuning coefficient λ, estimated from the gold labels, which is supposed to make the method
"essentially never worse" than classical inference.

That guarantee is asymptotic. In the regime PPI actually exists to serve — a gold set of tens of
items, because human labelling is the expensive part — a preliminary Monte-Carlo study conducted
during project selection found the standard PPI++ interval covering at roughly **0.878** when it
claims 0.95, under a bivariate-Gaussian working model at n = 10. The deficit was approximately
**invariant to predictor quality** and persisted (smaller) at n = 20 and n = 50.

The working hypothesis is that the PPI++ variance expression omits a **regression-leverage /
extrapolation term** that arises because the tuning coefficient λ̂ is estimated from the same small
gold sample used for the mean. Restoring that term — recognising the PPI++ point estimate as an OLS
regression-imputation estimator and using its exact normal-theory pivot — produced approximately
nominal coverage (0.9492–0.9500) across every cell tested in the limiting case where the number of
unlabelled items N → ∞.

**What remains genuinely open:** the correction is *not* exact at finite N (it over-covers, e.g.
≈ 0.9612 at N = 200, n = 10, ρ = 0.9); robustness outside the Gaussian model is untested; a BCa
bootstrap has not yet been compared; and the survey-sampling GREG / model-assisted literature has
**not** been mapped and could contain an equivalent result.

---

## What this project is NOT

- It is **not** an attempt to build a better prediction model. The predictor `f` is taken as given.
- It is **not** a deep-learning or reinforcement-learning project. No neural network is required
  anywhere on the critical path.
- It is **not** a benchmark-beating exercise. The deliverable is a **validity guarantee** plus an
  honest characterisation of what it costs in interval width.

---

## Computational profile

CPU only. Every result in this package was produced in NumPy/SciPy on 2 CPU threads in under
15 minutes total. No GPU is required at any stage.

---

## Document index

| File | Purpose |
|---|---|
| `01_PROJECT_OVERVIEW.md` | What the project is, and the demonstrated / supported / open split. |
| `02_RESEARCH_PROBLEM_AND_GAP.md` | Rigorous problem statement and the precise research gap. |
| `03_LITERATURE_AND_PRIOR_WORK.md` | Prior work with citation discipline; the GREG prior-art warning. |
| `04_MATHEMATICAL_FORMULATION.md` | **Mathematical source of truth.** Notation, estimators, variances, derivation status. |
| `05_PROPOSED_METHOD.md` | LC-PPI (provisional name) as an algorithm. |
| `06_BASELINES_AND_COMPARISONS.md` | B1–B10, each with formula, purpose, and expected behaviour. |
| `07_EXPERIMENT_PLAN.md` | Experiments A–I with objectives, variables, metrics, failure conditions. |
| `08_DATASETS_AND_REAL_DATA.md` | Tier 1 synthetic / Tier 2 semi-real / Tier 3 real AI annotations. |
| `09_REPRODUCTION_PROTOCOL.md` | **Reproduction source of truth.** Environment, seeds, parameters, expected tables, tolerances. |
| `10_HYPOTHESES_AND_KILL_CRITERIA.md` | H1–H5 and K1–K4 as pre-registered decision rules. |
| `11_IMPLEMENTATION_SPEC.md` | **Implementation source of truth.** Repo layout, modules, signatures, tests. |
| `12_CLAUDE_CODE_HANDOFF.md` | **Operational source of truth.** Mission, order, acceptance criteria, forbidden shortcuts. |
| `13_RESEARCH_LOG.md` | Chronological record, including ideas that failed. |
| `14_OPEN_QUESTIONS.md` | Every unresolved item with priority and success condition. |
| `15_ARCHITECTURE_SPECIFICATION.md` | **Architecture source of truth.** Layering, data flow, math→code map, public API, orchestration, storage, testing, MVP, and the conflict register. |

The package splits cleanly in two: **`01`–`10` and `13`–`14` are the science** (what is true, what is
preliminary, what is open, and the rules for deciding); **`11`, `12` and `15` are the build** (what
code exists, how the system is laid out, and in what order to write it).

---

## Provenance of the preliminary numbers

All preliminary numbers in this package were produced by three scripts written during project
selection: `probe.py` (MSE), `cover.py` (coverage of six interval variants), `exact.py` (the
leverage-corrected interval). They are shipped alongside this package as `preliminary_code/`.

> **These are existing preliminary results. They must be independently reproduced before any of
> them is treated as established.** Where this package repeats them, they are labelled as such.
