# 15 — Architecture Specification

> **This document is the architecture source of truth for the coding stage.**
> It is *derived from* the research package, not invented alongside it. Every architectural decision
> below is traced to the research document that forces it. Where a decision is not forced by the
> research, it is marked **`ARCHITECTURAL CHOICE`** and a rationale is given.
>
> Precedence: `04` (mathematics) > `09` (reproduction) > **this document** (architecture) > `11`
> (module contracts) > `12` (operational order). Where this document extends or supersedes `11`, it
> says so explicitly in §0.3.

---

## 0. Preliminaries

### 0.1 The derivation chain

The research package fixes the following chain, and the architecture is a direct rendering of it:

```
Research problem            02 §2.4   PPI++ intervals under-cover at small n
        ↓
Mathematical formulation    04 §4.6   the omitted term is s²·(f̄_N − f̄_n)²/S_ff
        ↓
Estimator                   04 §4.7   θ̂ = Ȳ + b(f̄_N − f̄_n) is an OLS fitted value
        ↓
Inference                   04 §4.8   V̂ = s²(1/n + lev) + b²σ̂_f²/N ; quantile t_{n−2}
        ↓
Baselines                   06        B1–B10, all on identical draws
        ↓
Simulation                  08 T1.1–T1.5, 09 §Seed policy
        ↓
Experiments                 07        A–I
        ↓
Evaluation                  06 §Reporting, 09 §Verification checks
        ↓
Results                     09 §Output files
```

**Every arrow is a module boundary.** That is the whole architecture in one sentence: the software
is the chain, one component per link, with nothing crossing a link it does not need to.

### 0.2 The five constraints that actually shape the design

These are not style preferences; each is forced by a specific research requirement and each rules
out an otherwise-reasonable design.

| # | Constraint | Forced by | What it rules out |
|---|---|---|---|
| C1 | **Common random numbers.** All methods in a cell see the identical dataset. | `09` §Seed policy rule 3; every paired comparison in `06`–`07` | Any design where a method can draw its own data. Methods must receive arrays, never a generator. |
| C2 | **Batched Monte Carlo.** Estimators operate on `(R, n)` arrays. | `R = 200,000` × ~9 cells × ~10 methods (`09`) | Per-replication Python loops. A scalar-first API retrofitted to batches. |
| C3 | **Variance is pluggable.** The variance expression is a parameter, not a hard-coded branch. | `04` §4.8.4 defines four competing finite-N repairs R1–R4 that must be compared on equal footing | Baking `V̂` into the interval constructor. |
| C4 | **Every formula lives in exactly one place.** | `11` §Repository structure rule; the project's whole claim is a variance expression | Duplicating an equation between `src/` and `experiments/`. |
| C5 | **Verification is executable and exits non-zero.** | `09` V1–V12; `12` Stage 7 is a hard gate | Eyeballing tables. Tolerances as documentation only. |

### 0.3 Relationship to `11_IMPLEMENTATION_SPEC.md`

`11` specifies module *contracts* (function signatures, inputs, outputs, tests). This document
specifies the *system* (layering, data flow, orchestration, storage, MVP order). They agree on
module names and responsibilities.

**This document supersedes `11` in exactly three places**, each an extension rather than a
contradiction, and each logged in `13_RESEARCH_LOG.md`:

1. The repository tree here adds `configs/`, `data/`, `requirements.txt`, and three modules
   (`config.py`, `io.py`, `plots.py`) that `11` did not name. `11`'s tree is a strict subset.
2. `11` lists one script per experiment (`exp_b_bootstrap.py`, …). This document replaces them with
   **one config-driven runner** plus thin per-experiment config files. See §7.1 and the conflict
   report in §13.
3. `11` gives `pyproject.toml` only; this document also requires `requirements.txt` (the user's
   stated stack) and pins the minimum Python to 3.11. See the conflict report in §13.

---

## 1. Architecture overview

### 1.1 Layering

Five layers, strictly downward-depending. A layer may import only from layers below it.

```
┌───────────────────────────────────────────────────────────────────────┐
│  L5  REPORTING           plots.py, tables, the paper's figures        │
├───────────────────────────────────────────────────────────────────────┤
│  L4  ORCHESTRATION       runner.py, config.py, io.py, verify.py       │
│                          sweeps, replication management, provenance   │
├───────────────────────────────────────────────────────────────────────┤
│  L3  EVALUATION          metrics.py   coverage, width, MSE, MC s.e.   │
├───────────────────────────────────────────────────────────────────────┤
│  L2  METHODS             registry.py → B1..B10                        │
│      ├── baselines       ppi.py, ppi_plus.py, bootstrap.py            │
│      └── proposed        lc_ppi.py                                    │
├───────────────────────────────────────────────────────────────────────┤
│  L1  STATISTICAL CORE    estimators.py, variance.py,                  │
│                          confidence_intervals.py, types.py            │
└───────────────────────────────────────────────────────────────────────┘
        ↑ data producers, orthogonal to the stack, feeding L2:
        simulation.py (synthetic)      realdata.py (Tier 2/3)
```

**Why this layering and not a flatter one.** The research asks one question repeatedly with one
thing varied each time: *the variance expression* (C3). Isolating L1 means a new candidate variance
(R1–R4 in `04` §4.8.4) is a new function in `variance.py` and a new registry entry — it touches
nothing else. That is the property the project needs most, because Stage 11 (`12`) will iterate on
exactly that.

### 1.2 The two spines

**Data spine.** `PPIData → IntervalResult → metric rows → CSV`. Three types, three transitions.
Everything in the system is one of those three transitions.

**Identity spine.** A method is a string (`"B9"`), a cell is a tuple of parameters, an experiment is
a config file. These three identifiers appear in the registry, in the CSV, in the figures, and in
`06`/`07`/`09`. **The same string, everywhere, no translation tables.**

### 1.3 What is deliberately absent

Derived from `12` forbidden shortcut 9 ("do not build architecture before the mathematics is
verified") and the minimal-stack requirement:

- No database. CSV + JSON sidecars + npz. (`17`)
- No web service, no API server, no dashboard.
- No plugin system, no abstract base classes, no dependency injection framework. The registry is a
  dict.
- No deep learning, no GPU, no RL. (`02` §2.7; `12` forbidden shortcut 8.)
- No parallelism framework. Vectorisation over `R` is the parallelism. If a cell is slow, chunk it
  over `R` inside the runner; do not introduce `joblib`/`multiprocessing` until measurement shows
  it is needed. **`ARCHITECTURAL CHOICE`** — rationale: reproducibility under C1 is easier to
  guarantee single-threaded, and the workload is already ~10⁷ float operations per cell, which
  NumPy handles in seconds.

---

## 2. Final repository tree

```text
ppi-finite-sample/
├── README.md                      # points at docs/README_RESEARCH_PACKAGE.md
├── pyproject.toml                 # package metadata, pinned minimums, pytest config
├── requirements.txt               # exact pinned versions for reproduction
├── Makefile                       # make test | reproduce | verify | figures | all
│
├── docs/                          # THIS RESEARCH PACKAGE, verbatim
│   ├── README_RESEARCH_PACKAGE.md
│   ├── 01_PROJECT_OVERVIEW.md  ...  15_ARCHITECTURE_SPECIFICATION.md
│   ├── prior_art_findings.md      # WRITTEN IN STAGE 0, before any code
│   ├── theory_finite_N.md         # written in Stage 11
│   └── tier2_<name>.md            # written in Stage 12
│
├── src/
│   └── ppi_finite/
│       ├── __init__.py            # public API re-exports (§6)
│       ├── types.py               # PPIData, IntervalResult, CellSpec  (L1)
│       ├── estimators.py          # sufficient statistics, point estimators, λ rules  (L1)
│       ├── variance.py            # every variance expression, incl. R1–R4, HC3  (L1)
│       ├── confidence_intervals.py# quantiles → intervals  (L1)
│       ├── ppi.py                 # B1, B2 — the coefficient-free methods  (L2)
│       ├── ppi_plus.py            # B3, B4, B5, B6 + shrink variants  (L2)
│       ├── lc_ppi.py              # B9, B9-nolev, B9-R1..R4, B10  (L2)
│       ├── bootstrap.py           # B7, B8  (L2)
│       ├── registry.py            # method-ID → callable  (L2)
│       ├── theory.py              # finite-N derivation objects, NOT methods  (L1)
│       ├── metrics.py             # coverage, width, MSE, oracle gain, MC s.e.  (L3)
│       ├── simulation.py          # DGPs T1.1–T1.5, run_cell  (producer)
│       ├── realdata.py            # Tier 2/3 loaders, subsampling protocol  (producer)
│       ├── config.py              # config schema, parsing, validation  (L4)
│       ├── runner.py              # sweep expansion, execution, provenance  (L4)
│       ├── io.py                  # tidy CSV, sidecar JSON, raw npz  (L4)
│       └── plots.py               # figure generation  (L5)
│
├── configs/
│   ├── experiment_A_reproduction.toml
│   ├── experiment_B_bootstrap.toml
│   ├── experiment_C_sample_size.toml
│   ├── experiment_D_predictor_quality.toml
│   ├── experiment_E_finite_N.toml
│   ├── experiment_F_heavy_tails.toml
│   ├── experiment_G_skew.toml
│   ├── experiment_H_heteroskedastic.toml
│   ├── experiment_I_real_data.toml
│   └── smoke.toml                 # R=2000, used by CI
│
├── experiments/
│   ├── run.py                     # single entry point: run.py --config configs/X.toml
│   ├── verify.py                  # V1–V12; exits non-zero on failure
│   └── make_figures.py            # results/*.csv → figures/
│
├── tests/
│   ├── test_estimators.py         # vs scipy.stats.linregress, hand-computed example
│   ├── test_variance.py           # non-negativity, reductions, leverage = 0 case
│   ├── test_intervals.py          # containment, monotonicity in alpha, vs scipy.stats.t
│   ├── test_identity.py           # V11: θ̂ == OLS fitted value at μ_f
│   ├── test_simulation.py         # DGP moments match targets
│   ├── test_registry.py           # every ID resolves; uniform signature
│   ├── test_crn.py                # common random numbers actually shared
│   ├── test_determinism.py        # same seed spec → identical CSV
│   ├── test_pivot_ks.py           # V12, @pytest.mark.slow
│   └── test_reproduction.py       # Tables A1–A5 at reduced R, @pytest.mark.slow
│
├── data/                          # Tier 2/3 cached inputs only; git-ignored
├── results/                       # tidy CSVs (tracked) + .meta.json + raw/ (ignored)
└── figures/                       # PNG for review, PDF for the paper
```

**`data/` exists but is empty in the MVP.** Tiers 1 and 2 need no stored data (`08`): Tier 1 is
generated, Tier 2 is bundled with scikit-learn. `data/` is for Tier-3 cached model outputs only, and
Tier 3 is explicitly not load-bearing (`08`).

---

## 3. Component responsibilities

### L1 — Statistical core

| Module | Responsibility | Must NOT |
|---|---|---|
| `types.py` | The three data contracts: `PPIData`, `IntervalResult`, `CellSpec`. Validation in `__post_init__`. | contain any statistics |
| `estimators.py` | Sufficient statistics (`Ȳ, f̄_n, S_ff, S_fY, b, s², se(b)`); point estimators; the four λ rules (plug-in, cross-fit, threshold, shrunk). | construct intervals; know about method IDs |
| `variance.py` | Every variance expression in the package, one function each, each docstring citing its equation number in `04`. | choose which one to use |
| `confidence_intervals.py` | `(θ̂, se, dof, α, dist) → (lo, hi)`. Nothing else. | compute a variance |

The split between `variance.py` and `confidence_intervals.py` is the direct expression of C3: the
research varies the variance while holding the interval construction fixed.

### L2 — Methods

| Module | Provides |
|---|---|
| `ppi.py` | **B1**, B2 — the two methods with no estimated coefficient. *(Amended 2026-08-19: this table originally gave B1 no module at all, while `11` put its constructor in `confidence_intervals.py`, which §3 restricts to quantiles-to-intervals. B1 belongs in L2 with the other coefficient-free method.)* |
| `ppi_plus.py` | B3, B4, B5, B6, `B6-shrink-c1`, `B6-shrink-c3` |
| `lc_ppi.py` | **B9** (proposed), `B9-nolev` (the decomposition control), `B9-R1`…`B9-R4`, `B9-HC`, **B10** |
| `bootstrap.py` | B7, B8 |
| `registry.py` | `METHODS: dict[str, MethodFn]` where `MethodFn = (PPIData, float, Generator|None) -> IntervalResult` |

Each method is a **thin composition** of L1 calls. If a method function is longer than about fifteen
lines, a formula has leaked out of L1 (violates C4).

`registry.py` is also the pre-registration record (`10` standing prohibition 7): adding an entry
after results are seen requires a log entry in `13`.

### L3 — Evaluation

`metrics.py` turns `(IntervalResult, θ_true)` into metric rows. It owns the Monte-Carlo standard
error, the failure accounting (`n_failed`, never silently dropped — `06` §Reporting), and the
oracle-gain fraction.

### L4 — Orchestration

| Module | Responsibility |
|---|---|
| `config.py` | Parse and **validate** a TOML config into a list of `CellSpec`. Reject unknown keys loudly. |
| `runner.py` | For each cell: derive the seed, call the DGP **once** (C1), evaluate every requested method on that one draw, compute metrics, emit rows. |
| `io.py` | Tidy CSV writing, provenance sidecar, raw `.npz`. |
| `verify.py` | V1–V12 against a results CSV. Exit code 0/1. |

`runner.py` is the only place C1 can be violated, and therefore the only place it needs to be
enforced. It is worth reading carefully in review.

### L5 — Reporting

`plots.py` consumes only `results/*.csv`. It never recomputes a statistic. This makes every figure
traceable to a row of a CSV that has a provenance sidecar.

---

## 4. Data flow

### 4.1 End-to-end

```
   ┌──────────────────── CONFIG (configs/*.toml) ────────────────────┐
   │  experiment_id · method IDs · n grid · N grid · ρ grid ·        │
   │  α levels · DGP family + params · R · seed entropy · modes      │
   └─────────────────────────────┬───────────────────────────────────┘
                                 │  config.py: expand to CellSpec list
                                 ▼
                        ┌─────────────────┐
                        │   CellSpec      │  one point of the sweep
                        └────────┬────────┘
                                 │  runner.py: derive child Generator
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │  DATA PRODUCTION   (exactly once per cell — C1)    │
        │                                                    │
        │  simulation.py                 realdata.py         │
        │   dgp_gaussian / heavy_tail     load_tier2/3       │
        │   skew / heteroskedastic        fit_weak_predictor │
        │   misspecified                  subsample_protocol │
        └────────────────────────┬───────────────────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │  PPIData  (R, n) batched   │
                    │  Y, f, f_unlabeled|f̄_N,    │
                    │  σ̂_f², N,  θ_true          │
                    └────────────┬───────────────┘
                                 │  θ_true is EVALUATION-ONLY.
                                 │  It never enters a method. (see §4.3)
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   ┌──────────────────────┐              ┌──────────────────────┐
   │ SUFFICIENT STATS     │              │ (bootstrap only)     │
   │ estimators.py        │              │ child Generator      │
   │ Ȳ f̄_n S_ff S_fY b s² │              │ resample indices     │
   └──────────┬───────────┘              └──────────┬───────────┘
              │                                     │
              ▼                                     │
   ┌──────────────────────┐                         │
   │ λ RULE               │                         │
   │ plug-in / cross-fit  │                         │
   │ threshold / shrunk   │                         │
   └──────────┬───────────┘                         │
              ▼                                     │
   ┌──────────────────────┐                         │
   │ VARIANCE   (C3)      │                         │
   │ ppi_plus | lc_ppi    │                         │
   │ v482 | v_full |      │                         │
   │ R1..R4 | HC3 | nolev │                         │
   └──────────┬───────────┘                         │
              ▼                                     ▼
        ┌───────────────────────────────────────────────┐
        │  INTERVAL CONSTRUCTION                        │
        │  B1 B2 B3 B4 B5 B6 │ B7 B8 │ B9 B9-* B10      │
        └───────────────────────┬───────────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │  IntervalResult        │
                    │  θ̂, lo, hi, se, dof,   │
                    │  extras, n_failed      │
                    └───────────┬────────────┘
                                │  + θ_true  ← first and only use
                                ▼
                    ┌────────────────────────┐
                    │  metrics.py            │
                    │  coverage ± MC s.e.    │
                    │  width (mean, median)  │
                    │  MSE, oracle gain      │
                    │  tail miscoverage      │
                    └───────────┬────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │  io.py                                        │
        │  results/<exp>.csv        tidy, one row per   │
        │                           (method, cell, metric)│
        │  results/<exp>.meta.json  provenance          │
        │  results/<exp>_raw/*.npz  per-replication     │
        └───────────┬───────────────────────┬───────────┘
                    ▼                       ▼
            experiments/verify.py     plots.py
            V1–V12, exit 0/1          figures/*.png|pdf
```

### 4.2 Inputs, transformations, intermediates, outputs

| Stage | Object | Shape | Notes |
|---|---|---|---|
| Input | config | TOML | the only hand-edited artifact |
| Input | `Y`, `f` | `(R, n)` | gold labels and predictions on the labelled sample |
| Input | `f_unlabeled` or `f̄_N` | `(R, N)` or `(R,)` | see the mode note in §4.4 |
| Intermediate | `Ȳ, f̄_n, S_ff, S_fY, b, s², se(b)` | `(R,)` each | computed once per cell, shared by every method (C1 + efficiency) |
| Intermediate | `λ̂` per rule | `(R,)` | plug-in, cross-fit, threshold, shrunk |
| Intermediate | `leverage`, `V̂` | `(R,)` | the quantities the project is about |
| Output | `θ̂, lo, hi` | `(R,)` | per method |
| Output | metric rows | tidy | the deliverable |
| Evaluation-only | `θ_true` | scalar | see §4.3 |

### 4.3 Evaluation-only ground truth — an enforced boundary

`θ_true` is known in Tier 1 by construction and in Tiers 2–3 because the full dataset is treated as
the population (`08`). **It must never reach a method.**

This is enforced structurally, not by convention:

- `PPIData` carries `theta_true`, but the registry signature passes methods a **`PPIView`** — a
  frozen projection of `PPIData` with `theta_true` removed.
- `runner.py` is the only code that holds both a method's `IntervalResult` and `theta_true`, and it
  brings them together only inside `metrics.py`.
- `tests/test_crn.py` asserts that mutating `theta_true` leaves every method's output bit-identical.

The same boundary protects assumption A5 (`04` §4.10): in `realdata.py`, `fit_weak_predictor` is
called on the predictor-fitting split and is structurally unable to see the inference split. This is
`12` forbidden shortcut 2, the error the handoff calls the most damaging possible one.

### 4.4 The unlabelled-mode branch

`08` §T1.1 defines two modes and warns they are not interchangeable:

- `mean_only` — draw `f̄_N ~ N(0, σ_f²/N)` directly and treat `σ_f²` as known. Makes `N = 10⁶` free.
  **This is what produced the preliminary numbers**, so Experiment A must use it.
- `explicit` — materialise `f_unlabeled` and estimate `σ̂_f²`. Required whenever `N ≤ 5000`, and for
  any bootstrap that resamples the pool.

`unlabeled_mode` is a required config key with no default. `runner.py` writes it into every CSV row.
`config.py` **rejects** `mean_only` with `N ≤ 5000` unless `allow_mean_only_small_N = true` is set
explicitly, so the trap in `08` cannot be fallen into silently.

---

## 5. Mathematical-object → software mapping

Every mathematically important object has exactly one implementation location. Equation references
are to `04_MATHEMATICAL_FORMULATION.md`.

| Mathematical object | Definition | Eq. | Module | Function / field |
|---|---|---|---|---|
| `n`, `N`, `R` | gold size, unlabelled size, replications | §4.1 | `types.py` | `PPIData.n`, `.N`; `CellSpec.R` |
| `Y`, `f` | gold labels, predictions on labelled sample | §4.1 | `types.py` | `PPIData.Y`, `.f` |
| `f̄_n` | labelled prediction mean | §4.1 | `estimators.py` | `labelled_summaries()['fbar_n']` |
| `f̄_N` | unlabelled prediction mean | §4.1 | `types.py` | `PPIData.fbar_N` (or derived from `f_unlabeled`) |
| `μ_f` | population prediction mean | §4.1 | `types.py` | `PPIData.mu_f` (set ⇒ `N = ∞` branch) |
| `σ̂_f²` | variance of `f` | §4.1 | `types.py` | `PPIData.sigma2_f` |
| `S_ff` | `Σ(f_i − f̄_n)²` — **sum, not variance** | §4.1 | `estimators.py` | `labelled_summaries()['S_ff']` |
| `S_fY` | `Σ(f_i − f̄_n)(Y_i − Ȳ)` | §4.1 | `estimators.py` | `labelled_summaries()['S_fY']` |
| `b` | OLS slope `S_fY/S_ff` | §4.7 | `estimators.py` | `labelled_summaries()['b']` |
| `s²` | residual variance `RSS/(n−2)` | §4.1 | `estimators.py` | `labelled_summaries()['s2']` |
| `se(b)` | `s/√S_ff` | §4.9 | `estimators.py` | `labelled_summaries()['se_b']` |
| `θ̂_classical` | `Ȳ` | §4.2 | `estimators.py` | `theta_classical()` |
| `θ̂_PPI` | `Ȳ − (f̄_n − f̄_N)` | §4.3 | `estimators.py` | `theta_ppi()` |
| `θ̂_λ` | `Ȳ − λ(f̄_n − f̄_N)` | §4.4 | `estimators.py` | `theta_ppi_plus()` |
| `λ*` | `β/(1+r)` (population) | §4.4 | `estimators.py` | `lambda_oracle()` |
| `λ̂` | plug-in `b/(1+r)` | §4.4 | `estimators.py` | `lambda_plugin()` |
| `λ̂_cf` | cross-fitted λ̂ | B5 | `estimators.py` | `lambda_crossfit(scheme=...)` |
| `λ_thr` | `λ̂` if `t² > t_crit²` else 0 | B6 | `estimators.py` | `lambda_threshold()` |
| `λ̃` | `λ̂(1 − c/t²)₊` | §4.9 | `estimators.py` | `lambda_shrunk(c=...)` |
| `V̂_classical` | `s_Y²/n` | §4.2 | `variance.py` | `var_classical()` |
| `V̂_PPI` | `V̂ar(Y−f)/n + V̂ar(f)/N`, **labelled-sample** `ddof=1` moments | §4.3 | `variance.py` | `var_ppi(Y, f, N)` |
| `V̂_PPI++` | `V̂ar(Y−λ̂f)/n + λ̂²V̂ar(f)/N`, **labelled-sample** `ddof=1` moments | §4.5 | `variance.py` | `var_ppi_plus(Y, f, lam, N)` |
| **leverage `h`** | **`(f̄_N − f̄_n)²/S_ff`** | **§4.6** | **`variance.py`** | **`leverage_term()`** |
| **`V̂_LC`** | **`s²(1/n + h) + b²σ̂_f²/N`** | **§4.8.2** | **`variance.py`** | **`var_lc_ppi(mode="v482")`** |
| `V̂_LC` (N→∞) | `s²(1/n + h)` | §4.8.1 | `variance.py` | `var_lc_ppi(N=inf)` |
| `V_full` | derived, **unverified** | §4.8.3 | `variance.py` | `var_lc_ppi(mode="v_full")` |
| R1 expectation form | `max(0, Δ² − σ̂_f²/N)` substitution | §4.8.4 | `variance.py` | `var_lc_ppi(mode="r1")` |
| R2 Satterthwaite df | effective ν | §4.8.4 | `variance.py` | `var_lc_ppi(mode="r2")` → also returns `dof` |
| R3 conditional form | conditional-bias correction | §4.8.4 | `variance.py` | `var_lc_ppi(mode="r3")` |
| R4 conservativeness | current form, claim only | §4.8.4 | — | no code; a theorem about `mode="v482"` |
| HC3 sandwich | heteroskedasticity-consistent | Exp. H | `variance.py` | `var_lc_ppi(mode="hc3")` |
| no-leverage control | `s²/n + b²σ̂_f²/N` | §4.8 note | `variance.py` | `var_lc_ppi(mode="nolev")` |
| Interval from `(θ̂, se, ν)` | `θ̂ ± q·se` | §4.2, §4.8 | `confidence_intervals.py` | `ci_from_se(dist="t"|"normal")` |
| **`CI_LC`** | **`θ̂_LC ± t_{n−2,1−α/2}√V̂_LC`** | **§4.8.2** | **`lc_ppi.py`** | **`lc_ppi_interval()`** |
| Bootstrap replicates | resampled `(Y,f)` pairs | B7/B8 | `bootstrap.py` | `bootstrap_replicates()` |
| Percentile CI | empirical quantiles | B7 | `bootstrap.py` | `ci_percentile()` |
| BCa CI | `ẑ₀`, `â`, adjusted percentiles | B8 | `bootstrap.py` | `ci_bca()` |
| Coverage | `P(θ ∈ CI)` | `06` | `metrics.py` | `coverage()` |
| MC standard error | `√(p(1−p)/R)` | `07` | `metrics.py` | `mc_se_proportion()` |
| Oracle gain fraction | `(V_ref−V_m)/(V_ref−V_oracle)` | `06` | `metrics.py` | `oracle_gain_fraction()` |
| Exact pivot `T` | `(θ̂−θ)/(s√(1/n+h))` | §4.7 | `metrics.py` | `studentised_pivot()` (for V12) |

The three bolded rows are the project's contribution. Everything else is scaffolding that exists so
those three can be measured honestly.

---

## 6. Public Python API

Re-exported from `ppi_finite/__init__.py`. Types use `numpy.typing.NDArray[np.float64]`; every
array argument accepts shape `(n,)` or `(R, n)` and the return shape follows.

### 6.1 Data construction

```python
def make_data(
    Y: ArrayLike,                       # (n,) or (R, n)
    f: ArrayLike,                       # same shape as Y
    *,
    f_unlabeled: ArrayLike | None = None,   # (N,) or (R, N)
    fbar_N: float | ArrayLike | None = None,
    mu_f: float | None = None,
    sigma2_f: float | None = None,
    N: int | float | None = None,
    theta_true: float | None = None,
) -> PPIData: ...
```

**Validation** (raises, never warns-and-guesses):
`ValueError` if `Y.shape != f.shape`; if neither `f_unlabeled` nor `(fbar_N, N)` nor `mu_f` is
given; if more than one of those three is given; if `n < 3`; if any NaN or non-finite value is
present. **Warns** (`UserWarning`) if `n == 3` (one degree of freedom) or if `N/n < 1000`
(the finite-N regime where `04` §4.8.2 is known inexact).

### 6.2 Estimators — one function per baseline family

```python
def estimate_classical(data: PPIData) -> PointEstimate: ...          # B1
def estimate_ppi(data: PPIData) -> PointEstimate: ...                # B2
def estimate_ppi_plus(
    data: PPIData, *,
    lambda_rule: Literal["plugin","crossfit","threshold","shrunk","oracle"] = "plugin",
    crossfit_scheme: Literal["2fold","kfold"] = "2fold",
    k: int = 5,
    t_crit: float = 2.0,
    shrinkage_c: float | None = None,
) -> PointEstimate: ...                                              # B3-B6
def estimate_lc_ppi(
    data: PPIData, *,
    coef_mode: Literal["ols","ppi_plus"] = "ols",
    shrinkage_c: float | None = None,
) -> PointEstimate: ...                                              # B9, B10
```

`PointEstimate` is a frozen dataclass carrying `theta_hat`, the λ actually used, and the sufficient
statistics, so the variance layer never recomputes them.

### 6.3 Variance and intervals

```python
def compute_variance(
    est: PointEstimate,
    data: PPIData,
    mode: Literal["classical","ppi","ppi_plus",
                  "lc_v482","lc_v_full","lc_r1","lc_r2","lc_r3",
                  "lc_hc3","lc_nolev"],
) -> VarianceResult: ...      # .value, .dof, .components dict, .leverage, .leverage_frac

def compute_interval(
    theta_hat: NDArray, se: NDArray, *,
    alpha_level: float = 0.05,
    dof: float | NDArray | None = None,
    dist: Literal["t","normal"] = "t",
) -> Interval: ...            # .low, .high, .width
```

`compute_variance` returning `dof` is what makes R2 (Satterthwaite) expressible without a special
case in the interval constructor — an instance of C3.

**Validation:** `compute_interval` raises if `alpha_level ∉ (0, 1)`, if `dist == "t"` and `dof` is
`None`, or if any `se < 0`. It asserts `low <= high` on return.

### 6.4 Bootstrap

```python
def bootstrap_ppi(
    data: PPIData, *,
    method: Literal["percentile","bca"],
    B: int = 2000,
    rng: np.random.Generator,
    lambda_rule: str = "plugin",
    resample_unlabeled: bool = False,
    alpha_level: float = 0.05,
) -> IntervalResult: ...
```

`rng` is **required and positional-by-keyword** — a bootstrap is the one method that legitimately
consumes randomness, and making it explicit keeps C1 auditable.

### 6.5 One-call convenience wrappers

These are what an applied user of the eventual package would call.

```python
def ppi_plus_interval(Y, f, f_unlabeled=None, *, alpha_level=0.05, **kw) -> IntervalResult: ...
def lc_ppi_interval(Y, f, f_unlabeled=None, *, alpha_level=0.05,
                    coef_mode="ols", variance_mode="v482",
                    shrinkage_c=None) -> IntervalResult: ...
```

`lc_ppi_interval` is the drop-in replacement for the `ppi-py` mean path and is the artifact a
practitioner would actually adopt. Its `extras` must include every diagnostic in `05` §Outputs.

### 6.6 Simulation, running, evaluation

```python
def simulate(
    dgp: Literal["gaussian","heavy_tail","skew","heteroskedastic","misspecified"],
    *, R: int, n: int, N: int | float, rho: float,
    rng: np.random.Generator,
    unlabeled_mode: Literal["mean_only","explicit"],
    **dgp_params,
) -> PPIData: ...

def run_cell(cell: CellSpec, methods: Sequence[str]) -> list[MetricRow]: ...
def run_experiment(config_path: str | Path, output_dir: str | Path) -> Path: ...

def evaluate_coverage(interval: Interval, theta_true: float) -> CoverageResult: ...
def evaluate_all(results: Mapping[str, IntervalResult], theta_true: float,
                 reference: str = "B1") -> list[MetricRow]: ...
```

`run_cell` takes **method ID strings**, not callables. That is C1 made structural: the caller cannot
hand in something that draws its own data.

---

## 7. Experiment orchestration

### 7.1 Config-driven, one runner

**`ARCHITECTURAL CHOICE`, and a deliberate departure from `11`.** `11` listed nine per-experiment
scripts. Experiments A–I differ only in their parameter grids, method lists, and DGP — never in
their logic. Nine scripts would be nine copies of the same loop, and C4 forbids that.

Resolution: **one runner, nine configs.** `experiments/run.py --config configs/experiment_X.toml`.
Thin named shortcuts may exist in the `Makefile` for discoverability, but they only invoke `run.py`.

### 7.2 Config format

TOML, parsed with the standard-library `tomllib`. Chosen over YAML because it needs no dependency,
and over JSON because it supports comments — and experiment configs need comments.

```toml
# configs/experiment_A_reproduction.toml
experiment_id   = "A_reproduction"
description     = "Reproduce the preliminary Gaussian result. Docs: 09 Tables A1-A3."
seed_entropy    = 20260819           # BASE_ENTROPY; never change without a log entry
R               = 200_000
alpha_levels    = [0.05, 0.10, 0.01]
unlabeled_mode  = "mean_only"
coef_mode       = "ols"

methods = ["B1", "B2", "B3", "B4", "B5", "B6", "B9-nolev", "B9"]

[dgp]
family   = "gaussian"
var_Y    = 1.0
var_f    = 1.0
mean_Y   = 0.0
mean_f   = 0.0

[sweep]
n   = [10, 20, 50]
rho = [0.0, 0.5, 0.9]
N   = [1_000_000, 1_000, 200]

[outputs]
save_raw   = true
figures    = ["coverage_vs_n", "width_vs_rho", "coverage_vs_N"]
```

`config.py` validates: every method ID exists in the registry; every sweep list is non-empty; `R`
is a positive integer; `alpha_levels ⊂ (0,1)`; **unknown keys raise** rather than being ignored — a
silently-ignored typo in a config is a silently-wrong experiment.

### 7.3 Sweep expansion and cell identity

The Cartesian product of `[sweep]` × `alpha_levels` × DGP-parameter lists gives the cells. Each cell
gets a deterministic identity:

```python
cell_id = f"{experiment_id}|n={n}|N={N}|rho={rho}|alpha={alpha}|dgp={dgp_hash}"
```

`cell_id` goes into the CSV, so any row can be traced back to the exact configuration.

### 7.4 Seed handling

Implements `09` §Seed policy exactly.

**Corrected 2026-08-19.** This block previously read `root.spawn_key_from(...)`, which is not
a NumPy API. The equivalent that does exist is a `SeedSequence` constructed with an explicit
`spawn_key`; it has both properties this section requires. The `root` line is kept for exposition.

```python
root = np.random.SeedSequence(entropy=config.seed_entropy)   # exposition only; see below
cell_ss = np.random.SeedSequence(                  # deterministic, collision-free
    entropy=config.seed_entropy,
    spawn_key=(experiment_index, n_index, N_index, rho_index, alpha_index, dgp_index))
cell_rng = np.random.default_rng(cell_ss)           # the ONLY data-drawing generator
boot_ss  = cell_ss.spawn(1)[0]                          # bootstrap only
boot_rng = np.random.default_rng(boot_ss)               # ONE PER STOCHASTIC METHOD, see below
```

> **Refined 2026-08-19 (extension E3).** `boot_rng` must be constructed **freshly for each
> stochastic method** from the same `boot_ss`, not created once and shared. A single shared
> generator object breaks the method-insertion-invariance this section promises: whichever
> bootstrap method ran first would consume the stream, so adding B8 to a config would silently
> change B7's intervals. Constructing one generator per method from the same seed also puts B7 and
> B8 on the **identical** resamples, which is the paired-comparison logic C1 already applies to the
> simulated data. `simulation.cell_rng` therefore returns a `SeedSequence`, and `run_cell` does the
> construction. `tests/test_crn.py` asserts both properties.

Two properties this buys, both required:

- **Reordering-invariance.** Indices come from the sorted sweep values, not from iteration order, so
  adding a value to a sweep does not change any existing cell's draws.
- **Method-insertion-invariance.** The bootstrap child generator is spawned separately, so adding
  B7/B8 to a config does not perturb the draws seen by B1–B6, B9, B10.

The legacy formulas from the preliminary scripts (`31*n + int(100*rho) + N` etc.) are recorded in
`09` for provenance and **must not be reused** — they collide.

### 7.5 Execution

```
for cell in expand(config):
    rng      = derive(cell)
    data     = simulate(...)            # ONCE  ← C1 enforced here and only here
    view     = data.view()              # theta_true stripped
    results  = {m: REGISTRY[m](view, cell.alpha, child_rng_if_bootstrap) for m in cell.methods}
    rows     = evaluate_all(results, data.theta_true, reference="B1")
    emit(rows, raw=results if cell.save_raw else None)
```

Chunking over `R` (for memory, in `explicit` unlabelled mode) happens **inside** `simulate` and the
method loop, accumulating metric counters — never by re-seeding, which would break C1.

### 7.6 Output directory structure

```
results/
├── A_reproduction.csv
├── A_reproduction.meta.json
├── A_reproduction_raw/
│   ├── cell=A_reproduction|n=10|N=1000000|rho=0.0|alpha=0.05.npz
│   └── ...
├── B_bootstrap.csv
└── ...
figures/
├── A_coverage_vs_n.png / .pdf
├── C_deficit_vs_inverse_n.png
└── ...
```

---

## 8. Result storage

**No database.** `17`'s requirement is transparency, and a CSV a human can open satisfies it.

### 8.1 Tidy CSV schema

One row per `(experiment, method, cell, metric)`. Long format, not wide — new metrics and new
methods then require no schema migration.

| Column | Type | Source |
|---|---|---|
| `experiment_id` | str | config |
| `cell_id` | str | §7.3 |
| `method` | str | registry ID, e.g. `B9` |
| `dgp` | str | e.g. `gaussian`, `heavy_tail:nu=5` |
| `n` | int | sweep |
| `N` | int/float | sweep (`inf` allowed) |
| `rho` | float | sweep |
| `alpha_level` | float | sweep |
| `R` | int | config |
| `unlabeled_mode` | str | config |
| `coef_mode` | str | config |
| `variance_mode` | str | method |
| `metric` | str | see below |
| `value` | float | computed |
| `mc_se` | float | Monte-Carlo s.e. of `value` (`NaN` where undefined) |
| `n_failed` | int | replications where construction failed |
| `seed_spec` | str | the spawn key, as a string |
| `timestamp_utc` | str | ISO-8601 |
| `python_version`, `numpy_version`, `scipy_version` | str | runtime |
| `git_commit` | str | `git rev-parse HEAD`, with `-dirty` if applicable |

`metric ∈ {coverage, coverage_failures_as_miss, width_mean, width_median, width_ratio_b1,
mse, mse_ratio_b1, oracle_gain_fraction, tail_miscoverage_left, tail_miscoverage_right,
theta_hat_mean, theta_hat_sd, seconds_per_interval, leverage_frac_mean}`.

> **Amended 2026-08-19.** `theta_hat_mean` and `theta_hat_sd` were missing, yet `09` V2 *requires*
> mean `θ̂` and, in its amended z-test form, `sd(θ̂)` as well — so the schema could not express its
> own verification check. `coverage_failures_as_miss` is the second coverage figure required by
> `06` §Reporting item 6.

### 8.2 Provenance sidecar

`results/<experiment_id>.meta.json` — the full resolved config, the command line, wall-clock
seconds, resolved package versions, the git commit, and the hostname. **A results file without a
sidecar is not a result** (`11` §Logging).

### 8.3 Raw per-replication outcomes

`.npz` per cell, containing `covered` (bool `(R,)`), `width` (`(R,)`), `theta_hat` (`(R,)`) per
method. This is what makes tail behaviour and failure accounting recomputable without re-running a
200,000-replication cell. Git-ignored; regenerable from the sidecar's command line.

Parquet is permitted **only** if the raw store exceeds a few hundred MB and `pyarrow` is already
present. Do not add the dependency pre-emptively.

---

## 9. Testing strategy

Five tiers, mapping to `11` §Testing requirements.

| Tier | File(s) | What it protects | Speed |
|---|---|---|---|
| **Unit** | `test_estimators.py`, `test_variance.py`, `test_intervals.py` | every L1 formula against an independent computation: `scipy.stats.linregress` to 1e-12, a hand-computed 5-point example, `scipy.stats.t.interval` | fast |
| **Mathematical** | `test_identity.py`, `test_pivot_ks.py` | V11 (θ̂ = OLS fitted value, 1e-10) and V12 (KS test of the studentised pivot against `t_{n−2}`) | V11 fast, V12 slow |
| **Invariant** | across the above | `V̂ ≥ 0`; `V̂_LC ≥ V̂_nolev`; `low ≤ θ̂ ≤ high`; width decreasing in `alpha_level`; `λ=0` ⇒ classical; `λ=1` ⇒ PPI; `leverage = 0` when `f̄_N = f̄_n`; `S_ff = 0` falls back to B1 | fast |
| **Reproduction** | `test_reproduction.py` | Tables A1–A5 at reduced `R = 20,000` (CI) and full `R` (nightly) | slow |
| **Regression** | `test_determinism.py`, golden CSVs | byte-identical output for a fixed seed spec; a stored `smoke.toml` result that must not drift | fast |

Plus two architectural tests that have no statistical content but protect C1 and §4.3:

- `test_crn.py` — within a cell, every method receives arrays that are `is`-identical, and mutating
  `theta_true` leaves all outputs bit-identical.
- `test_registry.py` — every ID in `06` resolves; every callable has the uniform signature; no ID
  exists that is not documented in `06`.

Default `pytest` run (excluding `@pytest.mark.slow`) must finish under 60 seconds.

---

## 10. Reproducibility strategy

1. **One command per artifact.** `make reproduce` regenerates every CSV; `make figures` regenerates
   every figure from CSVs; `make verify` runs V1–V12 and exits non-zero on failure.
2. **Configs are the experiment.** Changing a parameter means editing a TOML file that is in git,
   not editing code.
3. **Provenance in every row**, not only in the sidecar, so a CSV pasted into a paper draft is still
   traceable.
4. **Determinism is tested**, not assumed.
5. **No notebook is load-bearing.** `notebooks/` may exist for exploration; nothing in `results/` or
   `figures/` may originate there.
6. **Two pin files.** `pyproject.toml` holds minimum versions for installability;
   `requirements.txt` holds the exact versions used for the recorded results.
7. **Tolerances live in one place** — a `TOLERANCES` constant in `verify.py`, transcribed from `09`
   §Tolerances, with the doc reference in a comment. Changing it requires a `13` entry.

---

## 11. MVP

### 11.1 Definition

The MVP is the minimum system that can test the **central hypothesis** — H1 (PPI++ under-covers) and
H3 (the leverage correction fixes it). Nothing more.

**In the MVP:**

| Component | Scope |
|---|---|
| `types.py` | `PPIData`, `PPIView`, `IntervalResult`, `CellSpec` |
| `estimators.py` | sufficient statistics, `theta_classical`, `theta_ppi`, `theta_ppi_plus`, `lambda_plugin` |
| `variance.py` | `var_classical`, `var_ppi`, `var_ppi_plus`, `leverage_term`, `var_lc_ppi(mode="v482"\|"nolev")` |
| `confidence_intervals.py` | `ci_from_se` |
| Methods | **B1, B2, B3, B9-nolev, B9** |
| `simulation.py` | `dgp_gaussian` (T1.1), `mean_only` mode, `run_cell` |
| `metrics.py` | `coverage`, `mc_se_proportion`, `width_stats` |
| `runner.py`, `config.py`, `io.py` | enough to run `configs/experiment_A_reproduction.toml` |
| `verify.py` | **V1, V2, V3, V4, V5, V6, V7, V8, V9, V11** |
| Tests | unit + invariant + `test_identity` + `test_crn` + `test_determinism` |

**Why `B9-nolev` is in the MVP even though the user's Part 19 list omits it.** V7 in `09` requires
it: it is the control that decomposes the deficit into a quantile part (≈ 0.937) and a leverage part
(≈ 0.013). Without it, a successful reproduction cannot distinguish "the leverage term did the work"
from "the Student quantile did the work" — which is the project's central causal claim. It costs one
extra `variance_mode` string. **This is a deliberate, justified addition and is logged.**

**Why B2 (plain PPI) is in the MVP.** It is the only PPI-family method with **no estimated
coefficient**, so it isolates the effect of estimating λ. If B2 covers well and B3 does not, that is
direct evidence for the diagnosis in `04` §4.6 (`06` B2).

> **Amended 2026-08-19 — V4 and V9 promoted into the MVP gate.** The exclusion below was
> wrong on its own stated grounds. **V4** is the spread of B3 coverage across ρ at fixed n; it
> needs only B3, which is in the MVP. **V9** is B9 coverage at N = 200, n = 10, ρ = 0.9; the MVP's
> Experiment A config already sweeps `N = [10⁶, 1000, 200]`, so the cell exists. Both are computed
> from `results/A_reproduction.csv` with no new code, and V4 is the direct test of H1's
> ρ-invariance clause — the MVP should not be able to pass while it fails. V10 (needs B5,
> extension E1) and V12 (the slow KS test) remain excluded.

**Excluded from the MVP:** V10 and V12 (they need B5 or the slow KS test); B4, B5,
B6, B7, B8, B10; every DGP but T1.1; `realdata.py`; `plots.py`; `bootstrap.py`.

### 11.2 MVP acceptance

The MVP is done when, from a clean checkout:

```
make test        # fast suite green, < 60 s
make reproduce   # results/A_reproduction.csv at R = 200,000, N = 10⁶
make verify      # V1-V9 (bar V10) and V11 all pass → exit 0
```

and the CSV shows B3 ≈ 0.878 and B9 ≈ 0.950 at n = 10, within `09`'s ± 0.004.

At that moment H1 and H3 have been independently reproduced and the project's premise is confirmed
or refuted. Everything after is elaboration.

### 11.3 Optional extensions, in dependency order

| Ext. | Adds | Unlocks | Docs |
|---|---|---|---|
| E1 | B4, B5, B6 + `lambda_crossfit`, `lambda_threshold` | **H2**, V4, V10, Table A4 | `06`, `09` |
| E2 | `plots.py`, `make_figures.py` | Experiments C, D and their diagnostic figures | `07` |
| | *Done 2026-08-20. `plots.py` reads `results/` only — tidy CSVs and the per-replication `.npz` store — and recomputes no statistic. `make figures` regenerates all 16. Experiments C and D were found never to have been run and were run at the same time.* | | |
| E3 | `bootstrap.py` (B7, B8) | **K3**, Experiment B | `06`, `10` |
| | *Done 2026-08-19. K3 did not fire; see `10` K3 and `13`.* | | |
| E4 | `explicit` unlabelled mode, `var_lc_ppi` modes `v_full`, `r1`–`r3` | **Q1**, Experiment E, V9 | `04` §4.8.3–4.8.4 |
| | *E4c (2026-08-20) adds `theory.py` — the idealised statistic, its exact coverage integral and the ablations that separate the idealisations. It is L1 and deliberately **unregistered**: its objects use population quantities, so they are derivation tools, not methods. It lives in `src/` because `11` forbids a variance formula in `experiments/`.* | | |
| | *Done 2026-08-20. `04` §4.8.3–4.8.4 updated; R2 identified as the working repair; exactness still open. `make_data` now accepts `mu_f` alongside `fbar_N`, carried for the `v_full` diagnostic only and never used by an estimator.* | | |
| E5 | DGPs T1.2–T1.4, `var_lc_ppi(mode="hc3")` | **H4**, **K4**, Experiments F, G, H | `08` |
| | *Done 2026-08-20. H4 fails on its relative clause, K4 does not fire, K3's second clause settled. HC3 is implemented as `variance.var_lc_hc3` + method `B9-HC` rather than a `var_lc_ppi` mode, because it needs the raw `(Y, f)` and not just the sufficient statistics. `config.py` gained the ability to sweep **one** DGP parameter, which fills the `dgp_index` slot §7.4 already reserved.* | | |
| E6 | `realdata.py`, Tier 2 then Tier 3 | Experiment I | `08` |
| | *Done 2026-08-20. `realdata.py` implements `load_population` and `subsample_protocol` and registers a `galaxies` producer in `simulation.DGPS`, so the existing runner drives the real-data experiment unchanged. `plots.py` and `experiments/make_figures.py` (extension E2) were built here because E6 is the first experiment with a figure deliverable. `config.py` gained `FAMILY_SWEEPS`: real-data families omit `rho` and `N` from `[sweep]` because those are observed, and `run_cell` now returns a `realised` dict that the runner writes into the CSV in their place.* | | |
| E7 | M-estimation | H5 | `10` — do not start before E4 completes |

E3 before E4 is deliberate: E3 can change what the project claims (K3), and it is cheaper to learn
that before investing in the finite-N theory.

---

## 12. Claude Code implementation sequence

The architecture-level sequence. It refines `12`'s twelve stages without contradicting them; the
mapping is given in the last column.

| # | Task | Done when | `12` stage |
|---|---|---|---|
| 0 | **Prior art.** Cochran, Särndal–Swensson–Wretman, control-variates; full text of arXiv:2505.20178. Write `docs/prior_art_findings.md`. **No code before this.** | K1 answered with citations | 0 |
| 1 | Repo skeleton per §2; `pyproject.toml`, `requirements.txt`, `Makefile`, `pytest` config | `import ppi_finite` works; empty suite green | 1 |
| 2 | `types.py` — the three contracts, `PPIView` projection, validation | `test_crn.py`'s `theta_true` isolation test passes | 1 |
| 3 | `estimators.py` — batched sufficient statistics first | matches `linregress` to 1e-12; 5-point example by hand | 2 |
| 4 | `variance.py` — `classical`, `ppi`, `ppi_plus`, `leverage_term`, `lc(v482)`, `lc(nolev)` | invariants hold; `leverage = 0` when `f̄_N = f̄_n` | 2 |
| 5 | `confidence_intervals.py` | matches `scipy.stats.t.interval` to 1e-12; `low ≤ high` asserted | 2 |
| 6 | B1, B2 + `registry.py` | `λ=0` ⇒ classical, `λ=1` ⇒ PPI, exactly | 3 |
| 7 | B3 | normal multiplier is `1.959964…`; smoke coverage ≈ 0.878 at n = 10, R = 20,000 | 4 |
| 8 | **B9, B9-nolev** | **V11 passes to 1e-10** | 5 |
| 9 | `simulation.py` T1.1 + `run_cell` (C1 enforced here) | DGP moments match targets at R = 10⁶ | 2 |
| 10 | `metrics.py`, `config.py`, `runner.py`, `io.py` | `run.py --config configs/smoke.toml` writes a valid CSV + sidecar | 1/7 |
| 11 | `verify.py` with the MVP checks | exits non-zero on a deliberately broken variance | 7 |
| 12 | **Experiment A at R = 200,000; run `verify`** | **MVP acceptance (§11.2)** | 7 |
| 13 | E1 → B4, B5, B6; V4, V10; Table A4 | H2 confirmed or refuted, in writing | 8 |
| 14 | E2 → `plots.py`; Experiments C, D | `figures/C_deficit_vs_inverse_n.png` exists | 8 |
| 15 | E3 → `bootstrap.py`; Experiment B | **K3 verdict written in `13`**; BCa validated against `scipy.stats.bootstrap` | 9 |
| 16 | E5 → T1.2–T1.4, HC3; Experiments F, G, H | H4/K4 verdict; validity boundary quantified | 10 |
| 17 | E4 → `explicit` mode, `v_full`, R1–R3; Experiment E | **(V_full) verified or refuted**; `04` updated | 11 |
| 18 | E6 → `realdata.py`; Experiment I | Tier-2 CSV + `docs/tier2_<name>.md` + FPC caveat | 12 |

Steps 15 and 17 are transposed relative to a naive reading of `12`'s stage numbers (Stage 9 before
Stage 11 in `12`; here E3 before E4). This matches `12`'s own priority note — Stage 9 is labelled
"highest research priority after Stage 7" — and `14`'s work queue, which places Q3 (bootstrap) above
Q1 (finite-N). **Consistent, but flagged so the ordering is not read as an error.**

---

## 13. CONFLICT / RESOLUTION REQUIRED

Conflicts found while deriving this architecture against the existing package. Each is reported
rather than silently resolved, per Part 22 of the task and `12` forbidden shortcut 1.

### CONFLICT 1 — Minimum Python version

- `09` §Environment: "Python ≥ 3.10".
- This document §7.2: configs are TOML parsed with `tomllib`, which is **standard library only from
  Python 3.11**.

**RESOLUTION (applied):** raise the minimum to **Python 3.11**, matching the reference environment
in which the preliminary numbers were produced (3.11.15). `09` must be edited to say ≥ 3.11.
**Alternative if 3.10 support is required:** use JSON configs (stdlib, all versions) and lose
comments, or add `tomli` as a dependency. Recommendation: 3.11, since it is already the reference
environment and adds no dependency. **Action: edit `09` §Environment.**

### CONFLICT 2 — Per-experiment scripts vs. a single config-driven runner

- `11` §Repository structure lists nine scripts: `exp_b_bootstrap.py`, `exp_c_sample_size.py`, …
- Part 16 of the task, and C4, require a configuration-driven experiment system.

**RESOLUTION (applied):** one `experiments/run.py` plus nine files in `configs/`. The nine script
names survive only as `Makefile` targets that invoke `run.py --config …`. This preserves `11`'s
discoverability intent while satisfying C4. **Action: edit `11` §Repository structure, or treat §2
of this document as superseding it — the latter is recorded in §0.3.**

### CONFLICT 3 — Repository tree completeness

- `11`'s tree has no `configs/`, `data/`, `requirements.txt`, `Makefile`, or `notebooks/` (it has
  `notebooks/`, but not the others).
- The task's Part 9 example tree and this architecture require them.

**RESOLUTION (applied):** §2 of this document is a strict superset of `11`'s tree and supersedes it.
No formula, module responsibility, or test requirement in `11` changes. **Action: none required
beyond the note in §0.3.**

### CONFLICT 4 — `variance_mode` naming

- `11` §`lc_ppi.py` specifies `variance_mode ∈ {"v482","v_full","r1","r2","r3","hc3"}`.
- This document §6.3 uses the prefixed forms `{"lc_v482","lc_v_full","lc_r1",…}` in
  `compute_variance`, because that function also serves the non-LC modes (`classical`, `ppi`,
  `ppi_plus`).

**RESOLUTION (applied):** both spellings are kept, at different levels. `compute_variance` (the
generic L1 entry point) uses the prefixed names; `lc_ppi_interval` (the user-facing wrapper of `11`)
keeps the unprefixed ones and maps them internally. `registry.py` must expose the method IDs from
`06` (`B9`, `B9-R1`, …) and nothing else, so neither spelling leaks into results files.
**Action: document the mapping in the `lc_ppi.py` docstring.**

### CONFLICT 5 — Ordering of Stage 9 (bootstrap) and Stage 11 (finite-N)

- `12` numbers the bootstrap Stage 9 and finite-N Stage 11, but labels Stage 9 "highest research
  priority after Stage 7".
- `14`'s work queue places Q3 (bootstrap) above Q1 (finite-N).
- §12 above sequences E3 (bootstrap) before E4 (finite-N), i.e. step 15 before step 17.

**RESOLUTION:** not a true contradiction — `12`'s numbering is a dependency order, not a priority
order, and its own note says so. This document makes the priority explicit. **Action: none, but the
reader should not treat `12`'s stage numbers as a strict priority ranking.**

### NON-CONFLICT, recorded to prevent a future one

`09` Table A4's last two columns are labelled `B10-ish c=1` / `B10-ish c=3` and are explicitly **not**
the B10 defined in `06` — they apply shrinkage on top of the **B3** variance, not on top of B9. `09`
already instructs that they be coded as `B6-shrink-c1` / `B6-shrink-c3`. This architecture adopts
those IDs in `registry.py`. Anyone who reads only Table A4 will get this wrong; the registry names
are the defence.

---

## 14. Architecture status summary

| Element | Status |
|---|---|
| Layering, data flow, math→code mapping | **Derived** from `04`, `06`, `09` |
| C1–C5 constraints | **Forced** by the research documents cited |
| Config format (TOML), single runner, no parallelism framework | **`ARCHITECTURAL CHOICE`** with rationale given |
| MVP contents | **Derived** from H1/H3 and the V-checks they require |
| Extension order E1–E7 | **Derived** from `10` (kill criteria) and `14` (work queue) |
| Five conflicts above | **Reported**, four resolved with an action, one dismissed as apparent |

Nothing in this document changes a single equation, hypothesis, tolerance, or expected number. If it
appears to, `04` and `09` win and the discrepancy goes in `13`.
