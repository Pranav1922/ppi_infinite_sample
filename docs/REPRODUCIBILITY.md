# Reproducibility

Everything in the manuscript — every number, table and figure — is regenerable from this repository
by the commands below. Repository state when this was written: commit `e5f6a16`.

---

## 1. Environment

| component | version used |
|---|---|
| Python | 3.12.9 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| pandas | 3.0.2 |
| pytest | 9.1.1 |
| matplotlib | 3.10.8 |
| OS | macOS (darwin), single-threaded |

`pyproject.toml` pins **minimum** versions for installability (Python ≥ 3.11, NumPy ≥ 1.24,
SciPy ≥ 1.10, pandas ≥ 2.0); `requirements.txt` pins the **exact** versions above. Python 3.11 is
the floor because the config system uses `tomllib`. **No GPU is used anywhere.** The full
reproduction is roughly 20 minutes on one CPU thread, dominated by the bootstrap experiments.

The reference environment in which the project's initial preliminary numbers were produced
(Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, Linux x86-64) is recorded in
`docs/09_REPRODUCTION_PROTOCOL.md`. Results should be insensitive to these versions — the
computations are elementary arithmetic on arrays — and the resolved versions are written into
**every row** of every results CSV, not only the sidecar.

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No install of the package itself is needed: the `Makefile` sets `PYTHONPATH=src`.

## 3. Commands

| purpose | command | runtime |
|---|---|---|
| Fast test suite | `make test` | ~30 s |
| Slow statistical tests | `make test-slow` | minutes |
| Reproduce **all** experiments | `make reproduce` | ~20 min |
| — Experiment A (reproduction) | `make reproduce-A` | 17 s |
| — Experiments C, D (sweeps) | `make reproduce-C`, `make reproduce-D` | ~1 min each |
| — Experiment B (bootstrap) | `make reproduce-B`, `make reproduce-B-final` | 6 + 8 min |
| — Experiment E (finite N) | `make reproduce-E`, `-E-largeN`, `-E-convergence` | ~7 min |
| — Experiment I (real data) | `make reproduce-I` | 78 s |
| — Experiment J (MT-bench, LLM judge) | `python3 experiments/prepare_mtbench.py && make reproduce-J` | ~45 s + download |
| Verification checks | `make verify` | seconds |
| Regenerate the eight paper tables | `make tables` | seconds |
| Regenerate every figure | `make figures` | seconds |
| Assemble the manuscript | `make manuscript` | seconds |
| Claim-consistency check | `make check-claims` | seconds |
| Submission checklist | `make check-submission` | seconds |
| Typeset DOCX + PDF | `make manuscript-docx` | ~20 s |
| Verify the typeset PDF | `make check-pdf` | seconds |

Experiments F, G and H (robustness) are run by
`python experiments/run.py --config configs/experiment_{F,G,H}_*.toml --output results`.

**Expected state after a full run:** `make test` 274 passed; `make verify` 86 passed, 0 failed;
`make check-claims` 77 numeric claims, 0 problems; `make check-submission` 13 items, 0 failures;
`make check-pdf` 16 pages, 6 figures, 8 tables, 0 failures.

**Faculty-reviewed baseline.** The manuscript as reviewed by faculty (2026-09-02 build) is
preserved unchanged, with SHA-256 sums, in `docs/baseline_faculty_review_2026-09-02/`; the
2026-09-20 revision (mean-estimation scope, author–year references, Eq. (1) derivation, Corollary
1a open-problem framing, Experiment J) is built on top of it by the same pipeline.

**On page counts, and on inline maths.** `make check-pdf` reports LibreOffice's pagination, and
the `<Pages>` property stamped into the DOCX is that same number: it is **not** independent
evidence of how Word paginates. Word grows a line to the full height of any inline Office Math
object it contains, while LibreOffice packs those lines tightly, so a Word render of the same file
runs longer -- measured at +2 to +6 pages on earlier builds of this manuscript. `docxbuild`
therefore sets an inline span as an equation object **only** when it needs one (a script, accent,
radical or fraction: `equations.needs_equation`); plain notation such as `n = 10` is set as styled
text at prose line height. Treat a Word render as authoritative for any hard page limit.

**Typesetting.** `make manuscript-docx` renders `docs/MANUSCRIPT.md` to `docs/MANUSCRIPT.docx` and
converts it to PDF with LibreOffice (`soffice` must be on `PATH`). Display equations are real Office
Math, built from LaTeX that is keyed to the exact source text it replaces, so the build fails rather
than typeset a stale transcription. The body font is STIX Two Text with STIX Two Math as its
companion; both ship with macOS. Without them the build still succeeds and falls back to a single
face, but combining marks over Greek — the `λ̂` of the tuning coefficient — may not compose.

## 4. Random seed policy

* **One base entropy, `20260819`**, declared as `seed_entropy` in every config and never changed
  without an entry in `docs/13_RESEARCH_LOG.md`.
* Each experiment cell derives its own generator as
  `np.random.SeedSequence(entropy=seed_entropy, spawn_key=(experiment, n, N, rho, alpha, dgp))`,
  where the indices come from the **sorted** sweep values. Consequence: adding a value to a sweep
  does not change any existing cell's draws.
* **Common random numbers.** Within a cell the data are generated once and the identical arrays are
  passed to every method. This is structural — methods receive arrays, never a generator — and is
  asserted by `tests/test_crn.py`.
* **Bootstrap methods** receive a generator spawned from a separate child seed and **constructed
  afresh per method**, so adding one bootstrap method does not perturb another's draws, and B7 and
  B8 see identical resamples.
* The legacy seed formulas from the project's preliminary scripts are recorded in `docs/09` for
  provenance and are deliberately **not** reused; they can collide.
* Every results row carries its `seed_spec`.
* **Determinism is tested, not assumed:** two runs of a fixed configuration produce byte-identical
  CSVs apart from the timestamp columns and the single wall-clock metric
  (`seconds_per_interval`), which cannot be reproducible by nature.

## 5. Provenance

Every experiment writes `results/<id>.csv` plus `results/<id>.meta.json` recording the fully
resolved config, the command line, wall-clock seconds, hostname, git commit and all package
versions. **A results file without a sidecar is not a result.** The same provenance fields are
repeated on every CSV row so that a table pasted into a draft remains traceable.

## 6. The Galaxies dataset

**The raw file is not redistributed with this repository.** `data/` is git-ignored and contains no
tracked data.

**Why.** The distributor's licence (MIT, `ppi_py`) covers its **code, not its data**, and the
upstream Galaxy Zoo portal states no licence — only a request that the papers be cited. We could
locate no terms authorising redistribution, so we do not redistribute. See
`docs/tier2_galaxies.md` §1.

**Acquisition.**

```bash
pip install ppi-py gdown
python -c "from ppi_py.datasets import load_dataset; load_dataset('./data/', 'galaxies')"
```

**Verify you have the same bytes this work was run against:**

```bash
shasum -a 256 data/galaxies.npz
# 7be1d40a049f524fc51e150281541359a47c5eca80343dfac5b1c457fd7eae99
```

`tests/test_realdata.py` independently pins the population size (16,743), the estimand
(θ = 0.2592725318), the realised correlation (0.7407825907) and the mean prediction, so **a
different file fails the test suite rather than silently changing the results.** If
`data/galaxies.npz` is absent, those tests skip and every other test still runs.

**Provenance chain.** Galaxy Zoo 2 volunteer classifications of SDSS images (Willett et al., 2013)
→ a subset paired with computer-vision predictions → packaged as `galaxies.npz` and hosted on
Google Drive by `ppi_py` → the local file. Any publication using it should cite Willett et al.
(2013).

## 7. Repository layout

```
src/ppi_finite/     the package: types, estimators, variance, intervals, methods,
                    registry, metrics, simulation, realdata, theory, config, runner, io, plots
configs/            one TOML per experiment arm; the experiment IS the config
experiments/        run.py, verify.py, make_tables.py, make_figures.py,
                    make_manuscript.py, check_claims.py
tests/              274 tests
results/            tidy CSVs (tracked) + sidecars + per-replication npz (ignored)
figures/            the manuscript's PNGs
docs/               the research package, the theory, and the manuscript
```

## 8. Known non-reproducible element

`seconds_per_interval` is wall-clock and varies with the machine. It is excluded from the
determinism test and appears in the manuscript only as an order-of-magnitude comparison
(Figure 5, Table 3).
