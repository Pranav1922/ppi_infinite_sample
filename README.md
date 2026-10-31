# ppi-finite-sample

Finite-sample-valid prediction-powered inference. **Start with
[`docs/README_RESEARCH_PACKAGE.md`](docs/README_RESEARCH_PACKAGE.md)**; the research package in
`docs/` is the source of truth, and `docs/15_ARCHITECTURE_SPECIFICATION.md` is the architecture
source of truth for the code.

Current state: **MVP + E1 + E3 + E4 + E5 + E6** (architecture §11) — methods B1–B9 (B10 excepted); the Gaussian
DGP T1.1 in `mean_only` unlabelled mode; Experiment A with its Table A4/A5 companions, and
Experiment B; Experiment E (finite N) in explicit unlabelled mode;
robustness (E5); and Experiment I on the real Galaxies population (E6), with figures.
Only M-estimation (E7) is not started. Experiments A-I of `docs/07` have all been run.

```
make test        # fast suite
make reproduce   # every results/*.csv; ~15 min, the bootstrap cells dominate
make verify      # V1-V11 (bar V12) + Tables A4/A5 + Experiment B -> exit 0/1
```

## Dashboard

A read-only view of the saved results:

```
streamlit run dashboard/app.py
```

It **reads** `results/*.csv` and `data/galaxies.npz` and writes nothing — no result, config, seed,
figure or manuscript file is modified by it. `tests/test_dashboard.py` enforces that by hashing
every artefact before and after rendering all five pages. See
[`dashboard/README.md`](dashboard/README.md). Needs `streamlit` and `altair`.

No novelty is claimed for the estimator (it is the survey-sampling GREG estimator — Mozer 2026) or
for the finite-N conservativeness theorem (Mickey & Brown 1966). See `docs/theory_finite_N.md`
Part III for the prior-art matrix and the revised contribution statement.

Headline so far: PPI++ covers ~0.878 at n = 10 where it claims 0.95; a Student-*t* quantile, three
cross-fitting schemes, three hard thresholds, and both a percentile and a BCa bootstrap all fail to
reach 0.945; the leverage-corrected interval reaches 0.950. At finite N it is conservative rather
than exact, and the gap is a degrees-of-freedom effect that a Satterthwaite correction removes to
±0.003. See `docs/13_RESEARCH_LOG.md`.

No novelty is claimed for the leverage correction: see `docs/prior_art_findings.md` (K1 is open).
