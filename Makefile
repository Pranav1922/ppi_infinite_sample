PY      ?= python3
export PYTHONPATH := src

RESULTS := results

.PHONY: all test tables figures manuscript check-claims reproduce reproduce-C reproduce-D reproduce-A reproduce-A4 reproduce-A5 reproduce-B reproduce-B-final reproduce-E reproduce-E-largeN reproduce-E-convergence \
        reproduce-I figures-I reproduce-J figures-J verify smoke clean

all: test reproduce verify

test:
	$(PY) -m pytest

test-slow:
	$(PY) -m pytest -m slow

smoke:
	$(PY) experiments/run.py --config configs/smoke.toml --output $(RESULTS)

# Experiment A and its Table A4 / A5 companions (09 Tables A1-A5).
reproduce: reproduce-A reproduce-A4 reproduce-A5 reproduce-C reproduce-D \
           reproduce-B reproduce-B-final \
           reproduce-E reproduce-E-largeN reproduce-E-convergence

reproduce-A:
	$(PY) experiments/run.py --config configs/experiment_A_reproduction.toml --output $(RESULTS)

reproduce-A4:
	$(PY) experiments/run.py --config configs/experiment_A4_simple_fixes.toml --output $(RESULTS)

reproduce-A5:
	$(PY) experiments/run.py --config configs/experiment_A5_mse.toml --output $(RESULTS)

# Experiment B is the expensive one: ~6 min at B = 2000 and ~8 min at B = 9999 on one thread.
reproduce-B:
	$(PY) experiments/run.py --config configs/experiment_B_bootstrap.toml --output $(RESULTS)

reproduce-B-final:
	$(PY) experiments/run.py --config configs/experiment_B_final.toml --output $(RESULTS)

# Experiment E: the finite-N arm is explicit-mode and ~2 min; the convergence arm ~6 min.
reproduce-E:
	$(PY) experiments/run.py --config configs/experiment_E_finite_N.toml --output $(RESULTS)

reproduce-E-largeN:
	$(PY) experiments/run.py --config configs/experiment_E_largeN.toml --output $(RESULTS)

reproduce-E-convergence:
	$(PY) experiments/run.py --config configs/experiment_E_convergence.toml --output $(RESULTS)
	$(PY) experiments/run.py --config configs/experiment_E_convergence_meanonly.toml --output $(RESULTS)

reproduce-C:
	$(PY) experiments/run.py --config configs/experiment_C_sample_size.toml --output $(RESULTS)

reproduce-D:
	$(PY) experiments/run.py --config configs/experiment_D_predictor_quality.toml --output $(RESULTS)

# Paper tables, generated from saved results only. Never edit docs/17 by hand.
tables:
	$(PY) experiments/make_tables.py > docs/17_PAPER_TABLES.md

# Every figure in 07, from saved results only.
figures:
	$(PY) experiments/make_figures.py --all --results $(RESULTS)
	$(PY) experiments/make_figures.py --csv $(RESULTS)/I_real_data.csv --prefix I

# Experiment I: real-data validation on the Galaxies population (E6).
reproduce-I:
	$(PY) experiments/run.py --config configs/experiment_I_real_data.toml --output $(RESULTS)

figures-I:
	$(PY) experiments/make_figures.py --csv $(RESULTS)/I_real_data.csv --prefix I

# Experiment J: second real-data demonstration, MT-bench expert judgments with a GPT-4 judge.
data/mtbench_gpt35.npz:
	$(PY) experiments/prepare_mtbench.py

reproduce-J: data/mtbench_gpt35.npz
	$(PY) experiments/run.py --config configs/experiment_J_mtbench.toml --output $(RESULTS)

figures-J:
	$(PY) experiments/make_figures.py --csv $(RESULTS)/J_mtbench.csv --prefix J

# Final manuscript, assembled from the draft, the generated tables and the references.
manuscript:
	$(PY) experiments/make_manuscript.py > docs/MANUSCRIPT.md

# Claim-consistency check: the executable form of docs/FINAL_CLAIM_AUDIT.md.
check-claims:
	$(PY) experiments/check_claims.py

verify:
	$(PY) experiments/verify.py --tables $(RESULTS)/A_reproduction.csv \
	      --table-a4 $(RESULTS)/A4_simple_fixes.csv --table-a5 $(RESULTS)/A5_mse.csv \
	      --table-b $(RESULTS)/B_bootstrap.csv --table-b-final $(RESULTS)/B_final.csv \
	      --table-e $(RESULTS)/E_finite_N.csv $(RESULTS)/E_largeN.csv

clean:
	rm -rf $(RESULTS)/*.csv $(RESULTS)/*.meta.json $(RESULTS)/*_raw .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

check-submission:
	python3 experiments/check_submission.py

manuscript-docx:
	python3 experiments/make_docx.py

# Statistics and Computing: the journal's artwork, front-matter and reference rules, and the
# submission package (docs/STATISTICS_AND_COMPUTING_SUBMISSION_AUDIT.md).
journal-figures:
	python3 experiments/make_journal_figures.py

check-journal:
	python3 experiments/check_journal.py

submission-package: journal-figures manuscript manuscript-docx check-pdf check-journal
	python3 experiments/make_submission_package.py

check-pdf:
	python3 experiments/docxbuild/omml.py
	python3 experiments/docxbuild/equations.py
	python3 experiments/check_manuscript_pdf.py

gate1-validate:
	python3 experiments/gate1_validate.py

gate2:
	python3 experiments/gate2_run.py
	python3 experiments/gate2_figures.py

gate4:
	python3 experiments/gate4_run.py
	python3 experiments/gate4_figures.py
