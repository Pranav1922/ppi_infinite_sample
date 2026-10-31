"""ppi-finite-sample — finite-sample-valid prediction-powered inference.

Public API (`15` §6).  Research package and sources of truth are in `docs/`; the mathematics is
`docs/04_MATHEMATICAL_FORMULATION.md`.  **No novelty is claimed for the leverage correction**:
kill criterion K1 is open (`docs/prior_art_findings.md`).

Current scope is the MVP of `15` §11 plus extension E1: methods B1, B2, B3, B4, B5
(+`B5-kfold`, `B5-avglam`), B6, B7, B8, B9-nolev, B9; DGP T1.1 in `mean_only` mode;
Experiment A and its Table A4 / A5 companions.
"""
from .bootstrap import (bootstrap_ppi, bootstrap_replicates, ci_bca, ci_percentile,
                        jackknife_theta, method_b7, method_b7_b9999, method_b8,
                        method_b8_b9999)
from .confidence_intervals import ci_from_se
from .estimators import (crossfit_folds, labelled_summaries, lambda_crossfit,
                         lambda_plugin, lambda_threshold, theta_classical, theta_ppi,
                         theta_ppi_plus)
from .lc_ppi import (lc_ppi_interval, method_b9, method_b9_nolev, method_b9_ppiplus,
                     method_b9_hc, method_b9_r1, method_b9_r2, method_b9_r3, method_b9_r5,
                     method_b9_vfull)
from .metrics import coverage, evaluate_all, mc_se_proportion, mse, width_stats
from .ppi import method_b1, method_b2
from .ppi_plus import (method_b3, method_b4, method_b5, method_b5_avglam,
                       method_b5_kfold, method_b6, method_b6_t15, method_b6_t3,
                       ppi_plus_interval)
from .registry import METHODS, get
from .runner import run_experiment
from .realdata import load_population, subsample_protocol
from .simulation import dgp_gaussian, run_cell, simulate
from .theory import (ABLATIONS, ablation_coverages, exact_coverage_integral,
                     idealised_coverage, population_share)
from .types import CellSpec, IntervalResult, PPIData, PPIView, make_data
from .variance import (leverage_term, var_classical, var_lc_hc3, var_lc_ppi, var_ppi,
                       var_ppi_plus)

__version__ = "0.1.0"

__all__ = [
    "make_data", "PPIData", "PPIView", "IntervalResult", "CellSpec",
    "labelled_summaries", "theta_classical", "theta_ppi", "theta_ppi_plus", "lambda_plugin",
    "lambda_crossfit", "lambda_threshold", "crossfit_folds",
    "var_classical", "var_ppi", "var_ppi_plus", "leverage_term", "var_lc_ppi", "var_lc_hc3",
    "ci_from_se", "lc_ppi_interval", "ppi_plus_interval",
    "method_b1", "method_b2", "method_b3", "method_b4", "method_b5", "method_b5_kfold",
    "method_b5_avglam", "method_b6", "method_b6_t15", "method_b6_t3",
    "method_b7", "method_b8", "method_b7_b9999", "method_b8_b9999", "bootstrap_ppi", "bootstrap_replicates",
    "ci_percentile", "ci_bca", "jackknife_theta", "method_b9", "method_b9_nolev", "method_b9_vfull", "method_b9_r1",
    "method_b9_r2", "method_b9_r3", "method_b9_r5", "method_b9_hc", "method_b9_ppiplus",
    "METHODS", "get",
    "coverage", "mc_se_proportion", "width_stats", "mse", "evaluate_all",
    "dgp_gaussian", "simulate", "run_cell", "run_experiment",
    "load_population", "subsample_protocol",
    "exact_coverage_integral", "population_share", "idealised_coverage",
    "ablation_coverages", "ABLATIONS",
    "__version__",
]
