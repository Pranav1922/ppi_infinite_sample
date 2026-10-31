r"""Full repair-extension grid. Writes only under `results_repair_extension/`.

Uses the paper's own harness: the T1.1 Gaussian DGP, `explicit` unlabelled mode, one draw per
cell shared by every method (common random numbers), and the SeedSequence spawn-key scheme.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ppi_finite import registry                                    # noqa: E402
from ppi_finite.estimators import labelled_summaries               # noqa: E402
from ppi_finite.simulation import dgp_gaussian                     # noqa: E402
from repair_ext.applicability import as_rows                       # noqa: E402
from repair_ext.methods import method_ppboot, method_ridge_ppi     # noqa: E402

OUT = ROOT / "results_repair_extension"
BASE_ENTROPY = 20260825          # Gate-extension entropy; never reused from the paper's cells
R_REPS = 20_000                  # matches Experiment B (bootstraps); MC s.e. 0.0015 near 0.95
B_BOOT = 2_000                   # matches Experiment B
ALPHA = 0.05
TOLERANCE = 0.005                # the paper's own +-0.005 band for "attains nominal"

EXISTING = ["B1", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B9-tmin"]
STOCHASTIC = {"B7", "B8"}

GRID_N = [4, 5, 8, 10, 15, 20, 50]
GRID_RHO = [0.0, 0.5, 0.9]
GRID_POOL = [50, 200, 1000]
#: Q7 probe: the G8 region N < n-1 at a strong predictor, which the main grid does not reach.
PROBE = [(20, 0.99, 5), (20, 0.99, 10), (20, 0.99, 15), (20, 0.99, 25),
         (50, 0.99, 10), (50, 0.99, 25), (50, 0.99, 49), (50, 0.99, 60),
         (50, 0.9, 10), (50, 0.9, 25), (50, 0.9, 49), (50, 0.9, 60)]


def cell_seed(n, rho, N):
    ss = np.random.SeedSequence(entropy=BASE_ENTROPY,
                                spawn_key=(int(n), int(round(rho * 1000)), int(N)))
    return ss


def evaluate(n, rho, N, R=R_REPS):
    """One cell: draw once, evaluate every method on that draw."""
    ss = cell_seed(n, rho, N)
    rng = np.random.default_rng(ss)
    boot_ss = ss.spawn(1)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = dgp_gaussian(rng, R=R, n=n, rho=rho, N=float(N),
                            unlabeled_mode="explicit", keep_pool=True)
    view = data.view()
    summ = labelled_summaries(view.Y, view.f)
    truth = np.broadcast_to(np.asarray(data.theta_true, float), (R,))

    results = {}
    timings = {}
    for mid in EXISTING:
        mrng = np.random.default_rng(boot_ss) if mid in STOCHASTIC else None
        t0 = time.perf_counter()
        results[mid] = registry.get(mid)(view, ALPHA, mrng, summaries=summ)
        timings[mid] = time.perf_counter() - t0
    t0 = time.perf_counter()
    results["Ridge-PPI"] = method_ridge_ppi(view, ALPHA)
    timings["Ridge-PPI"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    results["PPBoot"] = method_ppboot(view, ALPHA, np.random.default_rng(boot_ss),
                                      B=B_BOOT, pool=data.f_unlabeled)
    timings["PPBoot"] = time.perf_counter() - t0

    rows = []
    for mid, res in results.items():
        lo = np.asarray(res.ci_low, float)
        hi = np.asarray(res.ci_high, float)
        th = np.asarray(res.theta_hat, float)
        bad = ~(np.isfinite(lo) & np.isfinite(hi) & np.isfinite(th)) | (hi < lo)
        ok = ~bad
        cover = float(np.mean((lo[ok] <= truth[ok]) & (truth[ok] <= hi[ok]))) if ok.any() else np.nan
        rows.append({
            "method": mid, "n": n, "rho": rho, "N": N, "R": R, "alpha_level": ALPHA,
            "coverage": cover,
            "coverage_error": abs(cover - (1 - ALPHA)) if np.isfinite(cover) else np.nan,
            "signed_coverage_error": cover - (1 - ALPHA) if np.isfinite(cover) else np.nan,
            "mc_se": float(np.sqrt(max(cover, 1e-9) * (1 - cover) / max(int(ok.sum()), 1)))
                     if np.isfinite(cover) else np.nan,
            "width_mean": float(np.mean(hi[ok] - lo[ok])) if ok.any() else np.nan,
            "width_median": float(np.median(hi[ok] - lo[ok])) if ok.any() else np.nan,
            "bias": float(np.mean(th[ok] - truth[ok])) if ok.any() else np.nan,
            "mse": float(np.mean((th[ok] - truth[ok]) ** 2)) if ok.any() else np.nan,
            "n_invalid": int(bad.sum()),
            "n_failed_reported": int(getattr(res, "n_failed", 0) or 0),
            "seconds": round(timings[mid], 3),
        })
    return rows


def main() -> int:
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    cells = [(n, r, N) for n in GRID_N for r in GRID_RHO for N in GRID_POOL] + PROBE
    print(f"repair extension: {len(cells)} cells, R = {R_REPS:,}, B = {B_BOOT:,}", flush=True)

    rows, t0 = [], time.time()
    for i, (n, rho, N) in enumerate(cells, 1):
        rows += evaluate(n, rho, N)
        if i % 5 == 0 or i == len(cells):
            print(f"  {i}/{len(cells)}  n={n} rho={rho} N={N}   [{time.time()-t0:.0f}s]",
                  flush=True)

    def dump(name, fields):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    key = ["method", "n", "rho", "N", "R", "alpha_level"]
    dump("coverage_results.csv", key + ["coverage", "coverage_error",
                                        "signed_coverage_error", "mc_se", "n_invalid"])
    dump("interval_width_results.csv", key + ["width_mean", "width_median"])
    dump("bias_mse_results.csv", key + ["bias", "mse"])
    dump("invalid_runs.csv", key + ["n_invalid", "n_failed_reported"])

    with open(OUT / "method_definitions.csv", "w", newline="") as fh:
        defs = as_rows()
        w = csv.DictWriter(fh, fieldnames=list(defs[0]))
        w.writeheader(); w.writerows(defs)

    import scipy
    params = {
        "alpha_level": ALPHA, "R_replications": R_REPS, "bootstrap_B": B_BOOT,
        "nominal_level": 1 - ALPHA, "tolerance_for_attains_nominal": TOLERANCE,
        "dgp": "T1.1 bivariate Gaussian, var_Y = var_f = 1, mean_Y = mean_f = 0",
        "unlabeled_mode": "explicit (pool materialised; sigma_f^2 estimated with ddof=1)",
        "coef_mode": "ols",
        "grid_n": GRID_N, "grid_rho": GRID_RHO, "grid_pool_N": GRID_POOL, "probe_cells": PROBE,
        "methods_existing": EXISTING,
        "methods_new": ["Ridge-PPI", "PPBoot"],
        "methods_excluded": {
            "Sigmoid-PPI": "not applicable to the Gaussian DGP (binary h / f in [0,1] only)",
            "cross-PPBoot": "not applicable (no covariates, no model-fitting step to cross-fit)",
            "Efron PPBoot": "not a published method",
        },
        "ridge_alpha_selection": "leave-one-out CV on the labelled set, squared prediction error",
        "ridge_alpha_grid": "Var_hat[f] * ([0] + logspace(-3, 3, 25))",
        "ridge_interval": "OURS: PPI++ interval with lambda_hat_alpha (paper defines none)",
        "ppboot_interval": "percentile, per Zrnic Algorithm 1",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    }
    (OUT / "method_parameters.json").write_text(json.dumps(params, indent=2))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "base_entropy": BASE_ENTROPY,
        "spawn_key": "(n, round(rho*1000), N)  -- disjoint from the paper's (exp_index, ...) keys",
        "bootstrap_seed": "ss.spawn(1)[0]; a fresh generator per stochastic method",
        "git_commit": commit,
        "cells": [{"n": n, "rho": r, "N": N,
                   "state0": int(cell_seed(n, r, N).generate_state(1)[0])}
                  for (n, r, N) in cells],
    }
    (OUT / "seed_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {len(rows)} rows across 4 result CSVs  [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
