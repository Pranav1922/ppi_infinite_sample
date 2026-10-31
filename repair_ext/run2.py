r"""Repair-comparison extension — checkpointed, restartable, one file per cell."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ppi_finite import registry                                       # noqa: E402
from ppi_finite.estimators import labelled_summaries                  # noqa: E402
from ppi_finite.simulation import dgp_gaussian                        # noqa: E402
from repair_ext.methods import (method_efron_ppboot2, method_ppboot,  # noqa: E402
                                method_ridge_ppi)

CFG = ROOT / "configs" / "experiment_repair_extension.toml"
OUT = ROOT / "results" / "repair_extension"
CELLS = OUT / "cells"
EXISTING = ["B1", "B3", "B9", "B9-tmin"]


def cell_ss(entropy, n, rho, N):
    return np.random.SeedSequence(entropy=entropy,
                                  spawn_key=(int(n), int(round(rho * 1000)), int(N)))


def run_cell(cfg, n, rho, N):
    R, B, alpha = cfg["R"], cfg["B_boot"], cfg["alpha_level"]
    ss = cell_ss(cfg["seed_entropy"], n, rho, N)
    boot_ss = ss.spawn(1)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = dgp_gaussian(np.random.default_rng(ss), R=R, n=n, rho=rho, N=float(N),
                            unlabeled_mode="explicit", keep_pool=True)
    view = data.view()
    summ = labelled_summaries(view.Y, view.f)
    truth = np.broadcast_to(np.asarray(data.theta_true, float), (R,))

    res = {}
    secs = {}
    for mid in EXISTING:
        t0 = time.perf_counter()
        res[mid] = registry.get(mid)(view, alpha, None, summaries=summ)
        secs[mid] = time.perf_counter() - t0
    for mid, fn in (("Ridge-PPI", lambda: method_ridge_ppi(view, alpha)),
                    ("PPBoot", lambda: method_ppboot(view, alpha,
                                                     np.random.default_rng(boot_ss), B=B,
                                                     pool=data.f_unlabeled)),
                    ("Efron-PPboot2", lambda: method_efron_ppboot2(
                        view, alpha, np.random.default_rng(boot_ss), B=B))):
        t0 = time.perf_counter()
        res[mid] = fn()
        secs[mid] = time.perf_counter() - t0

    rows = []
    for mid, r in res.items():
        lo, hi = np.asarray(r.ci_low, float), np.asarray(r.ci_high, float)
        th = np.asarray(r.theta_hat, float)
        bad = ~(np.isfinite(lo) & np.isfinite(hi) & np.isfinite(th)) | (hi < lo)
        ok = ~bad
        cov = float(np.mean((lo[ok] <= truth[ok]) & (truth[ok] <= hi[ok]))) if ok.any() else np.nan
        rows.append({
            "method": mid, "n": n, "rho": rho, "N": N, "R": R, "B": B, "alpha_level": alpha,
            "coverage": cov,
            "coverage_error": abs(cov - (1 - alpha)) if np.isfinite(cov) else np.nan,
            "mc_se": float(np.sqrt(max(cov, 1e-9) * (1 - cov) / max(int(ok.sum()), 1)))
                     if np.isfinite(cov) else np.nan,
            "width_mean": float(np.mean(hi[ok] - lo[ok])) if ok.any() else np.nan,
            "bias": float(np.mean(th[ok] - truth[ok])) if ok.any() else np.nan,
            "mse": float(np.mean((th[ok] - truth[ok]) ** 2)) if ok.any() else np.nan,
            "n_invalid": int(bad.sum()), "seconds": round(secs[mid], 2),
        })
    return rows


def main() -> int:
    cfg = tomllib.load(open(CFG, "rb"))
    CELLS.mkdir(parents=True, exist_ok=True)
    g = cfg["grid"]
    cells = [(n, r, N) for n in g["n"] for r in g["rho"] for N in g["N"]]
    print(f"repair extension: {len(cells)} cells, R = {cfg['R']:,}, B = {cfg['B_boot']}",
          flush=True)

    t0 = time.time()
    for i, (n, rho, N) in enumerate(cells, 1):
        path = CELLS / f"n{n}_rho{rho}_N{N}.json"
        if path.exists():                                     # restartable: skip completed cells
            print(f"  {i}/{len(cells)}  skip (done) n={n} rho={rho} N={N}", flush=True)
            continue
        rows = run_cell(cfg, n, rho, N)
        path.write_text(json.dumps(rows, indent=1))           # checkpoint immediately
        print(f"  {i}/{len(cells)}  n={n} rho={rho} N={N}  [{time.time()-t0:.0f}s]", flush=True)

    rows = []
    for p in sorted(CELLS.glob("*.json")):
        rows += json.loads(p.read_text())
    if not rows:
        print("no cells completed")
        return 1

    def dump(name, fields):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    key = ["method", "n", "rho", "N", "R", "B", "alpha_level"]
    dump("summary.csv", key + ["coverage", "coverage_error", "mc_se", "width_mean", "bias",
                               "mse", "n_invalid", "seconds"])
    dump("coverage.csv", key + ["coverage", "coverage_error", "mc_se"])
    dump("width.csv", key + ["width_mean"])
    dump("failures.csv", key + ["n_invalid"])

    import scipy
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except Exception:
        commit = "unknown"
    (OUT / "provenance.json").write_text(json.dumps({
        "config": str(CFG.relative_to(ROOT)), "git_commit": commit,
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seed_entropy": cfg["seed_entropy"],
        "spawn_key": "(n, round(rho*1000), N)",
        "cells": [{"n": n, "rho": r, "N": N,
                   "state0": int(cell_ss(cfg["seed_entropy"], n, r, N).generate_state(1)[0])}
                  for n, r, N in cells],
        "methods_run": EXISTING + ["Ridge-PPI", "PPBoot", "Efron-PPboot2"],
        "methods_excluded": cfg["excluded"],
        "interval_provenance": cfg["methods"],
    }, indent=2))
    print(f"\nwrote {len(rows)} rows to results/repair_extension/  [{time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
