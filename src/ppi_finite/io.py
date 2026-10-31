"""Results I/O and provenance (L4).  `09` §Output files, `15` §8.

A results file without a sidecar is not a result (`11` §Logging).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

__all__ = ["provenance", "write_csv", "write_sidecar", "write_raw", "COLUMNS"]

COLUMNS = [
    # 15 §8.1.  09 §Output files names the first column `experiment`; 15 names it
    # `experiment_id` and adds `cell_id` and `variance_mode`.  We follow 15 (the identity spine:
    # the same string everywhere) — noted as a naming discrepancy in 13.
    "experiment_id", "cell_id", "method", "dgp", "n", "N", "rho", "alpha_level", "R",
    "unlabeled_mode", "coef_mode", "variance_mode", "metric", "value", "mc_se", "n_failed",
    "seed_spec", "timestamp_utc", "python_version", "numpy_version", "scipy_version",
    "git_commit",
]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5)
        if out.returncode != 0:
            return "unknown"
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def provenance() -> dict:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_csv(rows: list[dict], path: Path) -> Path:
    df = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_sidecar(path: Path, *, config: dict, command: str, seconds: float, prov: dict,
                  n_cells: int) -> Path:
    payload = {"config": config, "command": command, "wall_clock_seconds": round(seconds, 3),
               "n_cells": n_cells, **prov}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def write_raw(directory: Path, cell_id: str, results: dict, theta_true: float) -> Path:
    """Per-replication outcomes for one cell (`15` §8.3): `covered`, `width`, `theta_hat`, `se`,
    and `lambda_hat` where the method estimates one."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for method, res in results.items():
        arrays[f"{method}|covered"] = (res.ci_low <= theta_true) & (theta_true <= res.ci_high)
        arrays[f"{method}|width"] = res.width
        arrays[f"{method}|theta_hat"] = res.theta_hat
        arrays[f"{method}|se"] = np.asarray(res.se)
        # lambda where the method estimates one: `plots.py` may only read stored per-replication
        # outputs, never recompute a statistic (`15` §L5), so it has to be stored here.
        lam = res.extras.get("lambda_hat")
        if lam is not None and np.ndim(lam) == 1:
            arrays[f"{method}|lambda_hat"] = np.asarray(lam)
    safe = cell_id.replace("|", "__").replace("=", "-")
    path = directory / f"cell={safe}.npz"
    np.savez_compressed(path, **arrays)
    return path
