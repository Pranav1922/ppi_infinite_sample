"""Read-only access to the saved research artefacts.

Every function here **opens files for reading only**. The dashboard is a presentation layer: it
never writes to `results/`, `configs/`, `docs/` or `data/`, never re-runs an experiment, and never
recomputes a saved statistic. Where a page needs a number that was never saved (a single interval,
say), it is computed live and labelled as such on the page -- never mixed into a research table.
"""
from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

#: Display names, kept in the order a reader meets them in the paper.
METHOD_LABELS = {
    "B1": "Classical (gold labels only)",
    "B2": "PPI",
    "B3": "PPI++",
    "B4": "PPI++ with a Student quantile",
    "B5": "PPI++ cross-fitted",
    "B8": "BCa bootstrap",
    "B9": "LC-PPI (leverage-corrected)",
    "B9-tmin": "LC-PPI + finite-pool safeguard",
    "B9-HC": "LC-PPI with HC3 variance",
    "B9-R2": "LC-PPI with Welch–Satterthwaite df",
}

#: Which saved metric answers which display column.
METRIC_LABELS = {
    "coverage": "empirical coverage",
    "width_mean": "mean interval width",
    "theta_hat_mean": "mean point estimate",
    "var_estimate_mean": "mean variance estimate",
    "seconds_per_interval": "seconds per interval",
    "mse": "mean squared error",
    "unlabeled_share_mean": "unlabelled variance share",
}


def rel(path: Path) -> str:
    """Repo-relative path — never an absolute filesystem path, which would leak the machine."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


@functools.lru_cache(maxsize=32)
def load_results(name: str) -> pd.DataFrame:
    """Load one saved results CSV, read-only and cached."""
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing saved result: {rel(path)}")
    return pd.read_csv(path)


@functools.lru_cache(maxsize=1)
def load_galaxies() -> tuple[np.ndarray, np.ndarray]:
    """The Galaxies population: gold labels `Y` and model predictions `Yhat`."""
    path = DATA / "galaxies.npz"
    with np.load(path) as d:
        return d["Y"].copy(), d["Yhat"].copy()


@functools.lru_cache(maxsize=1)
def provenance() -> dict:
    """Version stamps for the footer. The commit recorded *inside* the results is the one that
    matters for reproducibility; the working-tree commit is shown alongside it."""
    info = {"working_tree_commit": "unknown", "results_commit": "unknown"}
    try:
        info["working_tree_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        pass
    try:
        col = load_results("E_finite_N.csv").get("git_commit")
        if col is not None and len(col):
            info["results_commit"] = str(col.iloc[0])[:12]
    except Exception:
        pass
    return info


def pivot_metrics(df: pd.DataFrame, metrics: list[str], **filters) -> pd.DataFrame:
    """Filter a long-format results frame and pivot the requested metrics into columns."""
    sub = df
    for key, value in filters.items():
        if value is None or key not in sub.columns:
            continue
        sub = sub[sub[key] == value] if not isinstance(value, (list, tuple)) else sub[sub[key].isin(value)]
    sub = sub[sub["metric"].isin(metrics)]
    if sub.empty:
        return pd.DataFrame()
    out = sub.pivot_table(index="method", columns="metric", values="value", aggfunc="first")
    return out.reindex(columns=[m for m in metrics if m in out.columns])


def available(name: str) -> bool:
    return (RESULTS / name).exists()
