"""Intervals from a standard error (L1).  Nothing here computes a variance (`15` §3, C3)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = ["ci_from_se"]

Array = NDArray[np.float64]


def ci_from_se(theta_hat, se, *, alpha_level: float = 0.05, dof=None, dist: str = "t"):
    """`θ̂ ± q · se` (`04` §4.2, §4.8).

    `dist="t"` uses `t_{dof, 1−α/2}`, `dist="normal"` uses `z_{1−α/2}`.  `dof = inf` with
    `dist="t"` is the normal quantile in the limit and is allowed.

    Raises if `alpha_level ∉ (0,1)`, if `dist="t"` and `dof is None`, or if any `se < 0`.
    """
    if not (0.0 < alpha_level < 1.0):
        raise ValueError(f"alpha_level must be in (0,1); got {alpha_level}")
    if dist not in ("t", "normal"):
        raise ValueError(f"dist must be 't' or 'normal'; got {dist!r}")
    if dist == "t" and dof is None:
        raise ValueError("dist='t' requires dof")
    se = np.asarray(se, dtype=np.float64)
    if np.any(se < 0):
        raise ValueError("se must be non-negative")
    theta_hat = np.asarray(theta_hat, dtype=np.float64)

    if dist == "normal":
        q = stats.norm.ppf(1.0 - alpha_level / 2.0)
    else:
        q = stats.t.ppf(1.0 - alpha_level / 2.0, np.asarray(dof, dtype=np.float64))
    half = q * se
    low, high = theta_hat - half, theta_hat + half
    assert np.all(low <= high)
    return low, high
