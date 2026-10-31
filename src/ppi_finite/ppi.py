"""B1 (classical) and B2 (PPI) — the two methods with no estimated coefficient (L2).

`15` §1.1 assigns B2 to this module and gives B1 no module of its own; B1 lives here because it
is the other coefficient-free method and because L2 is where methods belong.  Each method is a
thin composition of L1 calls (`15` §3).
"""
from __future__ import annotations

import numpy as np

from . import variance as V
from .confidence_intervals import ci_from_se
from .estimators import theta_classical, theta_ppi
from .types import IntervalResult, PPIView

__all__ = ["method_b1", "method_b2"]


def method_b1(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None):
    """B1 — classical `t` interval: `Ȳ ± t_{n−1} s_Y/√n` (`04` §4.2, `06` B1).

    Ignores `f` and the unlabelled pool entirely.  The validity reference: under a normal model
    it is exact, so its measured coverage tests the harness (`09` V1).
    """
    theta = theta_classical(data.Y)
    v = V.var_classical(data.Y)
    se = np.sqrt(v.value)
    low, high = ci_from_se(theta, se, alpha_level=alpha_level, dof=v.dof, dist="t")
    return IntervalResult("B1", theta, low, high, se, v.dof,
                          extras={"variance_mode": "classical", "dist": "t"})


def method_b2(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None):
    """B2 — PPI: `Ȳ − (f̄_n − f̄_N)`, `V̂ar(Y−f)/n + σ̂_f²/N`, normal quantile (`04` §4.3, `06` B2)."""
    theta = theta_ppi(data.Y, data.f, data.fbar_N)
    v = V.var_ppi(data.Y, data.f, data.N)
    se = np.sqrt(v.value)
    low, high = ci_from_se(theta, se, alpha_level=alpha_level, dist="normal")
    return IntervalResult("B2", theta, low, high, se, np.inf,
                          extras={"variance_mode": "ppi", "dist": "normal"})
