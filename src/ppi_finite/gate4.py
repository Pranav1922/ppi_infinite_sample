r"""Gate 4 — is the finite-pool safeguard `t_{min(n−2, N−1)}` improvable?

Gate 3 registered `B9-tmin`. Its cost is width: when `N ≪ n` the quantile jumps from `t_{n−2}` to
`t_{N−1}`, which at `N = 2` is a factor 6. Gate 4 asks whether a narrower rule can keep the
protection.

The answer splits cleanly:

* **Among constant critical values** valid uniformly over the unknown variance ratio, `t_min` is
  **exactly optimal** (Theorem G12 in `docs/theory_finite_N.md` Part IX): Mickey & Brown's lower
  bound is *attained* at an endpoint of `γ ∈ [0,1]`, where `V_γ` is exactly `t_{min(ν₁,ν₂)}`.
* **Data-dependent** rules are not excluded by that theorem, because `γ` is partially observable
  through the realised variance share. This module implements the classical construction for that
  situation -- Berger & Boos (1994): eliminate the nuisance parameter over a confidence set for it
  rather than over its whole range.

Nothing here is a new statistical idea. Berger-Boos and the `F` interval for a variance ratio are
textbook; the only question Gate 4 asks is whether they buy anything in this problem.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .gate2 import _two_df_table

__all__ = ["exact_q_at_share", "variance_share_ci", "quantile_berger_boos", "GATE4_CANDIDATES"]

GATE4_CANDIDATES = ("B9_t_n2", "B9_tmin", "C5_bb")


def exact_q_at_share(s, nu1: float, nu2: float, alpha: float):
    """Exact two-df quantile at pool variance share `s = v₂/(v₁+v₂)`, i.e. `γ = 1 − s`."""
    s_grid, nu_star = _two_df_table(float(nu1), float(nu2), float(alpha))
    filled = np.where(np.isfinite(nu_star), nu_star, 1e7)
    return stats.t.ppf(1 - alpha / 2, np.interp(np.clip(s, 0.0, 1.0), s_grid, filled))


def variance_share_ci(V1, V2, nu1: float, nu2: float, eta: float):
    r"""A `1 − eta` confidence interval for the pool variance share `s = v₂/(v₁+v₂)`.

    `V̂₂/V̂₁ = (v₂/v₁)·(W₂/W₁)` with `W₂/W₁ ~ F_{ν₂,ν₁}` under the idealisation, so the classical
    `F` interval for a variance ratio applies verbatim, and `s = r/(1+r)` is monotone in `r`.
    """
    V1 = np.asarray(V1, float); V2 = np.asarray(V2, float)
    R = V2 / np.maximum(V1, 1e-300)
    f_hi = stats.f.ppf(1 - eta / 2, nu2, nu1)
    f_lo = stats.f.ppf(eta / 2, nu2, nu1)
    r_lo, r_hi = R / f_hi, R / f_lo
    return r_lo / (1.0 + r_lo), r_hi / (1.0 + r_hi)


def quantile_berger_boos(V1, V2, n: int, N: float, alpha: float = 0.05, eta: float = 0.01):
    r"""Berger-Boos critical value, clipped to `[t_{n−2}, t_{min(n−2,N−1)}]`.

    `q = max_{s ∈ Ŝ} q_exact(s; α − η)` over a `1 − η` confidence set `Ŝ` for the variance share.
    Validity: `P(miss) ≤ P(s ∉ Ŝ) + P(miss, s ∈ Ŝ) ≤ η + (α − η) = α`, which is Berger & Boos's
    argument applied unchanged.

    Two deliberate clips, each enforcing one of the Gate-4 criteria:

    * **never below `t_{n−2}`** — criterion 1 and 5. Without it the rule would narrow below the
      deployed interval where `N ≥ n−1`, which is exactly what sank the `C3`/`C4` candidates of
      Gate 2;
    * **never above `t_min`** — criterion 4, so it can only ever improve on `B9-tmin`.

    When `N ≥ n − 1` the two clips coincide and the rule *is* `t_{n−2}`.
    """
    nu1, nu2 = float(n - 2), float(N) - 1.0
    t_n2 = stats.t.ppf(1 - alpha / 2, nu1)
    t_min = stats.t.ppf(1 - alpha / 2, min(nu1, nu2))
    V1 = np.asarray(V1, float)
    if nu2 >= nu1:                       # N >= n-1: criterion 1, exactly the deployed interval
        return np.full(V1.shape, t_n2)
    s_lo, s_hi = variance_share_ci(V1, V2, nu1, nu2, eta)
    q = np.maximum(exact_q_at_share(s_lo, nu1, nu2, alpha - eta),
                   exact_q_at_share(s_hi, nu1, nu2, alpha - eta))
    return np.clip(q, t_n2, t_min)
