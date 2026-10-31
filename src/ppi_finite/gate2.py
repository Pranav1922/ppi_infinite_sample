r"""Gate 2 — validating the finite-`N` bound and mapping the under-coverage boundary.

Gate 1 proved `|coverage_act − coverage_id| ≤ B(n,ρ,α,N)` (Theorem G1) and identified the
strong-predictor boundary (Result G8): as `ρ → 1` the deployed statistic degenerates to `t_{N−1}`
while the interval keeps the `t_{n−2}` quantile, so coverage falls below nominal iff `N < n − 1`.

This module supplies the machinery for the four candidate quantile rules and for the
Rao-Blackwellised coverage/width evaluation used throughout Gate 2. It writes nothing and changes
no existing estimator: `registry` is untouched, so every Experiment A-I result is unaffected.

Notation follows `docs/theory_finite_N.md` §II.1 and Part VI.
"""
from __future__ import annotations

import functools

import numpy as np
from scipy import optimize, stats

from .theory_gate1 import reduction_1, reduction_2

__all__ = ["limit_coverage", "dof_ws2", "exact_two_df_nu", "candidate_quantiles",
           "simulate_cell", "CANDIDATES"]

#: The candidate rules of the Gate-2 brief, in report order.
CANDIDATES = ("B9_t_n2", "C1_t_N1", "C2_t_min", "C3_ws2", "C4_exact2", "R2_ws_inf")


def limit_coverage(n: int, N: float, alpha: float = 0.05) -> float:
    r"""`P(|t_{N−1}| ≤ t_{n−2,1−α/2})` — Result G8's `ρ → 1` prediction for the deployed interval.

    Below `1−α` exactly when `N − 1 < n − 2`. This is a *prediction from the limit*, never a fit.
    """
    q = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(2 * stats.t.cdf(q, N - 1) - 1)


def dof_ws2(V1, V2, nu1: float, nu2: float):
    r"""Welch-Satterthwaite effective df for `V̂ = V1 + V2` with **both** components estimated.

        nu_eff = (V1+V2)^2 / ( V1^2/nu1 + V2^2/nu2 )

    This is the classical formula (Welch 1947; Satterthwaite 1946), nothing more. The deployed
    `r2` variant is the same expression with `nu2 = inf`, i.e. with the pool's own degrees of
    freedom omitted; supplying `nu2 = N−1` is the only change. Limits, both exact:

    * `V2 → 0`  (pool effectively infinite)  ⇒ `nu_eff → nu1 = n−2`
    * `V1 → 0`  (strong predictor, `σ → 0`)  ⇒ `nu_eff → nu2 = N−1`
    """
    V1, V2 = np.asarray(V1, float), np.asarray(V2, float)
    total = V1 + V2
    denom = V1**2 / nu1 + V2**2 / nu2
    return np.where(denom > 0, total**2 / np.maximum(denom, 1e-300), nu1)


def _coverage_two_df(q, s, nu1, nu2, nodes):
    """`P(|T| ≤ q)` for `T = Z/sqrt((1-s)Q1/nu1 + s Q2/nu2)`, by product quadrature."""
    q1, w1, q2, w2 = nodes
    r = np.sqrt((1 - s) * q1[:, None] / nu1 + s * q2[None, :] / nu2)
    return float(np.sum(w1[:, None] * w2[None, :] * (2 * stats.norm.cdf(q * r) - 1)))


@functools.lru_cache(maxsize=256)
def _two_df_table(nu1: float, nu2: float, alpha: float, m: int = 81, k: int = 64):
    r"""Exact quantile of the **two-estimated-variance** idealisation, as an effective `t` df.

    `r5` solves `T = Z/sqrt((1-s)Q/nu + s)`, i.e. the case where the pool component's scale is
    known. With `sigma_f^2` estimated that component is `s·Q2/nu2` with `Q2 ~ chi2_{N-1}`, and the
    statistic is the genuine Behrens-Fisher pivot with *both* variances estimated. Its quantile is
    still an exact integral, now two-dimensional; solving it at level `1-alpha` and re-expressing
    the root as `t_{nu*,1-alpha/2}` keeps the interval constructor generic.

    Returns `(s_grid, nu_star_grid)` for interpolation. Cached on `(nu1, nu2, alpha)`.
    """
    gl_x, gl_w = np.polynomial.legendre.leggauss(k)
    p = 0.5 * (gl_x + 1.0)
    nodes = (stats.chi2.ppf(p, nu1), 0.5 * gl_w, stats.chi2.ppf(p, nu2), 0.5 * gl_w)

    s_grid = np.linspace(0.0, 1.0, m)
    nu_star = np.empty(m)
    lo, hi = min(nu1, nu2), max(nu1, nu2)
    for i, s_val in enumerate(s_grid):
        target = 1 - alpha
        f = lambda qq: _coverage_two_df(qq, s_val, nu1, nu2, nodes) - target
        a = stats.t.ppf(1 - alpha / 2, max(hi, 1e-6)) * 0.5
        b = stats.t.ppf(1 - alpha / 2, max(lo, 1e-6)) * 2.0 + 1.0
        try:
            q = optimize.brentq(f, a, b, xtol=1e-10)
        except ValueError:
            nu_star[i] = lo
            continue
        z = stats.norm.ppf(1 - alpha / 2)
        if q <= z * (1 + 1e-12):
            nu_star[i] = np.inf
        else:
            nu_star[i] = optimize.brentq(
                lambda v: stats.t.ppf(1 - alpha / 2, v) - q, 1e-3, 1e7, xtol=1e-8)
    return s_grid, nu_star


def exact_two_df_nu(s_hat, nu1: float, nu2: float, alpha: float):
    """Interpolate `_two_df_table` at the realised variance share `s_hat`."""
    s_grid, nu_star = _two_df_table(float(nu1), float(nu2), float(alpha))
    finite = np.isfinite(nu_star)
    filled = np.where(finite, nu_star, 1e7)
    return np.interp(np.clip(s_hat, 0.0, 1.0), s_grid, filled)


def candidate_quantiles(V1, V2, n: int, N: float, alpha: float) -> dict:
    """Every candidate rule's quantile, on one set of draws. Keys are `CANDIDATES`."""
    nu1, nu2 = n - 2, N - 1
    s_hat = V2 / np.maximum(V1 + V2, 1e-300)
    z = np.ones_like(np.asarray(V1, float))
    return {
        "B9_t_n2":   z * stats.t.ppf(1 - alpha / 2, nu1),                    # deployed
        "C1_t_N1":   z * stats.t.ppf(1 - alpha / 2, nu2),
        "C2_t_min":  z * stats.t.ppf(1 - alpha / 2, min(nu1, nu2)),
        "C3_ws2":    stats.t.ppf(1 - alpha / 2, dof_ws2(V1, V2, nu1, nu2)),
        "C4_exact2": stats.t.ppf(1 - alpha / 2, exact_two_df_nu(s_hat, nu1, nu2, alpha)),
        "R2_ws_inf": stats.t.ppf(1 - alpha / 2, dof_ws2(V1, V2, nu1, np.inf)),  # deployed r2
    }


def simulate_cell(n: int, rho: float, N: float, *, R: int = 400_000, alpha: float = 0.05,
                  seed: int = 0, chunk: int = 100_000, var_Y: float = 1.0,
                  sigma_f: float = 1.0) -> dict:
    r"""Coverage and mean width of every candidate at one `(n, ρ, N)` cell, on common randoms.

    Rao-Blackwellised: `reduction_1` gives `P(|T_act| ≤ q | F,Q,W,Z₁,σ̂_f²)` exactly, which is
    valid for a *data-dependent* `q` because every candidate's quantile is measurable w.r.t. that
    conditioning. `coverage_ideal` uses `reduction_2`, which needs a `Z₁`-free denominator.

    All candidates share one set of draws, so their differences carry no Monte-Carlo noise.
    """
    nu = n - 2
    sig = np.sqrt((1.0 - rho**2) * var_Y)
    beta = rho * np.sqrt(var_Y) / sigma_f
    c, lam = beta * sigma_f / np.sqrt(N), beta * sigma_f / (sig * np.sqrt(N))
    rng = np.random.default_rng(seed)

    cov = {k: 0.0 for k in CANDIDATES}
    wid = {k: 0.0 for k in CANDIDATES}
    cov_ideal = 0.0
    done = 0
    while done < R:
        r = min(chunk, R - done)
        f = rng.normal(0.0, sigma_f, size=(r, n))
        S_ff = ((f - f.mean(1)[:, None]) ** 2).sum(1)
        u = -f.mean(1) / np.sqrt(S_ff)
        kap = sigma_f / (np.sqrt(N) * np.sqrt(S_ff))
        Z1, W = rng.normal(size=(2, r))
        Q = rng.chisquare(nu, size=r)
        sh2 = sigma_f**2 * rng.chisquare(N - 1, size=r) / (N - 1)

        a = u + kap * W
        g_act, g_id = 1 / n + a**2, 1 / n + u**2
        s2 = sig**2 * Q / nu
        b = beta + sig * Z1 / np.sqrt(S_ff)
        V1 = s2 * g_act                       # labelled component, nu = n-2
        V2 = b**2 * sh2 / N                   # pool component,      nu = N-1
        V_act = V1 + V2
        m_act = sig * a * Z1 + c * W

        for name, q in candidate_quantiles(V1, V2, n, N, alpha).items():
            cov[name] += reduction_1(q, n, sig, V_act, m_act).sum()
            wid[name] += float(np.sum(2.0 * q * np.sqrt(V_act)))
        t_id = stats.t.ppf(1 - alpha / 2, nu)
        cov_ideal += reduction_2(t_id, nu, Q, lam, g_id, W).sum()
        done += r

    out = {"n": n, "rho": rho, "N": N, "alpha": alpha, "R": R, "seed": seed,
           "coverage_ideal": cov_ideal / R}
    for k in CANDIDATES:
        out[f"coverage_{k}"] = cov[k] / R
        out[f"width_{k}"] = wid[k] / R
    out["coverage_actual"] = out["coverage_B9_t_n2"]
    out["delta_cov"] = out["coverage_actual"] - out["coverage_ideal"]
    # MC s.e. of a Rao-Blackwellised mean of values in [0,1]; conservative binomial form
    out["mc_se"] = float(np.sqrt(0.25 / R))
    out["limit_pred"] = limit_coverage(n, N, alpha)
    return out
