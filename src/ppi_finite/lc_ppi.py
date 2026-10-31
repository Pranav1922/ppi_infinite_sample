"""The proposed method (L2): B9 and its decomposition control B9-nolev.

**No novelty is claimed here, and the prior art is now identified.**  The estimator is the survey
sampling difference/GREG estimator (Cassel et al. 1976; Särndal et al. 2003), recognised for PPI by
Mozer (2026), arXiv:2603.19160.  At finite `N` the studentised statistic is the Behrens–Fisher
statistic with one variance known, and its conservativeness is Mickey & Brown (1966).  See
`docs/prior_art_findings.md` §4d and `docs/theory_finite_N.md` Part III.  Docstrings below cite
`04` equation numbers only.

Variance-mode naming (`15` §13 CONFLICT 4): the user-facing `lc_ppi_interval` takes the
unprefixed names `variance_mode ∈ {"v482", "nolev"}` of `11`; the generic L1 entry point in
`variance.py` uses the same unprefixed strings for the LC family, so nothing has to be mapped in
the MVP.  Neither spelling ever reaches a results file — only the `06` method IDs do.
"""
from __future__ import annotations

import numpy as np

from . import variance as V
from .confidence_intervals import ci_from_se
from .estimators import labelled_summaries, lambda_plugin, summaries_of, theta_ppi_plus
from .types import IntervalResult, PPIView, make_data

__all__ = ["lc_ppi_interval", "method_b9", "method_b9_nolev", "method_b9_vfull",
           "method_b9_r1", "method_b9_r2", "method_b9_r3", "method_b9_r5", "method_b9_tmin",
           "method_b9_hc", "method_b9_ppiplus"]


def _lc(data: PPIView, alpha_level: float, mode: str, method_id: str,
        summaries: dict | None = None, coef_mode: str = "ols") -> IntervalResult:
    s = summaries if summaries is not None else summaries_of(data)
    n, N = s["n"], data.N

    # 04 §4.7 / 05 step 8: the point estimate uses the OLS slope b, not lambda_hat = b/(1+r);
    # the exactness argument is for the OLS fitted value.  coef_mode is an OPEN research
    # parameter (05 step 4 note) and defaults to "ols".
    if coef_mode not in ("ols", "ppi_plus"):
        raise ValueError(f"coef_mode must be 'ols' or 'ppi_plus'; got {coef_mode!r}")
    coef = s["b"] if coef_mode == "ols" else lambda_plugin(s["b"], n, N)

    theta = theta_ppi_plus(data.Y, data.f, data.fbar_N, coef)      # = Ȳ + b(f̄_N − f̄_n)
    v = V.var_lc_ppi(s, data.fbar_N, data.sigma2_f, N, mode=mode, mu_f=data.mu_f,
                     coef=coef, alpha_level=alpha_level)
    se = np.sqrt(v.value)
    low, high = ci_from_se(theta, se, alpha_level=alpha_level, dof=v.dof, dist="t")

    flags = []
    degenerate = np.asarray(s["degenerate"])
    n_degenerate = int(degenerate.sum())
    if n_degenerate:
        # 05 §Preconditions: S_ff = 0 => the slope is undefined; fall back to B1 on those
        # replications and flag it.
        flags.append("degenerate_S_ff")
        vc = V.var_classical(data.Y)
        se_c = np.sqrt(vc.value)
        lo_c, hi_c = ci_from_se(s["Ybar"], se_c, alpha_level=alpha_level, dof=vc.dof, dist="t")
        theta = np.where(degenerate, s["Ybar"], theta)
        se = np.where(degenerate, se_c, se)
        low = np.where(degenerate, lo_c, low)
        high = np.where(degenerate, hi_c, high)
    if n <= 3:
        flags.append("small_n")
    if np.isfinite(N) and N / n < 1000:
        flags.append("finite_N_regime")     # 04 §4.8.2-4.8.3: known inexact here

    with np.errstate(invalid="ignore", divide="ignore"):
        rho_hat = np.where(s["S_ff"] * s["S_YY"] > 0,
                           s["S_fY"] / np.sqrt(np.where(s["S_ff"] * s["S_YY"] > 0,
                                                        s["S_ff"] * s["S_YY"], 1.0)), 0.0)
    extras = {
        "variance_mode": mode, "dist": "t", "coef_mode": coef_mode,
        "lambda_hat": lambda_plugin(s["b"], n, N), "b": s["b"],
        "leverage": v.leverage, "leverage_frac": v.leverage_frac,
        "s2_resid": s["s2"], "var_components": v.components,
        "n": n, "N": N, "rho_hat": rho_hat, "flags": flags,
        "n_degenerate": n_degenerate,
    }
    return IntervalResult(method_id, theta, low, high, se, v.dof, extras=extras)


def method_b9(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None):
    """B9 — LC-PPI (`04` §4.8.2, `05` Algorithm, `06` B9).

    `θ̂ = Ȳ + b(f̄_N − f̄_n)`;  `V̂ = s²[1/n + (f̄_N − f̄_n)²/S_ff] + b²σ̂_f²/N`;  `t_{n−2}`.
    """
    return _lc(data, alpha_level, "v482", "B9", summaries)


def method_b9_nolev(data: PPIView, alpha_level: float = 0.05, rng=None,
                    summaries: dict | None = None):
    """B9-nolev — the decomposition **control**, not a proposal (`09` Table A1, V7).

    B9's point estimate and `t_{n−2}` quantile with the leverage term deleted:
    `V̂ = s²/n + b²σ̂_f²/N`.  It separates the part of the deficit the quantile fixes (≈ 0.937)
    from the part the leverage term fixes (the remaining ≈ 0.013).
    """
    return _lc(data, alpha_level, "nolev", "B9-nolev", summaries)


def method_b9_vfull(data: PPIView, alpha_level: float = 0.05, rng=None,
                    summaries: dict | None = None):
    """B9-vfull — `04` §4.8.3's (V_full) with `t_{n−2}`.

    A **diagnostic, not a method**: it needs `μ_f`, which a practitioner does not know. It exists
    so the derived expression can be tested against the empirical variance, which `12` Stage 11
    calls "the first numerical test that (V_full) has ever received". Its feasible counterpart is
    `B9-R1`.
    """
    return _lc(data, alpha_level, "v_full", "B9-vfull", summaries)


def method_b9_r1(data: PPIView, alpha_level: float = 0.05, rng=None,
                 summaries: dict | None = None):
    """B9-R1 — the expectation form of `04` §4.8.4 R1.  Status `PROPOSED`."""
    return _lc(data, alpha_level, "r1", "B9-R1", summaries)


def method_b9_r2(data: PPIView, alpha_level: float = 0.05, rng=None,
                 summaries: dict | None = None):
    """B9-R2 — Satterthwaite effective degrees of freedom, `04` §4.8.4 R2.  Status `PROPOSED`."""
    return _lc(data, alpha_level, "r2", "B9-R2", summaries)


def method_b9_r3(data: PPIView, alpha_level: float = 0.05, rng=None,
                 summaries: dict | None = None):
    """B9-R3 — the conditional form, `04` §4.8.4 R3, concretised as described in
    `variance.var_lc_ppi`.  Status `PROPOSED`; `04` specifies no formula for it."""
    return _lc(data, alpha_level, "r3", "B9-R3", summaries)


def method_b9_r5(data: PPIView, alpha_level: float = 0.05, rng=None,
                 summaries: dict | None = None):
    """B9-R5 — the exact quantile of the idealised finite-N law (E4b; not one of `04`'s R1-R4).

    Same variance as B9; the quantile inverts the exact coverage integral instead of
    Satterthwaite's moment match.  See `variance.var_lc_ppi` mode `r5`.
    """
    return _lc(data, alpha_level, "r5", "B9-R5", summaries)


def method_b9_tmin(data: PPIView, alpha_level: float = 0.05, rng=None, *,
                   summaries: dict | None = None) -> IntervalResult:
    """B9-tmin — B9's variance and point estimate with the quantile `t_{min(n−2, N−1)}`.

    The **finite-pool safeguard**. B9 uses `t_{n−2}`, which is Mickey & Brown's (1966) `t_min`
    bound instantiated with `f₂ = ∞` — i.e. with `σ_f²` treated as known. When `σ̂_f²` is
    estimated from the pool the data supply `f₂ = N−1`, and the same classical bound returns
    `t_{min(n−2, N−1)}`.

    **The rule is prior art** (Mickey & Brown 1966, *Ann. Math. Statist.* 37, 639–642). Nothing is
    claimed for it here; what this repository contributes is its identification as the correct
    finite-pool instantiation for the deployed LC-PPI interval, and the demonstration that
    `t_{n−2}` under-covers without it once `N < n − 1` (`theory_finite_N.md` §VI.8, §VII.4).

    `min(n−2, N−1) = n−2` whenever `N ≥ n−1`, so this is **identical** to B9 on every grid this
    paper reports, and widens only inside the region where B9 provably fails.
    """
    return _lc(data, alpha_level, "tmin", "B9-tmin", summaries)


def method_b9_hc(data: PPIView, alpha_level: float = 0.05, rng=None,
                 summaries: dict | None = None):
    """B9-HC — B9's point estimate with an HC3 sandwich variance (`07` Experiment H).

    For use when heteroskedasticity is suspected.  **Asymptotic only**: `07` Experiment H records
    that the HC variant almost certainly retains no exactness claim, and none is made.
    """
    s = summaries if summaries is not None else summaries_of(data)
    theta = theta_ppi_plus(data.Y, data.f, data.fbar_N, s["b"])
    v = V.var_lc_hc3(s, data.Y, data.f, data.fbar_N, data.sigma2_f, data.N)
    se = np.sqrt(v.value)
    low, high = ci_from_se(theta, se, alpha_level=alpha_level, dof=v.dof, dist="t")
    return IntervalResult("B9-HC", theta, low, high, se, v.dof,
                          extras={"variance_mode": "hc3", "dist": "t", "coef_mode": "ols",
                                  "leverage": v.leverage, "leverage_frac": v.leverage_frac,
                                  "var_components": v.components, "flags": ["asymptotic_only"]})


def method_b9_ppiplus(data: PPIView, alpha_level: float = 0.05, rng=None,
                      summaries: dict | None = None):
    """B9-ppiplus — B9 with `coef_mode="ppi_plus"`, i.e. the coefficient `b/(1+r)` instead of `b`.

    `05` step 4 leaves the choice OPEN and `07` Experiment E requires both to be reported. The
    two coincide numerically whenever `r = n/N` is negligible, which is every cell of Experiment
    A but **not** the small-N cells of Experiment E (`r = 0.2` at N = 50, n = 10).
    """
    return _lc(data, alpha_level, "v482", "B9-ppiplus", summaries, coef_mode="ppi_plus")


def lc_ppi_interval(Y, f, f_unlabeled=None, *, mu_f=None, N=None, sigma2_f=None,
                    alpha_level: float = 0.05, coef_mode: str = "ols",
                    variance_mode: str = "v482") -> IntervalResult:
    """One-call wrapper (`11` §lc_ppi.py, `15` §6.5) — the drop-in for the `ppi-py` mean path.

    Supply exactly one of `f_unlabeled`, `mu_f`, or `(fbar_N via N + sigma2_f)`.  `shrinkage_c`
    (B10) and the finite-N variance modes are not implemented in the MVP; see `variance.py`.
    """
    data = make_data(Y, f, f_unlabeled=f_unlabeled, mu_f=mu_f, N=N, sigma2_f=sigma2_f)
    s = labelled_summaries(data.Y, data.f)
    method_id = "B9" if variance_mode == "v482" else f"B9-{variance_mode}"
    return _lc(data.view(), alpha_level, variance_mode, method_id, s, coef_mode=coef_mode)
