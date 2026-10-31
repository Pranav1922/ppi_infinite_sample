"""Variance expressions (L1).  One function per expression; each cites its equation in `04`.

This module never chooses which expression to use — that is the method's job (`15` §3, C3).

Extension E4 adds the finite-N family — `v_full` (`04` §4.8.3) and the candidate repairs `r1`,
`r2`, `r3` (`04` §4.8.4).  Each is a **separately identifiable** mode and each carries its status
tag in its docstring; none is presented as established.  `r4` has no code by construction: it is
the claim that `v482` is conservative, i.e. a theorem about an existing mode.  HC3 remains E5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import stats
from numpy.typing import NDArray

__all__ = ["VarianceResult", "var_classical", "var_ppi", "var_ppi_plus",
           "leverage_term", "var_lc_ppi", "var_lc_hc3", "LC_MODES"]

Array = NDArray[np.float64]

LC_MODES = ("v482", "tmin", "nolev", "v_full", "r1", "r2", "r3", "r5")
#: Modes that need the population prediction mean and are therefore diagnostics, not
#: methods a practitioner could run (`04` §4.8.3 is written in terms of `μ_f`).
LC_MODES_REQUIRING_MU_F = ("v_full",)


@dataclass(frozen=True)
class VarianceResult:
    """`15` §6.3.  `dof` travels with the variance so the interval constructor stays generic."""

    value: Array
    dof: Array | float
    components: dict = field(default_factory=dict)
    leverage: Array | float | None = None
    leverage_frac: Array | float | None = None


def _unlabeled_term(coef, sigma2_f, N: float):
    """`coef² σ̂_f² / N`, and exactly zero when `N` is infinite (`04` §4.8.1 vs §4.8.2)."""
    if not np.isfinite(N):
        return np.zeros_like(np.asarray(coef, dtype=np.float64))
    return np.asarray(coef, dtype=np.float64) ** 2 * np.asarray(sigma2_f, dtype=np.float64) / N


def var_classical(Y: Array) -> VarianceResult:
    """`V̂ = s_Y²/n` with `ddof = 1`; quantile `t_{n−1}` (`04` §4.2)."""
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.shape[-1]
    v = Y.var(axis=-1, ddof=1) / n
    return VarianceResult(value=v, dof=float(n - 1), components={"mean_term": v})


def var_ppi(Y: Array, f: Array, N: float) -> VarianceResult:
    """`V̂ = V̂ar(Y − f)/n + V̂ar(f)/N` with `ddof = 1`; normal quantile (`04` §4.3).

    `V̂ar(f)` is the **labelled-sample** `ddof = 1` variance, i.e. `S_ff/(n−1)`: `04` §4.4
    defines the `V̂ar(·)` notation used throughout §4.3–§4.5 as "computed on the same n gold
    pairs", and both preliminary scripts do exactly that.  The LC family is different by
    design — `04` §4.8.2 defines its `σ̂_f²` as estimated from the N unlabelled predictions.
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = Y.shape[-1]
    mean_term = (Y - f).var(axis=-1, ddof=1) / n
    unl = _unlabeled_term(np.ones_like(mean_term), f.var(axis=-1, ddof=1), N)
    return VarianceResult(value=mean_term + unl, dof=np.inf,
                          components={"mean_term": mean_term, "unlabeled_term": unl})


def var_ppi_plus(Y: Array, f: Array, lam, N: float) -> VarianceResult:
    """`V̂ = V̂ar(Y − λ̂ f)/n + λ̂² V̂ar(f)/N` with `ddof = 1`; normal quantile (`04` §4.5).

    This is the expression under investigation: it is the variance `θ̂_λ` would have if λ were
    the fixed constant λ*, and carries no term for λ̂ having been estimated on the same n pairs.

    Both `V̂ar(·)` are **labelled-sample** `ddof = 1` moments (`04` §4.4: "computed on the same
    n gold pairs"; `09` §Baseline configuration: "`ddof = 1` sample moments"), so
    `V̂ar(f) = S_ff/(n−1)`.  This is what `cover.py` and `exact.py` compute.  Contrast the LC
    family, whose `σ̂_f²` comes from the unlabelled pool (`04` §4.8.2).  See `13` entry L1.
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = Y.shape[-1]
    lam = np.asarray(lam, dtype=np.float64)
    mean_term = (Y - lam[..., None] * f).var(axis=-1, ddof=1) / n
    unl = _unlabeled_term(lam, f.var(axis=-1, ddof=1), N)
    return VarianceResult(value=mean_term + unl, dof=np.inf,
                          components={"mean_term": mean_term, "unlabeled_term": unl})


def _leverage(fbar_N, fbar_n, S_ff):
    """`h = (f̄_N − f̄_n)² / S_ff` (`04` §4.6).  Zero where `S_ff = 0` (degenerate, see B1 fallback)."""
    S_ff = np.asarray(S_ff, dtype=np.float64)
    gap = np.asarray(fbar_N, dtype=np.float64) - np.asarray(fbar_n, dtype=np.float64)
    return np.where(S_ff > 0.0, gap ** 2 / np.where(S_ff > 0.0, S_ff, 1.0), 0.0)


def leverage_term(f: Array, fbar_N) -> Array:
    """`h = (f̄_N − f̄_n)² / S_ff` from the raw predictions (`04` §4.6).

    Public wrapper over the same one formula used inside `var_lc_ppi`; it recomputes `f̄_n` and
    `S_ff`, so batched callers that already hold the summaries should use `var_lc_ppi`.
    """
    f = np.asarray(f, dtype=np.float64)
    fbar_n = f.mean(axis=-1)
    S_ff = np.sum((f - fbar_n[..., None]) ** 2, axis=-1)
    return _leverage(fbar_N, fbar_n, S_ff)


@lru_cache(maxsize=64)
def _r5_dof_table(nu: float, alpha_level: float, m: int = 161):
    """Tabulate the effective `t` degrees of freedom that reproduces the EXACT quantile of the
    idealised finite-N law, on a grid of the unlabelled variance share `s`.

    Under the idealisation of `04` §4.8.3 (E4b): `T = Z / sqrt((1-s)Q/nu + s)` with `Z ~ N(0,1)`
    and `Q ~ chi2_nu` independent, so

        P(|T| <= q)  =  E_Q[ 2*Phi(q*sqrt((1-s)Q/nu + s)) - 1 ]

    is an exact one-dimensional integral.  Solving it for `q` at level `1-alpha` gives the exact
    quantile; because `q` always lies in `[z_{1-alpha/2}, t_{nu,1-alpha/2}]`, it can be expressed
    as `t_{nu*, 1-alpha/2}` for a unique `nu* >= nu`, which is what is returned so that the
    interval constructor stays generic (C3).

    Returns `(s_grid, nu_star_grid)`; callers interpolate.  Cached because it depends only on
    `(nu, alpha_level)`, which are constant within a cell.
    """
    from scipy import optimize

    from .theory import exact_coverage_integral
    s_grid = np.linspace(0.0, 1.0, m)
    q_t = float(stats.t.ppf(1 - alpha_level / 2, nu))
    q_z = float(stats.norm.ppf(1 - alpha_level / 2))

    cover = lambda q, s: float(exact_coverage_integral(q, nu, s))

    nu_star = np.empty(m)
    for i, s_val in enumerate(s_grid):
        if s_val <= 0.0:
            nu_star[i] = nu
            continue
        if s_val >= 1.0:
            nu_star[i] = np.inf
            continue
        q = optimize.brentq(lambda qq: cover(qq, s_val) - (1 - alpha_level),
                            q_z * (1 - 1e-9), q_t * (1 + 1e-9), xtol=1e-12)
        # invert nu -> t_{nu, 1-alpha/2} to express q as an effective t degrees of freedom
        if q <= q_z * (1 + 1e-12):
            nu_star[i] = np.inf
        else:
            nu_star[i] = optimize.brentq(
                lambda lv: float(stats.t.ppf(1 - alpha_level / 2, np.exp(lv))) - q,
                np.log(nu), np.log(1e12), xtol=1e-10)
            nu_star[i] = np.exp(nu_star[i])
    return s_grid, nu_star


def var_lc_ppi(summaries: dict, fbar_N, sigma2_f, N: float, *, mode: str = "v482",
               mu_f: float | None = None, coef=None, alpha_level: float = 0.05) -> VarianceResult:
    """Leverage-corrected variance.  Quantile `t_{n−2}` except where a mode says otherwise.

    Modes, each separately identifiable and each carrying its `04` status tag:

    * **`v482`** — `V̂ = s²(1/n + h) + b² σ̂_f²/N`, `h = (f̄_N − f̄_n)²/S_ff` (`04` §4.8.2).
      With `N = inf` it reduces to `s²(1/n + h)` of `04` §4.8.1, the case with an exactness
      argument.  Status: `IMPLEMENTED AND MEASURED`, known inexact at finite N.
    * **`nolev`** — `V̂ = s²/n + b² σ̂_f²/N`; the leverage term deleted (`09` Table A1's
      "no-leverage + t" control).  A **control, not a proposal**.
    * **`v_full`** — `V̂ = s²/n + s²[(f̄_n − μ_f)² + σ̂_f²/N]/S_ff + b² σ̂_f²/N`, the expression
      derived in `04` §4.8.3, transcribed term by term.  **Requires `μ_f`**, which `04` §4.8.3
      is written in terms of and which a practitioner does not know — so this is a *diagnostic*
      for testing the derivation, not a method.  Its feasible counterpart is `r1`.
      Status on arrival: `DERIVED`, **never numerically verified**; see `13` for the E4 test.
    * **`r1`** — the expectation form of `04` §4.8.4: `(f̄_n − μ_f)²` estimated by
      `max(0, Δ² − σ̂_f²/N)`, and no additive term beyond `b² σ̂_f²/N`.  Status: `PROPOSED`.
      Note `r1 == v482` algebraically whenever `Δ² ≥ σ̂_f²/N`, and is *wider* otherwise.
    * **`r2`** — Satterthwaite: the `v482` **value**, with an effective degrees of freedom from
      the relative sizes of the `t_{n−2}` component `s²(1/n + h)` and the essentially
      known-variance component `b² σ̂_f²/N` (`04` §4.8.4).  Status: `PROPOSED`.
    * **`r5`** — **new in E4b, not in `04` §4.8.4.**  The `v482` value with the **exact** quantile
      of the idealised finite-N law of `04` §4.8.3 (E4b update), obtained by inverting the exact
      coverage integral rather than matching moments as `r2` does.  Returned as an effective `t`
      degrees of freedom so the interval constructor is unchanged.  Status: `DERIVED` under the
      stated idealisation; the implemented estimator differs from the idealisation by `O(1/N)`
      terms, so it is `NUMERICALLY VERIFIED` at best.
    * **`r3`** — the conditional form (`04` §4.8.4, "least developed").  See the note below:
      `04` specifies no formula for it, so what is implemented is **this repository's
      concretisation**, and it is labelled as such.

    `r4` is deliberately absent: it is the claim that `v482` is conservative, i.e. a theorem
    about `mode="v482"` rather than a new expression (`04` §4.8.4, `15` §5).

    **What `r3` is here.** `04` §4.8.4 R3 says only "accept the conditional-on-`f̄_N` view, and
    correct the conditional bias rather than the variance".  Conditional on the labelled design
    and on `f̄_N`, `θ̂_LC` is exactly `t_{n−2}`-pivotal about `θ + β(f̄_N − μ_f)`, so the
    conditional construction is `V̂ = s²(1/n + h)` with `t_{n−2}` and **no unlabelled term** —
    which is what is implemented.  The bias `β(f̄_N − μ_f)` is **not** corrected, because it is
    not estimable without `μ_f`; that is a finding about R3, recorded in `13`, not an omission.

    **`coef`** is the coefficient the point estimate actually used, defaulting to the OLS slope
    `b`.  With `coef_mode="ppi_plus"` (`05` step 4, OPEN) it is `b/(1+r)`, and the variance must
    follow: `Var(c·) = k²Var(b)` for a deterministic `k = 1/(1+r)`, so the leverage term is scaled
    by `k²` and the unlabelled term uses `coef²`.  **Note this does not make that estimator
    correct** — with `k ≠ 1` it acquires a bias `(k−1)β(μ_f − f̄_n)` that no variance expression
    addresses.  That is the finding, not an omission; see `13` (E4).

    Takes precomputed `summaries` (`estimators.labelled_summaries`) because every method in a
    cell shares them (`15` §4.2).
    """
    if mode not in LC_MODES:
        raise ValueError(f"unknown LC variance mode {mode!r}; available: {LC_MODES}. "
                         "hc3 is extension E5 and is not implemented.")
    if mode in LC_MODES_REQUIRING_MU_F and mu_f is None:
        raise ValueError(f"mode={mode!r} is written in terms of the population mean mu_f "
                         "(04 §4.8.3) and cannot be evaluated without it; its feasible "
                         "counterpart is mode='r1'")
    n = summaries["n"]
    s2 = summaries["s2"]
    S_ff = summaries["S_ff"]
    b = summaries["b"]
    ok = np.asarray(S_ff) > 0.0
    S_ff_safe = np.where(ok, S_ff, 1.0)
    sigma2_f = np.asarray(sigma2_f, dtype=np.float64)
    per_N = 0.0 if not np.isfinite(N) else sigma2_f / N

    coef = b if coef is None else np.asarray(coef, dtype=np.float64)
    nz = np.asarray(b) != 0.0
    k2 = np.where(nz, (coef / np.where(nz, b, 1.0)) ** 2, 1.0)   # (coef/b)^2, = 1 for coef = b

    h = _leverage(fbar_N, summaries["fbar_n"], S_ff) * k2     # (f̄_N − f̄_n)^2 / S_ff, scaled
    mean_term = s2 / n
    unl = _unlabeled_term(coef, sigma2_f, N)                  # coef^2 σ̂_f^2 / N
    dof = float(n - 2)

    if mode in ("v482", "tmin"):
        lev_term = s2 * h
    elif mode == "nolev":
        lev_term = np.zeros_like(h)
    elif mode == "r3":
        lev_term = s2 * h
        unl = np.zeros_like(h)                                # conditional-on-f̄_N view
    elif mode in ("r2", "r5"):
        lev_term = s2 * h
    elif mode == "v_full":
        gap2 = (np.asarray(summaries["fbar_n"]) - mu_f) ** 2
        lev_term = s2 * k2 * np.where(ok, (gap2 + per_N) / S_ff_safe, 0.0)
    elif mode == "r1":
        Delta2 = (np.asarray(fbar_N) - np.asarray(summaries["fbar_n"])) ** 2
        gap2_hat = np.maximum(0.0, Delta2 - per_N)
        lev_term = s2 * k2 * np.where(ok, (gap2_hat + per_N) / S_ff_safe, 0.0)

    value = mean_term + lev_term + unl
    if mode == "r5":
        # The exact quantile of the idealised law, expressed as an effective t degrees of
        # freedom.  Unlike r2 this is not a moment match: it inverts the exact coverage integral.
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(value > 0, unl / np.where(value > 0, value, 1.0), 0.0)
        grid, nu_star = _r5_dof_table(float(n - 2), float(alpha_level))
        dof = np.interp(np.clip(share, 0.0, 1.0), grid, nu_star)
    if mode == "tmin":
        # Mickey & Brown (1966) with the pool's OWN degrees of freedom supplied. Their theorem
        # bounds the Behrens-Fisher statistic below by Student-t on min(f1, f2) df; here
        # f1 = n-2 (from s^2) and f2 = N-1 (from the pool's sigma_hat_f^2). `v482` is the same
        # rule instantiated with f2 = inf, i.e. with sigma_f^2 treated as known, which is why it
        # under-covers once N-1 < n-2 (theory_finite_N.md Result G8/G9).
        #
        # PRIOR ART: the t_min conservativeness is Mickey & Brown's, not ours. Nothing is claimed
        # for the rule itself; what is ours is its identification as the right instantiation here.
        #
        # min(n-2, N-1) = n-2 whenever N >= n-1 (including N = inf), so this reduces to `v482`
        # EXACTLY on every grid the paper reports -- no A-I number can move.
        dof = float(min(float(n - 2), float(N) - 1.0))

    if mode == "r2":
        # Satterthwaite: nu_eff = (V1+V2)^2 / (V1^2/(n-2)), the V2 component having nu = inf.
        # nu_eff >= n-2 always, and -> n-2 as V2 -> 0, so it can only narrow the interval.
        V1 = mean_term + lev_term
        with np.errstate(divide="ignore", invalid="ignore"):
            dof = np.where(V1 > 0, (n - 2) * (1.0 + unl / np.where(V1 > 0, V1, 1.0)) ** 2,
                           float(n - 2))

    return VarianceResult(
        value=value,
        dof=dof,
        components={"mean_term": mean_term, "leverage_term": lev_term, "unlabeled_term": unl},
        leverage=h,
        leverage_frac=h / (1.0 / n + h),
    )


def var_lc_hc3(summaries: dict, Y, f, fbar_N, sigma2_f, N: float) -> VarianceResult:
    """HC3 sandwich variance for the LC-PPI estimator (`07` Experiment H, extension E5).

    `θ̂_LC` is a linear estimator in `Y`: writing `Δ = f̄_N − f̄_n`,

        θ̂_LC = Σ_i w_i Y_i ,     w_i = 1/n + (f_i − f̄_n)·Δ/S_ff

    so conditionally on the design `Var(θ̂_LC) = Σ_i w_i² Var(Y_i | f_i)`.  Under homoskedasticity
    `Σ_i w_i² = 1/n + Δ²/S_ff` **exactly**, which recovers `04` §4.8.2's bracket — HC3 is a strict
    generalisation of the leverage term, not a different idea.  Replacing the common `σ²` by the
    HC3 estimate of each `Var(Y_i | f_i)` gives

        V̂_HC3 = Σ_i w_i² · e_i²/(1 − h_ii)²  +  b² σ̂_f²/N ,
        e_i = Y_i − Ȳ − b(f_i − f̄_n),   h_ii = 1/n + (f_i − f̄_n)²/S_ff

    **No exactness is claimed and none is available.** `07` Experiment H says so explicitly:
    "Whether the HC variant retains any exactness claim. Almost certainly not; it would be an
    asymptotic refinement. Do not claim otherwise."  The `t_{n−2}` quantile is retained as a
    convention for comparability with B9, not because it is derived.
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = summaries["n"]
    S_ff = summaries["S_ff"]
    ok = np.asarray(S_ff) > 0.0
    S_ff_safe = np.where(ok, S_ff, 1.0)

    fc = f - summaries["fbar_n"][..., None]
    Yc = Y - summaries["Ybar"][..., None]
    e = Yc - summaries["b"][..., None] * fc
    h_ii = 1.0 / n + fc ** 2 / S_ff_safe[..., None]
    h_ii = np.minimum(h_ii, 1.0 - 1e-10)                 # guard the HC3 denominator
    gap = (np.asarray(fbar_N) - summaries["fbar_n"])[..., None]
    w = 1.0 / n + fc * gap / S_ff_safe[..., None]

    mean_term = np.sum(w ** 2 * e ** 2 / (1.0 - h_ii) ** 2, axis=-1)
    mean_term = np.where(ok, mean_term, np.asarray(Y).var(axis=-1, ddof=1) / n)
    unl = _unlabeled_term(summaries["b"], sigma2_f, N)
    h = _leverage(fbar_N, summaries["fbar_n"], S_ff)
    return VarianceResult(value=mean_term + unl, dof=float(n - 2),
                          components={"mean_term": mean_term,
                                      "leverage_term": np.zeros_like(mean_term),
                                      "unlabeled_term": unl},
                          leverage=h, leverage_frac=h / (1.0 / n + h))
