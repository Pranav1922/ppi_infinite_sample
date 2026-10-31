"""Bootstrap intervals: B7 (percentile) and B8 (BCa).  Extension E3 (`15` §11.3).

`06` B7/B8 fix the construction: resample the **n gold pairs** `(Y_i, f_i)` with replacement,
recompute `λ̂` and `θ̂*` on each resample, and take empirical quantiles of the `θ̂*`; BCa adds a
bias correction `ẑ₀` from the position of `θ̂` in the bootstrap distribution and an acceleration
`â` from a jackknife over the n gold pairs.

The bootstrapped statistic is the **PPI++ point estimate** with the plug-in λ̂ (`06` §Summary
table, B7/B8 column "point estimate"), so B7/B8 are bootstrap intervals *for B3's estimator* —
which is what makes them the relevant competitor.

The unlabelled pool is **not** resampled by default (`06` B7): it is large and its contribution
is `O(1/N)`.  `resample_unlabeled` is exposed because `06` B7 leaves the small-`N/n` case open
and Experiment E must test both; in `mean_only` mode it resamples `f̄_N`'s sampling distribution
rather than a materialised pool, and is not used by Experiment B.

`theta_true` cannot reach any function here: they all take a `PPIView` or raw arrays (`15` §4.3).

**Convention note for `ẑ₀`.** `06` B8 says "the proportion of bootstrap replicates below the
observed estimate".  This module uses the tie-splitting form
`(#{θ̂* < θ̂} + #{θ̂* ≤ θ̂}) / 2B`, which equals that proportion whenever there are no ties (the
generic case for a continuous statistic) and is what `scipy.stats.bootstrap` uses, so the two
implementations can be compared exactly.  See `tests/test_bootstrap.py`.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .confidence_intervals import ci_from_se  # noqa: F401  (kept for API symmetry)
from .estimators import labelled_summaries, lambda_plugin
from .types import IntervalResult, PPIView

__all__ = ["bootstrap_indices", "bootstrap_replicates", "jackknife_theta", "ci_percentile",
           "ci_bca", "bootstrap_ppi", "method_b7", "method_b8", "method_b7_b9999",
           "method_b8_b9999", "MAX_GATHER_ELEMENTS"]

Array = NDArray[np.float64]

#: Chunking budget for the `(chunk, B, n)` gather.  Chunking is stream-invariant — successive
#: `rng.integers` calls consume the same values in the same order as one big call — so the
#: results do not depend on this number.  `tests/test_bootstrap.py` asserts it.
MAX_GATHER_ELEMENTS = 8_000_000


def _theta_ppi_plus_from_sums(sum_Y, sum_f, sum_fY, sum_ff, m: int, fbar_N, N: float):
    """`θ̂_λ̂` for a sample given its raw sums.  `04` §4.4, §4.5.

    `S_ff = Σf² − (Σf)²/m`, `S_fY = ΣfY − ΣfΣY/m`, `b = S_fY/S_ff`, `λ̂ = b/(1 + m/N)`.
    `b = 0` where `S_ff = 0` (a resample can be constant in `f`), which makes the estimator fall
    back to the resample mean — the same convention as `estimators.labelled_summaries`.
    """
    Ybar = sum_Y / m
    fbar = sum_f / m
    S_ff = sum_ff - sum_f * sum_f / m
    S_fY = sum_fY - sum_f * sum_Y / m
    ok = S_ff > 0
    b = np.where(ok, S_fY / np.where(ok, S_ff, 1.0), 0.0)
    lam = lambda_plugin(b, m, N)
    return Ybar - lam * (fbar - fbar_N)


def bootstrap_indices(n: int, B: int, rng: np.random.Generator, R: int | None = None):
    """Resample indices, drawn exactly as `scipy.stats.bootstrap(paired=True)` draws them:
    `rng.integers(0, n, (B, n))`, or `(R, B, n)` for a batch of R datasets."""
    shape = (B, n) if R is None else (R, B, n)
    return rng.integers(0, n, shape)


def bootstrap_replicates(Y: Array, f: Array, fbar_N, N: float, B: int,
                         rng: np.random.Generator, resample_unlabeled: bool = False,
                         sigma2_f: float = 1.0) -> Array:
    """`θ̂*` for each of B resamples of the n gold pairs (`06` B7).

    Shapes: `Y`, `f` are `(n,)` or `(R, n)`; the result is `(B,)` or `(R, B)`.
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = Y.shape[-1]
    batched = Y.ndim == 2
    fbar_N = np.asarray(fbar_N, dtype=np.float64)

    if not batched:
        idx = bootstrap_indices(n, B, rng)
        Yb, fb = Y[idx], f[idx]
        fbar_N_b = fbar_N
        if resample_unlabeled:
            fbar_N_b = fbar_N + rng.standard_normal(B) * np.sqrt(sigma2_f / N)
        return _theta_ppi_plus_from_sums(Yb.sum(-1), fb.sum(-1), (Yb * fb).sum(-1),
                                         (fb * fb).sum(-1), n, fbar_N_b, N)

    R = Y.shape[0]
    chunk = max(1, MAX_GATHER_ELEMENTS // (B * n))
    out = np.empty((R, B), dtype=np.float64)
    for start in range(0, R, chunk):
        stop = min(start + chunk, R)
        idx = bootstrap_indices(n, B, rng, R=stop - start)
        rows = np.arange(stop - start)[:, None, None]
        Yb, fb = Y[start:stop][rows, idx], f[start:stop][rows, idx]
        fbar_N_b = fbar_N[start:stop, None]
        if resample_unlabeled:
            fbar_N_b = fbar_N_b + rng.standard_normal((stop - start, B)) * np.sqrt(sigma2_f / N)
        out[start:stop] = _theta_ppi_plus_from_sums(
            Yb.sum(-1), fb.sum(-1), (Yb * fb).sum(-1), (fb * fb).sum(-1), n, fbar_N_b, N)
    return out


def jackknife_theta(Y: Array, f: Array, fbar_N, N: float) -> Array:
    """Leave-one-*pair*-out `θ̂_(i)`, `i = 1..n`, for the BCa acceleration (`06` B8).

    Closed form from the raw sums, so it costs one pass rather than n.  Each replicate is the
    statistic applied to the reduced sample of size `m = n − 1`, so its `λ̂` uses `r = m/N`.
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = Y.shape[-1]
    m = n - 1
    fbar_N = np.asarray(fbar_N, dtype=np.float64)
    if Y.ndim == 2:
        fbar_N = fbar_N[..., None]
    return _theta_ppi_plus_from_sums(
        Y.sum(-1, keepdims=True) - Y,
        f.sum(-1, keepdims=True) - f,
        (Y * f).sum(-1, keepdims=True) - Y * f,
        (f * f).sum(-1, keepdims=True) - f * f,
        m, fbar_N, N)


def ci_percentile(theta_star: Array, alpha_level: float = 0.05):
    """B7 — the `[α/2, 1−α/2]` empirical quantiles of the bootstrap distribution (`06` B7).

    Linear-interpolated quantiles, matching `scipy.stats.bootstrap(method="percentile")`, which
    uses `scipy.stats.quantile(..., method="linear")`.
    """
    q = np.quantile(theta_star, [alpha_level / 2, 1 - alpha_level / 2], axis=-1,
                    method="linear")
    return q[0], q[1]


def ci_bca(theta_star: Array, theta_hat: Array, theta_jack: Array, alpha_level: float = 0.05):
    """B8 — bias-corrected and accelerated interval (`06` B8; Efron & Tibshirani 1993).

        ẑ₀ = Φ⁻¹( #{θ̂* ≤ θ̂ with ties split} / B )
        U_i = θ̄_(·) − θ̂_(i)
        â   = Σ U_i³ / ( 6 (Σ U_i²)^{3/2} )
        α₁  = Φ( ẑ₀ + (ẑ₀ + z_{α/2}) / (1 − â(ẑ₀ + z_{α/2})) )      (α₂ with z_{1−α/2})
        CI  = [ θ̂*_(α₁) , θ̂*_(α₂) ]

    Returns `(low, high, z0, a_hat)`.  Endpoints are `nan` where the construction degenerates
    (`ẑ₀ = ±∞` because every replicate is on one side of `θ̂`, or a zero jackknife spread); the
    caller counts those as failures rather than silently repairing them (`06` §Reporting 6).
    """
    theta_star = np.asarray(theta_star, dtype=np.float64)
    B = theta_star.shape[-1]
    theta_hat = np.asarray(theta_hat, dtype=np.float64)[..., None]

    # tie-splitting rank of theta_hat in the bootstrap distribution (see module docstring)
    prop = ((theta_star < theta_hat).sum(-1) + (theta_star <= theta_hat).sum(-1)) / (2.0 * B)
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = stats.norm.ppf(prop)

        U = theta_jack.mean(-1, keepdims=True) - theta_jack
        num = (U ** 3).sum(-1)
        den = 6.0 * ((U ** 2).sum(-1)) ** 1.5
        a_hat = np.where(den != 0, num / np.where(den != 0, den, 1.0), np.nan)

        z_lo, z_hi = stats.norm.ppf(alpha_level / 2), stats.norm.ppf(1 - alpha_level / 2)
        alphas = []
        for z_q in (z_lo, z_hi):
            num_q = z0 + z_q
            alphas.append(stats.norm.cdf(z0 + num_q / (1 - a_hat * num_q)))

    lo = _quantile_along_last(theta_star, alphas[0])
    hi = _quantile_along_last(theta_star, alphas[1])
    return lo, hi, z0, a_hat


def _quantile_along_last(theta_star: Array, probs) -> Array:
    """`np.quantile(method="linear")` with a different probability per row, which is what BCa
    needs and what `np.quantile` cannot do.  Rows whose adjusted probability is not a finite
    number in [0, 1] — the degenerate constructions — return `nan` rather than raising."""
    probs = np.asarray(probs, dtype=np.float64)
    scalar = probs.ndim == 0
    p = np.atleast_1d(probs).ravel()
    B = theta_star.shape[-1]
    ts = theta_star.reshape(-1, B)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p) & (p >= 0.0) & (p <= 1.0)
    if np.any(ok):
        srt = np.sort(ts[ok], axis=-1)
        pos = p[ok] * (B - 1)
        lo_i = np.floor(pos).astype(np.int64)
        hi_i = np.minimum(lo_i + 1, B - 1)
        frac = pos - lo_i
        rows = np.arange(srt.shape[0])
        out[ok] = srt[rows, lo_i] * (1 - frac) + srt[rows, hi_i] * frac
    return out[0] if scalar else out.reshape(probs.shape)


def bootstrap_ppi(data: PPIView, *, method: str, B: int = 2000,
                  rng: np.random.Generator, alpha_level: float = 0.05,
                  resample_unlabeled: bool = False,
                  method_id: str | None = None) -> IntervalResult:
    """`15` §6.4.  `rng` is required and keyword-only: a bootstrap is the one method that
    legitimately consumes randomness, and making it explicit keeps C1 auditable."""
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca'; got {method!r}")
    if rng is None:
        raise ValueError("bootstrap_ppi requires an explicit rng (15 §6.4)")

    s = labelled_summaries(data.Y, data.f)
    lam = lambda_plugin(s["b"], s["n"], data.N)
    theta_hat = s["Ybar"] - lam * (s["fbar_n"] - np.asarray(data.fbar_N))

    theta_star = bootstrap_replicates(data.Y, data.f, data.fbar_N, data.N, B, rng,
                                      resample_unlabeled, float(np.mean(data.sigma2_f)))
    extras = {"variance_mode": "bootstrap", "dist": "empirical", "B": B,
              "resample_unlabeled": resample_unlabeled}

    if method == "percentile":
        low, high = ci_percentile(theta_star, alpha_level)
    else:
        theta_jack = jackknife_theta(data.Y, data.f, data.fbar_N, data.N)
        low, high, z0, a_hat = ci_bca(theta_star, theta_hat, theta_jack, alpha_level)
        extras |= {"z0_mean": float(np.mean(z0[np.isfinite(z0)])) if np.any(np.isfinite(z0))
                   else np.nan,
                   "a_hat_mean": float(np.mean(a_hat[np.isfinite(a_hat)]))
                   if np.any(np.isfinite(a_hat)) else np.nan}

    bad = ~(np.isfinite(low) & np.isfinite(high))
    n_failed = int(np.sum(bad))
    # The bootstrap standard error is reported for diagnostics only; the interval is not built
    # from it (06 B7/B8: the interval is the empirical quantiles).
    se = np.std(theta_star, axis=-1, ddof=1)
    return IntervalResult(method_id or ("B7" if method == "percentile" else "B8"),
                          theta_hat, low, high, se, np.nan, extras=extras, n_failed=n_failed)


def method_b7(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None):
    """B7 — percentile bootstrap of the PPI++ estimator (`06` B7).  `B = 2000` (`06` default)."""
    return bootstrap_ppi(data, method="percentile", B=2000, rng=rng, alpha_level=alpha_level)


def method_b8(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None):
    """B8 — BCa bootstrap of the PPI++ estimator (`06` B8).  Kill criterion K3."""
    return bootstrap_ppi(data, method="bca", B=2000, rng=rng, alpha_level=alpha_level)


# `07` Experiment B: "B = 2000 for exploration and 9999 for the final table".  B is part of the
# method's configuration, so it is part of the method ID (`15` §1.2's identity spine) rather
# than a hidden argument, exactly as the B6 thresholds are.
def method_b7_b9999(data: PPIView, alpha_level: float = 0.05, rng=None,
                    summaries: dict | None = None):
    """B7 at `B = 9999` — the final-table setting of `07` Experiment B."""
    return bootstrap_ppi(data, method="percentile", B=9999, rng=rng, alpha_level=alpha_level,
                         method_id="B7-B9999")


def method_b8_b9999(data: PPIView, alpha_level: float = 0.05, rng=None,
                    summaries: dict | None = None):
    """B8 at `B = 9999` — the final-table setting on which K3 is decided."""
    return bootstrap_ppi(data, method="bca", B=9999, rng=rng, alpha_level=alpha_level,
                         method_id="B8-B9999")
