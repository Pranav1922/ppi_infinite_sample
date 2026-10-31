"""Sufficient statistics, point estimators and lambda rules (L1).

Every function is batched: arrays of shape `(n,)` or `(R, n)` both work and the return shape
follows (scalar or `(R,)`).  Reductions are over the last axis.

Equation references are to `04_MATHEMATICAL_FORMULATION.md`.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import PPIView

__all__ = [
    "labelled_summaries", "theta_classical", "theta_ppi", "theta_ppi_plus",
    "lambda_plugin", "lambda_crossfit", "lambda_threshold", "crossfit_folds", "summaries_of",
]

Array = NDArray[np.float64]


def labelled_summaries(Y: Array, f: Array) -> dict:
    """Sufficient statistics of the gold sample (`04` §4.1, §4.7).

    Returns `Ybar`, `fbar_n`, `S_ff`, `S_fY`, `S_YY`, `b`, `s2`, `se_b`, `n`, `degenerate`.

    `S_ff = Σ_i (f_i − f̄_n)²` is a **sum of squares, not a variance** (`04` §4.1).
    `b = S_fY / S_ff` is the OLS slope; `s2 = RSS/(n−2)`; `se_b = s/√S_ff` (`04` §4.9).

    Where `S_ff == 0` (all predictions identical) the slope is undefined; `b` and `se_b` are
    set to 0 and `degenerate` marks those replications, so the caller can fall back to the
    classical interval (`05` §Preconditions).
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = Y.shape[-1]
    Ybar = Y.mean(axis=-1)
    fbar_n = f.mean(axis=-1)
    Yc = Y - Ybar[..., None]
    fc = f - fbar_n[..., None]
    S_ff = np.sum(fc * fc, axis=-1)
    S_fY = np.sum(fc * Yc, axis=-1)
    S_YY = np.sum(Yc * Yc, axis=-1)
    degenerate = S_ff <= 0.0
    S_ff_safe = np.where(degenerate, 1.0, S_ff)
    b = np.where(degenerate, 0.0, S_fY / S_ff_safe)
    resid = Yc - b[..., None] * fc
    s2 = np.sum(resid * resid, axis=-1) / (n - 2)
    se_b = np.where(degenerate, 0.0, np.sqrt(s2 / S_ff_safe))
    return {
        "Ybar": Ybar, "fbar_n": fbar_n, "S_ff": S_ff, "S_fY": S_fY, "S_YY": S_YY,
        "b": b, "s2": s2, "se_b": se_b, "n": n, "degenerate": degenerate,
    }


def summaries_of(data: PPIView) -> dict:
    """`labelled_summaries` for a dataset.  Computed once per cell and shared (`15` §4.2)."""
    return labelled_summaries(data.Y, data.f)


def theta_classical(Y: Array) -> Array:
    """`θ̂ = Ȳ` (`04` §4.2)."""
    return np.asarray(Y, dtype=np.float64).mean(axis=-1)


def theta_ppi(Y: Array, f: Array, fbar_N) -> Array:
    """`θ̂ = Ȳ − (f̄_n − f̄_N)` (`04` §4.3).  The λ = 1 case of `theta_ppi_plus`."""
    return theta_ppi_plus(Y, f, fbar_N, 1.0)


def theta_ppi_plus(Y: Array, f: Array, fbar_N, lam) -> Array:
    """`θ̂_λ = Ȳ − λ (f̄_n − f̄_N)` (`04` §4.4).

    λ = 0 gives `theta_classical`; λ = 1 gives `theta_ppi`.  With λ = b this is also the
    LC-PPI point estimate `Ȳ + b(f̄_N − f̄_n)` (`04` §4.7).
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    return Y.mean(axis=-1) - np.asarray(lam) * (f.mean(axis=-1) - np.asarray(fbar_N))


def lambda_plugin(b: Array, n: int, N: float) -> Array:
    """`λ̂ = b / (1 + n/N)` (`04` §4.4); `r = n/N`, and `r = 0` when `N` is infinite."""
    r = 0.0 if not np.isfinite(N) else n / N
    return np.asarray(b) / (1.0 + r)


def crossfit_folds(n: int, scheme: str = "2fold", k: int = 5) -> list[np.ndarray]:
    """Index sets for cross-fitting (`06` B5).

    * `"2fold"` — two contiguous folds of sizes `⌊n/2⌋` and `⌈n/2⌉`, exactly as `06` B5 states
      and as `cover.py` / `probe.py` construct them.
    * `"kfold"` — `k` contiguous folds via `np.array_split`, the K-fold alternative `06` B5
      requires the implementation to offer as well.

    Contiguous rather than randomised: the gold pairs are i.i.d., so a contiguous split has the
    same distribution as a random one, and using no generator keeps C1 (`15` §0.2) trivially
    satisfied — a method must never consume randomness it was not given.
    """
    idx = np.arange(n)
    if scheme == "2fold":
        return [idx[: n // 2], idx[n // 2:]]
    if scheme == "kfold":
        if not 2 <= k <= n:
            raise ValueError(f"kfold needs 2 <= k <= n; got k={k}, n={n}")
        return list(np.array_split(idx, k))
    raise ValueError(f"scheme must be '2fold' or 'kfold'; got {scheme!r}")


def lambda_crossfit(Y: Array, f: Array, N: float, *, scheme: str = "2fold", k: int = 5):
    """Per-fold plug-in λ̂, each estimated on the **complement** of the fold it is used on.

    Returns `(lams, folds)` where `lams[j]` has the batch shape and is the λ̂ to apply to
    `folds[j]`, and `folds` comes from `crossfit_folds`.  For `"2fold"` this is `06` B5's
    `λ̂_A`/`λ̂_B` exactly (fold B's λ̂ is estimated on fold A and vice versa).

    Each fold λ̂ carries the `1/(1 + n/N)` factor of `04` §4.4, as `probe.py` does.  `06` B5
    writes the fold coefficients as bare slopes `S_fY/S_ff`; the two differ by `r = n/N ≤ 10⁻³`
    in every configuration in `09` and the choice is logged in `13`.
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = Y.shape[-1]
    folds = crossfit_folds(n, scheme, k)
    lams = []
    for j in range(len(folds)):
        complement = np.concatenate([folds[i] for i in range(len(folds)) if i != j])
        b_c = labelled_summaries(Y[..., complement], f[..., complement])["b"]
        lams.append(lambda_plugin(b_c, n, N))
    return lams, folds


def lambda_threshold(b: Array, se_b: Array, n: int, N: float, *, t_crit: float = 2.0) -> Array:
    """`λ_thr = λ̂` if `t² > t_crit²`, else 0 (`06` B6).

    `t² = (b / se(b))²` with `se(b) = s/√S_ff`, i.e. the studentised OLS slope of `04` §4.9.
    `t_crit` is a parameter, not a constant: `06` B6 records that `|t| > 2` has **no principled
    justification** and requires Experiment D to sweep `{1.5, 2, 3}`.

    `cover.py` and `probe.py` put `λ̂` rather than `b` in the numerator (`09` §Provenance, known
    issue 1); `04` §4.9 says a general implementation should use `b`, which is what this does.
    They differ by the factor `(1 + n/N)`, ≤ 10⁻³ in every configuration in `09`.
    """
    b = np.asarray(b, dtype=np.float64)
    se_b = np.asarray(se_b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(se_b > 0, (b / np.where(se_b > 0, se_b, 1.0)) ** 2, 0.0)
    return np.where(t2 > t_crit ** 2, lambda_plugin(b, n, N), 0.0)
