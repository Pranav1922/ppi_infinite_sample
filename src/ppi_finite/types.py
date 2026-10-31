"""Data contracts (L1).  Architecture `15` §3, `11` §Data contracts.

Three types cross module boundaries: `PPIData` (a dataset), `PPIView` (a dataset with the
ground truth structurally removed, which is what methods see), and `IntervalResult` (what a
method returns).  `CellSpec` is one point of an experiment sweep.

This module contains no statistics.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["PPIData", "PPIView", "IntervalResult", "CellSpec", "make_data"]

Array = NDArray[np.float64]


def _as_finite_array(name: str, a: Any) -> Array:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape (n,) or (R, n); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or non-finite values")
    return arr


@dataclass(frozen=True)
class PPIView:
    """A dataset as a *method* sees it: `theta_true` is not a field (`15` §4.3).

    Shapes: `Y`, `f` are `(n,)` or `(R, n)`; `fbar_N` is a scalar or `(R,)`.
    """

    Y: Array
    f: Array
    fbar_N: Array | float
    N: float
    sigma2_f: float | Array
    f_unlabeled: Array | None = None
    mu_f: float | None = None

    @property
    def n(self) -> int:
        return int(self.Y.shape[-1])

    @property
    def R(self) -> int | None:
        return int(self.Y.shape[0]) if self.Y.ndim == 2 else None


@dataclass(frozen=True)
class PPIData:
    """A dataset plus its ground truth.  `theta_true` is EVALUATION-ONLY (`15` §4.3).

    Exactly one of `f_unlabeled`, `(fbar_N, N)`, `mu_f` must be supplied; use `make_data`.
    """

    Y: Array
    f: Array
    f_unlabeled: Array | None
    fbar_N: Array | float | None
    sigma2_f: float | Array | None
    N: float
    theta_true: float | None
    mu_f: float | None = None

    def view(self) -> PPIView:
        """Project away `theta_true`.  The only way a method receives data."""
        return PPIView(
            Y=self.Y,
            f=self.f,
            fbar_N=self.fbar_N,
            N=self.N,
            sigma2_f=self.sigma2_f,
            f_unlabeled=self.f_unlabeled,
            mu_f=self.mu_f,
        )

    def with_theta_true(self, theta_true: float | None) -> "PPIData":
        """Same arrays (identical objects), different ground truth.  Used by `test_crn`."""
        return replace(self, theta_true=theta_true)


def make_data(
    Y,
    f,
    *,
    f_unlabeled=None,
    fbar_N=None,
    mu_f: float | None = None,
    sigma2_f: float | None = None,
    N: int | float | None = None,
    theta_true: float | None = None,
) -> PPIData:
    """Construct and validate a `PPIData` (`15` §6.1).

    Raises (never warns-and-guesses) on shape mismatch, on a missing or over-specified
    unlabelled specification, on `n < 3`, and on non-finite values.  Warns on `n == 3`
    (one degree of freedom) and on `N / n < 1000` (`04` §4.8.2, known inexact there).
    """
    Y = _as_finite_array("Y", Y)
    f = _as_finite_array("f", f)
    if Y.shape != f.shape:
        raise ValueError(f"Y.shape {Y.shape} != f.shape {f.shape}")
    n = Y.shape[-1]
    if n < 3:
        raise ValueError(f"n must be >= 3 for s2 to be defined (04 §4.10 A4); got n={n}")

    # Exactly one input determines the unlabelled summary the estimator uses.  `mu_f` may
    # additionally accompany `f_unlabeled` or `fbar_N`, in which case it is carried for
    # DIAGNOSTIC use only (`04` §4.8.3's V_full is written in terms of mu_f) and the estimator
    # still uses the finite-pool mean.  See `13` (2026-08-20, E4).
    determines = [f_unlabeled is not None, fbar_N is not None]
    if sum(determines) > 1:
        raise ValueError("supply at most one of f_unlabeled, fbar_N; both were given")
    if sum(determines) == 0 and mu_f is None:
        raise ValueError(
            "supply exactly one of f_unlabeled, fbar_N (with N), mu_f; none was given")

    if f_unlabeled is not None:
        f_unlabeled = _as_finite_array("f_unlabeled", f_unlabeled)
        N_eff = float(f_unlabeled.shape[-1])
        fbar_N_eff: Array | float = f_unlabeled.mean(axis=-1)
        sigma2_eff = sigma2_f if sigma2_f is not None else f_unlabeled.var(axis=-1, ddof=1)
    elif fbar_N is not None:
        if N is None:
            raise ValueError("fbar_N requires N")
        N_eff = float(N)
        fbar_N_eff = np.asarray(fbar_N, dtype=np.float64) if np.ndim(fbar_N) else float(fbar_N)
        if sigma2_f is None:
            raise ValueError("mean-only mode requires sigma2_f (it cannot be estimated from fbar_N)")
        sigma2_eff = sigma2_f
    else:
        N_eff = float(np.inf)
        fbar_N_eff = float(mu_f)  # type: ignore[arg-type]
        sigma2_eff = sigma2_f if sigma2_f is not None else 0.0

    if np.ndim(fbar_N_eff) and Y.ndim == 2 and np.shape(fbar_N_eff) != (Y.shape[0],):
        raise ValueError(f"fbar_N shape {np.shape(fbar_N_eff)} incompatible with Y shape {Y.shape}")
    if not np.all(np.isfinite(np.asarray(sigma2_eff, dtype=np.float64))):
        raise ValueError("sigma2_f must be finite")

    if n == 3:
        warnings.warn("n == 3: t_{n-2} has one degree of freedom; the interval is degenerate "
                      "in practice (04 §4.10 A4)", UserWarning, stacklevel=2)
    if np.isfinite(N_eff) and N_eff / n < 1000:
        warnings.warn(f"N/n = {N_eff / n:.3g} < 1000: finite-N regime where the interval of "
                      "04 §4.8.2 is known to be inexact", UserWarning, stacklevel=2)

    return PPIData(
        Y=Y, f=f, f_unlabeled=f_unlabeled, fbar_N=fbar_N_eff,
        sigma2_f=sigma2_eff, N=N_eff, theta_true=theta_true, mu_f=mu_f,
    )


@dataclass(frozen=True)
class IntervalResult:
    """What every method returns (`11` §Data contracts)."""

    method: str
    theta_hat: Array
    ci_low: Array
    ci_high: Array
    se: Array
    dof: Array | float
    extras: dict = field(default_factory=dict)
    n_failed: int = 0

    def __post_init__(self) -> None:
        if np.any(self.ci_low > self.ci_high):
            raise ValueError(f"{self.method}: ci_low > ci_high")

    @property
    def width(self) -> Array:
        return self.ci_high - self.ci_low


@dataclass(frozen=True)
class CellSpec:
    """One point of an experiment sweep (`15` §7.3)."""

    experiment_id: str
    dgp: str
    dgp_params: dict
    n: int
    N: float
    rho: float
    alpha_level: float
    R: int
    methods: tuple[str, ...]
    unlabeled_mode: str
    coef_mode: str
    seed_spawn_key: tuple[int, ...]
    seed_entropy: int
    save_raw: bool = False
    #: name of the DGP parameter being swept, if any; its value goes in `dgp_params`
    dgp_swept_key: str | None = None

    @property
    def dgp_label(self) -> str:
        """`gaussian`, or `heavy_tail:nu=5` when a DGP parameter is swept (`15` §8.1)."""
        if self.dgp_swept_key is None:
            return self.dgp
        return f"{self.dgp}:{self.dgp_swept_key}={self.dgp_params[self.dgp_swept_key]}"

    @property
    def cell_id(self) -> str:
        return (f"{self.experiment_id}|n={self.n}|N={self.N:g}|rho={self.rho}"
                f"|alpha={self.alpha_level}|dgp={self.dgp_label}")

    @property
    def seed_spec(self) -> str:
        return f"SeedSequence(entropy={self.seed_entropy}, spawn_key={self.seed_spawn_key})"
