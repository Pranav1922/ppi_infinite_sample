"""The PPI++ family (L2): B3, B4, B5, B6 (extension E1, `15` §11.3).

All four share one point-estimate form, `θ̂_λ = Ȳ − λ(f̄_n − f̄_N)` (`04` §4.4), and one variance,
`V̂ar(Y − λf)/n + λ²V̂ar(f)/N` (`04` §4.5).  They differ only in **which λ** and **which
quantile** — which is the whole point of the decomposition `06` B4–B6 is built to provide:

| ID | λ | quantile | isolates |
|---|---|---|---|
| B3 | plug-in `b/(1+r)` | `z` | the method as shipped |
| B4 | plug-in `b/(1+r)` | `t_{n−2}` | how much of B3's deficit is merely the wrong quantile |
| B5 | cross-fitted | `t_{n−2}` | the fix recommended by arXiv:2505.20178 |
| B6 | plug-in or 0, by significance | `t_{n−2}` | whether a selection rule can replace a variance |

B10's shrunk λ̃ (`04` §4.9) is **not** here: it is demoted to an efficiency mechanism and is not
part of E1.
"""
from __future__ import annotations

import numpy as np

from . import variance as V
from .confidence_intervals import ci_from_se
from .estimators import (lambda_crossfit, lambda_plugin, lambda_threshold, summaries_of,
                         theta_ppi_plus)
from .types import IntervalResult, PPIView

__all__ = ["method_b3", "method_b4", "method_b5", "method_b5_kfold", "method_b5_avglam",
           "method_b6", "method_b6_t15", "method_b6_t3", "ppi_plus_interval"]


def method_b3(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None):
    """B3 — PPI++ as shipped (`04` §4.4–4.5, `06` B3, `09` §Baseline configuration).

    `λ̂ = b/(1 + n/N)`; `θ̂ = Ȳ − λ̂(f̄_n − f̄_N)`; `V̂ = V̂ar(Y − λ̂f)/n + λ̂²V̂ar(f)/N` with labelled-sample `ddof = 1` moments; **normal**
    quantile.  This is the object of the critique: `V̂` is the variance θ̂_λ would have at a fixed
    λ, with no term for λ̂ having been estimated on the same n gold pairs.
    """
    s = summaries if summaries is not None else summaries_of(data)
    lam = lambda_plugin(s["b"], s["n"], data.N)
    theta = theta_ppi_plus(data.Y, data.f, data.fbar_N, lam)
    v = V.var_ppi_plus(data.Y, data.f, lam, data.N)
    se = np.sqrt(v.value)
    low, high = ci_from_se(theta, se, alpha_level=alpha_level, dist="normal")
    return IntervalResult("B3", theta, low, high, se, np.inf,
                          extras={"variance_mode": "ppi_plus", "dist": "normal",
                                  "lambda_mean": float(np.mean(lam))})


def _interval(method_id: str, data: PPIView, alpha_level: float, lam, dist: str, dof,
              theta=None, extras: dict | None = None) -> IntervalResult:
    """The one composition B3-B6 share: point estimate, B3's variance, a quantile."""
    if theta is None:
        theta = theta_ppi_plus(data.Y, data.f, data.fbar_N, lam)
    v = V.var_ppi_plus(data.Y, data.f, lam, data.N)
    se = np.sqrt(v.value)
    low, high = ci_from_se(theta, se, alpha_level=alpha_level, dof=dof, dist=dist)
    return IntervalResult(method_id, theta, low, high, se, dof,
                          extras={"variance_mode": "ppi_plus", "dist": dist,
                                  "lambda_mean": float(np.mean(lam)), **(extras or {})})


def method_b4(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None):
    """B4 — PPI++ with a Student-`t` quantile (`06` B4).

    Identical to B3 except `t_{n−2, 1−α/2}` replaces `z_{1−α/2}`.  It exists to show that the
    quantile alone does not suffice: `09` Table A4 puts it at ≈ 0.924 against a 0.950 target,
    recovering about 64 % of B3's deficit.
    """
    s = summaries if summaries is not None else summaries_of(data)
    lam = lambda_plugin(s["b"], s["n"], data.N)
    return _interval("B4", data, alpha_level, lam, "t", float(s["n"] - 2))


def _crossfit(method_id: str, data: PPIView, alpha_level: float, summaries: dict | None,
              scheme: str, k: int, average_lambda: bool) -> IntervalResult:
    s = summaries if summaries is not None else summaries_of(data)
    n = s["n"]
    lams, folds = lambda_crossfit(data.Y, data.f, data.N, scheme=scheme, k=k)
    lam_bar = sum(lams) / len(lams)

    if average_lambda:
        # cover.py's scheme: average the fold lambdas, then apply the average to the WHOLE
        # gold sample.  This is what produced 09 Table A4's B5 column.
        theta = None
        lam_for_variance = lam_bar
    else:
        # 06 B5 as written (and probe.py): apply each fold's held-out lambda to that fold only,
        # then average the fold-level estimates unweighted, as 06's "1/2 (theta_A + theta_B)".
        parts = []
        for lam_j, fold in zip(lams, folds):
            Yj, fj = data.Y[..., fold], data.f[..., fold]
            parts.append(Yj.mean(axis=-1) - lam_j * (fj.mean(axis=-1)
                                                     - np.asarray(data.fbar_N)))
        theta = sum(parts) / len(parts)
        lam_for_variance = lam_bar

    return _interval(method_id, data, alpha_level, lam_for_variance, "t", float(n - 2),
                     theta=theta,
                     extras={"crossfit_scheme": scheme, "crossfit_k": len(folds),
                             "crossfit_average_lambda": average_lambda})


def method_b5(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None):
    """B5 — cross-fitted PPI++, 2-fold, exactly as `06` B5 writes it.

        λ̂_A = S_fY(A)/S_ff(A),  λ̂_B = S_fY(B)/S_ff(B)
        θ̂_B = mean_{i∈B}(Y_i − λ̂_A f_i) + λ̂_A f̄_N ,   θ̂_A likewise with λ̂_B
        θ̂   = ½(θ̂_A + θ̂_B)

    Variance as in B3 on the combined sample (with the averaged λ̂), quantile `t_{n−2}`.
    This is the incumbent fix — the variant analysed by arXiv:2505.20178 — so showing that it
    does not restore coverage is load-bearing for the project's argument.
    """
    return _crossfit("B5", data, alpha_level, summaries, "2fold", 2, average_lambda=False)


def method_b5_kfold(data: PPIView, alpha_level: float = 0.05, rng=None,
                    summaries: dict | None = None):
    """B5-kfold — the K-fold alternative `06` B5 requires so that "cross-fitting does not fix
    it" is not an artefact of one particular split.  `k = 5`, folds contiguous."""
    return _crossfit("B5-kfold", data, alpha_level, summaries, "kfold", 5, average_lambda=False)


def method_b5_avglam(data: PPIView, alpha_level: float = 0.05, rng=None,
                     summaries: dict | None = None):
    """B5-avglam — average the fold λ̂'s and apply the average to the whole gold sample.

    This is the third scheme named in `06` B5's implementation note ("averaging λ̂ rather than
    averaging estimates"), and it is what `cover.py` actually computed, so it is the scheme that
    produced `09` Table A4's B5 column.  Registered so that the Table A4 target has an exact
    counterpart and the two schemes can be compared on identical draws.
    """
    return _crossfit("B5-avglam", data, alpha_level, summaries, "2fold", 2, average_lambda=True)


def method_b6(data: PPIView, alpha_level: float = 0.05, rng=None, summaries: dict | None = None,
              t_crit: float = 2.0):
    """B6 — hard-threshold PPI++ (`06` B6): use λ̂ only if the slope is significant.

    `λ_thr = λ̂` if `(b/se(b))² > t_crit²` else 0, then B3's variance and `t_{n−2}`.  The crudest
    plausible repair, included to demonstrate that a selection rule is not a substitute for a
    correct variance: `09` Table A4 shows it non-monotone in ρ and **worst where the predictor is
    good** (0.9194 at ρ = 0.8, n = 10).
    """
    s = summaries if summaries is not None else summaries_of(data)
    lam = lambda_threshold(s["b"], s["se_b"], s["n"], data.N, t_crit=t_crit)
    method_id = "B6" if t_crit == 2.0 else f"B6-t{t_crit:g}"
    return _interval(method_id, data, alpha_level, lam, "t", float(s["n"] - 2),
                     extras={"t_crit": t_crit,
                             "frac_lambda_zero": float(np.mean(np.asarray(lam) == 0.0))})


def method_b6_t15(data: PPIView, alpha_level: float = 0.05, rng=None,
                  summaries: dict | None = None):
    """B6 at `|t| > 1.5`.  `10` H2's guard requires the hypothesis to be tested against three
    thresholds, not the one arbitrary value `06` B6 happens to name."""
    return method_b6(data, alpha_level, rng, summaries, t_crit=1.5)


def method_b6_t3(data: PPIView, alpha_level: float = 0.05, rng=None,
                 summaries: dict | None = None):
    """B6 at `|t| > 3`.  See `method_b6_t15`."""
    return method_b6(data, alpha_level, rng, summaries, t_crit=3.0)


def ppi_plus_interval(Y, f, f_unlabeled=None, *, mu_f=None, N=None, sigma2_f=None,
                      alpha_level: float = 0.05, lambda_rule: str = "plugin",
                      dist: str = "normal", k: int = 5, t_crit: float = 2.0) -> IntervalResult:
    """One-call wrapper over the PPI++ family (`15` §6.5)."""
    from .types import make_data
    data = make_data(Y, f, f_unlabeled=f_unlabeled, mu_f=mu_f, N=N, sigma2_f=sigma2_f).view()
    rules = {"plugin": method_b3 if dist == "normal" else method_b4,
             "crossfit": method_b5, "crossfit_kfold": method_b5_kfold,
             "crossfit_avglam": method_b5_avglam, "threshold": method_b6}
    if lambda_rule not in rules:
        raise ValueError(f"lambda_rule must be one of {sorted(rules)}; got {lambda_rule!r}")
    if lambda_rule == "threshold":
        return method_b6(data, alpha_level, t_crit=t_crit)
    return rules[lambda_rule](data, alpha_level)
