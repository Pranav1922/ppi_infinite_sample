"""Evaluation (L3).  `(IntervalResult, θ_true) -> metric rows`.  `06` §Reporting, `09` §Output.

`θ_true` enters the system here and nowhere else (`15` §4.3).

Failed constructions are never silently dropped: `n_failed` is carried on every row, and
coverage is reported both among successes and with failures counted as non-covering.
"""
from __future__ import annotations

import numpy as np

__all__ = ["coverage", "mc_se_proportion", "width_stats", "mse", "tail_miscoverage",
           "evaluate_all", "METRIC_NAMES"]

METRIC_NAMES = (
    "coverage", "coverage_failures_as_miss", "width_mean", "width_median", "width_ratio_b1",
    "mse", "mse_ratio_b1", "tail_miscoverage_left", "tail_miscoverage_right", "theta_hat_mean", "theta_hat_sd",
    "leverage_frac_mean", "seconds_per_interval",
    # E4 finite-N diagnostics: is the variance expression right, or is it the quantile?
    "var_estimate_mean", "var_empirical", "var_ratio", "var_bias", "unlabeled_share_mean",
)


def mc_se_proportion(p: float, R: int) -> float:
    """`√(p(1−p)/R)` — the Monte-Carlo standard error of a coverage estimate (`07`)."""
    return float(np.sqrt(p * (1.0 - p) / R))


def coverage(ci_low, ci_high, theta_true: float):
    """Empirical coverage among the replications where an interval could be constructed.

    Non-finite endpoints (a failed construction — e.g. a degenerate BCa) are **excluded** here
    and counted by the caller as `n_failed`; `06` §Reporting 6 requires both figures and forbids
    dropping failures from the denominator silently, which is what
    `coverage_failures_as_miss` is for.
    """
    lo, hi = np.asarray(ci_low), np.asarray(ci_high)
    ok = np.isfinite(lo) & np.isfinite(hi)
    covered = (lo[ok] <= theta_true) & (theta_true <= hi[ok])
    p = float(covered.mean()) if covered.size else np.nan
    return p, mc_se_proportion(p, max(covered.size, 1)), covered


def tail_miscoverage(ci_low, ci_high, theta_true: float):
    """Left and right miscoverage separately (`06` §Reporting, needed for Experiment G).

    "Left" means the whole interval sits above θ, i.e. the estimate was too high.
    """
    lo, hi = np.asarray(ci_low), np.asarray(ci_high)
    ok = np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(lo[ok] > theta_true)), float(np.mean(hi[ok] < theta_true))


def width_stats(width):
    width = np.asarray(width, dtype=np.float64)
    width = width[np.isfinite(width)]
    return float(width.mean()), float(np.median(width))


def mse(theta_hat, theta_true: float) -> float:
    return float(np.mean((np.asarray(theta_hat) - theta_true) ** 2))


def evaluate_all(results: dict, theta_true: float, R: int, reference: str = "B1",
                 seconds: dict | None = None) -> list[dict]:
    """Metric rows for every method in a cell.  Widths and MSEs are reported relative to
    `reference` (B1) as well as absolutely (`06` §Reporting)."""
    ref = results.get(reference)
    ref_width = float(np.mean(ref.width)) if ref is not None else np.nan
    ref_mse = mse(ref.theta_hat, theta_true) if ref is not None else np.nan

    rows: list[dict] = []
    for method, res in results.items():
        p, se_p, covered = coverage(res.ci_low, res.ci_high, theta_true)
        se_fin = np.asarray(res.se)[np.isfinite(res.se)]
        v_est = float(np.mean(se_fin ** 2)) if se_fin.size else np.nan
        v_emp = float(np.var(res.theta_hat, ddof=1))
        w_mean, w_median = width_stats(res.width)
        m = mse(res.theta_hat, theta_true)
        left, right = tail_miscoverage(res.ci_low, res.ci_high, theta_true)
        p_fail_as_miss = float(covered.sum()) / R

        vals = {
            "coverage": (p, se_p),
            "coverage_failures_as_miss": (p_fail_as_miss, mc_se_proportion(p_fail_as_miss, R)),
            "width_mean": (w_mean, np.nan),
            "width_median": (w_median, np.nan),
            "width_ratio_b1": (w_mean / ref_width, np.nan),
            "mse": (m, np.nan),
            "mse_ratio_b1": (m / ref_mse, np.nan),
            "tail_miscoverage_left": (left, mc_se_proportion(left, R)),
            "tail_miscoverage_right": (right, mc_se_proportion(right, R)),
            # theta_hat_mean is required by V2 (09) but is absent from the metric enum of
            # 09 §Output files / 15 §8.1.  Added here; see the discrepancy note in 13.
            "seconds_per_interval": ((seconds or {}).get(method, np.nan) / R, np.nan),
            # E4: mean(se^2) is what the method claims the variance is; var(theta_hat) across
            # replications is what it actually is. Their ratio separates "the variance
            # expression is wrong" from "the quantile is wrong" (04 §4.8.3).
            "var_estimate_mean": (v_est, np.nan),
            "var_empirical": (v_emp, np.nan),
            "var_ratio": (v_est / v_emp if v_emp > 0 else np.nan, np.nan),
            "var_bias": (v_est - v_emp, np.nan),
            "theta_hat_mean": (float(np.mean(res.theta_hat)),
                               float(np.std(res.theta_hat, ddof=1) / np.sqrt(R))),
            # theta_hat_sd makes V2's Monte-Carlo error computable from the CSV alone.
            "theta_hat_sd": (float(np.std(res.theta_hat, ddof=1)), np.nan),
        }
        lf = res.extras.get("leverage_frac")
        if lf is not None:
            vals["leverage_frac_mean"] = (float(np.mean(lf)), np.nan)
        comps = res.extras.get("var_components")
        if comps and "unlabeled_term" in comps:
            # The share of the variance carried by the essentially-normal unlabelled component.
            # 04 §4.8.3's conjecture says the over-coverage is driven by this share, because a
            # t_{n-2} quantile is applied to a standard error that is partly not t-pivotal.
            tot = np.asarray(res.se) ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(tot > 0, np.asarray(comps["unlabeled_term"]) / tot, np.nan)
            vals["unlabeled_share_mean"] = (float(np.nanmean(share)), np.nan)

        for metric, (value, mc_se) in vals.items():
            rows.append({"method": method, "metric": metric, "value": value, "mc_se": mc_se,
                         "n_failed": res.n_failed,
                         "variance_mode": res.extras.get("variance_mode", "")})
    return rows
