"""Gate 4 — optimality of `t_min` and the Berger-Boos candidate."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ppi_finite.gate4 import (exact_q_at_share, quantile_berger_boos, variance_share_ci)


def _cov_gamma(q, gamma, nu1, nu2, k=160):
    """Exact P(|V_gamma| <= q) for V_gamma = Z/sqrt(gamma W1 + (1-gamma) W2), by quadrature."""
    x, w = np.polynomial.legendre.leggauss(k)
    p = 0.5 * (x + 1.0); w = 0.5 * w
    q1, q2 = stats.chi2.ppf(p, nu1) / nu1, stats.chi2.ppf(p, nu2) / nu2
    r = np.sqrt(gamma * q1[:, None] + (1 - gamma) * q2[None, :])
    return float(np.sum(w[:, None] * w[None, :] * (2 * stats.norm.cdf(q * r) - 1)))


# --------------------------------------------------------------------------- Theorem G12

@pytest.mark.parametrize("nu1,nu2", [(48, 24), (8, 24), (18, 24), (8, 4), (48, 1), (48, 49)])
def test_mickey_brown_lower_bound_is_attained_at_an_endpoint(nu1, nu2):
    """The infimum over gamma of the coverage equals 1-alpha exactly, at gamma in {0,1}.

    This is what makes t_min unimprovable among constant critical values: any smaller q fails at
    that endpoint, where V_gamma is exactly t_{min(nu1,nu2)}.
    """
    q = stats.t.ppf(0.975, min(nu1, nu2))
    gg = np.linspace(0.0, 1.0, 101)
    cov = np.array([_cov_gamma(q, g, nu1, nu2) for g in gg])
    assert cov.min() == pytest.approx(0.95, abs=2e-4)
    assert np.all(cov >= 0.95 - 2e-4)                     # Mickey & Brown's bound holds
    assert gg[int(cov.argmin())] in (0.0, 1.0)            # and is attained at an endpoint


@pytest.mark.parametrize("nu1,nu2", [(48, 24), (8, 24), (8, 4)])
def test_any_smaller_constant_quantile_under_covers(nu1, nu2):
    """Theorem G12's contrapositive: shrink q below t_min and some gamma drops below 1-alpha."""
    q = stats.t.ppf(0.975, min(nu1, nu2)) * 0.99
    worst = min(_cov_gamma(q, g, nu1, nu2) for g in (0.0, 1.0))
    assert worst < 0.95


def test_endpoints_are_exactly_student_t():
    for nu1, nu2 in [(8, 24), (48, 2)]:
        for gamma, nu in ((1.0, nu1), (0.0, nu2)):
            q = stats.t.ppf(0.975, nu)
            assert _cov_gamma(q, gamma, nu1, nu2) == pytest.approx(0.95, abs=2e-4)


# --------------------------------------------------------------------------- Berger-Boos

def test_berger_boos_respects_both_clips():
    """Criterion 1 and 4: never below t_{n-2}, never above t_min."""
    rng = np.random.default_rng(0)
    for n, N in [(50, 25), (50, 2), (20, 5), (10, 25), (50, 1000)]:
        V1 = rng.gamma(3.0, 1.0, 500); V2 = rng.gamma(3.0, 1.0, 500)
        q = quantile_berger_boos(V1, V2, n, N)
        lo = stats.t.ppf(0.975, n - 2)
        hi = stats.t.ppf(0.975, min(n - 2, N - 1))
        assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)


def test_berger_boos_reduces_to_the_deployed_rule_when_N_at_least_n_minus_1():
    """Criteria 1 and 5, analytically: no dependence on the data at all in that region."""
    rng = np.random.default_rng(1)
    for n, N in [(10, 9), (10, 25), (50, 49), (20, 1000)]:
        q = quantile_berger_boos(rng.gamma(2.0, 1.0, 200), rng.gamma(2.0, 1.0, 200), n, N)
        assert np.allclose(q, stats.t.ppf(0.975, n - 2), rtol=0, atol=0)


def test_variance_share_ci_brackets_and_is_ordered():
    lo, hi = variance_share_ci(np.array([1.0, 1.0]), np.array([1.0, 4.0]), 48.0, 24.0, 0.01)
    assert np.all(lo >= 0) and np.all(hi <= 1) and np.all(lo < hi)
    lo2, hi2 = variance_share_ci(np.array([1.0]), np.array([4.0]), 48.0, 24.0, 0.01)
    assert lo2[0] > lo[0]                                  # more pool variance -> larger share


def test_exact_q_at_share_endpoints():
    nu1, nu2, a = 48.0, 24.0, 0.05
    assert exact_q_at_share(0.0, nu1, nu2, a) == pytest.approx(stats.t.ppf(1 - a / 2, nu1), rel=2e-3)
    assert exact_q_at_share(1.0, nu1, nu2, a) == pytest.approx(stats.t.ppf(1 - a / 2, nu2), rel=2e-2)


@pytest.mark.slow
def test_saved_gate4_results_show_no_under_coverage_for_tmin_or_bb():
    import csv
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "results" / "gate4_quantiles.csv"
    rows = list(csv.DictReader(open(p)))
    se = float(rows[0]["mc_se"])
    for cand in ("B9_tmin", "C5_bb"):
        cov = [float(r["coverage"]) for r in rows if r["candidate"] == cand]
        assert len(cov) > 100
        assert min(cov) >= 0.95 - 3 * se, f"{cand} under-covered at {min(cov)}"
    # and Berger-Boos is never wider than t_min
    assert max(float(r["width_rel_tmin"]) for r in rows if r["candidate"] == "C5_bb") <= 1.0 + 1e-9
