"""B9-tmin — the registered finite-pool safeguard, `t_{min(n−2, N−1)}`.

The rule is Mickey & Brown (1966); these tests check only that it is *instantiated* correctly:
which degrees of freedom are selected, and that it collapses onto B9 exactly when `N ≥ n − 1`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from ppi_finite import registry
from ppi_finite.estimators import labelled_summaries
from ppi_finite.simulation import dgp_gaussian
from ppi_finite.variance import LC_MODES, var_lc_ppi


def _draw(n, N, rho=0.9, R=64, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = dgp_gaussian(np.random.default_rng(seed), R=R, n=n, rho=rho, N=N,
                         unlabeled_mode="explicit")
    v = d.view()
    return v, labelled_summaries(v.Y, v.f)


def _dof(res_or_dof):
    """The single degrees-of-freedom value, whether stored as a scalar or a per-row array."""
    d = np.unique(np.asarray(getattr(res_or_dof, "dof", res_or_dof), dtype=float))
    assert d.size == 1, f"expected one dof value, got {d}"
    return float(d[0])


def _pair(n, N, rho=0.9, seed=0):
    v, sm = _draw(n, N, rho, seed=seed)
    return (registry.get("B9")(v, 0.05, None, summaries=sm),
            registry.get("B9-tmin")(v, 0.05, None, summaries=sm))


def test_registered_and_deterministic_kind():
    assert "B9-tmin" in registry.METHODS
    assert "B9-tmin" not in registry.STOCHASTIC_METHODS
    assert "tmin" in LC_MODES
    assert "B9" in registry.METHODS          # the original is preserved


@pytest.mark.parametrize("n,N", [(10, 9), (10, 25), (10, 1000), (20, 19), (50, 49), (4, 3)])
def test_N_at_least_n_minus_1_selects_t_n_minus_2(n, N):
    """min(n−2, N−1) = n−2, so the safeguard IS the deployed interval here."""
    b9, tm = _pair(n, N)
    assert _dof(tm) == float(n - 2)
    assert np.array_equal(b9.ci_low, tm.ci_low)
    assert np.array_equal(b9.ci_high, tm.ci_high)
    assert np.array_equal(b9.theta_hat, tm.theta_hat)
    assert np.array_equal(b9.se, tm.se)


@pytest.mark.parametrize("n,N", [(50, 25), (50, 2), (20, 10), (10, 5), (50, 48)])
def test_N_below_n_minus_1_selects_t_N_minus_1(n, N):
    """min(n−2, N−1) = N−1, and the interval strictly widens."""
    b9, tm = _pair(n, N)
    assert _dof(tm) == float(N - 1)
    assert np.all(tm.ci_high - tm.ci_low >= b9.ci_high - b9.ci_low)
    assert np.any(tm.ci_high - tm.ci_low > b9.ci_high - b9.ci_low)
    # the point estimate and the standard error are untouched: only the quantile changes
    assert np.array_equal(b9.theta_hat, tm.theta_hat)
    assert np.allclose(b9.se, tm.se, rtol=0, atol=0)


def test_the_boundary_is_exactly_N_equals_n_minus_1():
    n = 50
    assert _dof(_pair(n, n - 1)[1]) == float(n - 2)     # N = n-1 -> t_{n-2}
    assert _dof(_pair(n, n - 2)[1]) == float(n - 3)     # N = n-2 -> t_{N-1}


def test_width_ratio_matches_the_quantile_ratio_exactly():
    """Widening is exactly t_{N−1}/t_{n−2}: the variance is B9's, unchanged."""
    n, N = 50, 25
    b9, tm = _pair(n, N)
    want = stats.t.ppf(0.975, N - 1) / stats.t.ppf(0.975, n - 2)
    got = (tm.ci_high - tm.ci_low) / (b9.ci_high - b9.ci_low)
    assert np.allclose(got, want, rtol=1e-12)


def test_infinite_pool_reduces_to_the_deployed_rule():
    v, sm = _draw(10, 25)
    r = var_lc_ppi(sm, v.fbar_N, v.sigma2_f, np.inf, mode="tmin")
    assert _dof(r) == 8.0


def test_variance_value_is_identical_to_v482():
    """Only `dof` differs between the two modes -- the variance expression is the same."""
    for n, N in [(10, 25), (50, 25), (20, 3)]:
        v, sm = _draw(n, N)
        a = var_lc_ppi(sm, v.fbar_N, v.sigma2_f, N, mode="v482")
        b = var_lc_ppi(sm, v.fbar_N, v.sigma2_f, N, mode="tmin")
        assert np.array_equal(a.value, b.value)
        assert np.array_equal(a.leverage, b.leverage)


def test_deterministic_across_repeated_calls():
    a = _pair(50, 25, seed=3)[1]
    b = _pair(50, 25, seed=3)[1]
    c = _pair(50, 25, seed=4)[1]
    assert np.array_equal(a.ci_low, b.ci_low) and np.array_equal(a.ci_high, b.ci_high)
    assert not np.array_equal(a.ci_low, c.ci_low)


@pytest.mark.slow
def test_experiment_E_grid_is_unchanged_by_the_safeguard():
    """Regression on the saved Gate-3 re-run: B9-tmin == B9 on every Experiment E cell."""
    import csv
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "results" / "gate3" / "E_finite_N.csv"
    rows = list(csv.DictReader(open(p)))
    key = lambda r: (r["n"], r["N"], r["rho"], r["alpha_level"], r["metric"])
    b9 = {key(r): float(r["value"]) for r in rows if r["method"] == "B9"}
    tm = {key(r): float(r["value"]) for r in rows if r["method"] == "B9-tmin"}
    assert b9 and set(b9) == set(tm)
    worst = max(abs(b9[k] - tm[k]) for k in b9 if k[-1] != "seconds_per_interval")
    assert worst == 0.0, f"B9-tmin diverged from B9 on Experiment E by {worst}"
