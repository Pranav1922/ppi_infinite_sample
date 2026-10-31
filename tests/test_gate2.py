"""Gate 2 — the finite-N boundary, the candidate quantile rules, and bound evaluation."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ppi_finite.gate2 import (CANDIDATES, candidate_quantiles, dof_ws2, exact_two_df_nu,
                              limit_coverage, simulate_cell)
from ppi_finite.theory_gate1 import coverage_bound


# --------------------------------------------------------------------------- limiting law

def test_limit_coverage_is_the_t_N_minus_1_probability():
    """Result G8's prediction, checked against a direct evaluation."""
    for n, N in [(10, 25), (50, 25), (20, 100), (5, 3)]:
        q = stats.t.ppf(0.975, n - 2)
        assert limit_coverage(n, N) == pytest.approx(2 * stats.t.cdf(q, N - 1) - 1, abs=1e-12)


def test_boundary_is_exactly_N_less_than_n_minus_1():
    """coverage limit < 1-alpha iff N-1 < n-2. Checked on both sides for every n."""
    for n in (4, 5, 10, 20, 50):
        for N in range(2, 3 * n):
            below = limit_coverage(n, N) < 0.95
            assert below == (N < n - 1), f"n={n} N={N}"
        assert limit_coverage(n, n - 1) == pytest.approx(0.95, abs=1e-12)   # exactly nominal


def test_limit_is_approached_as_rho_goes_to_one():
    """Simulated coverage converges to the predicted limit as rho -> 1 (not fitted to it)."""
    n, N = 50, 25
    pred = limit_coverage(n, N)
    errs = [abs(simulate_cell(n, rho, N, R=60_000, seed=3)["coverage_actual"] - pred)
            for rho in (0.9, 0.99, 0.999)]
    assert errs[0] > errs[-1], "convergence should tighten with rho"
    assert errs[-1] < 0.003


# --------------------------------------------------------------------------- candidate rules

def test_ws2_recovers_each_degree_of_freedom_in_its_limit():
    assert dof_ws2(1.0, 1e-14, 8, 24) == pytest.approx(8.0, rel=1e-6)     # pool negligible
    assert dof_ws2(1e-14, 1.0, 8, 24) == pytest.approx(24.0, rel=1e-6)    # sigma -> 0
    # the deployed r2 is the same formula with nu2 = infinity
    assert dof_ws2(1.0, 1.0, 8, np.inf) == pytest.approx(4.0 * 8, rel=1e-9)
    assert np.all(dof_ws2(np.array([1.0, 2.0]), np.array([1.0, 0.5]), 8, 24) > 0)


def test_t_min_equals_the_deployed_rule_exactly_when_N_at_least_n_minus_1():
    """C2 changes nothing outside the under-coverage region -- so no existing result moves."""
    for n, N in [(10, 9), (10, 25), (20, 19), (50, 49), (50, 1000)]:
        V1, V2 = np.array([1.0, 0.3]), np.array([0.2, 0.9])
        q = candidate_quantiles(V1, V2, n, N, 0.05)
        assert np.allclose(q["C2_t_min"], q["B9_t_n2"], rtol=0, atol=0)
    # and it strictly widens below the boundary
    q = candidate_quantiles(np.array([1.0]), np.array([0.2]), 50, 25, 0.05)
    assert q["C2_t_min"][0] > q["B9_t_n2"][0]


def test_exact_two_df_quantile_brackets_and_endpoints():
    nu1, nu2, alpha = 8.0, 24.0, 0.05
    assert exact_two_df_nu(0.0, nu1, nu2, alpha) == pytest.approx(nu1, rel=1e-3)
    assert exact_two_df_nu(1.0, nu1, nu2, alpha) == pytest.approx(nu2, rel=1e-2)
    mid = exact_two_df_nu(np.array([0.25, 0.5, 0.75]), nu1, nu2, alpha)
    assert np.all(mid >= min(nu1, nu2) - 1e-6) and np.all(np.diff(mid) > 0)


def test_all_candidates_are_produced_and_positive():
    q = candidate_quantiles(np.array([1.0, 0.5]), np.array([0.4, 2.0]), 10, 25, 0.05)
    assert set(q) == set(CANDIDATES)
    for k, v in q.items():
        assert np.all(np.isfinite(v)) and np.all(v > 0), k


# --------------------------------------------------------------------------- bound + determinism

@pytest.mark.parametrize("n,rho,N", [(10, 0.0, 25.0), (20, 0.5, 100.0), (50, 0.9, 1000.0)])
def test_measured_delta_respects_theorem_G1(n, rho, N):
    res = simulate_cell(n, rho, N, R=120_000, seed=11)
    assert abs(res["delta_cov"]) <= coverage_bound(n, rho, N).total


def test_simulate_cell_is_deterministic_in_its_seed():
    a = simulate_cell(10, 0.5, 25, R=40_000, seed=7)
    b = simulate_cell(10, 0.5, 25, R=40_000, seed=7)
    c = simulate_cell(10, 0.5, 25, R=40_000, seed=8)
    for k in ("coverage_actual", "coverage_ideal", "width_C2_t_min"):
        assert a[k] == b[k], k
    assert a["coverage_actual"] != c["coverage_actual"]


def test_chunking_does_not_change_the_answer():
    a = simulate_cell(10, 0.5, 25, R=40_000, seed=5, chunk=40_000)
    b = simulate_cell(10, 0.5, 25, R=40_000, seed=5, chunk=10_000)
    assert a["coverage_actual"] == pytest.approx(b["coverage_actual"], abs=5e-4)


@pytest.mark.slow
def test_t_min_never_under_covers_on_the_saved_grid():
    """Regression test on the persisted Gate-2 results: C2 has no significant under-coverage."""
    import csv
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "results" / "gate2_quantiles.csv"
    rows = [r for r in csv.DictReader(open(p)) if r["candidate"] == "C2_t_min"]
    assert len(rows) > 400
    worst = min(float(r["coverage"]) for r in rows)
    se = float(rows[0]["mc_se"])
    assert worst >= 0.95 - 3 * se, f"C2 under-covered at {worst}"
