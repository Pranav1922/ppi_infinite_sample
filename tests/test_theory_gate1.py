"""Gate 1 — the rigorous finite-N coverage-error bound.

Fast tests cover the exact identities and the bound's structural properties; the Monte-Carlo
validations that need millions of replications are marked `slow`.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ppi_finite.theory_gate1 import (
    PHI0, CoverageBound, coverage_bound, denominator_moment, design_moments,
    inv_sqrt_chi2_moment, psi, reduction_1, reduction_2,
)


# --------------------------------------------------------------------------- primitives

def draw(n, rho, N, R, rng):
    """The primitives of theory_finite_N.md §II.1, with sigma_f = 1, mu_f = 0, Var(Y) = 1."""
    nu, sig = n - 2, np.sqrt(1 - rho**2)
    beta = rho
    f = rng.normal(0.0, 1.0, size=(R, n))
    S_ff = ((f - f.mean(1)[:, None]) ** 2).sum(1)
    u = -f.mean(1) / np.sqrt(S_ff)
    kap = 1.0 / (np.sqrt(N) * np.sqrt(S_ff))
    Z0, Z1, W = rng.normal(size=(3, R))
    Q = rng.chisquare(nu, size=R)
    sh2 = rng.chisquare(N - 1, size=R) / (N - 1)
    return dict(nu=nu, sig=sig, beta=beta, c=beta / np.sqrt(N), lam=beta / (sig * np.sqrt(N)),
                S_ff=S_ff, u=u, kap=kap, Z0=Z0, Z1=Z1, W=W, Q=Q, sh2=sh2)


def test_psi_is_a_probability_with_the_stated_derivative_bounds():
    x = np.linspace(0.01, 6, 200)[:, None]
    y = np.linspace(-6, 6, 201)[None, :]
    p = psi(x, y)
    assert np.all((p >= 0) & (p <= 1))
    # psi(x, y) = P(|Z + y| <= x): symmetric in y, increasing in x
    assert np.allclose(psi(x, y), psi(x, -y))
    assert np.all(np.diff(psi(x, 0.3), axis=0) > 0)
    # sup|psi_x| = 2 phi(0), sup|psi_y| = phi(0), the only smoothness the proof uses
    h = 1e-6
    dx = (psi(x + h, y) - psi(x - h, y)) / (2 * h)
    dy = (psi(x, y + h) - psi(x, y - h)) / (2 * h)
    assert np.max(np.abs(dx)) <= 2 * PHI0 + 1e-9
    assert np.max(np.abs(dy)) <= PHI0 + 1e-9


def test_inv_sqrt_chi2_moment_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for df in (2, 3, 8, 48):
        mc = np.mean(1.0 / np.sqrt(rng.chisquare(df, size=400_000)))
        assert abs(inv_sqrt_chi2_moment(df) - mc) < 4e-3 * mc
    assert inv_sqrt_chi2_moment(1) == np.inf          # diverges at df <= 1


def test_quadrature_moments_match_monte_carlo():
    rng = np.random.default_rng(1)
    n, N, nu = 10, 100.0, 8
    for lam in (0.0, 0.3, 2.0):
        q = rng.chisquare(nu, size=400_000)
        a = np.sqrt(q / (nu * n))
        mc = np.mean(1.0 / (a + np.sqrt(a**2 + lam**2)))
        assert abs(denominator_moment(nu, n, lam) - mc) < 5e-3 * mc

    A = rng.normal(0.0, 1.0 / np.sqrt(n), size=400_000)
    S = rng.chisquare(n - 1, size=400_000)
    m_abs, m_sq = design_moments(n, N)
    assert abs(m_abs - np.mean(np.abs(A) / (S / n + A**2)) / np.sqrt(N)) < 5e-3 * m_abs
    assert abs(m_sq - np.mean(1.0 / (S / n + A**2)) / N) < 5e-3 * m_sq


# --------------------------------------------------------------------------- exact reductions

@pytest.mark.slow
@pytest.mark.parametrize("n,rho,N", [(10, 0.0, 25.0), (5, 0.9, 25.0), (10, 0.9, 100.0)])
def test_reductions_agree_with_brute_force_coverage(n, rho, N):
    """Both conditional reductions are exact, so they must match direct simulation of |T| <= t."""
    R = 2_000_000
    rng = np.random.default_rng(4)
    d = draw(n, rho, N, R, rng)
    t = stats.t.ppf(0.975, d["nu"])
    a = d["u"] + d["kap"] * d["W"]
    g_act, g_id = 1 / n + a**2, 1 / n + d["u"] ** 2
    s2 = d["sig"] ** 2 * d["Q"] / d["nu"]
    b = d["beta"] + d["sig"] * d["Z1"] / np.sqrt(d["S_ff"])
    X_act = d["sig"] * (d["Z0"] / np.sqrt(n) + a * d["Z1"]) + d["c"] * d["W"]
    X_id = d["sig"] * (d["Z0"] / np.sqrt(n) + d["u"] * d["Z1"]) + d["c"] * d["W"]
    V_act = s2 * g_act + b**2 * d["sh2"] / N
    V_id = s2 * g_id + d["beta"] ** 2 / N

    m_act = d["sig"] * a * d["Z1"] + d["c"] * d["W"]
    brute_act = np.mean(np.abs(X_act) <= t * np.sqrt(V_act))
    exact_act = reduction_1(t, n, d["sig"], V_act, m_act).mean()
    brute_id = np.mean(np.abs(X_id) <= t * np.sqrt(V_id))
    exact_id = reduction_2(t, d["nu"], d["Q"], d["lam"], g_id, d["W"]).mean()

    tol = 5 * np.sqrt(0.25 / R)
    assert abs(brute_act - exact_act) < tol
    assert abs(brute_id - exact_id) < tol


@pytest.mark.slow
def test_the_two_reductions_agree_on_the_intermediate_statistic():
    """T_mid has a Z1-free denominator, so reduction_1 and reduction_2 both apply and must agree."""
    n, rho, N, R = 10, 0.5, 100.0, 1_000_000
    d = draw(n, rho, N, R, np.random.default_rng(5))
    t = stats.t.ppf(0.975, d["nu"])
    a = d["u"] + d["kap"] * d["W"]
    g_act = 1 / n + a**2
    V_mid = d["sig"] ** 2 * d["Q"] / d["nu"] * g_act + d["beta"] ** 2 / N
    m_act = d["sig"] * a * d["Z1"] + d["c"] * d["W"]
    r1 = reduction_1(t, n, d["sig"], V_mid, m_act).mean()
    r2 = reduction_2(t, d["nu"], d["Q"], d["lam"], g_act, d["W"]).mean()
    assert abs(r1 - r2) < 5 * np.sqrt(0.25 / R)


def test_at_rho_zero_the_product_and_leverage_idealisations_vanish_identically():
    """Corollary: beta = 0 makes lambda = 0, so (x, y) coincide pointwise -- not just in mean."""
    rng = np.random.default_rng(6)
    n, N, R = 10, 25.0, 20_000
    d = draw(n, 0.0, N, R, rng)
    t = stats.t.ppf(0.975, d["nu"])
    a = d["u"] + d["kap"] * d["W"]
    g_act, g_id = 1 / n + a**2, 1 / n + d["u"] ** 2
    r_act = reduction_2(t, d["nu"], d["Q"], d["lam"], g_act, d["W"])
    r_id = reduction_2(t, d["nu"], d["Q"], d["lam"], g_id, d["W"])
    assert d["lam"] == 0.0
    assert np.allclose(r_act, r_id, atol=0, rtol=0)
    assert coverage_bound(n, 0.0, N).bound_PD == 0.0


# --------------------------------------------------------------------------- the bound

def test_bound_requires_n_at_least_four_and_rho_below_one():
    with pytest.raises(ValueError):
        coverage_bound(3, 0.5, 100.0)
    with pytest.raises(ValueError):
        coverage_bound(10, 1.0, 100.0)
    assert np.isfinite(coverage_bound(4, 0.5, 100.0).total)


def test_bound_is_order_one_over_N():
    """C = N x bound must converge, and the bound itself must fall like 1/N."""
    for n, rho in [(10, 0.0), (10, 0.5), (20, 0.9)]:
        Cs = [coverage_bound(n, rho, N).C for N in (1e4, 1e5, 1e6, 1e7)]
        assert all(np.isfinite(Cs))
        # successive values converge: the gap to the limit never grows. At rho = 0 the constant
        # is exactly free of N (lambda = 0 and the sigma_f^2 term carries a beta^2), so the gaps
        # are identically zero -- hence <= rather than <.
        gaps = [abs(Cs[i + 1] - Cs[i]) for i in range(len(Cs) - 1)]
        assert gaps[-1] <= gaps[0] + 1e-12
        assert abs(Cs[-1] - Cs[-2]) < 0.02 * Cs[-1]
        ratio = coverage_bound(n, rho, 1e5).total / coverage_bound(n, rho, 1e6).total
        assert 9.0 < ratio < 11.0            # a decade in N buys a decade in the bound


def test_bound_is_capped_at_one_and_is_a_sum_of_its_parts():
    b = coverage_bound(10, 0.5, 200.0)
    assert isinstance(b, CoverageBound)
    assert b.total == pytest.approx(b.bound_B + b.bound_PD)
    assert coverage_bound(50, 0.99, 25.0).total == 1.0        # vacuous, but still valid
    assert 0.0 <= coverage_bound(5, 0.99, 25.0).total <= 1.0


def test_sigma_f_known_bound_is_no_larger():
    """Dropping the sigma_f^2 estimation term can only shrink the bound."""
    for n, rho, N in [(10, 0.5, 100.0), (50, 0.9, 50.0), (20, 0.0, 25.0)]:
        assert (coverage_bound(n, rho, N, sigma_f_known=True).total
                <= coverage_bound(n, rho, N).total + 1e-15)


@pytest.mark.slow
@pytest.mark.parametrize("n,rho,N", [(5, 0.0, 25.0), (10, 0.0, 100.0), (10, 0.5, 100.0),
                                     (20, 0.5, 1000.0), (50, 0.0, 25.0), (10, 0.9, 1000.0)])
def test_measured_delta_respects_the_bound(n, rho, N):
    """|coverage_act - coverage_id| <= bound, measured on common random numbers."""
    R = 1_000_000
    d = draw(n, rho, N, R, np.random.default_rng(8))
    t = stats.t.ppf(0.975, d["nu"])
    a = d["u"] + d["kap"] * d["W"]
    g_act, g_id = 1 / n + a**2, 1 / n + d["u"] ** 2
    s2 = d["sig"] ** 2 * d["Q"] / d["nu"]
    b = d["beta"] + d["sig"] * d["Z1"] / np.sqrt(d["S_ff"])
    V_act = s2 * g_act + b**2 * d["sh2"] / N
    m_act = d["sig"] * a * d["Z1"] + d["c"] * d["W"]
    cov_act = reduction_1(t, n, d["sig"], V_act, m_act).mean()
    cov_id = reduction_2(t, d["nu"], d["Q"], d["lam"], g_id, d["W"]).mean()
    delta = cov_act - cov_id
    bd = coverage_bound(n, rho, N)
    assert abs(delta) <= bd.total, f"bound violated: |{delta:.6f}| > {bd.total:.6f}"


@pytest.mark.slow
def test_deployed_interval_under_covers_at_large_n_small_N_high_rho():
    """Counterexample to conservativeness of the *deployed* interval (Gate 1, step 9).

    theory_finite_N.md §II.8 recorded that the practical interval never under-covered; that
    ablation held sigma_f^2 known. With sigma_f^2 estimated from the pool -- which is what B9
    does in `explicit` mode -- coverage falls below nominal at n = 50, rho = 0.99, N = 25.
    """
    from ppi_finite import registry
    from ppi_finite.estimators import labelled_summaries
    from ppi_finite.simulation import dgp_gaussian

    n, rho, N, R = 50, 0.99, 25.0, 400_000
    rng = np.random.default_rng(20260824)
    hits = done = 0
    while done < R:
        r = min(100_000, R - done)
        data = dgp_gaussian(rng, R=r, n=n, rho=rho, N=N, unlabeled_mode="explicit")
        v = data.view()
        res = registry.get("B9")(v, 0.05, None, summaries=labelled_summaries(v.Y, v.f))
        hits += int(np.sum((res.ci_low <= data.theta_true) & (data.theta_true <= res.ci_high)))
        done += r
    cov = hits / R
    se = np.sqrt(cov * (1 - cov) / R)
    assert cov < 0.95 - 5 * se, f"expected under-coverage, got {cov:.5f} (s.e. {se:.5f})"
