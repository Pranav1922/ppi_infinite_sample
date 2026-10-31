"""V11 — the algebraic identity of `04` §4.7, to 1e-10 (`09` V11).

`θ̂_b = Ȳ + b(μ_f − f̄_n)` is exactly the OLS fitted value at `f = μ_f`.
"""
import numpy as np
import pytest
from scipy import stats

from ppi_finite import labelled_summaries, lc_ppi_interval, make_data, method_b1, method_b9

TOL = 1e-10


@pytest.mark.parametrize("n", [4, 10, 50])
@pytest.mark.parametrize("mu_f", [0.0, 0.37, -2.5])
def test_theta_hat_is_the_ols_fitted_value(n, mu_f):
    rng = np.random.default_rng(1000 + n)
    f = rng.standard_normal(n)
    Y = 0.6 * f + rng.standard_normal(n)
    s = labelled_summaries(Y, f)
    theta_hat = s["Ybar"] + s["b"] * (mu_f - s["fbar_n"])
    lr = stats.linregress(f, Y)
    assert abs(theta_hat - (lr.intercept + lr.slope * mu_f)) < TOL
    # and via numpy.polyfit, a second independent implementation
    slope, intercept = np.polyfit(f, Y, 1)
    assert abs(theta_hat - (intercept + slope * mu_f)) < TOL


def test_method_b9_point_estimate_is_the_fitted_value():
    rng = np.random.default_rng(5)
    f = rng.standard_normal((13, 8))
    Y = 0.5 * f + rng.standard_normal((13, 8))
    fbar_N = rng.standard_normal(13) / 1000
    data = make_data(Y, f, fbar_N=fbar_N, N=1e6, sigma2_f=1.0, theta_true=0.0)
    res = method_b9(data.view(), 0.05)
    for r in range(13):
        lr = stats.linregress(f[r], Y[r])
        assert abs(res.theta_hat[r] - (lr.intercept + lr.slope * fbar_N[r])) < TOL


def test_mu_f_branch_agrees_with_a_large_explicit_pool():
    """`11` §lc_ppi.py: mu_f supplied vs f_unlabeled at N = 1e6 agree to 4 decimals."""
    rng = np.random.default_rng(6)
    f = rng.standard_normal(20)
    Y = 0.5 * f + rng.standard_normal(20)
    pool = rng.standard_normal(1_000_000)
    known = lc_ppi_interval(Y, f, mu_f=float(pool.mean()), sigma2_f=float(pool.var(ddof=1)))
    estimated = lc_ppi_interval(Y, f, f_unlabeled=pool)
    assert np.isclose(known.theta_hat, estimated.theta_hat, atol=1e-4)
    assert np.isclose(known.ci_low, estimated.ci_low, atol=1e-4)
    assert np.isclose(known.ci_high, estimated.ci_high, atol=1e-4)
    assert np.isinf(known.extras["N"]) and estimated.extras["N"] == 1_000_000


def test_degenerate_S_ff_falls_back_to_classical():
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    f = np.ones(5)
    data = make_data(Y, f, mu_f=1.0, sigma2_f=1.0, theta_true=3.0)
    b9 = method_b9(data.view(), 0.05)
    b1 = method_b1(data.view(), 0.05)
    assert "degenerate_S_ff" in b9.extras["flags"]
    assert np.allclose(b9.theta_hat, b1.theta_hat)
    assert np.allclose([b9.ci_low, b9.ci_high], [b1.ci_low, b1.ci_high])


def test_n_too_small_raises_and_n_three_warns():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        make_data(rng.standard_normal(2), rng.standard_normal(2), mu_f=0.0)
    with pytest.warns(UserWarning, match="one degree of freedom"):
        make_data(rng.standard_normal(3), rng.standard_normal(3), mu_f=0.0)
