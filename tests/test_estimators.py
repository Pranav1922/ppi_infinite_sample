"""Unit tests for the L1 estimators, against independent implementations (`11` §Testing)."""
import numpy as np
import pytest
from scipy import stats

from ppi_finite import (labelled_summaries, lambda_plugin, theta_classical, theta_ppi,
                        theta_ppi_plus)

TOL = 1e-12


def test_b_and_se_b_match_linregress():
    rng = np.random.default_rng(0)
    for n in (5, 10, 50):
        f = rng.standard_normal(n)
        Y = 0.7 * f + rng.standard_normal(n)
        s = labelled_summaries(Y, f)
        lr = stats.linregress(f, Y)
        assert abs(s["b"] - lr.slope) < TOL
        assert abs(s["se_b"] - lr.stderr) < TOL


def test_s2_matches_independent_rss():
    rng = np.random.default_rng(1)
    f, Y = rng.standard_normal(12), rng.standard_normal(12)
    s = labelled_summaries(Y, f)
    lr = stats.linregress(f, Y)
    rss = float(np.sum((Y - (lr.intercept + lr.slope * f)) ** 2))
    assert abs(s["s2"] - rss / (12 - 2)) < TOL


def test_hand_computed_five_point_example():
    # f = 1..5, Y = 2,3,5,4,6.  fbar = 3, Ybar = 4.
    # S_ff = 4+1+0+1+4 = 10;  S_fY = (-2)(-2)+(-1)(-1)+0+1*0+2*2 = 9;  b = 0.9.
    f = np.array([1.0, 2, 3, 4, 5])
    Y = np.array([2.0, 3, 5, 4, 6])
    s = labelled_summaries(Y, f)
    assert s["Ybar"] == 4.0 and s["fbar_n"] == 3.0
    assert s["S_ff"] == 10.0 and s["S_fY"] == 9.0
    assert abs(s["b"] - 0.9) < TOL
    resid = (Y - 4.0) - 0.9 * (f - 3.0)
    assert abs(s["s2"] - float(resid @ resid) / 3) < TOL
    assert abs(s["se_b"] - np.sqrt(s["s2"] / 10.0)) < TOL


def test_S_ff_is_a_sum_of_squares_not_a_variance():
    f = np.array([1.0, 2, 3, 4, 5])
    s = labelled_summaries(f, f)
    assert s["S_ff"] == pytest.approx(10.0)          # not 2.5 (ddof=1) and not 2.0 (ddof=0)


def test_batched_matches_looped():
    rng = np.random.default_rng(2)
    Y, f = rng.standard_normal((7, 9)), rng.standard_normal((7, 9))
    batched = labelled_summaries(Y, f)
    for r in range(7):
        single = labelled_summaries(Y[r], f[r])
        for k in ("Ybar", "fbar_n", "S_ff", "S_fY", "b", "s2", "se_b"):
            assert abs(batched[k][r] - single[k]) < TOL


def test_lambda_reductions_are_exact():
    rng = np.random.default_rng(3)
    Y, f = rng.standard_normal((4, 11)), rng.standard_normal((4, 11))
    fbar_N = rng.standard_normal(4) / 1000
    assert np.array_equal(theta_ppi_plus(Y, f, fbar_N, 0.0), theta_classical(Y))
    assert np.array_equal(theta_ppi_plus(Y, f, fbar_N, 1.0), theta_ppi(Y, f, fbar_N))


def test_lambda_plugin_shrinks_by_one_plus_r():
    b = np.array([2.0])
    assert lambda_plugin(b, 10, 100)[0] == pytest.approx(2.0 / 1.1)
    assert lambda_plugin(b, 10, np.inf)[0] == 2.0          # r = 0 when N is infinite


def test_degenerate_S_ff_gives_zero_slope_not_nan():
    f = np.ones(6)
    Y = np.arange(6.0)
    s = labelled_summaries(Y, f)
    assert s["S_ff"] == 0.0 and s["b"] == 0.0 and s["se_b"] == 0.0
    assert np.isfinite(s["s2"]) and bool(s["degenerate"])


def test_equation_1_decomposition_is_an_exact_identity():
    """Manuscript Eq. (1): θ̂_LC − θ = ε̄ + (b − β)(μ_f − f̄_n) + b·δ, δ = f̄_N − μ_f, for any n, N.

    The symbolic derivation is in §4.1 of the manuscript; this is its numerical companion, at
    arbitrary (non-Gaussian) data, so it checks the algebra rather than the model.
    """
    rng = np.random.default_rng(7)
    alpha, beta, mu_f = 0.3, -1.7, 2.5
    for n, N in ((4, 3), (10, 50), (50, 1000)):
        f = mu_f + rng.exponential(2.0, n)                      # any design
        eps = rng.standard_t(3, n)                               # any residual law
        Y = alpha + beta * f + eps
        theta = alpha + beta * mu_f
        fbar_N = mu_f + rng.standard_normal() / np.sqrt(N)
        b = labelled_summaries(Y, f)["b"]
        theta_lc = Y.mean() + b * (fbar_N - f.mean())
        rhs = eps.mean() + (b - beta) * (mu_f - f.mean()) + b * (fbar_N - mu_f)
        assert abs((theta_lc - theta) - rhs) < TOL
