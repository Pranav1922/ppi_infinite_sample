"""Extension E5 — the robustness DGPs T1.2-T1.4 and the HC3 variance (`08`, `07` F/G/H).

These test that the *generators* produce what `08` specifies and that HC3 reduces correctly.
They do **not** test coverage claims: outside the Gaussian model there are no theorems to test,
only measurements, and those live in `results/` and `13`.
"""
import numpy as np
import pytest
from scipy import stats

from ppi_finite import METHODS, labelled_summaries, var_lc_hc3
from ppi_finite.simulation import (_binary_params, _standardised_skew, dgp_heavy_tail,
                                   dgp_heteroskedastic, dgp_skew)


# ---------------------------------------------------------------- T1.2 heavy tails

@pytest.mark.parametrize("nu", [3.0, 5.0, 10.0, 30.0])
def test_heavy_tail_is_standardised_and_hits_the_target_correlation(nu):
    d = dgp_heavy_tail(np.random.default_rng(int(nu)), R=40_000, n=20, rho=0.5, N=1e6, nu=nu)
    Y, f = d.Y.ravel(), d.f.ravel()
    assert abs(Y.var() - 1.0) < 0.06 and abs(f.var() - 1.0) < 0.03
    assert abs(np.corrcoef(Y, f)[0, 1] - 0.5) < 0.02
    assert d.theta_true == 0.0


def test_heavy_tail_residuals_really_are_heavy():
    """Kurtosis must rise as nu falls; that is the whole point of the experiment."""
    k = []
    for nu in (30.0, 10.0, 5.0):
        d = dgp_heavy_tail(np.random.default_rng(1), R=60_000, n=10, rho=0.0, N=1e6, nu=nu)
        k.append(float(stats.kurtosis(d.Y.ravel())))
    assert k[0] < k[1] < k[2]


def test_heavy_f_variant_makes_the_predictor_heavy_not_just_the_residual():
    a = dgp_heavy_tail(np.random.default_rng(2), R=40_000, n=10, rho=0.0, N=1e6, nu=5.0)
    b = dgp_heavy_tail(np.random.default_rng(2), R=40_000, n=10, rho=0.0, N=1e6, nu=5.0,
                       heavy_f=True)
    assert stats.kurtosis(b.f.ravel()) > stats.kurtosis(a.f.ravel()) + 1.0


def test_nu_below_two_raises():
    with pytest.raises(ValueError, match="finite variance"):
        dgp_heavy_tail(np.random.default_rng(0), R=10, n=5, rho=0.0, N=1e6, nu=2.0)


# ---------------------------------------------------------------- T1.3 skew and binary

@pytest.mark.parametrize("fam,expect_skew", [("exponential", 2.0), ("gamma1", 2.0),
                                             ("gamma2", 1.41), ("gamma5", 0.89),
                                             ("lognormal_mild", 1.75), ("lognormal_severe", 6.18)])
def test_skew_families_are_standardised_with_the_right_skewness(fam, expect_skew):
    x = _standardised_skew(np.random.default_rng(3), (4_000_000,), fam)
    assert abs(x.mean()) < 0.01 and abs(x.var() - 1.0) < 0.02
    assert stats.skew(x) == pytest.approx(expect_skew, rel=0.08)


def test_unknown_skew_family_raises():
    with pytest.raises(ValueError, match="unknown skew family"):
        _standardised_skew(np.random.default_rng(0), (10,), "gaussian")


def test_binary_dgp_hits_prevalence_and_target_correlation():
    d = dgp_skew(np.random.default_rng(4), R=60_000, n=20, rho=0.5, N=1e6, skew_family="binary")
    assert set(np.unique(d.Y)) <= {0.0, 1.0}
    assert d.theta_true == 0.5                       # theta = prevalence, NOT 0
    assert abs(d.Y.mean() - 0.5) < 0.005
    assert abs(np.corrcoef(d.Y.ravel(), d.f.ravel())[0, 1] - 0.5) < 0.02


def test_binary_correlation_is_capped_at_the_achievable_maximum():
    """`07` Experiment G: "or the closest achievable correlation for the discrete case". With
    Gaussian `f` and prevalence 0.5 the ceiling is `phi(0)/0.5 = 0.7979`, attained by the
    deterministic threshold rule, so the rho = 0.9 cell cannot exist as specified."""
    _, _, achieved = _binary_params(0.5, 0.9)
    assert achieved == pytest.approx(2 * stats.norm.pdf(0.0), abs=0.005)
    assert achieved < 0.8
    for target in (0.0, 0.3, 0.5, 0.7):
        assert _binary_params(0.5, target)[2] == pytest.approx(target, abs=1e-3)


# ---------------------------------------------------------------- T1.4 heteroskedasticity

@pytest.mark.parametrize("form", ["exp", "step"])
def test_heteroskedastic_gamma_zero_recovers_the_gaussian_dgp(form):
    from ppi_finite import dgp_gaussian
    kw = dict(R=500, n=10, rho=0.5, N=1e6)
    a = dgp_heteroskedastic(np.random.default_rng(5), gamma=0.0, form=form, **kw)
    b = dgp_gaussian(np.random.default_rng(5), **kw)
    assert np.allclose(a.Y, b.Y) and np.allclose(a.f, b.f)


@pytest.mark.parametrize("form,gamma", [("exp", 0.5), ("exp", 1.0), ("step", 0.5), ("step", 1.0)])
def test_heteroskedastic_keeps_unit_variance_and_creates_real_heteroskedasticity(form, gamma):
    d = dgp_heteroskedastic(np.random.default_rng(6), R=200_000, n=10, rho=0.0, N=1e6,
                            gamma=gamma, form=form)
    Y, f = d.Y.ravel(), d.f.ravel()
    assert abs(Y.var() - 1.0) < 0.15               # exp(gamma f) has a heavy right tail at gamma=1
    lo, hi = Y[f < -0.5], Y[f > 0.5]
    assert hi.var() > lo.var() * 1.2               # residual scale really does grow with f


def test_bad_hetero_form_raises():
    with pytest.raises(ValueError, match="form must be"):
        dgp_heteroskedastic(np.random.default_rng(0), R=10, n=5, rho=0.0, N=1e6, form="linear")


# ---------------------------------------------------------------- HC3

def test_hc3_weights_sum_to_the_leverage_bracket_exactly():
    """`Σ_i w_i² = 1/n + Δ²/S_ff` exactly, so HC3 is a strict generalisation of `04` §4.8.2's
    bracket rather than a different construction."""
    rng = np.random.default_rng(7)
    f = rng.standard_normal((50, 12))
    fbar_N = rng.standard_normal(50) / 100
    s = labelled_summaries(rng.standard_normal((50, 12)), f)
    fc = f - s["fbar_n"][:, None]
    w = 1 / 12 + fc * (fbar_N - s["fbar_n"])[:, None] / s["S_ff"][:, None]
    lhs = (w ** 2).sum(-1)
    rhs = 1 / 12 + (fbar_N - s["fbar_n"]) ** 2 / s["S_ff"]
    assert np.allclose(lhs, rhs, atol=1e-14)


def test_hc3_is_close_to_b9_under_homoskedasticity_and_wider_under_heteroskedasticity():
    from ppi_finite import dgp_gaussian
    homo = dgp_gaussian(np.random.default_rng(8), R=20_000, n=30, rho=0.5, N=1e6)
    r_b9 = METHODS["B9"](homo.view(), 0.05)
    r_hc = METHODS["B9-HC"](homo.view(), 0.05)
    assert np.mean(r_hc.se) / np.mean(r_b9.se) == pytest.approx(1.0, abs=0.05)

    het = dgp_heteroskedastic(np.random.default_rng(8), R=20_000, n=30, rho=0.5, N=1e6, gamma=1.0)
    assert (np.mean(METHODS["B9-HC"](het.view(), 0.05).se)
            > np.mean(METHODS["B9"](het.view(), 0.05).se))


def test_hc3_claims_no_exactness():
    """`07` Experiment H: "Do not claim otherwise." The flag is the machine-readable form."""
    from ppi_finite import dgp_gaussian
    d = dgp_gaussian(np.random.default_rng(9), R=100, n=10, rho=0.5, N=1e6)
    assert "asymptotic_only" in METHODS["B9-HC"](d.view(), 0.05).extras["flags"]


def test_hc3_falls_back_when_S_ff_is_zero():
    from ppi_finite import make_data
    Y = np.arange(6.0)[None, :]
    f = np.ones((1, 6))
    data = make_data(Y, f, fbar_N=np.zeros(1), N=1e6, sigma2_f=1.0, mu_f=0.0, theta_true=0.0)
    s = labelled_summaries(Y, f)
    v = var_lc_hc3(s, Y, f, data.fbar_N, 1.0, 1e6)
    assert np.all(np.isfinite(v.value)) and np.all(v.value > 0)
