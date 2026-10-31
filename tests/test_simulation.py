"""The DGP produces what `08` §T1.1 says it produces."""
import numpy as np
import pytest

from ppi_finite import dgp_gaussian


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_moments_match_targets(rho):
    rng = np.random.default_rng(int(100 * rho))
    data = dgp_gaussian(rng, R=20_000, n=50, rho=rho, N=1e6)
    Y, f = data.Y.ravel(), data.f.ravel()
    assert abs(Y.mean()) < 0.01
    assert abs(f.mean()) < 0.01
    assert abs(Y.var() - 1.0) < 0.01
    assert abs(f.var() - 1.0) < 0.01
    assert abs(np.corrcoef(Y, f)[0, 1] - rho) < 0.01
    assert data.theta_true == 0.0


def test_fbar_N_has_variance_one_over_N():
    rng = np.random.default_rng(3)
    N = 1000.0
    data = dgp_gaussian(rng, R=200_000, n=10, rho=0.5, N=N)
    assert abs(np.var(data.fbar_N) * N - 1.0) < 0.02
    assert data.N == N and data.sigma2_f == 1.0


def test_non_default_scales():
    rng = np.random.default_rng(4)
    data = dgp_gaussian(rng, R=20_000, n=20, rho=0.5, N=1e6, var_Y=4.0, var_f=9.0,
                        mean_Y=2.0, mean_f=-1.0)
    assert abs(data.Y.mean() - 2.0) < 0.02 and abs(data.f.mean() + 1.0) < 0.02
    assert abs(data.Y.var() - 4.0) < 0.05 and abs(data.f.var() - 9.0) < 0.1
    assert data.theta_true == 2.0


def test_unknown_unlabeled_mode_raises():
    with pytest.raises(ValueError, match="unlabeled_mode"):
        dgp_gaussian(np.random.default_rng(0), R=10, n=5, rho=0.0, N=100,
                     unlabeled_mode="nonsense")


# ---------------------------------------------------------------- explicit mode (E4)

def test_explicit_mode_estimates_sigma2_f_and_does_not_treat_it_as_known():
    """`08` §T1.1: mean_only treats sigma_f^2 as known; explicit must estimate it from the pool."""
    d = dgp_gaussian(np.random.default_rng(1), R=5000, n=10, rho=0.5, N=200,
                     unlabeled_mode="explicit")
    s2 = np.asarray(d.sigma2_f)
    assert s2.shape == (5000,)                    # per-replication, not a constant
    assert abs(s2.mean() - 1.0) < 0.01            # unbiased for Var(f) = 1
    assert abs(s2.std() - np.sqrt(2 / 199)) < 0.005   # sd of a chi2_{N-1}/(N-1) variance
    assert d.mu_f == 0.0                          # carried for the v_full diagnostic only


def test_explicit_fbar_N_has_the_right_sampling_variance():
    d = dgp_gaussian(np.random.default_rng(2), R=200_000, n=10, rho=0.0, N=500,
                     unlabeled_mode="explicit")
    assert abs(np.var(d.fbar_N) * 500 - 1.0) < 0.02


def test_explicit_chunked_path_equals_the_whole_pool_path():
    """Chunking the (R, N) pool is stream-invariant, so keep_pool=True must give the same
    f_bar_N and sigma_hat_f^2 as the chunked path."""
    import ppi_finite.simulation as sim
    kw = dict(R=40, n=8, rho=0.5, N=50, unlabeled_mode="explicit")
    whole = dgp_gaussian(np.random.default_rng(3), keep_pool=True, **kw)
    old = sim.MAX_POOL_ELEMENTS
    try:
        sim.MAX_POOL_ELEMENTS = 100               # forces several chunks
        chunked = dgp_gaussian(np.random.default_rng(3), **kw)
    finally:
        sim.MAX_POOL_ELEMENTS = old
    assert np.allclose(whole.fbar_N, chunked.fbar_N, atol=0)
    assert np.allclose(whole.sigma2_f, chunked.sigma2_f, atol=0)
    assert whole.f_unlabeled.shape == (40, 50)    # the pool really is materialised
    assert np.allclose(whole.f_unlabeled.mean(-1), whole.fbar_N)


def test_explicit_converges_to_mean_only_as_N_grows():
    """`15` §4.4: the two modes are not interchangeable at small N but must agree at large N."""
    prev = None
    for N in (200, 5_000, 200_000):
        a = dgp_gaussian(np.random.default_rng(4), R=4000, n=10, rho=0.9, N=N,
                         unlabeled_mode="explicit")
        b = dgp_gaussian(np.random.default_rng(4), R=4000, n=10, rho=0.9, N=N,
                         unlabeled_mode="mean_only")
        # Y and f are drawn first and identically in both modes: only the pool differs.
        assert np.array_equal(a.Y, b.Y) and np.array_equal(a.f, b.f)
        gap = abs(float(np.mean(np.asarray(a.sigma2_f))) - 1.0)
        assert np.std(a.fbar_N) == pytest.approx(np.std(b.fbar_N), rel=0.1)
        if prev is not None:
            assert gap < prev + 1e-6              # sigma2_f estimate tightens onto the truth
        prev = max(gap, 1e-9)
