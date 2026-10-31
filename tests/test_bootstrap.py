"""B7/B8 (extension E3), validated against `scipy.stats.bootstrap` (`11` §bootstrap.py).

`11` requires `ci_bca` to be validated against an independent implementation *before it is
trusted*, because a wrong BCa would silently decide kill criterion K3.

The validation is **exact, not statistical**: `scipy.stats.bootstrap(paired=True)` resamples via
`rng.integers(0, n, (B, n))`, which is what `bootstrap_indices` does, so seeding both with the
same generator makes the two implementations compute from the identical resamples and any
disagreement is a formula difference rather than Monte-Carlo noise.
"""
import time

import numpy as np
import pytest
from scipy import stats

from ppi_finite import make_data
from ppi_finite.bootstrap import (MAX_GATHER_ELEMENTS, bootstrap_indices, bootstrap_ppi,
                                  bootstrap_replicates, ci_bca, ci_percentile, jackknife_theta)

SEED = 4242
B = 999
ALPHA = 0.05


def _data(n=12, rho=0.6, seed=7, N=1e6, fbar_N=0.0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    Y = rho * f + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    return Y, f, N, fbar_N


def _ppi_plus_statistic(N, fbar_N):
    """The statistic B7/B8 bootstrap: the PPI++ point estimate with the plug-in lambda."""
    def stat(Y, f, axis=-1):
        m = Y.shape[axis]
        Ybar, fbar = Y.mean(axis), f.mean(axis)
        S_ff = ((f - np.expand_dims(fbar, axis)) ** 2).sum(axis)
        S_fY = ((f - np.expand_dims(fbar, axis)) * (Y - np.expand_dims(Ybar, axis))).sum(axis)
        b = np.where(S_ff > 0, S_fY / np.where(S_ff > 0, S_ff, 1.0), 0.0)
        return Ybar - b / (1 + m / N) * (fbar - fbar_N)
    return stat


# ------------------------------------------------------------------ exact SciPy agreement

@pytest.mark.parametrize("method,scipy_method", [("percentile", "percentile"), ("bca", "BCa")])
@pytest.mark.parametrize("n,rho", [(8, 0.0), (12, 0.6), (25, 0.9)])
def test_matches_scipy_bootstrap_exactly_on_the_same_resamples(method, scipy_method, n, rho):
    Y, f, N, fbar_N = _data(n=n, rho=rho, seed=n)
    stat = _ppi_plus_statistic(N, fbar_N)

    ref = stats.bootstrap((Y, f), stat, paired=True, method=scipy_method, n_resamples=B,
                          batch=B, confidence_level=1 - ALPHA,
                          rng=np.random.default_rng(SEED))
    data = make_data(Y, f, fbar_N=fbar_N, N=N, sigma2_f=1.0).view()
    mine = bootstrap_ppi(data, method=method, B=B, rng=np.random.default_rng(SEED),
                         alpha_level=ALPHA)

    assert mine.ci_low == pytest.approx(ref.confidence_interval.low, abs=1e-12)
    assert mine.ci_high == pytest.approx(ref.confidence_interval.high, abs=1e-12)
    # the bootstrap SD is reported for diagnostics only; it differs from SciPy's at the
    # level of floating-point summation order, not formula.
    assert mine.se == pytest.approx(ref.standard_error, rel=1e-9)


def test_matches_scipy_for_the_plain_mean():
    """`07` Experiment B failure condition: bootstrapping the plain mean must reproduce the
    classical interval closely.  Here also checked against SciPy exactly."""
    Y, f, N, fbar_N = _data(n=15, rho=0.0, seed=3)
    f = np.zeros_like(f)                       # S_ff = 0 => b = 0 => the statistic is mean(Y)
    stat = _ppi_plus_statistic(N, fbar_N)
    ref = stats.bootstrap((Y, f), stat, paired=True, method="BCa", n_resamples=B, batch=B,
                          confidence_level=1 - ALPHA, rng=np.random.default_rng(SEED))
    data = make_data(Y, f, fbar_N=fbar_N, N=N, sigma2_f=1.0).view()
    mine = bootstrap_ppi(data, method="bca", B=B, rng=np.random.default_rng(SEED),
                         alpha_level=ALPHA)
    assert mine.theta_hat == pytest.approx(Y.mean())
    assert mine.ci_low == pytest.approx(ref.confidence_interval.low, abs=1e-12)
    # and it lands near the classical t interval
    classical = stats.t.interval(0.95, 14, loc=Y.mean(), scale=Y.std(ddof=1) / np.sqrt(15))
    assert mine.ci_low == pytest.approx(classical[0], abs=0.25)
    assert mine.ci_high == pytest.approx(classical[1], abs=0.25)


def test_bca_components_match_scipys_bca_interval():
    """Compare `z0` and `a_hat` themselves, not just the endpoints, against SciPy's private
    `_bca_interval` fed the identical bootstrap distribution."""
    from scipy.stats._resampling import _bca_interval
    import scipy._lib.array_api_compat.numpy as xp_np

    Y, f, N, fbar_N = _data(n=10, rho=0.5, seed=11)
    stat = _ppi_plus_statistic(N, fbar_N)
    theta_star = bootstrap_replicates(Y, f, fbar_N, N, B, np.random.default_rng(SEED))

    def paired_stat(i, axis=-1):
        return stat(Y[i], f[i], axis=axis)

    a1_ref, a2_ref, a_ref = _bca_interval((np.arange(10),), paired_stat, axis=-1,
                                          alpha=ALPHA / 2, theta_hat_b=theta_star, batch=None,
                                          xp=xp_np)
    theta_hat = stat(Y, f)
    theta_jack = jackknife_theta(Y, f, fbar_N, N)
    lo, hi, z0, a_hat = ci_bca(theta_star, theta_hat, theta_jack, ALPHA)

    assert a_hat == pytest.approx(float(a_ref), abs=1e-12)
    assert lo == pytest.approx(float(np.quantile(theta_star, float(a1_ref))), abs=1e-12)
    assert hi == pytest.approx(float(np.quantile(theta_star, float(a2_ref))), abs=1e-12)


# ------------------------------------------------------------------ internals

def test_jackknife_matches_a_direct_leave_one_out_loop():
    Y, f, N, fbar_N = _data(n=9, rho=0.4, seed=5)
    stat = _ppi_plus_statistic(N, fbar_N)
    direct = [stat(np.delete(Y, i), np.delete(f, i)) for i in range(9)]
    assert np.allclose(jackknife_theta(Y, f, fbar_N, N), direct, atol=1e-12)


def test_batched_replicates_equal_the_unbatched_ones():
    Y, f, N, fbar_N = _data(n=10, seed=2)
    a = bootstrap_replicates(Y, f, fbar_N, N, 200, np.random.default_rng(1))
    b = bootstrap_replicates(Y[None, :], f[None, :], np.array([fbar_N]), N, 200,
                             np.random.default_rng(1))
    assert np.allclose(a, b[0], atol=0)


def test_chunking_does_not_change_the_result(monkeypatch):
    """Chunking is stream-invariant: successive `rng.integers` calls consume the same values in
    the same order as one large call, so `MAX_GATHER_ELEMENTS` must not affect any number."""
    rng_Y = np.random.default_rng(8)
    Y, f = rng_Y.standard_normal((7, 10)), rng_Y.standard_normal((7, 10))
    fbar_N = np.zeros(7)
    big = bootstrap_replicates(Y, f, fbar_N, 1e6, 100, np.random.default_rng(3))
    monkeypatch.setattr("ppi_finite.bootstrap.MAX_GATHER_ELEMENTS", 200)
    small = bootstrap_replicates(Y, f, fbar_N, 1e6, 100, np.random.default_rng(3))
    assert np.array_equal(big, small)
    assert MAX_GATHER_ELEMENTS == 8_000_000        # restored by monkeypatch teardown


def test_resample_indices_have_the_scipy_shape_and_range():
    idx = bootstrap_indices(6, 50, np.random.default_rng(0))
    assert idx.shape == (50, 6) and idx.min() >= 0 and idx.max() <= 5


def test_percentile_interval_is_the_empirical_quantiles():
    x = np.arange(1001.0)[None, :]
    lo, hi = ci_percentile(x, 0.05)
    assert lo == pytest.approx(25.0) and hi == pytest.approx(975.0)


# ------------------------------------------------------------------ contracts

def test_rng_is_required():
    Y, f, N, fbar_N = _data()
    data = make_data(Y, f, fbar_N=fbar_N, N=N, sigma2_f=1.0).view()
    with pytest.raises(ValueError, match="explicit rng"):
        bootstrap_ppi(data, method="bca", B=10, rng=None)
    with pytest.raises(ValueError, match="percentile"):
        bootstrap_ppi(data, method="basic", B=10, rng=np.random.default_rng(0))


def test_degenerate_replicates_are_reported_as_failures_not_repaired():
    """All Y identical => every bootstrap replicate equals theta_hat => z0 is infinite."""
    Y = np.full(8, 2.0)
    f = np.zeros(8)
    data = make_data(Y, f, fbar_N=0.0, N=1e6, sigma2_f=1.0).view()
    res = bootstrap_ppi(data, method="bca", B=100, rng=np.random.default_rng(0))
    assert res.n_failed == 1 and not np.isfinite(res.ci_low)


def test_bootstrap_does_not_touch_theta_true():
    Y, f, N, fbar_N = _data()
    a = make_data(Y, f, fbar_N=fbar_N, N=N, sigma2_f=1.0, theta_true=0.0)
    b = a.with_theta_true(99.0)
    ra = bootstrap_ppi(a.view(), method="bca", B=200, rng=np.random.default_rng(1))
    rb = bootstrap_ppi(b.view(), method="bca", B=200, rng=np.random.default_rng(1))
    assert ra.ci_low == rb.ci_low and ra.ci_high == rb.ci_high


@pytest.mark.parametrize("name", ["B7", "B8"])
def test_registered_methods_run_batched(name):
    from ppi_finite import METHODS
    rng = np.random.default_rng(12)
    Y, f = rng.standard_normal((6, 10)), rng.standard_normal((6, 10))
    data = make_data(Y, f, fbar_N=np.zeros(6), N=1e6, sigma2_f=1.0, theta_true=0.0).view()
    res = METHODS[name](data, 0.05, np.random.default_rng(0))
    assert res.method == name and res.ci_low.shape == (6,)
    assert np.all(res.ci_low <= res.ci_high)
