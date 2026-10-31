"""Interval construction (`11` §confidence_intervals.py)."""
import numpy as np
import pytest
from scipy import stats

from ppi_finite import ci_from_se


def test_matches_scipy_t_interval():
    rng = np.random.default_rng(4)
    Y = rng.standard_normal(15)
    se = Y.std(ddof=1) / np.sqrt(15)
    low, high = ci_from_se(Y.mean(), se, alpha_level=0.05, dof=14, dist="t")
    ref = stats.t.interval(0.95, 14, loc=Y.mean(), scale=se)
    assert abs(low - ref[0]) < 1e-12 and abs(high - ref[1]) < 1e-12


def test_normal_multiplier_is_1_959964():
    low, high = ci_from_se(0.0, 1.0, alpha_level=0.05, dist="normal")
    assert high == pytest.approx(1.959964, abs=1e-6)


def test_interval_contains_point_estimate_and_widens_as_alpha_shrinks():
    low95, high95 = ci_from_se(1.0, 0.5, alpha_level=0.05, dof=8)
    low99, high99 = ci_from_se(1.0, 0.5, alpha_level=0.01, dof=8)
    assert low95 <= 1.0 <= high95
    assert (high99 - low99) > (high95 - low95)


def test_batched_shape_follows_input():
    low, high = ci_from_se(np.zeros(5), np.ones(5), alpha_level=0.05, dof=7)
    assert low.shape == high.shape == (5,)


def test_validation():
    with pytest.raises(ValueError):
        ci_from_se(0.0, 1.0, alpha_level=0.0, dof=5)
    with pytest.raises(ValueError):
        ci_from_se(0.0, 1.0, alpha_level=0.05, dof=None, dist="t")
    with pytest.raises(ValueError):
        ci_from_se(0.0, -1.0, alpha_level=0.05, dof=5)
    with pytest.raises(ValueError):
        ci_from_se(0.0, 1.0, alpha_level=0.05, dist="cauchy")
