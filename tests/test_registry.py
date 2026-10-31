"""Registry contract (`15` §9): every ID resolves and every callable has the uniform signature."""
import inspect

import numpy as np
import pytest

from ppi_finite import METHODS, get, make_data
from ppi_finite.registry import STOCHASTIC_METHODS
from ppi_finite.types import IntervalResult

# MVP (15 §11.1) + extension E1 (15 §11.3).  Every ID here is documented in 06: B1-B6 and B9
# by name, B9-nolev by 09 Table A1's "no-leverage + t" control, and the two B5 variants by 06
# B5's implementation note requiring at least the 2-fold scheme and one K-fold alternative to be
# offered and reported, and the B6 thresholds by 06 B6 + 10 H2's guard requiring three of them.
# B7 and B8 arrived with E3, and the B9-* finite-N variants with E4 (04 §4.8.3-4.8.4,
# named B9-R1..B9-R4 by 07 Experiment E).  B10 is documented but not implemented; R4
# has no entry because it is a theorem about B9, not a separate expression. B9-R5 is new in
# E4b (docs/theory_finite_N.md §9) and is not one of 04's R1-R4. B9-HC arrived with E5
# (07 Experiment H).  B9-tmin is the Gate 3 finite-pool safeguard, B9 with the min(n-2, N-1)
# quantile of theory_finite_N.md Part VIII.  B7, B8, B10 are documented but not implemented yet.
E1_IDS = {"B1", "B2", "B3", "B4", "B5", "B5-kfold", "B5-avglam", "B6", "B6-t1.5", "B6-t3",
          "B7", "B7-B9999", "B8", "B8-B9999", "B9-nolev", "B9",
          "B9-vfull", "B9-R1", "B9-R2", "B9-R3", "B9-R5", "B9-HC", "B9-ppiplus",
          "B9-tmin"}
NOT_YET = {"B10"}


def test_registry_contains_exactly_the_expected_ids():
    assert set(METHODS) == E1_IDS


def test_unimplemented_methods_are_absent_not_stubbed():
    assert not (set(METHODS) & NOT_YET)


def test_uniform_signature():
    for name, fn in METHODS.items():
        params = list(inspect.signature(fn).parameters)
        assert params[:3] == ["data", "alpha_level", "rng"], name


def test_stochastic_methods_are_exactly_the_bootstrap_ones():
    from ppi_finite.registry import STOCHASTIC_METHODS
    assert STOCHASTIC_METHODS == {"B7", "B7-B9999", "B8", "B8-B9999"}


def test_every_method_returns_an_interval_result_of_the_right_shape():
    rng = np.random.default_rng(8)
    Y = rng.standard_normal((17, 10))
    f = rng.standard_normal((17, 10))
    data = make_data(Y, f, fbar_N=rng.standard_normal(17) / 1000, N=1e6, sigma2_f=1.0,
                     theta_true=0.0, mu_f=0.0)   # mu_f: the B9-vfull diagnostic needs it
    for name in METHODS:
        rng = np.random.default_rng(0) if name in STOCHASTIC_METHODS else None
        res = get(name)(data.view(), 0.05, rng)
        assert isinstance(res, IntervalResult) and res.method == name
        assert res.theta_hat.shape == res.ci_low.shape == res.ci_high.shape == (17,)
        assert np.all(res.ci_low <= res.theta_hat) and np.all(res.theta_hat <= res.ci_high)


def test_unknown_id_raises():
    with pytest.raises(KeyError):
        get("B42")
