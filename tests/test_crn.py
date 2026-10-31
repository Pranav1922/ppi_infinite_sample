"""Common random numbers and the theta_true boundary (`15` §4.3, §9; C1)."""
import numpy as np

from ppi_finite import METHODS, make_data
from ppi_finite.registry import STOCHASTIC_METHODS
from ppi_finite.simulation import run_cell
from ppi_finite.types import CellSpec


def _cell(**kw):
    base = dict(experiment_id="crn", dgp="gaussian",
                dgp_params={"var_Y": 1.0, "var_f": 1.0, "mean_Y": 0.0, "mean_f": 0.0},
                n=10, N=1e6, rho=0.5, alpha_level=0.05, R=500,
                methods=("B1", "B2", "B3", "B9-nolev", "B9"), unlabeled_mode="mean_only",
                coef_mode="ols", seed_spawn_key=(1, 0, 0, 0, 0, 0), seed_entropy=20260819)
    return CellSpec(**{**base, **kw})


def test_theta_true_is_absent_from_the_view():
    data = make_data(np.zeros((3, 5)) + 1.0, np.arange(15.0).reshape(3, 5),
                     fbar_N=np.zeros(3), N=1e6, sigma2_f=1.0, theta_true=42.0)
    view = data.view()
    assert not hasattr(view, "theta_true")
    assert "theta_true" not in vars(view)


def test_mutating_theta_true_leaves_every_method_bit_identical():
    rng = np.random.default_rng(9)
    Y, f = rng.standard_normal((50, 10)), rng.standard_normal((50, 10))
    fbar_N = rng.standard_normal(50) / 1000
    a = make_data(Y, f, fbar_N=fbar_N, N=1e6, sigma2_f=1.0, theta_true=0.0, mu_f=0.0)
    b = a.with_theta_true(1234.5)
    for name, fn in METHODS.items():
        mk = (lambda: np.random.default_rng(5)) if name in STOCHASTIC_METHODS else (lambda: None)
        ra, rb = fn(a.view(), 0.05, mk()), fn(b.view(), 0.05, mk())
        assert np.array_equal(ra.theta_hat, rb.theta_hat), name
        assert np.array_equal(ra.ci_low, rb.ci_low) and np.array_equal(ra.ci_high, rb.ci_high)


def test_all_methods_in_a_cell_see_the_same_arrays():
    seen = {}
    original = dict(METHODS)
    try:
        for name, fn in original.items():
            def spy(view, alpha, rng, summaries=None, _fn=fn, _name=name):
                seen[_name] = (view.Y, view.f, view.fbar_N)
                return _fn(view, alpha, rng, summaries=summaries)
            METHODS[name] = spy
        run_cell(_cell())
    finally:
        METHODS.clear()
        METHODS.update(original)

    arrays = list(seen.values())
    for Y, f, fbar_N in arrays[1:]:
        assert Y is arrays[0][0] and f is arrays[0][1] and fbar_N is arrays[0][2]


def _drop_timing(rows):
    """`seconds_per_interval` is wall-clock and is the one row that is not reproducible by
    construction; everything else must be bit-identical."""
    return [r for r in rows if r["metric"] != "seconds_per_interval"]


def test_same_cell_spec_gives_identical_draws():
    rows_a, res_a, _, _ = run_cell(_cell())
    rows_b, res_b, _, _ = run_cell(_cell())
    for name in res_a:
        assert np.array_equal(res_a[name].theta_hat, res_b[name].theta_hat)
    assert _drop_timing(rows_a) == _drop_timing(rows_b)


def test_bootstrap_methods_share_resamples_and_do_not_perturb_each_other():
    """`15` §7.4 requires that adding a bootstrap method not change any other method's draws.
    A single shared generator would break that — whichever bootstrap ran first would consume
    the stream — so each stochastic method gets a fresh generator from the same seed, which
    also puts B7 and B8 on identical resamples (the paired comparison C1 exists for)."""
    _, with_both, _, _ = run_cell(_cell(methods=("B1", "B3", "B7", "B8", "B9")))
    _, without_b7, _, _ = run_cell(_cell(methods=("B1", "B3", "B8", "B9")))
    _, only_b7, _, _ = run_cell(_cell(methods=("B7",)))

    for name in ("B1", "B3", "B9"):
        assert np.array_equal(with_both[name].theta_hat, without_b7[name].theta_hat)
    assert np.array_equal(with_both["B8"].ci_low, without_b7["B8"].ci_low)
    assert np.array_equal(with_both["B7"].ci_low, only_b7["B7"].ci_low)
    # B7 and B8 are built from the same bootstrap distribution, so their point estimates and
    # bootstrap standard errors agree exactly while their intervals differ.
    assert np.array_equal(with_both["B7"].se, with_both["B8"].se)
    assert not np.array_equal(with_both["B7"].ci_low, with_both["B8"].ci_low)


def test_seed_depends_on_the_cell_coordinates():
    _, res_a, _, _ = run_cell(_cell())
    _, res_b, _, _ = run_cell(_cell(rho=0.9, seed_spawn_key=(1, 0, 0, 2, 0, 0)))
    assert not np.array_equal(res_a["B1"].theta_hat, res_b["B1"].theta_hat)
