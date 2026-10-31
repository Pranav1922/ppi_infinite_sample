"""Extension E6 — the Galaxies adapter and the real-data pipeline (`07` Experiment I, `08` Tier 2).

These test the **adapter**, not the science: on one real dataset there is nothing to prove, only
to demonstrate. The containment numbers live in `results/I_real_data.csv` and `13`.
"""
import numpy as np
import pytest

from ppi_finite import METHODS, load_population, subsample_protocol
from ppi_finite.realdata import _sample_without_replacement, dgp_galaxies

pytestmark = pytest.mark.skipif(not __import__("pathlib").Path("data/galaxies.npz").exists(),
                                reason="data/galaxies.npz absent; E6 needs the population file")


@pytest.fixture(scope="module")
def pop():
    return load_population()


# ---------------------------------------------------------------- the file itself

def test_expected_keys_shapes_and_dtypes(pop):
    assert pop["M"] == 16_743
    assert pop["Y"].shape == pop["f"].shape == (16_743,)
    assert pop["Y"].dtype == np.float64 and pop["f"].dtype == np.float64


def test_labels_are_binary_and_predictions_are_probabilities(pop):
    assert set(np.unique(pop["Y"])) == {0.0, 1.0}
    assert pop["binary_Y"]
    assert 0.0 < pop["f"].min() and pop["f"].max() < 1.0


def test_no_missing_values_anywhere(pop):
    assert np.all(np.isfinite(pop["Y"])) and np.all(np.isfinite(pop["f"]))


def test_population_quantities_are_the_documented_ones(pop):
    """Pinned in `docs/tier2_galaxies.md` §3; a change here means a different file."""
    assert pop["theta"] == pytest.approx(0.2592725318043361, abs=1e-12)
    assert pop["rho"] == pytest.approx(0.7407825907116699, abs=1e-12)
    assert pop["mu_f"] == pytest.approx(0.2603170228486923, abs=1e-12)


def test_missing_file_and_missing_keys_raise_rather_than_guess(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_population(path=tmp_path / "nope.npz")
    bad = tmp_path / "bad.npz"
    np.savez(bad, Y=np.zeros(5))
    with pytest.raises(KeyError, match="Yhat"):
        load_population(path=bad)


# ---------------------------------------------------------------- the sampling protocol

def test_sampling_is_without_replacement_and_in_range():
    idx = _sample_without_replacement(np.random.default_rng(0), 16_743, 50, 500)
    assert idx.shape == (500, 50)
    assert idx.min() >= 0 and idx.max() < 16_743
    assert all(len(np.unique(r)) == 50 for r in idx)


def test_sampling_is_reproducible_and_seed_dependent():
    a = _sample_without_replacement(np.random.default_rng(7), 1000, 20, 50)
    b = _sample_without_replacement(np.random.default_rng(7), 1000, 20, 50)
    c = _sample_without_replacement(np.random.default_rng(8), 1000, 20, 50)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_gold_and_pool_are_disjoint_and_pool_summaries_are_exact(pop):
    """`f̄_N` and `σ̂_f²` are computed from population totals minus gold totals rather than by
    materialising the pool; check that against a direct computation on a few replications."""
    d = subsample_protocol(pop, n=25, R=40, rng=np.random.default_rng(3))
    assert d.N == pop["M"] - 25
    idx = _sample_without_replacement(np.random.default_rng(3), pop["M"], 25, 40)
    for r in range(5):
        mask = np.ones(pop["M"], bool); mask[idx[r]] = False
        assert np.allclose(d.fbar_N[r], pop["f"][mask].mean())
        assert np.allclose(d.sigma2_f[r], pop["f"][mask].var(ddof=1))
        assert np.allclose(np.sort(d.Y[r]), np.sort(pop["Y"][idx[r]]))


def test_n_out_of_range_raises(pop):
    with pytest.raises(ValueError, match="n must satisfy"):
        subsample_protocol(pop, n=2, R=5, rng=np.random.default_rng(0))


# ---------------------------------------------------------------- PPIData construction

def test_ppidata_is_well_formed_and_carries_the_evaluation_only_truth(pop):
    d = subsample_protocol(pop, n=20, R=100, rng=np.random.default_rng(1))
    assert d.Y.shape == d.f.shape == (100, 20)
    assert np.all(np.isfinite(d.Y)) and np.all(np.isfinite(d.f))
    assert d.theta_true == pytest.approx(pop["theta"])
    assert not hasattr(d.view(), "theta_true")


def test_population_theta_cannot_leak_into_any_method(pop):
    """The E6 form of the `15` §4.3 boundary: the answer is in the dataset, so this matters more
    here than anywhere else in the project."""
    a = subsample_protocol(pop, n=20, R=200, rng=np.random.default_rng(2))
    b = a.with_theta_true(0.999)
    for name in ("B1", "B2", "B3", "B4", "B5", "B9", "B9-R2", "B9-HC"):
        ra, rb = METHODS[name](a.view(), 0.05), METHODS[name](b.view(), 0.05)
        assert np.array_equal(ra.ci_low, rb.ci_low), name
        assert np.array_equal(ra.ci_high, rb.ci_high), name


def test_no_nan_reaches_any_interval(pop):
    d = subsample_protocol(pop, n=10, R=500, rng=np.random.default_rng(4))
    for name in ("B1", "B2", "B3", "B9", "B9-HC"):
        r = METHODS[name](d.view(), 0.05)
        assert np.all(np.isfinite(r.ci_low)) and np.all(np.isfinite(r.ci_high)), name
        assert np.all(r.ci_low <= r.ci_high)


def test_producer_ignores_rho_and_N_because_they_are_observed_not_controlled():
    a = dgp_galaxies(np.random.default_rng(5), R=50, n=10, rho=0.0, N=999.0)
    b = dgp_galaxies(np.random.default_rng(5), R=50, n=10, rho=0.9, N=1e9)
    assert np.array_equal(a.Y, b.Y) and a.N == b.N
    assert a.N == 16_743 - 10


def test_repeated_runs_of_the_cell_are_deterministic():
    from ppi_finite.simulation import run_cell
    from ppi_finite.types import CellSpec
    spec = dict(experiment_id="I_real_data", dgp="galaxies", dgp_params={"dataset": "galaxies"},
                n=10, N=float("nan"), rho=float("nan"), alpha_level=0.05, R=300,
                methods=("B1", "B3", "B9"), unlabeled_mode="mean_only", coef_mode="ols",
                seed_spawn_key=(1, 0, 0, 0, 0, 0), seed_entropy=20260819)
    _, ra, _, rl_a = run_cell(CellSpec(**spec))
    _, rb, _, rl_b = run_cell(CellSpec(**spec))
    for m in ra:
        assert np.array_equal(ra[m].theta_hat, rb[m].theta_hat)
    assert rl_a == rl_b and rl_a["N"] == 16_733


# ---------------------------------------------------------------- Experiment J: MT-bench

@pytest.mark.skipif(not __import__("pathlib").Path("data/mtbench_gpt35.npz").exists(),
                    reason="data/mtbench_gpt35.npz absent; run experiments/prepare_mtbench.py")
def test_mtbench_population_is_the_documented_one():
    """Pinned in `docs/tier3_mtbench.md`; a change here means a different population."""
    pop = load_population("mtbench")
    assert pop["M"] == 1_478
    assert set(np.unique(pop["Y"])) == {0.0, 0.5, 1.0} and set(np.unique(pop["f"])) == {0.0, 0.5, 1.0}
    assert pop["theta"] == pytest.approx(0.6518944519621109, abs=1e-12)
    assert pop["rho"] == pytest.approx(0.4886994008697362, abs=1e-12)
    assert pop["mu_f"] == pytest.approx(0.6204330175913396, abs=1e-12)
