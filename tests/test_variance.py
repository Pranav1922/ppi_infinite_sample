"""Unit and invariant tests for the variance layer (`11` §variance.py, `15` §9)."""
import numpy as np
import pytest

from ppi_finite import (labelled_summaries, leverage_term, var_classical, var_lc_ppi, var_ppi,
                        var_ppi_plus)

rng = np.random.default_rng(7)
Y = rng.standard_normal((200, 10))
f = rng.standard_normal((200, 10))
fbar_N = rng.standard_normal(200) / 100.0
S = labelled_summaries(Y, f)


def test_var_classical_matches_definition():
    assert np.allclose(var_classical(Y).value, Y.var(axis=-1, ddof=1) / 10)
    assert var_classical(Y).dof == 9


def test_leverage_is_zero_when_fbar_N_equals_fbar_n():
    assert np.allclose(leverage_term(f, f.mean(axis=-1)), 0.0)


def test_leverage_term_matches_var_lc_ppi_internals():
    v = var_lc_ppi(S, fbar_N, 1.0, np.inf, mode="v482")
    assert np.allclose(v.leverage, leverage_term(f, fbar_N))


def test_var_lc_reduces_to_s2_over_n_at_infinite_N_without_leverage():
    v = var_lc_ppi(S, f.mean(axis=-1), 1.0, np.inf, mode="v482")
    assert np.allclose(v.value, S["s2"] / 10)


def test_var_lc_at_infinite_N_drops_the_unlabeled_term():
    v = var_lc_ppi(S, fbar_N, 1.0, np.inf, mode="v482")
    assert np.allclose(v.components["unlabeled_term"], 0.0)
    assert np.allclose(v.value, S["s2"] * (1 / 10 + v.leverage))


def test_var_lc_ge_var_nolev_and_both_nonnegative():
    full = var_lc_ppi(S, fbar_N, 1.0, 1000.0, mode="v482")
    nolev = var_lc_ppi(S, fbar_N, 1.0, 1000.0, mode="nolev")
    assert np.all(full.value >= nolev.value)
    assert np.all(nolev.value >= S["s2"] / 10)
    assert np.all(full.value >= 0) and np.all(nolev.value >= 0)
    assert full.dof == nolev.dof == 8


def test_ppi_variances_are_nonnegative_and_normal_dof():
    assert np.all(var_ppi(Y, f, 1000.0).value >= 0)
    v = var_ppi_plus(Y, f, S["b"], 1000.0)
    assert np.all(v.value >= 0) and np.isinf(v.dof)


def test_ppi_plus_variance_matches_explicit_formula():
    lam = S["b"] / (1 + 10 / 1000)
    v = var_ppi_plus(Y, f, lam, 1000.0)
    expected = (Y - lam[:, None] * f).var(axis=-1, ddof=1) / 10 + lam ** 2 * (S["S_ff"] / 9) / 1000
    assert np.allclose(v.value, expected)


def test_ppi_family_var_f_is_the_labelled_sample_moment():
    """`04` §4.5 (disambiguated) and `09` §Baseline configuration: the `V̂ar(·)` of §4.3-§4.5 are
    `ddof = 1` moments of the n gold pairs, so `V̂ar(f) = S_ff/(n-1)` — NOT the dataset-level
    sigma_f^2 that the LC family uses. This is what cover.py and exact.py compute. Regression
    test for the decision recorded in `13` entry L1."""
    lam, N = S["b"], 1000.0
    labelled = S["S_ff"] / (10 - 1)
    v = var_ppi_plus(Y, f, lam, N)
    assert np.allclose(v.components["unlabeled_term"], lam ** 2 * labelled / N)
    # the two readings genuinely differ: sigma_f^2 = 1 would give lam^2 / N
    assert not np.allclose(v.components["unlabeled_term"], lam ** 2 * 1.0 / N)
    assert np.allclose(var_ppi(Y, f, N).components["unlabeled_term"], labelled / N)


def test_lc_family_var_f_is_the_dataset_level_sigma2_f():
    """`04` §4.8.2: the LC unlabelled term uses sigma_f^2 estimated from the N unlabelled
    predictions, which in mean_only mode is the known value passed on PPIData."""
    v = var_lc_ppi(S, fbar_N, 0.25, 1000.0, mode="v482")
    assert np.allclose(v.components["unlabeled_term"], S["b"] ** 2 * 0.25 / 1000.0)


def test_unknown_lc_mode_raises():
    with pytest.raises(ValueError, match="v_full"):
        var_lc_ppi(S, fbar_N, 1.0, np.inf, mode="v_full")
