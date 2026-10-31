"""B4, B5, B6 and the lambda rules they need (extension E1; `06` B4-B6, `11` §estimators.py)."""
import numpy as np
import pytest

from ppi_finite import (METHODS, crossfit_folds, labelled_summaries, lambda_crossfit,
                        lambda_plugin, lambda_threshold, make_data, method_b1, method_b3,
                        method_b4, method_b5, method_b5_avglam, method_b5_kfold, method_b6)

rng = np.random.default_rng(21)
R, n, N = 40, 11, 1e5
f = rng.standard_normal((R, n))
Y = 0.6 * f + rng.standard_normal((R, n))
DATA = make_data(Y, f, fbar_N=rng.standard_normal(R) / np.sqrt(N), N=N, sigma2_f=1.0,
                 theta_true=0.0).view()
S = labelled_summaries(Y, f)


# ---------------------------------------------------------------- fold construction

def test_2fold_sizes_and_partition():
    a, b = crossfit_folds(11, "2fold")
    assert len(a) == 5 and len(b) == 6                 # 06 B5: floor(n/2), ceil(n/2)
    assert sorted(np.concatenate([a, b])) == list(range(11))


def test_kfold_partitions_exactly():
    folds = crossfit_folds(11, "kfold", 5)
    assert len(folds) == 5
    assert sorted(np.concatenate(folds)) == list(range(11))


def test_kfold_validates_k():
    with pytest.raises(ValueError):
        crossfit_folds(5, "kfold", 9)
    with pytest.raises(ValueError):
        crossfit_folds(5, "nonsense")


def test_fold_lambda_is_estimated_on_the_complement():
    """`06` B5: fold B's estimate uses the lambda fitted on fold A, and vice versa."""
    lams, folds = lambda_crossfit(Y, f, N, scheme="2fold")
    for j, other in ((0, 1), (1, 0)):
        b_other = labelled_summaries(Y[:, folds[other]], f[:, folds[other]])["b"]
        assert np.allclose(lams[j], lambda_plugin(b_other, n, N))


# ---------------------------------------------------------------- B4

def test_b4_is_b3_with_a_t_quantile():
    b3, b4 = method_b3(DATA, 0.05), method_b4(DATA, 0.05)
    assert np.allclose(b3.theta_hat, b4.theta_hat)     # same point estimate
    assert np.allclose(b3.se, b4.se)                   # same variance
    assert b4.dof == n - 2 and np.isinf(b3.dof)
    assert np.all(b4.width > b3.width)                 # t_{n-2} is wider than z


# ---------------------------------------------------------------- B5

def test_b5_matches_06_formula_computed_directly():
    lams, folds = lambda_crossfit(Y, f, N, scheme="2fold")
    A, B = folds
    tB = (Y[:, B] - lams[1][:, None] * f[:, B]).mean(1) + lams[1] * DATA.fbar_N
    tA = (Y[:, A] - lams[0][:, None] * f[:, A]).mean(1) + lams[0] * DATA.fbar_N
    assert np.allclose(method_b5(DATA, 0.05).theta_hat, 0.5 * (tA + tB))


def test_b5_avglam_is_a_different_scheme_from_b5():
    a, b = method_b5(DATA, 0.05), method_b5_avglam(DATA, 0.05)
    assert not np.allclose(a.theta_hat, b.theta_hat)   # different point estimates
    assert np.allclose(a.se, b.se)                     # same variance (averaged lambda)


def test_b5_avglam_applies_the_averaged_lambda_to_the_whole_sample():
    lams, _ = lambda_crossfit(Y, f, N, scheme="2fold")
    lam_bar = 0.5 * (lams[0] + lams[1])
    expected = Y.mean(1) - lam_bar * (f.mean(1) - DATA.fbar_N)
    assert np.allclose(method_b5_avglam(DATA, 0.05).theta_hat, expected)


def test_b5_kfold_uses_five_folds_and_the_t_quantile():
    res = method_b5_kfold(DATA, 0.05)
    assert res.extras["crossfit_k"] == 5 and res.dof == n - 2


# ---------------------------------------------------------------- B6

def test_threshold_keeps_or_zeroes_lambda_by_significance():
    lam = lambda_threshold(S["b"], S["se_b"], n, N, t_crit=2.0)
    t2 = (S["b"] / S["se_b"]) ** 2
    assert np.allclose(lam[t2 > 4], lambda_plugin(S["b"], n, N)[t2 > 4])
    assert np.all(lam[t2 <= 4] == 0.0)


def test_threshold_is_parameterised_not_hardcoded():
    """`06` B6: |t| > 2 has no principled justification; Experiment D must sweep it."""
    kept = [np.mean(lambda_threshold(S["b"], S["se_b"], n, N, t_crit=c) != 0)
            for c in (1.5, 2.0, 3.0)]
    assert kept[0] > kept[1] > kept[2]                 # a higher bar keeps fewer


def test_b6_with_an_unreachable_threshold_reduces_to_the_classical_mean():
    res = method_b6(DATA, 0.05, t_crit=1e9)
    assert np.allclose(res.theta_hat, method_b1(DATA, 0.05).theta_hat)
    assert res.dof == n - 2                            # but with t_{n-2}, not t_{n-1}


def test_b6_with_a_zero_threshold_reduces_to_b4():
    res = method_b6(DATA, 0.05, t_crit=0.0)
    assert np.allclose(res.theta_hat, method_b4(DATA, 0.05).theta_hat)


# ---------------------------------------------------------------- family invariants

def test_b6_threshold_variants_are_registered_and_differ():
    """`10` H2's guard: the hypothesis must be tested against three thresholds."""
    ids = [METHODS[i](DATA, 0.05, None) for i in ("B6-t1.5", "B6", "B6-t3")]
    assert [r.method for r in ids] == ["B6-t1.5", "B6", "B6-t3"]
    assert [r.extras["t_crit"] for r in ids] == [1.5, 2.0, 3.0]
    zeroed = [r.extras["frac_lambda_zero"] for r in ids]
    assert zeroed[0] < zeroed[1] < zeroed[2]


@pytest.mark.parametrize("name", ["B4", "B5", "B5-kfold", "B5-avglam", "B6", "B6-t1.5", "B6-t3"])
def test_family_uses_t_with_n_minus_2(name):
    res = METHODS[name](DATA, 0.05, None)
    assert res.dof == n - 2 and res.extras["dist"] == "t"
    assert np.all(res.ci_low <= res.theta_hat) and np.all(res.theta_hat <= res.ci_high)
