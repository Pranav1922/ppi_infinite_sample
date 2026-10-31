"""Extension E4 — the finite-N variance family (`04` §4.8.3-4.8.4).

The load-bearing test in this file is `test_exact_decomposition_identity`: it establishes, to
machine precision, what `θ̂_LC − θ` actually is at finite N, and everything else follows from it.
"""
import numpy as np
import pytest

from ppi_finite import METHODS, dgp_gaussian, labelled_summaries, make_data
from ppi_finite.variance import LC_MODES, var_lc_ppi


def _fixture(R=300, n=12, rho=0.6, N=200.0, seed=0):
    d = dgp_gaussian(np.random.default_rng(seed), R=R, n=n, rho=rho, N=N,
                     unlabeled_mode="explicit")
    return d, labelled_summaries(d.Y, d.f)


# ---------------------------------------------------------------- the exact decomposition

def test_exact_decomposition_identity():
    """`θ̂_LC − θ = ε̄ + (b − β)(μ_f − f̄_n) + b(f̄_N − μ_f)`, exactly.

    This is an algebraic identity, not an approximation, and it is what makes the finite-N
    variance derivable: given the labelled design the three terms are uncorrelated, `E[b] = β`,
    and `δ = f̄_N − μ_f` is independent of the labelled sample.  Derived in `13` (E4).
    """
    rho, n, N = 0.9, 10, 200.0
    beta, sigma2, mu_f = rho, 1 - rho ** 2, 0.0
    d = dgp_gaussian(np.random.default_rng(1), R=2000, n=n, rho=rho, N=N,
                     unlabeled_mode="explicit")
    s = labelled_summaries(d.Y, d.f)
    theta_hat = s["Ybar"] + s["b"] * (d.fbar_N - s["fbar_n"])
    eps = d.Y - beta * d.f                       # alpha = 0 because E[Y] = mu_f = 0
    rhs = eps.mean(-1) + (s["b"] - beta) * (mu_f - s["fbar_n"]) + s["b"] * (d.fbar_N - mu_f)
    assert np.abs(theta_hat - rhs).max() < 1e-12
    assert abs(sigma2 - (1 - rho ** 2)) < 1e-15


def test_v_full_matches_the_exact_conditional_variance_regrouping():
    """`04` §4.8.3's (V_full) equals `σ²/n + σ²(μ_f−f̄_n)²/S_ff + (β² + σ²/S_ff)σ_f²/N`.

    The regrouped form is what the decomposition above yields directly, since
    `Var(bδ | design) = E[b²]Var(δ)` and `E[b²] = β² + σ²/S_ff`.  Their equality is why (V_full)
    is exact rather than approximate.
    """
    d, s = _fixture()
    n, N, sigma2_f = s["n"], d.N, np.asarray(d.sigma2_f)
    as_written = (s["s2"] / n
                  + s["s2"] * ((s["fbar_n"] - d.mu_f) ** 2 + sigma2_f / N) / s["S_ff"]
                  + s["b"] ** 2 * sigma2_f / N)
    regrouped = (s["s2"] / n + s["s2"] * (d.mu_f - s["fbar_n"]) ** 2 / s["S_ff"]
                 + (s["b"] ** 2 + s["s2"] / s["S_ff"]) * sigma2_f / N)
    assert np.allclose(as_written, regrouped, atol=1e-15)
    got = var_lc_ppi(s, d.fbar_N, d.sigma2_f, d.N, mode="v_full", mu_f=d.mu_f)
    assert np.allclose(got.value, as_written, atol=1e-15)


# ---------------------------------------------------------------- mode contracts

def test_all_modes_are_separately_identifiable():
    d, s = _fixture()
    vals = {m: var_lc_ppi(s, d.fbar_N, d.sigma2_f, d.N, mode=m, mu_f=d.mu_f).value
            for m in LC_MODES}
    for m in LC_MODES:
        assert np.all(np.isfinite(vals[m])) and np.all(vals[m] >= 0), m
    # every pair differs somewhere except r2, which shares v482's value and differs in dof
    assert np.allclose(vals["r2"], vals["v482"])
    for a, b in (("v482", "nolev"), ("v482", "v_full"), ("v482", "r3"), ("v_full", "r1")):
        assert not np.allclose(vals[a], vals[b]), (a, b)


def test_v_full_requires_mu_f():
    d, s = _fixture()
    with pytest.raises(ValueError, match="mu_f"):
        var_lc_ppi(s, d.fbar_N, d.sigma2_f, d.N, mode="v_full")


def test_r1_equals_v482_when_delta_squared_exceeds_sigma2_over_N():
    """R1 replaces `(f̄_n−μ_f)²` by `max(0, Δ² − σ̂_f²/N)`; adding back `σ̂_f²/N` recovers `Δ²`
    exactly whenever the max is not clipped.  So R1 differs from v482 only in the clipped
    region, and is wider there — it can only raise coverage."""
    d, s = _fixture(R=4000, N=200.0)
    v482 = var_lc_ppi(s, d.fbar_N, d.sigma2_f, d.N, mode="v482").value
    r1 = var_lc_ppi(s, d.fbar_N, d.sigma2_f, d.N, mode="r1").value
    Delta2 = (d.fbar_N - s["fbar_n"]) ** 2
    unclipped = Delta2 >= np.asarray(d.sigma2_f) / d.N
    assert np.allclose(r1[unclipped], v482[unclipped], atol=1e-15)
    assert np.all(r1 >= v482 - 1e-15)
    assert unclipped.mean() < 1.0                # the clipped region is non-empty at N = 200


def test_r2_dof_exceeds_n_minus_2_and_tends_to_it_as_N_grows():
    # large N via mean_only: materialising a pool of 1e8 per replication is not the point here
    d0, s0 = _fixture(N=200.0)
    assert np.all(var_lc_ppi(s0, d0.fbar_N, d0.sigma2_f, 200.0, mode="r2").dof > s0["n"] - 2)
    big = dgp_gaussian(np.random.default_rng(0), R=300, n=12, rho=0.6, N=1e8)
    sb = labelled_summaries(big.Y, big.f)
    v = var_lc_ppi(sb, big.fbar_N, big.sigma2_f, big.N, mode="r2")
    assert np.all(v.dof >= sb["n"] - 2 - 1e-12)
    assert np.allclose(v.dof, sb["n"] - 2, atol=1e-3)


def test_r3_drops_the_unlabelled_term_entirely():
    d, s = _fixture()
    v = var_lc_ppi(s, d.fbar_N, d.sigma2_f, d.N, mode="r3")
    assert np.allclose(v.components["unlabeled_term"], 0.0)
    assert np.allclose(v.value, s["s2"] * (1 / s["n"] + v.leverage))


@pytest.mark.parametrize("mode", LC_MODES)
def test_every_mode_reduces_to_the_N_infinity_expression(mode):
    """As `N → ∞` every finite-N variant must collapse onto `04` §4.8.1's `s²(1/n + h)`."""
    rng = np.random.default_rng(9)
    d = make_data(rng.standard_normal((300, 12)), rng.standard_normal((300, 12)),
                  mu_f=0.0, sigma2_f=1.0, theta_true=0.0)
    s = labelled_summaries(d.Y, d.f)
    v = var_lc_ppi(s, d.fbar_N, d.sigma2_f, d.N, mode=mode, mu_f=0.0)
    h = (np.asarray(d.fbar_N) - s["fbar_n"]) ** 2 / s["S_ff"]
    expected = s["s2"] * (1 / s["n"] + (0.0 if mode == "nolev" else h))
    assert np.allclose(v.value, expected, atol=1e-15)


def test_coef_mode_ppi_plus_scales_the_variance_consistently():
    d, s = _fixture(N=50.0)
    r = s["n"] / d.N
    b9 = METHODS["B9"](d.view(), 0.05)
    pp = METHODS["B9-ppiplus"](d.view(), 0.05)
    assert not np.allclose(b9.theta_hat, pp.theta_hat)          # r = 0.24 here, not negligible
    # Only the coefficient-dependent terms scale by k^2 = 1/(1+r)^2; the s^2/n term comes from
    # Ybar and is untouched.
    mean_term = s["s2"] / s["n"]
    assert np.allclose(pp.se ** 2 - mean_term, (b9.se ** 2 - mean_term) / (1 + r) ** 2,
                       rtol=1e-12)


# ---------------------------------------------------------------- boundary and isolation

def test_theta_true_is_not_reachable_by_any_finite_N_variant():
    d, _ = _fixture()
    other = d.with_theta_true(99.0)
    for name in ("B9", "B9-vfull", "B9-R1", "B9-R2", "B9-R3", "B9-ppiplus"):
        a, b = METHODS[name](d.view(), 0.05), METHODS[name](other.view(), 0.05)
        assert np.array_equal(a.ci_low, b.ci_low) and np.array_equal(a.ci_high, b.ci_high)


def test_variances_are_non_negative_across_a_wide_grid():
    for N in (50.0, 200.0, 5000.0):
        for rho in (0.0, 0.9):
            d, s = _fixture(R=500, N=N, rho=rho, seed=int(N))
            for mode in LC_MODES:
                v = var_lc_ppi(s, d.fbar_N, d.sigma2_f, N, mode=mode, mu_f=d.mu_f)
                assert np.all(v.value >= 0), (N, rho, mode)
                assert np.all(np.asarray(v.dof) > 0), (N, rho, mode)


# ---------------------------------------------------------------- E4b: the exact theory

def test_conditional_pivot_is_exactly_t_n_minus_2():
    """Theorem A' of `docs/theory_finite_N.md`: with `δ = f̄_N − μ_f` and `Δ = f̄_N − f̄_n`,

        (θ̂_LC − θ − βδ) / (s·√(1/n + Δ²/S_ff))  ~  t_{n−2}   exactly,

    conditionally on the labelled design AND on `f̄_N`, hence marginally, and independent of `δ`.
    This is the one exact distributional statement available at finite N.
    """
    from scipy import stats
    n, rho, N, R = 12, 0.7, 100.0, 60_000
    d = dgp_gaussian(np.random.default_rng(31), R=R, n=n, rho=rho, N=N,
                     unlabeled_mode="explicit")
    s = labelled_summaries(d.Y, d.f)
    Delta = d.fbar_N - s["fbar_n"]
    theta_hat = s["Ybar"] + s["b"] * Delta
    G = (theta_hat - rho * d.fbar_N) / np.sqrt(s["s2"] * (1 / n + Delta ** 2 / s["S_ff"]))
    assert stats.kstest(G, "t", args=(n - 2,)).pvalue > 0.01
    # and independent of the unlabelled draw: the pivot still holds in each tail of delta
    lo, hi = np.quantile(d.fbar_N, [0.2, 0.8])
    assert stats.kstest(G[d.fbar_N < lo], "t", args=(n - 2,)).pvalue > 0.01
    assert stats.kstest(G[d.fbar_N > hi], "t", args=(n - 2,)).pvalue > 0.01
    assert abs(np.corrcoef(G, d.fbar_N)[0, 1]) < 0.02


def test_idealised_law_and_its_exact_coverage_integral():
    """Theorem B: under the idealisation, `T = Z/√((1−s)Q/ν + s)` with `Q ~ χ²_ν ⟂ Z ~ N(0,1)`,
    so `P(|T| ≤ q) = E_Q[2Φ(q√((1−s)Q/ν + s)) − 1]` exactly."""
    from scipy import integrate, stats
    nu, rng = 8, np.random.default_rng(3)
    q = float(stats.t.ppf(0.975, nu))
    for s_share in (0.1, 0.45):
        Q, Z = rng.chisquare(nu, 400_000), rng.standard_normal(400_000)
        mc = float(np.mean(np.abs(Z / np.sqrt((1 - s_share) * Q / nu + s_share)) <= q))
        exact = integrate.quad(
            lambda x: (2 * stats.norm.cdf(q * np.sqrt((1 - s_share) * x / nu + s_share)) - 1)
            * stats.chi2.pdf(x, nu), 0, nu * 60, limit=400)[0]
        assert abs(mc - exact) < 0.003


def test_coverage_is_monotone_increasing_in_the_unlabelled_share():
    """Theorem C: the conservativeness result. Coverage of the `t_{n−2}` interval under the
    idealised law is non-decreasing in `s`, so it is never below nominal."""
    from scipy import integrate, stats
    for nu in (3, 8, 48):
        q = float(stats.t.ppf(0.975, nu))
        cs = [integrate.quad(lambda x: (2 * stats.norm.cdf(q * np.sqrt((1 - sv) * x / nu + sv)) - 1)
                             * stats.chi2.pdf(x, nu), 0, nu * 60, limit=400)[0]
              for sv in np.linspace(0, 0.9, 10)]
        assert np.all(np.diff(cs) > -1e-9)
        assert abs(cs[0] - 0.95) < 1e-6          # s = 0 recovers exact nominal coverage
        assert cs[-1] > 0.95


def test_r5_dof_exceeds_satterthwaite_and_reduces_to_n_minus_2():
    """R5 inverts the exact coverage integral; R2 matches two moments of the variance estimator.
    R5's effective dof is therefore larger (a narrower interval) at every positive share."""
    from ppi_finite.variance import _r5_dof_table
    grid, nu_star = _r5_dof_table(8.0, 0.05)
    sat = 8.0 / (1 - grid[:-1]) ** 2
    assert nu_star[0] == 8.0
    assert np.all(nu_star[1:-1] > sat[1:])
    assert np.isinf(nu_star[-1])


def test_r5_is_registered_and_narrower_than_r2():
    d, _ = _fixture(R=2000, n=10, rho=0.9, N=50.0)
    r2, r5 = METHODS["B9-R2"](d.view(), 0.05), METHODS["B9-R5"](d.view(), 0.05)
    assert np.allclose(r2.se, r5.se)                       # same variance, different quantile
    assert np.mean(r5.ci_high - r5.ci_low) < np.mean(r2.ci_high - r2.ci_low)


# ---------------------------------------------------------------- E4c: the idealisation error

def test_ablations_share_draws_and_isolate_one_effect_each():
    """`A0 → A1 → A2 → A3` removes the product term, then the W-dependence, then the b²-vs-β²
    dependence. All four run on common random numbers, so their differences are attributable."""
    from ppi_finite import ablation_coverages, ABLATIONS
    r = ablation_coverages(np.random.default_rng(0), n=10, N=50.0, rho=0.9, R=200_000)
    assert set(ABLATIONS) <= set(r)
    for k in ABLATIONS:
        assert 0.9 < r[k] < 1.0
    # the paired difference must be materially more precise than differencing two coverages
    assert r["diff_mc_se"] < r["mc_se"] / 2
    assert abs(r["diff_A0_A3"] - (r["A0_practical"] - r["A3_idealised"])) < 1e-12


def test_idealisation_error_decays_like_one_over_N():
    """Proposition II.1: `Δ = O(1/N)`, tested where it matters — the rate must be much faster
    than `N^{-1/2}`, which is what the raw fluctuation of the `W`-dependence would give if it did
    not cancel. Ratios rather than a tight `N·Δ` comparison, because `Δ` at large `N` is only a
    few Monte-Carlo standard errors."""
    from ppi_finite import ablation_coverages
    diffs = {}
    for N in (100.0, 400.0, 1600.0):
        r = ablation_coverages(np.random.default_rng(int(N)), n=10, N=N, rho=0.0, R=1_500_000)
        diffs[N] = r["diff_A0_A3"]
    assert all(d > 0 for d in diffs.values())
    # 16x more unlabelled data: O(1/N) predicts a 16x drop, O(N^-1/2) only 4x.
    ratio = diffs[100.0] / diffs[1600.0]
    assert ratio > 8.0, diffs
    # and N*Delta stays bounded rather than growing
    assert all(0.05 < N * d < 0.6 for N, d in diffs.items()), diffs


def test_tau_bounds_the_product_term_share():
    """`v3/V <= n σ_f²/(N S_ff)`, so `τ = E[v3/V] <= n/(N(n-3))` for `n >= 4`."""
    rng = np.random.default_rng(2)
    for n, N, rho in ((10, 100.0, 0.5), (5, 25.0, 0.9), (50, 1000.0, 0.0)):
        f = rng.standard_normal((200_000, n))
        S_ff = ((f - f.mean(1, keepdims=True)) ** 2).sum(1)
        h = f.mean(1) ** 2 / S_ff
        sigma2, beta2 = 1 - rho ** 2, rho ** 2
        v1, v2, v3 = sigma2 * (1 / n + h), beta2 / N, sigma2 / (N * S_ff)
        tau = float(np.mean(v3 / (v1 + v2 + v3)))
        assert np.all(v3 / (v1 + v2 + v3) <= n / (N * S_ff) + 1e-12)
        assert tau <= n / (N * (n - 3))


def test_idealisation_error_is_not_sign_definite():
    """§II.8 counterexample: the tempting proof route `coverage_act >= coverage_id` is FALSE.
    This test exists so that the counterexample cannot be quietly lost."""
    from ppi_finite import ablation_coverages
    r = ablation_coverages(np.random.default_rng(77), n=5, N=25.0, rho=0.9, R=1_000_000)
    assert r["diff_A0_A3"] < -3 * r["diff_mc_se"]         # significantly negative
    assert r["A0_practical"] > 0.95                        # yet still conservative


def test_practical_coverage_stays_at_or_above_nominal_in_adversarial_cells():
    """Evidence for R4 where it is hardest: tiny n, tiny N, rho near 1. Not a proof."""
    from ppi_finite import ablation_coverages
    for n, N, rho in ((5, 25.0, 0.0), (5, 25.0, 0.99), (10, 25.0, 0.9), (50, 25.0, 0.0)):
        r = ablation_coverages(np.random.default_rng(n * 100 + int(N)), n=n, N=N, rho=rho,
                               R=400_000)
        assert r["A0_practical"] > 0.95 - 4 * r["mc_se"], (n, N, rho, r["A0_practical"])


def test_exact_coverage_integral_endpoints():
    from scipy import stats
    from ppi_finite import exact_coverage_integral
    q, nu = float(stats.t.ppf(0.975, 8)), 8
    assert abs(float(exact_coverage_integral(q, nu, 0.0)) - 0.95) < 1e-8
    assert abs(float(exact_coverage_integral(q, nu, 1.0)) - (2 * stats.norm.cdf(q) - 1)) < 1e-8
    vals = exact_coverage_integral(q, nu, np.array([0.0, 0.25, 0.5, 0.75]))
    assert np.all(np.diff(vals) > 0)                       # Theorem C, tabulated


def test_theorem_C_statistic_is_the_behrens_fisher_statistic_with_one_variance_known():
    """PRIOR ART LOCK-IN (E4d).  Our Theorem B/C statistic is *literally* the two-sample
    Behrens-Fisher statistic with the second variance known (`ν₂ = ∞`):

        (x̄₁ − x̄₂ − δ) / √(s₁²/n₁ + σ₂²/n₂)   with   ν₁s₁²/σ₁² ~ χ²_{ν₁}

    equals `Z/√((1−s)Q/ν₁ + s)` with `s = V₂/(V₁+V₂)`.  The `t_{min(ν₁,ν₂)} = t_{ν₁}` rule is then
    exactly our `t_{n−2}` rule, and "the actual confidence is greater than 100(1−α)%" is
    Mickey & Brown (1966), *Ann. Math. Statist.* 37, 639–642.

    This test exists so that the equivalence — and therefore the fact that **Theorem C is not
    ours** — cannot be quietly lost from the codebase.
    """
    from scipy import stats
    from ppi_finite import exact_coverage_integral
    rng = np.random.default_rng(0)
    nu1, R = 9, 500_000
    t_star = float(stats.t.ppf(0.975, nu1))
    for V1, V2 in ((1.0, 0.0), (1.0, 0.5), (0.3, 1.2)):
        D = rng.standard_normal(R) * np.sqrt(V1 + V2)
        s1 = V1 * rng.chisquare(nu1, R) / nu1
        cov_bf = float(np.mean(np.abs(D / np.sqrt(s1 + V2)) <= t_star))
        cov_ours = float(exact_coverage_integral(t_star, nu1, V2 / (V1 + V2)))
        assert abs(cov_bf - cov_ours) < 0.002, (V1, V2, cov_bf, cov_ours)
        assert cov_bf >= 0.95 - 0.002                      # Mickey & Brown conservativeness


def test_mickey_brown_two_sided_bound_holds_and_is_attained():
    """PRIOR ART LOCK-IN (E6b). Mickey & Brown (1966), Ann. Math. Statist. 37, 639-642, Theorem:

        P{|t_min(f1,f2)| < v}  <=  P{|V_gamma| < v}  <=  P{|t_{f1+f2}| < v}

    At `f2 -> infinity` this is `1-alpha <= coverage <= 2*Phi(t_{nu,1-alpha/2}) - 1`. The upper
    bound is a cap on the over-coverage that this project did NOT have, and both bounds are
    attained at the endpoints `s = 0` and `s = 1` — which is what their concavity proof predicts.
    """
    from scipy import stats
    from ppi_finite import exact_coverage_integral
    for nu in (3, 8, 18, 48):
        v = float(stats.t.ppf(0.975, nu))
        lo, hi = 0.95, 2 * stats.norm.cdf(v) - 1
        vals = np.array([float(exact_coverage_integral(v, nu, s))
                         for s in (0.0, 0.25, 0.5, 0.75, 1.0)])
        assert np.all(vals >= lo - 1e-9) and np.all(vals <= hi + 1e-9), (nu, vals)
        assert vals[0] == pytest.approx(lo, abs=1e-8)      # attained at s = 0
        assert vals[-1] == pytest.approx(hi, abs=1e-8)     # attained at s = 1
