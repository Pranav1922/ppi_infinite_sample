"""Finite-N theory objects: the idealised statistic, its exact coverage, and the ablations that
separate the idealisations (L1).  See `docs/theory_finite_N.md`.

**These are derivation objects, not methods.**  Nothing here is registered in `registry.py` and
nothing here may be used to build a reported interval: the ablations deliberately use population
quantities (`beta`, `sigma2`, `mu_f`) that no practitioner has.  They live in `src/` rather than
in a script because `11` §Repository structure forbids a variance formula from appearing in
`experiments/`, and because C4 requires each formula to exist in exactly one place.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats

__all__ = ["exact_coverage_integral", "population_share", "idealised_coverage",
           "ablation_coverages", "ABLATIONS"]

#: The four statistics compared, from the practical one to the idealised one.  Each row removes
#: one idealisation, so consecutive differences attribute the total error to a single cause.
ABLATIONS = ("A0_practical", "A1_no_product", "A2_no_W_dependence", "A3_idealised")


def exact_coverage_integral(q: float, nu: float, s) -> np.ndarray:
    """`P(|T| <= q)` for the idealised law of Theorem B, `T = Z/sqrt((1-s)Q/nu + s)`.

        P(|T| <= q)  =  E_Q[ 2*Phi(q*sqrt((1-s)Q/nu + s)) - 1 ],   Q ~ chi2_nu

    Exact (a one-dimensional integral), vectorised over `s`.  `s = 0` gives `P(|t_nu| <= q)` and
    `s = 1` gives `2*Phi(q) - 1`.
    """
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
    out = np.empty_like(s_arr)
    for i, sv in enumerate(s_arr):
        sv = float(np.clip(sv, 0.0, 1.0))
        f = lambda x: ((2 * stats.norm.cdf(q * np.sqrt((1 - sv) * x / nu + sv)) - 1)
                       * stats.chi2.pdf(x, nu))
        out[i] = integrate.quad(f, 0.0, nu * 60.0, limit=400)[0]
    return out if np.ndim(s) else out[0]


def population_share(S_ff, fbar_n, *, n: int, N: float, mu_f: float, beta: float,
                     sigma2: float, sigma2_f: float):
    """`s = v2/(v1+v2)`, the share of the total variance carried by the component estimated
    without error, as it enters Theorem B.

        v1 = sigma^2 (1/n + h),  h = (mu_f - fbar_n)^2 / S_ff      (the t-pivotal part)
        v2 = beta^2 sigma_f^2 / N                                  (the known-variance part)

    A **population** quantity: it uses `beta` and `sigma2`, not `b` and `s2`.  The plug-in that
    `R2`/`R5` actually use is `b^2 sigma_hat_f^2 / N / V_hat`, reported by the
    `unlabeled_share_mean` metric.
    """
    h = (np.asarray(fbar_n) - mu_f) ** 2 / np.asarray(S_ff)
    v1 = sigma2 * (1.0 / n + h)
    v2 = beta ** 2 * sigma2_f / N
    return v2 / (v1 + v2)


def idealised_coverage(S_ff, fbar_n, *, n: int, N: float, mu_f: float, beta: float,
                       sigma2: float, sigma2_f: float, alpha_level: float = 0.05,
                       grid: int = 401) -> float:
    """Coverage of the idealised statistic, marginalised over the labelled design.

    Theorem B gives coverage as a function of `s` alone, so this is `E_F[C(s(F))]` evaluated on
    the supplied design draws.  `C` is tabulated on a grid in `s` and interpolated, because the
    quadrature is the expensive part and `s` is smooth.
    """
    q = float(stats.t.ppf(1 - alpha_level / 2, n - 2))
    s = population_share(S_ff, fbar_n, n=n, N=N, mu_f=mu_f, beta=beta, sigma2=sigma2,
                         sigma2_f=sigma2_f)
    s_grid = np.linspace(0.0, min(1.0, float(np.max(s)) * 1.05 + 1e-9), grid)
    c_grid = exact_coverage_integral(q, n - 2, s_grid)
    return float(np.mean(np.interp(s, s_grid, c_grid)))


def ablation_coverages(rng: np.random.Generator, *, n: int, N: float, rho: float, R: int,
                       alpha_level: float = 0.05, chunk_elements: int = 20_000_000) -> dict:
    """Coverage of each statistic in `ABLATIONS`, on **common random numbers**.

    All four use the identical draws of `(Z0, Z1, W, Q, F)`, so their differences are attributable
    to the idealisation removed and not to Monte-Carlo noise.  Under T1.1 with `Var(f)=Var(Y)=1`
    and `E[Y]=E[f]=0`: `beta = rho`, `sigma2 = 1 - rho^2`, `mu_f = 0`, `sigma2_f = 1`.

    Writing `u = (mu_f - fbar_n)/sqrt(S_ff)` and `kappa = sigma_f/(sqrt(N) sqrt(S_ff))`, the
    numerator is exactly

        X = sigma[ Z0/sqrt(n) + (u + kappa W) Z1 ] + (beta sigma_f/sqrt(N)) W

    and the practical variance estimate is `s^2 (1/n + (u + kappa W)^2) + b^2 sigma_f^2/N`.
    The ablations are, in order:

    * `A0_practical`    — exactly the above (what B9 computes).
    * `A1_no_product`   — drop `kappa W Z1` from the numerator only.  Removes the variance
      `v3 = sigma^2 sigma_f^2/(N S_ff)`, of relative order `1/N`.
    * `A2_no_W_dependence` — additionally replace `(u + kappa W)^2` by `u^2 + kappa^2` in the
      variance estimate, i.e. by its conditional expectation.  Removes the dependence between the
      denominator and `W`, which appears in the numerator.
    * `A3_idealised`    — additionally replace `b^2` by `beta^2` and drop `kappa^2` from the
      leverage.  This is exactly Theorem B's statistic.
    """
    beta, sigma2, sigma2_f, mu_f = rho, 1.0 - rho ** 2, 1.0, 0.0
    sigma, nu = np.sqrt(sigma2), n - 2
    q = float(stats.t.ppf(1 - alpha_level / 2, nu))
    counts = {k: 0 for k in ABLATIONS}
    # The ablations share their draws, so the PAIRED difference A0 - A3 has far smaller
    # Monte-Carlo error than either coverage: track it directly rather than differencing two
    # noisy numbers.
    d_sum = d_sq = 0.0
    s_sum, done = 0.0, 0
    chunk = max(1, chunk_elements // max(n, 1))

    while done < R:
        m = min(chunk, R - done)
        f = rng.standard_normal((m, n))                      # sigma_f = 1, mu_f = 0
        fbar_n = f.mean(1)
        S_ff = ((f - fbar_n[:, None]) ** 2).sum(1)
        Z0, Z1, W = (rng.standard_normal(m) for _ in range(3))
        Q = rng.chisquare(nu, m)

        u = (mu_f - fbar_n) / np.sqrt(S_ff)
        kappa = np.sqrt(sigma2_f / (N * S_ff))
        s2 = sigma2 * Q / nu
        b = beta + sigma * Z1 / np.sqrt(S_ff)
        a2 = beta * np.sqrt(sigma2_f / N)

        core = sigma * (Z0 / np.sqrt(n) + u * Z1) + a2 * W   # numerator without the product term
        X_act = core + sigma * kappa * W * Z1
        lev_act = (u + kappa * W) ** 2

        V_A0 = s2 * (1 / n + lev_act) + b ** 2 * sigma2_f / N
        V_A2 = s2 * (1 / n + u ** 2 + kappa ** 2) + b ** 2 * sigma2_f / N
        V_A3 = s2 * (1 / n + u ** 2) + beta ** 2 * sigma2_f / N
        ind = {}
        for key, X, V in (("A0_practical", X_act, V_A0), ("A1_no_product", core, V_A0),
                          ("A2_no_W_dependence", core, V_A2), ("A3_idealised", core, V_A3)):
            ind[key] = np.abs(X) <= q * np.sqrt(V)
            counts[key] += int(np.count_nonzero(ind[key]))
        d = ind["A0_practical"].astype(np.int8) - ind["A3_idealised"].astype(np.int8)
        d_sum += float(d.sum()); d_sq += float((d.astype(np.float64) ** 2).sum())

        s_sum += float(np.sum(population_share(S_ff, fbar_n, n=n, N=N, mu_f=mu_f, beta=beta,
                                               sigma2=sigma2, sigma2_f=sigma2_f)))
        done += m

    out = {k: counts[k] / R for k in ABLATIONS}
    out["mean_share"] = s_sum / R
    out["mc_se"] = float(np.sqrt(0.95 * 0.05 / R))
    out["diff_A0_A3"] = d_sum / R
    out["diff_mc_se"] = float(np.sqrt(max(d_sq / R - (d_sum / R) ** 2, 0.0) / R))
    return out
