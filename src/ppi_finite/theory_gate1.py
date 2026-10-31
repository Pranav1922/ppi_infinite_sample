r"""Gate 1 — a rigorous finite-`N` coverage-error bound for the deployed LC-PPI interval.

Part II of `docs/theory_finite_N.md` derived the rate `|Δ| = O(1/N)` by a formal second-order
expansion whose remainder was not controlled, and paired it with an *empirically fitted* constant
(`K ≈ 0.43`). This module replaces both with a proof.

The proof rests on two **exact** conditional reductions (`reduction_1`, `reduction_2` below). In
the coordinates they produce, every perturbation separating the practical statistic from the
idealised one is already `O(1/N)` -- so a first-order Lipschitz argument suffices and no Stein,
anti-concentration or Berry-Esseen step is needed. The apparent `O(N^{-1/2})` size of the
perturbation is an artefact of comparing `T_act` with `T_id` directly instead of comparing their
conditional coverage functions.

Notation follows `docs/theory_finite_N.md` §II.1 exactly:

    u = (μ_f − f̄_n)/√S_ff      κ = σ_f/(√N·√S_ff)      c = βσ_f/√N      λ = c/σ
    g_act = 1/n + (u+κW)²       g_id = 1/n + u²          ν = n−2
    Z₀, Z₁, W ~ N(0,1),  Q ~ χ²_ν,  σ̂_f² = σ_f²·χ²_{N−1}/(N−1),  all mutually independent

All bounds are stated for the working model M1 and `n ≥ 4`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

__all__ = ["psi", "reduction_1", "reduction_2", "CoverageBound", "coverage_bound",
           "inv_sqrt_chi2_moment", "inv_sqrt_shifted_chi2_moment", "PHI0"]

PHI0 = 1.0 / np.sqrt(2.0 * np.pi)        # φ(0); sup|ψ_x| = 2φ(0), sup|ψ_y| = φ(0)
SQRT_2_PI = np.sqrt(2.0 / np.pi)         # E|W| for W ~ N(0,1); E|W|³ = 2·E|W|


def psi(x, y):
    """ψ(x, y) = P(|Z + y| ≤ x) for Z ~ N(0,1) = Φ(x − y) − Φ(−x − y).

    Bounded with bounded derivatives: |ψ_x| ≤ 2φ(0), |ψ_y| ≤ φ(0), uniformly in (x, y). Those two
    constants are the only smoothness input the proof uses.
    """
    return stats.norm.cdf(x - y) - stats.norm.cdf(-x - y)


def inv_sqrt_chi2_moment(df: float) -> float:
    """E[(χ²_k)^{−1/2}] = Γ((k−1)/2) / (√2 · Γ(k/2)); finite iff k > 1."""
    if df <= 1:
        return np.inf
    return np.exp(special.gammaln((df - 1) / 2) - special.gammaln(df / 2)) / np.sqrt(2.0)


def reduction_1(t, n, sigma, V, m):
    """Exact P(|T| ≤ t | F, Q, W, Z₁, σ̂_f²), obtained by integrating Z₀ out.

    Valid for any statistic whose denominator is measurable w.r.t. the conditioning -- in
    particular for `T_act`, whose `V̂` depends on Z₁ through `b²`.
    """
    return psi(t * np.sqrt(n * V) / sigma, np.sqrt(n) * m / sigma)


def reduction_2(t, nu, Q, lam, g, W):
    """Exact P(|T| ≤ t | F, Q, W), obtained by integrating (Z₀, Z₁) out jointly.

    Valid only when `V̂` is free of Z₁ -- i.e. for `T_mid` and `T_id`, not for `T_act`. σ cancels
    identically here, which is why this parametrisation exposes the `O(1/N)` perturbation sizes.
    """
    return psi(t * np.sqrt(Q / nu + lam**2 / g), lam * W / np.sqrt(g))


def _quantile_nodes(dist, k: int = 600):
    """Gauss-Legendre nodes/weights transported to a distribution: E[h(X)] = ∫₀¹ h(F⁻¹(p)) dp."""
    nodes, weights = np.polynomial.legendre.leggauss(k)
    return dist.ppf(0.5 * (nodes + 1.0)), 0.5 * weights


def denominator_moment(nu: float, n: int, lam: float) -> float:
    r"""`E[ 1 / (√(Q/(νn)) + √(Q/(νn) + λ²)) ]` for `Q ~ χ²_ν`, by quadrature.

    This is the exact `Q`-expectation appearing in the term-B bound. Keeping `λ²` rather than
    discarding it matters: `λ` grows as `ρ → 1`, exactly where dropping it would cost the bound a
    factor `1/σ`; keeping the first square root preserves sharpness at `λ = 0`.
    """
    q, w = _quantile_nodes(stats.chi2(nu))
    a = np.sqrt(q / (nu * n))
    return float(np.sum(w / (a + np.sqrt(a**2 + lam**2))))


def design_moments(n: int, N: float, sigma_f: float = 1.0) -> tuple[float, float]:
    r"""`(E[κ|u|/g_id], E[κ²/g_id])` over the labelled design, by 2-D quadrature.

    With `A = μ_f − f̄_n ~ N(0, σ_f²/n)` independent of `S = S_ff ~ σ_f²·χ²_{n−1}`, both reduce to
    expectations of explicit functions of `(A, S)`:

        κ|u|/g_id = σ_f·|A| / (√N·(S/n + A²)),        κ²/g_id = σ_f² / (N·(S/n + A²)).

    Evaluating these exactly beats bounding `|u|/g_id ≤ √n/2` and `1/g_id ≤ n`, which is where the
    algebraic route loses most of its sharpness at large `n`. Both are finite for `n ≥ 4`.
    """
    a, wa = _quantile_nodes(stats.norm(0.0, sigma_f / np.sqrt(n)), k=400)
    ss, ws = _quantile_nodes(stats.chi2(n - 1, scale=sigma_f**2), k=400)
    denom = ss[None, :] / n + a[:, None] ** 2
    weight = wa[:, None] * ws[None, :]
    m_abs = float(np.sum(weight * np.abs(a)[:, None] / denom)) * sigma_f / np.sqrt(N)
    m_sq = float(np.sum(weight / denom)) * sigma_f**2 / N
    return m_abs, m_sq


@dataclass(frozen=True)
class CoverageBound:
    """A rigorous bound on |coverage_act − coverage_id|, split by which idealisation it removes."""
    n: int
    rho: float
    N: float
    alpha: float
    bound_B: float          # b²σ̂_f²/N vs β²σ_f²/N in the denominator
    bound_PD: float         # the product term and the leverage's dependence on W
    total: float
    C: float                # N × total: the explicit constant, → finite as N → ∞

    def holds_for(self, delta: float) -> bool:
        return abs(delta) <= self.total


def coverage_bound(n: int, rho: float, N: float, alpha: float = 0.05,
                   sigma_f: float = 1.0, var_Y: float = 1.0,
                   sigma_f_known: bool = False) -> CoverageBound:
    r"""Rigorous bound on `|coverage_act − coverage_id|` under M1, for `n ≥ 4`.

    Parametrised as the simulations are: `β = ρ·σ_Y/σ_f`, `σ² = (1−ρ²)·var_Y`.

    `sigma_f_known=True` drops the `σ̂_f²` estimation term (the `mean_only` pool mode).

    Returns a `CoverageBound`. The bound is **valid but not tight**: it is a sum of absolute
    values, so it cannot exploit the cancellation between the three idealisations documented in
    §II.5, and it degrades as `ρ → 1` because the proof divides by `σ`.
    """
    if n < 4:
        raise ValueError("the bound requires n >= 4: E[1/S_ff] and E[Q^{-1/2}] diverge below it")
    nu = n - 2
    sigma = np.sqrt((1.0 - rho**2) * var_Y)
    beta = rho * np.sqrt(var_Y) / sigma_f
    t = stats.t.ppf(1 - alpha / 2, nu)

    if sigma <= 0:
        raise ValueError("the bound requires rho^2 < 1: the proof divides by sigma")

    # moments of the primitives; S_ff = σ_f²·χ²_{n−1}, Q ~ χ²_ν, both independent of everything
    E_inv_sqrt_Q = inv_sqrt_chi2_moment(nu)
    E_inv_sqrt_Sff = inv_sqrt_chi2_moment(n - 1) / sigma_f
    E_inv_Sff = 1.0 / (sigma_f**2 * (n - 3))
    lam = abs(beta) * sigma_f / (sigma * np.sqrt(N))
    m_abs, m_sq = design_moments(n, N, sigma_f)      # E[κ|u|/g_id], E[κ²/g_id]

    # --- term B ------------------------------------------------------------------------------
    # V̂_mid ≥ s²/n + c², so |δx_B| ≤ t√n·|D_B| / (σ²·√(Q/(νn) + λ²)) with D_B = δV̂/N.
    E_abs_b2_minus_beta2 = (2 * abs(beta) * sigma * SQRT_2_PI * E_inv_sqrt_Sff
                            + sigma**2 * E_inv_Sff)
    E_abs_sighat_dev = 0.0 if sigma_f_known else sigma_f**2 * np.sqrt(2.0 / (N - 1))
    B_moment = sigma_f**2 * E_abs_b2_minus_beta2 + beta**2 * E_abs_sighat_dev
    G = denominator_moment(nu, n, lam)
    bound_B = 2 * PHI0 * t * np.sqrt(n) / (sigma**2 * N) * G * B_moment

    # --- terms P and D -----------------------------------------------------------------------
    # |δy| ≤ λ|W|(2|u|κ|W| + κ²W²)·√n/g_id
    E_abs_dy = lam * np.sqrt(n) * (2 * m_abs + 2 * SQRT_2_PI * m_sq)
    # |δx_PD| ≤ t·λ²·(2|u|κ|W| + κ²W²)·n / (2·g_id·√(Q/ν))
    E_abs_dx_PD = (t * lam**2 * n * np.sqrt(nu) * E_inv_sqrt_Q
                   * (2 * SQRT_2_PI * m_abs + m_sq) / 2)
    bound_PD = 2 * PHI0 * E_abs_dx_PD + PHI0 * E_abs_dy

    total = min(bound_B + bound_PD, 1.0)   # a coverage difference never exceeds 1
    return CoverageBound(n=n, rho=rho, N=N, alpha=alpha, bound_B=bound_B,
                         bound_PD=bound_PD, total=total, C=total * N)
