r"""Published small-sample PPI repairs, implemented from their primary sources.

Every formula below is transcribed from the paper named beside it in `applicability.py`. Where a
paper leaves something unspecified (an alpha grid; an interval construction), that gap is marked
`DEVIATION` in a comment and recorded in the validation report — it is never filled silently.

All functions are vectorised over the leading replication axis, as the rest of the harness is.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ppi_finite.confidence_intervals import ci_from_se
from ppi_finite.types import IntervalResult

__all__ = ["ridge_lambda", "method_ridge_ppi", "method_ppboot", "sigmoid_fit",
           "method_sigmoid_ppi", "ALPHA_GRID_MULTIPLIERS"]

#: DEVIATION. Eyre & Madras say alpha is chosen "by cross validation over the labelled set" but do
#: not give the grid searched. We search a scale-free log grid in units of Var_hat[f], which makes
#: the choice invariant to the scale of the predictor, and we include 0 so that PPI++ itself is
#: always in the candidate set. alpha is chosen by leave-one-out prediction error, never by
#: coverage.
ALPHA_GRID_MULTIPLIERS = np.concatenate([[0.0], np.logspace(-3, 3, 25)])


def _moments(Y, f):
    """ddof = 1 labelled-sample moments, matching the harness's PPI++ convention."""
    Y = np.asarray(Y, float)
    f = np.asarray(f, float)
    n = Y.shape[-1]
    Yc = Y - Y.mean(-1, keepdims=True)
    fc = f - f.mean(-1, keepdims=True)
    cov = (Yc * fc).sum(-1) / (n - 1)
    var = (fc * fc).sum(-1) / (n - 1)
    return cov, var, n


def _loocv_alpha(Y, f, grid_mult):
    r"""Choose alpha by leave-one-out cross-validation of the post-hoc regression on the labelled
    set, as Eyre & Madras Section 5.2 specifies.

    The regression being cross-validated is theirs: predict `h(X_i)` from `lambda*f(X_i) + b`,
    where `b` is the intercept `mean(h) - lambda*mean(f)` (their Section 4.1). The selection
    criterion is squared prediction error -- **not** coverage, and not interval width.
    """
    Y = np.asarray(Y, float)
    f = np.asarray(f, float)
    R, n = Y.shape
    _, var_all, _ = _moments(Y, f)
    grid = var_all[:, None] * grid_mult[None, :]                 # (R, A), scale-free
    err = np.zeros((R, grid.shape[1]))
    for i in range(n):                                           # leave out point i
        keep = np.ones(n, bool)
        keep[i] = False
        Yk, fk = Y[:, keep], f[:, keep]
        m = keep.sum()
        Yc = Yk - Yk.mean(-1, keepdims=True)
        fc = fk - fk.mean(-1, keepdims=True)
        cov_k = (Yc * fc).sum(-1) / (m - 1)
        var_k = (fc * fc).sum(-1) / (m - 1)
        lam = cov_k[:, None] / (var_k[:, None] + grid)           # Eq. (8) on the retained points
        b = Yk.mean(-1)[:, None] - lam * fk.mean(-1)[:, None]    # Section 4.1 intercept
        pred = lam * f[:, i][:, None] + b
        err += (Y[:, i][:, None] - pred) ** 2
    best = np.argmin(err, axis=1)
    return grid[np.arange(R), best], best


def ridge_lambda(Y, f, N, grid_mult=ALPHA_GRID_MULTIPLIERS):
    r"""Eyre & Madras Eq. (8) with the Section 4.4 scaling.

        lambda_hat_alpha = Cov_hat[h, f] / (Var_hat[f] + alpha),  then scaled by (1 + n/N)^-1.
    """
    cov, var, n = _moments(Y, f)
    alpha, idx = _loocv_alpha(Y, f, grid_mult)
    lam = cov / (var + alpha)
    r = 0.0 if not np.isfinite(N) else n / N
    return lam / (1.0 + r), alpha, idx


def method_ridge_ppi(data, alpha_level=0.05, rng=None, summaries=None) -> IntervalResult:
    r"""Ridge-PPI (Eyre & Madras 2024), with OUR interval.

    Point estimate and lambda are exactly theirs. The interval is the PPI++ interval with
    `lambda_hat_alpha` substituted -- **DEVIATION**: the paper defines no interval at all
    ('confidence interval', 'coverage' and 'CI' appear zero times in it), so no coverage number
    for this method is a claim its authors made.
    """
    Y = np.asarray(data.Y, float)
    f = np.asarray(data.f, float)
    lam, alpha, _ = ridge_lambda(Y, f, data.N)
    theta = Y.mean(-1) - lam * (f.mean(-1) - np.asarray(data.fbar_N, float))
    n = Y.shape[-1]
    mean_term = (Y - lam[..., None] * f).var(axis=-1, ddof=1) / n
    unl = 0.0 if not np.isfinite(data.N) else lam**2 * f.var(axis=-1, ddof=1) / data.N
    se = np.sqrt(mean_term + unl)
    low, high = ci_from_se(theta, se, alpha_level=alpha_level, dist="normal")
    return IntervalResult("Ridge-PPI", theta, low, high, se, np.inf,
                          extras={"lambda_mean": float(np.mean(lam)),
                                  "alpha_mean": float(np.mean(alpha)),
                                  "interval_source": "OURS (paper defines none)"})


def method_ppboot(data, alpha_level=0.05, rng=None, summaries=None, B: int = 2000,
                  pool=None) -> IntervalResult:
    r"""PPBoot (Zrnic, Algorithm 1) — exact, for theta_hat = the sample mean.

        theta*_b = mean(f~*) + mean(Y*) - mean(f*),  percentile interval.

    Both the labelled sample and the unlabelled pool are resampled with replacement, as the
    algorithm requires; `pool` must therefore be the materialised pool.
    """
    if pool is None:
        raise ValueError("PPBoot resamples the unlabelled pool, so the pool must be materialised "
                         "(dgp_gaussian(..., keep_pool=True)). It is not approximated.")
    if rng is None:
        raise ValueError("PPBoot is stochastic and requires an explicit generator")
    Y = np.asarray(data.Y, float)
    f = np.asarray(data.f, float)
    pool = np.asarray(pool, float)
    R, n = Y.shape
    N = pool.shape[-1]

    theta_star = np.empty((R, B))
    for b in range(B):
        li = rng.integers(0, n, size=(R, n))                      # labelled resample, paired
        ui = rng.integers(0, N, size=(R, N))                      # pool resample
        Ys = np.take_along_axis(Y, li, axis=1).mean(-1)
        fs = np.take_along_axis(f, li, axis=1).mean(-1)
        us = np.take_along_axis(pool, ui, axis=1).mean(-1)
        theta_star[:, b] = us + Ys - fs
    low = np.quantile(theta_star, alpha_level / 2, axis=1)
    high = np.quantile(theta_star, 1 - alpha_level / 2, axis=1)
    theta = pool.mean(-1) + Y.mean(-1) - f.mean(-1)
    se = theta_star.std(axis=1, ddof=1)
    return IntervalResult("PPBoot", theta, low, high, se, np.inf,
                          extras={"B": B, "interval": "percentile (Algorithm 1)"})


def sigmoid_fit(Y, f, l2_grid=None, iters: int = 60):
    r"""Fit g(f) = 1 / (1 + exp(-alpha*f + beta)) on the labelled set (Eyre & Madras Eq. 9).

    Newton/IRLS on the L2-penalised binomial log-likelihood -- i.e. logistic regression of the
    binary label on the single predictor `f`, which is what a sigmoid link with learned
    (alpha, beta) is. The penalty is chosen by leave-one-out cross-validation, as Section 5.2
    specifies.
    """
    Y = np.asarray(Y, float)
    f = np.asarray(f, float)
    if l2_grid is None:
        l2_grid = np.concatenate([[0.0], np.logspace(-3, 2, 12)])

    def _fit(Yt, ft, lam2):
        R, m = Yt.shape
        a = np.zeros(R)
        b = np.zeros(R)
        for _ in range(iters):
            z = a[:, None] * ft - b[:, None]
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            w = np.clip(p * (1 - p), 1e-9, None)
            r = Yt - p
            g_a = (r * ft).sum(-1) - lam2 * a
            g_b = -r.sum(-1) - lam2 * b
            h_aa = (w * ft * ft).sum(-1) + lam2
            h_ab = -(w * ft).sum(-1)
            h_bb = w.sum(-1) + lam2
            det = np.clip(h_aa * h_bb - h_ab * h_ab, 1e-12, None)
            a += (h_bb * g_a - h_ab * g_b) / det
            b += (h_aa * g_b - h_ab * g_a) / det
        return a, b

    R, n = Y.shape
    err = np.zeros((R, len(l2_grid)))
    for i in range(n):
        keep = np.ones(n, bool)
        keep[i] = False
        for j, lam2 in enumerate(l2_grid):
            a, b = _fit(Y[:, keep], f[:, keep], lam2)
            p = 1.0 / (1.0 + np.exp(-np.clip(a * f[:, i] - b, -30, 30)))
            err[:, j] += (Y[:, i] - p) ** 2
    best = np.argmin(err, axis=1)
    out_a = np.empty(R)
    out_b = np.empty(R)
    for j, lam2 in enumerate(l2_grid):
        m = best == j
        if m.any():
            a, b = _fit(Y[m], f[m], lam2)
            out_a[m], out_b[m] = a, b
    return out_a, out_b, np.asarray(l2_grid)[best]


def method_sigmoid_ppi(data, alpha_level=0.05, rng=None, summaries=None,
                       pool=None) -> IntervalResult:
    r"""Sigmoid-PPI (Eyre & Madras Eqs. 9-10). Binary `h`, predictions in [0,1] only.

    **DEVIATION**: as with Ridge-PPI the paper gives no interval. We use the PPI++ interval form
    with `g(f)` in place of `lambda*f`, which is the direct analogue of their Eq. (10).
    """
    Y = np.asarray(data.Y, float)
    f = np.asarray(data.f, float)
    a, b, l2 = sigmoid_fit(Y, f)
    g_lab = 1.0 / (1.0 + np.exp(-np.clip(a[:, None] * f - b[:, None], -30, 30)))
    if pool is not None:
        p = np.asarray(pool, float)
        g_unl = (1.0 / (1.0 + np.exp(-np.clip(a[:, None] * p - b[:, None], -30, 30)))).mean(-1)
    else:
        raise ValueError("Sigmoid-PPI needs the pool predictions to evaluate g on them")
    n = Y.shape[-1]
    theta = g_unl + (Y - g_lab).mean(-1)
    mean_term = (Y - g_lab).var(axis=-1, ddof=1) / n
    unl = 0.0 if not np.isfinite(data.N) else g_lab.var(axis=-1, ddof=1) / data.N
    se = np.sqrt(mean_term + unl)
    low, high = ci_from_se(theta, se, alpha_level=alpha_level, dist="normal")
    return IntervalResult("Sigmoid-PPI", theta, low, high, se, np.inf,
                          extras={"l2_mean": float(np.mean(l2)),
                                  "interval_source": "OURS (paper defines none)"})


def method_efron_ppboot2(data, alpha_level=0.05, rng=None, summaries=None, B: int = 1000,
                         pool=None) -> IntervalResult:
    r"""Efron (2026) PPboot2, for `θ = E{y}` — the quantitative-response algorithm of his §5.

    His `modelq` (5.1) is `y_a(i) = μ_a(i) + σ_a(i)·ε(i)` with `μ = m(f)`, `σ = s(f)` and
    `ε ~ p` iid, mean 0 variance 1. Figure 4's steps, with `t(x,y) = mean(y)`:

        1. y*_a(i) = μ̂_a(i) + σ̂_a(i)·ε̂_i          (5.7)
        2. refit the model on (f_a, y*_a)
        3. π̂*_b = (m̂*(f_b), ŝ*(f_b), p̂)          (5.8)
        4. y*_b(i) = μ̂*_b(i) + σ̂*_b(i)·ε̂_i        (5.9)
        5. θ*_b = mean(y*_b)
        6. θ̂_b = mean_b θ*_b ,  sd_b = sd_b θ*_b   (3.2a, 3.2b)

    and the interval is `θ̂ ± bsd` (his Figure 3 discussion), taken here at the normal quantile.

    **Two documented adaptations**, both forced by our homoskedastic Gaussian model and both
    recorded in `applicability.py`:

    * `ŝ`. Efron estimates a *varying* `s(f)` from successive differences of the responses
      (Remark 13). Model T1.1 is homoskedastic, so `s(f) ≡ σ` is constant and the corresponding
      low-dimensional estimate is the residual standard deviation on `n−2` degrees of freedom.
      A successive-difference estimator would estimate the same constant less efficiently.
    * `p̂`. Taken as the empirical distribution of the standardised residuals, which is the
      standard reading of "ε̂_i iid ~ p̂"; the paper does not pin the estimator down further.

    For the mean, step 4 does **not** require materialising `y*_b`: `mean(y*_b) = â₀* + â₁*·f̄_N
    + σ̂*·mean(ε*)`, and `mean(ε*)` over `N` draws from an `n`-atom distribution is obtained
    exactly from a multinomial over those atoms. That is an algebraic simplification of the
    published step, not an approximation of it.
    """
    if rng is None:
        raise ValueError("PPboot2 is stochastic and requires an explicit generator")
    Y = np.asarray(data.Y, float)
    f = np.asarray(data.f, float)
    R, n = Y.shape
    N = float(data.N)
    fbar_N = np.asarray(data.fbar_N, float)

    # step 0: fit modelq on the labelled data -- m by least squares, s as the residual sd
    fbar = f.mean(-1, keepdims=True)
    Ybar = Y.mean(-1, keepdims=True)
    fc = f - fbar
    S_ff = (fc * fc).sum(-1)
    slope = np.where(S_ff > 0, ((Y - Ybar) * fc).sum(-1) / np.where(S_ff > 0, S_ff, 1.0), 0.0)
    inter = Ybar[..., 0] - slope * fbar[..., 0]
    resid = Y - (inter[:, None] + slope[:, None] * f)
    sigma = np.sqrt(np.maximum((resid * resid).sum(-1) / max(n - 2, 1), 1e-300))
    eps_hat = resid / np.where(sigma[:, None] > 0, sigma[:, None], 1.0)     # p-hat's atoms
    eps_hat = eps_hat - eps_hat.mean(-1, keepdims=True)                     # p has mean 0 (5.3)

    theta_star = np.empty((R, B))
    rows = np.arange(R)[:, None]
    for b in range(B):
        # step 1: y*_a from the fitted model
        j = rng.integers(0, n, size=(R, n))
        e_a = eps_hat[rows, j]
        Ys = inter[:, None] + slope[:, None] * f + sigma[:, None] * e_a
        # step 2: refit on (f_a, y*_a)
        Ys_bar = Ys.mean(-1, keepdims=True)
        slope_s = np.where(S_ff > 0, ((Ys - Ys_bar) * fc).sum(-1) / np.where(S_ff > 0, S_ff, 1.0),
                           0.0)
        inter_s = Ys_bar[..., 0] - slope_s * fbar[..., 0]
        res_s = Ys - (inter_s[:, None] + slope_s[:, None] * f)
        sigma_s = np.sqrt(np.maximum((res_s * res_s).sum(-1) / max(n - 2, 1), 1e-300))
        # steps 3-5: mean of y*_b over the N unlabelled points, exactly
        counts = rng.multinomial(int(N), np.full(n, 1.0 / n), size=R)
        mean_eps = (counts * eps_hat).sum(-1) / N
        theta_star[:, b] = inter_s + slope_s * fbar_N + sigma_s * mean_eps

    theta = theta_star.mean(axis=1)                                          # (3.2a)
    sd = theta_star.std(axis=1, ddof=1)                                      # (3.2b)
    z = stats.norm.ppf(1 - alpha_level / 2)
    return IntervalResult("Efron-PPboot2", theta, theta - z * sd, theta + z * sd, sd, np.inf,
                          extras={"B": B, "interval": "theta_hat +- z*bsd (Efron Fig. 3)"})
