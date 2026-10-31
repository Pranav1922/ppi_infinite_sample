"""Data production: the Gaussian DGP T1.1, and `run_cell` (the common-random-numbers gate).

`08` §T1.1–T1.4.  Scope: `dgp_gaussian` in both `mean_only` and `explicit` unlabelled mode (E4),
plus the robustness families `dgp_heavy_tail` (T1.2), `dgp_skew` (T1.3) and
`dgp_heteroskedastic` (T1.4), added by E5.  T1.5 (misspecification) is optional and not built.

**All robustness DGPs standardise the residual to mean 0 and unit variance**, so that `Var(Y) = 1`
and width comparisons stay on the same scale as T1.1 (`08` §T1.4 requires this explicitly for the
heteroskedastic family; it is applied to every family for comparability).
"""
from __future__ import annotations

import time

import numpy as np

from .estimators import labelled_summaries
from .metrics import evaluate_all
from .registry import METHODS, STOCHASTIC_METHODS
from scipy import stats

from .types import CellSpec, PPIData, make_data

__all__ = ["dgp_gaussian", "simulate", "cell_rng", "run_cell", "MAX_POOL_ELEMENTS"]

#: Chunking budget for materialising the `(R, N)` unlabelled pool.  Chunking is stream-invariant
#: (successive `rng.standard_normal` calls consume the same values in the same order as one big
#: call), so no number depends on it; `tests/test_simulation.py` asserts that.
MAX_POOL_ELEMENTS = 8_000_000


def dgp_gaussian(rng: np.random.Generator, *, R: int, n: int, rho: float, N: float,
                 unlabeled_mode: str = "mean_only", var_Y: float = 1.0, var_f: float = 1.0,
                 mean_Y: float = 0.0, mean_f: float = 0.0, keep_pool: bool = False) -> PPIData:
    """T1.1 bivariate Gaussian (`08` §T1.1, `07` Experiment A).

        f   = mean_f + sd_f * Z1
        eps = Z2
        Y   = mean_Y + sd_Y * (rho*Z1 + sqrt(1-rho^2)*eps)

    so `Var(f)=var_f`, `Var(Y)=var_Y`, `Corr(Y,f)=rho`, and `θ = E[Y] = mean_Y`.

    Unlabelled pool, two modes (`08` §T1.1), which are **not** interchangeable:

    * **`mean_only`** — what produced the preliminary numbers.  Only `f̄_N` and `σ_f²` enter any
      estimator, and under T1.1 `f̄_N ~ N(mean_f, σ_f²/N)` exactly, so it is drawn directly and
      `σ̂_f²` is treated as **known** (= `var_f`).  Harmless at large N, not at `N = 200`;
      `09` §Provenance records it as a known limitation that Experiment A reproduces deliberately.
    * **`explicit`** — the N unlabelled predictions are actually drawn, `f̄_N` is their mean and
      `σ̂_f²` is **estimated** from them with `ddof = 1`.  Required for `N ≤ 5000` (`08` §T1.1)
      and used throughout Experiment E.  The pool is materialised in chunks over replications and
      only its two sufficient statistics are retained, because `(R, N)` does not fit in memory at
      the replication counts `07` specifies; pass `keep_pool=True` to keep it (tests only).

    `mu_f = mean_f` is attached to the returned `PPIData` in both modes.  It is used **only** by
    the diagnostic variance mode `v_full` (`04` §4.8.3), never by the estimator.
    """
    if unlabeled_mode not in ("mean_only", "explicit"):
        raise ValueError(f"unlabeled_mode must be 'mean_only' or 'explicit'; "
                         f"got {unlabeled_mode!r}")
    if not 0.0 <= abs(rho) <= 1.0:
        raise ValueError(f"rho must lie in [-1, 1]; got {rho}")
    sd_f, sd_Y = np.sqrt(var_f), np.sqrt(var_Y)

    Z1 = rng.standard_normal((R, n))
    Z2 = rng.standard_normal((R, n))
    f = mean_f + sd_f * Z1
    Y = mean_Y + sd_Y * (rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2)

    if unlabeled_mode == "mean_only":
        fbar_N = mean_f + sd_f * rng.standard_normal(R) / np.sqrt(N)
        return make_data(Y, f, fbar_N=fbar_N, N=N, sigma2_f=var_f, theta_true=mean_Y,
                         mu_f=mean_f)

    N_int = int(N)
    if N_int < 2:
        raise ValueError(f"explicit mode needs N >= 2 to estimate sigma_f^2; got {N}")
    if keep_pool:
        # The whole pool is retained and `make_data` derives f̄_N and σ̂_f² from it. Only for
        # small (R, N): this is the path tests use to prove the chunked path agrees with it.
        pool = mean_f + sd_f * rng.standard_normal((R, N_int))
        return make_data(Y, f, f_unlabeled=pool, theta_true=mean_Y, mu_f=mean_f)

    fbar_N = np.empty(R, dtype=np.float64)
    sigma2_hat = np.empty(R, dtype=np.float64)
    chunk = max(1, MAX_POOL_ELEMENTS // N_int)
    for start in range(0, R, chunk):
        stop = min(start + chunk, R)
        pool = mean_f + sd_f * rng.standard_normal((stop - start, N_int))
        fbar_N[start:stop] = pool.mean(axis=-1)
        sigma2_hat[start:stop] = pool.var(axis=-1, ddof=1)
    return make_data(Y, f, fbar_N=fbar_N, N=N, sigma2_f=sigma2_hat, theta_true=mean_Y,
                     mu_f=mean_f)


def _unlabeled(rng, *, R, N, unlabeled_mode, mean_f, var_f, keep_pool=False):
    """The unlabelled pool, shared by every DGP.  `f` is Gaussian in T1.2-T1.4 except in the
    heavy-`f` variant, which passes its own draw function."""
    sd_f = np.sqrt(var_f)
    if unlabeled_mode == "mean_only":
        return dict(fbar_N=mean_f + sd_f * rng.standard_normal(R) / np.sqrt(N), N=N,
                    sigma2_f=var_f)
    raise NotImplementedError("robustness DGPs use mean_only; explicit is E4's Experiment E")


def dgp_heavy_tail(rng: np.random.Generator, *, R: int, n: int, rho: float, N: float,
                   unlabeled_mode: str = "mean_only", nu: float = 5.0,
                   heavy_f: bool = False) -> PPIData:
    """T1.2 — heavy tails (`08` §T1.2, `07` Experiment F).

        eps = t_nu / sqrt(nu/(nu-2))          standardised to unit variance
        Y   = rho*f + sqrt(1-rho^2)*eps

    With `heavy_f=True` the **predictor** is drawn from the same standardised `t_nu` family
    instead of a Gaussian, which is where the leverage term is most exposed (`08` §T1.2): the
    leverage `(f̄_N − f̄_n)²/S_ff` depends directly on the tails of `f`.

    `nu > 2` is required for the variance to exist; `nu > 4` for the kurtosis.  The unlabelled
    pool mean is still drawn as `N(mean_f, var_f/N)` — at `N = 10⁶` the CLT makes the pool mean
    Gaussian to far beyond any resolution here, and this keeps `f̄_N` comparable across `nu`.
    """
    if nu <= 2:
        raise ValueError(f"nu must exceed 2 for a finite variance; got {nu}")
    scale = np.sqrt(nu / (nu - 2.0))
    if heavy_f:
        f = rng.standard_t(nu, (R, n)) / scale
    else:
        f = rng.standard_normal((R, n))
    eps = rng.standard_t(nu, (R, n)) / scale
    Y = rho * f + np.sqrt(1.0 - rho ** 2) * eps
    return make_data(Y, f, theta_true=0.0, mu_f=0.0,
                     **_unlabeled(rng, R=R, N=N, unlabeled_mode=unlabeled_mode,
                                  mean_f=0.0, var_f=1.0))


def _standardised_skew(rng, shape, family: str):
    """Centred, unit-variance draws from the skew families of `08` §T1.3.

    `lognormal_mild` uses `sigma = 0.5` (skewness ~1.75) and `lognormal_severe` `sigma = 1.0`
    (skewness ~6.2).  `08` says "mild and severe" without fixing values; these are an
    **implementation decision** (`12` §What to do when you are stuck), logged in `13`.
    """
    if family == "exponential":
        return rng.exponential(1.0, shape) - 1.0                      # mean 1, var 1
    if family.startswith("gamma"):
        k = float(family[5:])
        return (rng.gamma(k, 1.0, shape) - k) / np.sqrt(k)            # mean k, var k
    if family.startswith("lognormal"):
        sig = {"lognormal_mild": 0.5, "lognormal_severe": 1.0}[family]
        x = rng.lognormal(0.0, sig, shape)
        m = np.exp(sig ** 2 / 2)
        v = (np.exp(sig ** 2) - 1) * np.exp(sig ** 2)
        return (x - m) / np.sqrt(v)
    raise ValueError(f"unknown skew family {family!r}")


def _binary_params(prevalence: float, rho_target: float):
    """Solve for `(a, b)` with `Y ~ Bernoulli(sigmoid(a + b f))`, `f ~ N(0,1)`, hitting a target
    prevalence and target `Corr(Y, f)` (`08` §T1.3).

    Returns `(a, b, achieved_rho)`.  **The target is not always achievable**: with `f` Gaussian
    and `Y` binary at prevalence 0.5 the correlation is capped at `2φ(0) ≈ 0.798` (attained by the
    deterministic threshold rule), so `07`'s `ρ = 0.9` cell cannot exist for the binary DGP.
    `07` anticipates this — "or the closest achievable correlation for the discrete case" — and the
    achieved value is what is returned and reported.
    """
    from scipy import integrate, optimize, special
    B_MAX = 40.0                       # effectively the deterministic threshold rule

    def moments(a, b):
        g = lambda z: special.expit(a + b * z)          # numerically stable sigmoid
        p = integrate.quad(lambda z: g(z) * stats.norm.pdf(z), -12, 12, limit=400)[0]
        c = integrate.quad(lambda z: z * g(z) * stats.norm.pdf(z), -12, 12, limit=400)[0]
        return p, c

    def solve_a(b):
        return optimize.brentq(lambda aa: moments(aa, b)[0] - prevalence, -80, 80, xtol=1e-13)

    def rho_of(b):
        a = solve_a(b)
        p, c = moments(a, b)
        return a, c / np.sqrt(p * (1 - p))

    # The ceiling is attained by the deterministic rule Y = 1{f > c}, c = Phi^{-1}(1-prevalence):
    # Cov(Y, f) = phi(c), Var(Y) = p(1-p).
    c_thr = stats.norm.ppf(1.0 - prevalence)
    rho_max = float(stats.norm.pdf(c_thr) / np.sqrt(prevalence * (1 - prevalence)))
    if rho_target <= 0.0:
        return float(special.logit(prevalence)), 0.0, 0.0
    if rho_target >= rho_max - 1e-4:
        a, achieved = rho_of(B_MAX)
        return a, B_MAX, achieved
    b = optimize.brentq(lambda bb: rho_of(bb)[1] - rho_target, 1e-6, B_MAX, xtol=1e-10)
    a, achieved = rho_of(b)
    return a, b, achieved


def dgp_skew(rng: np.random.Generator, *, R: int, n: int, rho: float, N: float,
             unlabeled_mode: str = "mean_only", skew_family: str = "exponential",
             prevalence: float = 0.5) -> PPIData:
    """T1.3 — skewed and discrete (`08` §T1.3, `07` Experiment G).

    Continuous families: `Y = rho*f + sqrt(1-rho^2)*eps` with `eps` a centred, unit-variance
    draw from the named family.  `skew_family="binary"` instead draws
    `Y ~ Bernoulli(sigmoid(a + b f))` — the case that matters for LLM annotation, where the gold
    label is a human binary judgement — and then `θ = E[Y] = prevalence`, not 0.
    """
    f = rng.standard_normal((R, n))
    if skew_family == "binary":
        a, b, achieved = _binary_params(prevalence, rho)
        Y = (rng.random((R, n)) < 1.0 / (1.0 + np.exp(-(a + b * f)))).astype(np.float64)
        theta = prevalence
    else:
        Y = rho * f + np.sqrt(1.0 - rho ** 2) * _standardised_skew(rng, (R, n), skew_family)
        theta = 0.0
    return make_data(Y, f, theta_true=theta, mu_f=0.0,
                     **_unlabeled(rng, R=R, N=N, unlabeled_mode=unlabeled_mode,
                                  mean_f=0.0, var_f=1.0))


def dgp_heteroskedastic(rng: np.random.Generator, *, R: int, n: int, rho: float, N: float,
                        unlabeled_mode: str = "mean_only", gamma: float = 0.5,
                        form: str = "exp") -> PPIData:
    """T1.4 — heteroskedasticity (`08` §T1.4, `07` Experiment H).

        Y = rho*f + sqrt(1-rho^2) * sigma(f) * eps / c

    with `sigma(f) = exp(gamma*f)` (`form="exp"`) or a two-group step
    `sigma(f) = 1 + gamma*1{f > 0}` (`form="step"`, "closer to realistic annotation settings where
    the model is reliable on easy items and noisy on hard ones", `08` §T1.4).  `c` normalises
    `E[sigma(f)^2] = 1` so that `Var(Y) = 1` and widths stay comparable.

    `gamma = 0` recovers T1.1 exactly, which is the control the experiment needs.
    """
    f = rng.standard_normal((R, n))
    if form == "exp":
        sig = np.exp(gamma * f)
        c = np.exp(gamma ** 2)                       # sqrt(E[exp(2 gamma f)]) = exp(gamma^2)
    elif form == "step":
        sig = 1.0 + gamma * (f > 0.0)
        c = np.sqrt(0.5 * 1.0 + 0.5 * (1.0 + gamma) ** 2)
    else:
        raise ValueError(f"form must be 'exp' or 'step'; got {form!r}")
    eps = rng.standard_normal((R, n))
    Y = rho * f + np.sqrt(1.0 - rho ** 2) * sig * eps / c
    return make_data(Y, f, theta_true=0.0, mu_f=0.0,
                     **_unlabeled(rng, R=R, N=N, unlabeled_mode=unlabeled_mode,
                                  mean_f=0.0, var_f=1.0))


def _galaxies(*a, **kw):
    from .realdata import dgp_galaxies                 # imported lazily: it touches the disk
    return dgp_galaxies(*a, **kw)


DGPS = {"gaussian": dgp_gaussian, "heavy_tail": dgp_heavy_tail, "skew": dgp_skew,
        "heteroskedastic": dgp_heteroskedastic, "galaxies": _galaxies}


def simulate(dgp: str, *, rng: np.random.Generator, R: int, n: int, N: float, rho: float,
             unlabeled_mode: str, **dgp_params) -> PPIData:
    """`15` §6.6.  Dispatch to a DGP by name."""
    if dgp not in DGPS:
        raise ValueError(f"unknown dgp {dgp!r}; available: {sorted(DGPS)}")
    return DGPS[dgp](rng, R=R, n=n, rho=rho, N=N, unlabeled_mode=unlabeled_mode, **dgp_params)


def cell_rng(cell: CellSpec) -> tuple[np.random.Generator, np.random.SeedSequence]:
    """Derive the cell's data generator and the bootstrap seed (`09` §Seed policy, `15` §7.4).

    The spawn key is built from *sorted sweep indices*, not iteration order, so adding a value
    to a sweep does not change any existing cell's draws.  The bootstrap seed is a separate
    child, so adding a stochastic method does not perturb the analytic methods' draws.

    A `SeedSequence` rather than a `Generator` is returned so that `run_cell` can hand **each**
    stochastic method a *freshly constructed* generator from the same seed.  That buys two
    things at once: adding B8 cannot change what B7 draws (method-insertion-invariance, which a
    single shared generator would break, since whichever method ran first would consume the
    stream), and B7 and B8 are compared on the *identical* bootstrap resamples — the same
    paired-comparison logic that C1 applies to the simulated data.
    """
    ss = np.random.SeedSequence(entropy=cell.seed_entropy, spawn_key=cell.seed_spawn_key)
    return np.random.default_rng(ss), ss.spawn(1)[0]


def run_cell(cell: CellSpec) -> tuple[list[dict], dict, float, dict]:
    """Draw **once**, evaluate every requested method on that one draw (C1, `15` §7.5).

    Common random numbers are enforced structurally: methods receive arrays (a `PPIView`),
    never a generator, and the view has no `theta_true` (`15` §4.3).  Returns the metric rows
    and the per-method `IntervalResult`s, `theta_true` (for the raw store), and a `realised`
    dict.  The last exists for real data, where `rho` and `N` are **observed properties of the
    dataset rather than controls**: the runner writes the realised values into the CSV instead of
    the configured placeholders.
    """
    rng, boot_ss = cell_rng(cell)
    data = simulate(cell.dgp, rng=rng, R=cell.R, n=cell.n, N=cell.N, rho=cell.rho,
                    unlabeled_mode=cell.unlabeled_mode, **cell.dgp_params)
    view = data.view()
    summaries = labelled_summaries(view.Y, view.f)      # shared by every method (15 §4.2)

    results, seconds = {}, {}
    for m in cell.methods:
        # A fresh generator from the same seed for every stochastic method: see `cell_rng`.
        method_rng = np.random.default_rng(boot_ss) if m in STOCHASTIC_METHODS else None
        t0 = time.perf_counter()
        results[m] = METHODS[m](view, cell.alpha_level, method_rng, summaries=summaries)
        seconds[m] = time.perf_counter() - t0

    rows = evaluate_all(results, data.theta_true, cell.R, seconds=seconds)
    Yc = view.Y - view.Y.mean(axis=-1, keepdims=True)
    fc = view.f - view.f.mean(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_hat = np.nanmean((Yc * fc).sum(-1)
                             / np.sqrt((Yc ** 2).sum(-1) * (fc ** 2).sum(-1)))
    realised = {"N": float(data.N), "rho": float(cell.rho) if np.isfinite(cell.rho)
                else float(rho_hat)}
    return rows, results, data.theta_true, realised
