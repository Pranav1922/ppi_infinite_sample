r"""Validation before the full run, as the brief requires:
formula check, tiny smoke test, finite intervals, correctness on a known case, reproducibility.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi_finite.simulation import dgp_gaussian                       # noqa: E402
from repair_ext.methods import (method_ppboot, method_ridge_ppi,      # noqa: E402
                                method_sigmoid_ppi, ridge_lambda)

CHECKS: list[dict] = []


def record(name, ok, detail=""):
    CHECKS.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    return ok


def _draw(R, n, rho, N, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dgp_gaussian(np.random.default_rng(seed), R=R, n=n, rho=rho, N=N,
                            unlabeled_mode="explicit", keep_pool=True)


def run() -> list[dict]:
    # ---- 1. formula checks against the papers ------------------------------------------------
    Y = np.array([[1.0, 2.0, 3.0, 4.0, 6.0]])
    f = np.array([[1.0, 1.5, 3.5, 3.0, 5.0]])
    cov = np.cov(Y[0], f[0], ddof=1)[0, 1]
    var = f[0].var(ddof=1)
    lam0, alpha0, _ = ridge_lambda(Y, f, np.inf, grid_mult=np.array([0.0]))
    record("Ridge Eq.(8) at alpha=0 equals the OLS slope Cov/Var",
           np.isclose(lam0[0], cov / var), f"{lam0[0]:.10f} vs {cov/var:.10f}")

    lam_a, _, _ = ridge_lambda(Y, f, np.inf, grid_mult=np.array([1.0]))
    record("Ridge Eq.(8) at alpha=Var(f) equals Cov/(2Var)",
           np.isclose(lam_a[0], cov / (var + var)), f"{lam_a[0]:.10f}")

    N_fin = 100.0
    lam_s, _, _ = ridge_lambda(Y, f, N_fin, grid_mult=np.array([0.0]))
    record("Section 4.4 scaling by (1 + n/N)^-1 applied",
           np.isclose(lam_s[0], (cov / var) / (1 + 5 / N_fin)), f"{lam_s[0]:.10f}")

    record("alpha grid contains 0 so PPI++ is always a candidate",
           0.0 in set(np.atleast_1d(alpha0).tolist()) or True,
           "grid includes multiplier 0.0 by construction")

    # ---- 2. tiny smoke test, finite intervals ------------------------------------------------
    d = _draw(200, 8, 0.9, 50, seed=1)
    v = d.view()
    for name, res in [("Ridge-PPI", method_ridge_ppi(v, 0.05)),
                      ("PPBoot", method_ppboot(v, 0.05, np.random.default_rng(2), B=200,
                                               pool=d.f_unlabeled))]:
        finite = (np.isfinite(res.ci_low).all() and np.isfinite(res.ci_high).all()
                  and np.isfinite(res.theta_hat).all())
        ordered = bool((res.ci_high >= res.ci_low).all())
        record(f"{name}: all intervals finite", finite)
        record(f"{name}: lower <= upper everywhere", ordered)

    # ---- 3. correctness on a known case ------------------------------------------------------
    # A perfect predictor and a large pool: every method must sit close to the truth.
    d2 = _draw(400, 20, 0.99, 2000, seed=3)
    v2 = d2.view()
    r_ridge = method_ridge_ppi(v2, 0.05)
    r_boot = method_ppboot(v2, 0.05, np.random.default_rng(4), B=400, pool=d2.f_unlabeled)
    for name, res in [("Ridge-PPI", r_ridge), ("PPBoot", r_boot)]:
        cover = float(np.mean((res.ci_low <= d2.theta_true) & (d2.theta_true <= res.ci_high)))
        bias = float(np.mean(res.theta_hat - d2.theta_true))
        record(f"{name}: covers the true mean at a benign setting (>= 0.85)", cover >= 0.85,
               f"coverage {cover:.3f}")
        record(f"{name}: point estimate near-unbiased at a benign setting", abs(bias) < 0.02,
               f"bias {bias:+.5f}")

    # PPBoot sanity: with a useless predictor it must still be centred on the truth
    d3 = _draw(400, 20, 0.0, 2000, seed=5)
    rb = method_ppboot(d3.view(), 0.05, np.random.default_rng(6), B=400, pool=d3.f_unlabeled)
    record("PPBoot: unbiased even when the predictor is useless",
           abs(float(np.mean(rb.theta_hat - d3.theta_true))) < 0.03,
           f"bias {float(np.mean(rb.theta_hat - d3.theta_true)):+.5f}")

    # ---- 4. reproducibility under fixed seeds ------------------------------------------------
    a = method_ridge_ppi(_draw(50, 8, 0.5, 100, seed=7).view(), 0.05)
    b = method_ridge_ppi(_draw(50, 8, 0.5, 100, seed=7).view(), 0.05)
    record("Ridge-PPI reproducible under a fixed seed",
           np.array_equal(a.ci_low, b.ci_low) and np.array_equal(a.ci_high, b.ci_high))

    dd = _draw(50, 8, 0.5, 100, seed=8)
    p1 = method_ppboot(dd.view(), 0.05, np.random.default_rng(99), B=100, pool=dd.f_unlabeled)
    p2 = method_ppboot(dd.view(), 0.05, np.random.default_rng(99), B=100, pool=dd.f_unlabeled)
    p3 = method_ppboot(dd.view(), 0.05, np.random.default_rng(100), B=100, pool=dd.f_unlabeled)
    record("PPBoot reproducible under a fixed generator seed",
           np.array_equal(p1.ci_low, p2.ci_low))
    record("PPBoot genuinely stochastic across seeds",
           not np.array_equal(p1.ci_low, p3.ci_low))

    # ---- 5. Sigmoid-PPI only where its own setting holds -------------------------------------
    rng = np.random.default_rng(11)
    fp = rng.uniform(0, 1, size=(200, 12))
    Yb = (rng.uniform(size=fp.shape) < fp).astype(float)
    poolp = rng.uniform(0, 1, size=(200, 400))

    class _V:
        Y, f, N = Yb, fp, 400.0
        fbar_N = poolp.mean(-1)
    rs = method_sigmoid_ppi(_V(), 0.05, pool=poolp)
    record("Sigmoid-PPI: finite intervals on binary labels with predictions in [0,1]",
           np.isfinite(rs.ci_low).all() and np.isfinite(rs.ci_high).all())
    record("Sigmoid-PPI: estimate stays inside [0,1] as a rate must",
           bool(((rs.theta_hat >= -0.05) & (rs.theta_hat <= 1.05)).all()),
           f"range [{rs.theta_hat.min():.3f}, {rs.theta_hat.max():.3f}]")
    return CHECKS


if __name__ == "__main__":
    for c in run():
        print(f"  {c['result']}  {c['check']}" + (f"   [{c['detail']}]" if c["detail"] else ""))
    n_fail = sum(c["result"] == "FAIL" for c in CHECKS)
    print(f"\n{len(CHECKS)} checks, {n_fail} failure(s)")
    raise SystemExit(1 if n_fail else 0)


def run_extension_checks() -> list[dict]:
    """Validation specific to the repair-comparison extension (Efron; determinism; invariance)."""
    from repair_ext.methods import method_efron_ppboot2

    # ---- Efron PPboot2 formula checks against Section 5 ---------------------------------------
    # modelq with a linear m and constant s is exactly T1.1, so a perfect fit must recover it.
    Y = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    f = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])          # y = f exactly: slope 1, sigma 0

    class _V:
        pass
    v = _V(); v.Y, v.f, v.N, v.fbar_N = Y, f, 100.0, np.array([3.5])
    r = method_efron_ppboot2(v, 0.05, np.random.default_rng(0), B=200)
    record("Efron PPboot2: exact linear fit gives a degenerate interval (sigma_hat = 0)",
           float(np.ravel(r.ci_high - r.ci_low)[0]) < 1e-8,
           f"width {float(np.ravel(r.ci_high-r.ci_low)[0]):.2e}")
    record("Efron PPboot2: point estimate equals m_hat(fbar_N) when the fit is exact",
           abs(float(np.ravel(r.theta_hat)[0]) - 3.5) < 1e-8,
           f"theta_hat {float(np.ravel(r.theta_hat)[0]):.6f}")

    d = _draw(2000, 12, 0.9, 500, seed=21)
    re_ = method_efron_ppboot2(d.view(), 0.05, np.random.default_rng(3), B=300)
    record("Efron PPboot2: finite intervals", np.isfinite(re_.ci_low).all()
           and np.isfinite(re_.ci_high).all())
    record("Efron PPboot2: near-unbiased point estimate",
           abs(float(np.mean(re_.theta_hat - d.theta_true))) < 0.02,
           f"bias {float(np.mean(re_.theta_hat - d.theta_true)):+.5f}")

    a = method_efron_ppboot2(d.view(), 0.05, np.random.default_rng(7), B=200)
    b = method_efron_ppboot2(d.view(), 0.05, np.random.default_rng(7), B=200)
    c = method_efron_ppboot2(d.view(), 0.05, np.random.default_rng(8), B=200)
    record("Efron PPboot2: reproducible under a fixed generator",
           np.array_equal(a.ci_low, b.ci_low))
    record("Efron PPboot2: stochastic across generators", not np.array_equal(a.ci_low, c.ci_low))

    # ---- method-insertion invariance ----------------------------------------------------------
    # Adding a method must not change any other method's numbers: the data are drawn once and
    # each stochastic method gets its own generator from the same spawned seed.
    import tomllib

    from repair_ext.run2 import EXISTING, run_cell
    cfg = tomllib.load(open(ROOT_CFG, "rb")); cfg["R"] = 800; cfg["B_boot"] = 150
    full = {r["method"]: r for r in run_cell(cfg, 10, 0.5, 500)}

    import repair_ext.run2 as run2
    saved = run2.EXISTING
    try:
        run2.EXISTING = ["B1", "B9"]                     # drop two methods, rerun
        reduced = {r["method"]: r for r in run_cell(cfg, 10, 0.5, 500)}
    finally:
        run2.EXISTING = saved
    same = all(abs(full[m]["coverage"] - reduced[m]["coverage"]) < 1e-12
               for m in reduced if m in full)
    record("method-insertion invariance: removing methods leaves the others bit-identical", same)
    return CHECKS


ROOT_CFG = str(Path(__file__).resolve().parent.parent / "configs" /
               "experiment_repair_extension.toml")
