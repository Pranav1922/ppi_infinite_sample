#!/usr/bin/env python3
"""Verification checks against a results CSV (`09` §Verification checks).  Exits non-zero on failure.

MVP gate (`15` §11.1, amended 2026-08-19): **V1, V2, V3, V4, V5, V6, V7, V8, V9, V11**.
V4 and V9 were promoted into the gate because both are computable from Experiment A alone —
V4 is the spread of B3 coverage across rho, V9 is the N = 200 cell that Experiment A already
runs.  V12 (the slow KS test) remains outside.

With `--table-a4` / `--table-a5` (extension E1) this also checks **V4 on the wider rho grid**,
**V10**, and every target value of `09` Tables A4 and A5.  With `--table-b` / `--table-b-final`
(extension E3) it checks `09` §Experiment B results, the `07` Experiment B failure condition, and
prints the pre-registered **K3** trigger evaluated against the observed numbers.  With
`--table-e` (extension E4) it checks (V_full) against the empirical variance, the
conservativeness of `v482` (R4), `14` Q1's success condition for the repairs, and the derived
mechanism prediction.

**V2 is a z-test, not a fixed absolute tolerance** — see `09` V2 and `13` (2026-08-19). A fixed
+/- 0.002 is not a bias test: its stringency depends entirely on R and on the estimator's own
dispersion, so it fires on a correct implementation at small n and can never fire at large R.

Tolerances live here and only here (`15` §10.7), transcribed from `09` §Tolerances.
Changing one requires an entry in `13_RESEARCH_LOG.md`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppi_finite import labelled_summaries  # noqa: E402

# 09 §Tolerances
TOLERANCES = {
    "coverage_R200k": 0.004,      # ~8 Monte-Carlo standard errors at R = 200,000
    "coverage_V7": 0.005,         # 09 V7 states +/- 0.005 explicitly
    "bias_family_alpha": 0.001,   # 09 V2: family-wise level of the z-test, Bonferroni-corrected
                                  # across sub-checks. 0.001 is the same family-wise level 09 V12
                                  # already sets for the KS test.
    "width_ratio": 0.010,         # 09 V8 / §Tolerances
    "identity": 1e-10,            # 09 V11
    "rho_spread": 0.006,          # 09 V4
    "coverage_R60k": 0.005,       # 09 §Tolerances, R = 60,000 (Table A4)
    "mse_ratio_rel": 0.020,       # 09 §Tolerances, MSE ratios, RELATIVE
    "v10_abs": 0.030,             # 09 V10 states 1.479 +/- 0.030 explicitly
    "coverage_R20k": 0.006,       # 09 §Tolerances, R = 20,000 (Experiment B), ~4 MC s.e.
    # E4. The variance ratio mean(se^2)/Var(theta_hat) is limited by Monte-Carlo error in the
    # empirical variance, which is a few percent for a heavy-tailed theta_hat at R = 200,000.
    # 0.05 catches a structurally wrong variance (dropping b^2 sigma_f^2/N gives 0.20-0.58) but
    # NOT the sigma_f^2/(N S_ff) interaction term, which is ~1% and is established algebraically
    # in tests/test_finite_n.py, not here. See 13 (E4).
    "var_ratio_tol": 0.05,
    "e_mechanism_tol": 0.005,     # observed max 0.0029 across 81 cells; see 13 (E4)
    "q1_success": 0.004,          # 14 Q1 success condition (a): within 0.004 of nominal
}

#: 10 K3's pre-registered trigger, quoted: "B8 attains coverage >= 0.945 at n = 10 across all rho".
#: Transcribed here so the verdict is computed, not asserted.
K3_COVERAGE_TRIGGER = 0.945
PRIMARY_ALPHA = 0.05
PRIMARY_N = 1_000_000            # Table A1 is at N = 10^6


class Checks:
    def __init__(self) -> None:
        self.failed = 0
        self.passed = 0

    def check_z(self, name: str, what: str, observed: float, mc_se: float, crit: float) -> None:
        """A two-sided z-test: `|observed| <= crit * mc_se`.  Scale-free and R-independent."""
        z = observed / mc_se
        ok = abs(z) <= crit
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {what}  observed={observed:+.5f} "
              f"mc_se={mc_se:.5f} z={z:+.2f} crit=+/-{crit:.2f}")
        self.passed += ok
        self.failed += not ok

    def check_table(self, name: str, what: str, diffs: dict, tol: float,
                    info: bool = False, note: str = "") -> None:
        """Aggregate a whole target table into one verdict: the worst deviation must be within
        tolerance.  Failing cells are listed individually so a failure is actionable."""
        worst_cell, worst = max(diffs.items(), key=lambda kv: abs(kv[1]))
        ok = abs(worst) <= tol
        tag = "INFO" if info else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {name}: {what}  {len(diffs)} cells, worst {worst:+.4f} at {worst_cell} "
              f"tol={tol:.4f}{note}")
        for cell, d in sorted(diffs.items(), key=lambda kv: -abs(kv[1])):
            if abs(d) > tol:
                print(f"       out of tolerance: {cell}  diff={d:+.4f}")
        if not info:
            self.passed += ok
            self.failed += not ok

    def check_bound(self, name: str, what: str, observed: float, limit: float,
                    upper: bool = True) -> None:
        """A one-sided condition: `observed <= limit` (or `>=` when `upper` is False).

        Some pass conditions in the research package are genuinely one-sided — `07` Experiment
        B's "coverage above 0.96 at every cell indicates a bug" is a ceiling, not a tolerance
        band around a target — and forcing them into a two-sided `check` tests the wrong thing.
        """
        ok = observed <= limit if upper else observed >= limit
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {what}  observed={observed:.4f} "
              f"{'<=' if upper else '>='} {limit:.4f}")
        self.passed += ok
        self.failed += not ok

    def check(self, name: str, what: str, observed: float, expected: float, tol: float,
              info: bool = False, note: str = "") -> None:
        ok = abs(observed - expected) <= tol
        tag = "INFO" if info else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {name}: {what}  observed={observed:.4f} expected={expected:.4f} "
              f"tol={tol:.4f} diff={observed - expected:+.4f}{note}")
        if not info:
            self.passed += ok
            self.failed += not ok


def _val(df, method, metric, n, rho, N=PRIMARY_N, alpha=PRIMARY_ALPHA) -> float:
    sel = df[(df.method == method) & (df.metric == metric) & (df.n == n) & (df.rho == rho)
             & (df.N == N) & (df.alpha_level == alpha)]
    if len(sel) != 1:
        raise SystemExit(f"expected exactly one row for {method}/{metric}/n={n}/rho={rho}"
                         f"/N={N}/alpha={alpha}; got {len(sel)}")
    return float(sel.value.iloc[0])


A1_RHOS = (0.0, 0.5, 0.9)
A1_NS = (10, 20, 50)


def run_checks(df: pd.DataFrame) -> Checks:
    c = Checks()

    # V1 — harness sanity: B1 must sit at nominal at every cell of Table A1.
    for n in A1_NS:
        for rho in A1_RHOS:
            c.check("V1", f"B1 coverage n={n} rho={rho}", _val(df, "B1", "coverage", n, rho),
                    0.950, TOLERANCES["coverage_R200k"])

    # V2 — estimand sanity: the point estimates are unbiased for theta = 0.  Stated as a
    # two-sided z-test on mean(theta_hat) / (sd(theta_hat)/sqrt(R)), with the critical value
    # Bonferroni-corrected across the sub-checks at family-wise level 0.001 (09 V2, amended).
    R = int(df.R.iloc[0])
    v2_methods = ("B1", "B9")
    m = len(v2_methods) * len(A1_NS) * len(A1_RHOS)
    crit = float(stats.norm.ppf(1.0 - TOLERANCES["bias_family_alpha"] / (2 * m)))
    print(f"       V2 is a z-test: {m} sub-checks, family-wise alpha "
          f"{TOLERANCES['bias_family_alpha']}, Bonferroni critical value {crit:.2f}")
    for method in v2_methods:
        for n in A1_NS:
            for rho in A1_RHOS:
                mean = _val(df, method, "theta_hat_mean", n, rho)
                mc = _val(df, method, "theta_hat_sd", n, rho) / np.sqrt(R)
                c.check_z("V2", f"{method} mean(theta_hat) n={n} rho={rho}", mean, mc, crit)

    # V3 — the B3 deficit reproduces at n = 10.
    for rho in A1_RHOS:
        c.check("V3", f"B3 coverage n=10 rho={rho}", _val(df, "B3", "coverage", 10, rho),
                0.878, TOLERANCES["coverage_R200k"])

    # V5 — decay in n: 0.878 / 0.919 / 0.939.
    for n, expected in zip(A1_NS, (0.878, 0.919, 0.939)):
        for rho in A1_RHOS:
            c.check("V5", f"B3 coverage n={n} rho={rho}", _val(df, "B3", "coverage", n, rho),
                    expected, TOLERANCES["coverage_R200k"])

    # V6 — B9 exactness at N -> infinity.
    for n in A1_NS:
        for rho in A1_RHOS:
            c.check("V6", f"B9 coverage n={n} rho={rho}", _val(df, "B9", "coverage", n, rho),
                    0.950, TOLERANCES["coverage_R200k"])

    # V7 — decomposition: the quantile alone reaches ~0.937, the leverage term supplies the rest.
    for rho in A1_RHOS:
        c.check("V7", f"B9-nolev coverage n=10 rho={rho}",
                _val(df, "B9-nolev", "coverage", 10, rho), 0.937, TOLERANCES["coverage_V7"])

    # V8 — width honesty: a valid interval at rho = 0, n = 10 is 8.1% wider than classical.
    c.check("V8", "B9 width ratio to B1, n=10 rho=0.0",
            _val(df, "B9", "width_ratio_b1", 10, 0.0), 1.081, TOLERANCES["width_ratio"])

    # V11 — analytic identity: theta_hat == the OLS fitted value at mu_f, to 1e-10.
    rng = np.random.default_rng(11)
    f = rng.standard_normal(25)
    Y = 0.4 * f + rng.standard_normal(25)
    mu_f = 0.37
    s = labelled_summaries(Y, f)
    theta_hat = s["Ybar"] + s["b"] * (mu_f - s["fbar_n"])
    lr = stats.linregress(f, Y)
    c.check("V11", "theta_hat vs scipy.stats.linregress fitted value at mu_f",
            float(theta_hat), float(lr.intercept + lr.slope * mu_f), TOLERANCES["identity"])

    # V4 — rho-invariance of the B3 deficit (the signature of an estimated-parameter cost
    # rather than a predictor-quality problem).  Computable from Experiment A alone.
    for n in A1_NS:
        cov = [_val(df, "B3", "coverage", n, rho) for rho in A1_RHOS]
        c.check("V4", f"B3 coverage spread across rho at n={n}",
                max(cov) - min(cov), 0.0, TOLERANCES["rho_spread"])

    # V9 — the known finite-N over-coverage of B9.  Needs only the N = 200 cells that
    # Experiment A already runs.
    c.check("V9", "B9 coverage N=200 n=10 rho=0.9",
            _val(df, "B9", "coverage", 10, 0.9, N=200), 0.9612,
            TOLERANCES["coverage_R200k"])
    return c


# 09 Table A4 (source: cover.py).  The "B5 cross-fit" column is the averaged-lambda scheme
# cover.py computed, which is registered as B5-avglam; 06 B5's own formula is B5.  See 13.
TABLE_A4 = {
    (10, 0.00): (0.9506, 0.8783, 0.9241, 0.9254, 0.9409),
    (10, 0.20): (0.9501, 0.8793, 0.9247, 0.9261, 0.9367),
    (10, 0.50): (0.9490, 0.8777, 0.9234, 0.9245, 0.9237),
    (10, 0.80): (0.9492, 0.8796, 0.9240, 0.9248, 0.9194),
    (10, 0.95): (0.9503, 0.8795, 0.9249, 0.9258, 0.9248),
    (20, 0.00): (0.9525, 0.9225, 0.9401, 0.9403, 0.9485),
    (20, 0.50): (0.9500, 0.9198, 0.9389, 0.9387, 0.9349),
    (20, 0.95): (0.9500, 0.9196, 0.9375, 0.9377, 0.9375),
    (50, 0.00): (0.9504, 0.9395, 0.9461, 0.9459, 0.9488),
    (50, 0.50): (0.9517, 0.9399, 0.9461, 0.9457, 0.9455),
    (50, 0.95): (0.9506, 0.9404, 0.9460, 0.9458, 0.9460),
    (100, 0.00): (0.9506, 0.9461, 0.9489, 0.9489, 0.9500),
    (100, 0.50): (0.9510, 0.9447, 0.9478, 0.9478, 0.9478),
    (100, 0.95): (0.9499, 0.9422, 0.9453, 0.9453, 0.9453),
}
A4_METHODS = ("B1", "B3", "B4", "B5-avglam", "B6")

# 09 Table A5 (source: probe.py), MSE ratio to classical.  The oracle column needs
# lambda_oracle, which is not part of extension E1 and is not reproduced.
TABLE_A5 = {
    (10, 0.0): (1.1416, 1.4792, 1.0440), (10, 0.1): (1.1341, 1.4826, 1.0519),
    (10, 0.3): (1.0438, 1.3734, 1.0368), (10, 0.5): (0.8523, 1.1170, 0.9388),
    (10, 0.9): (0.2137, 0.2766, 0.2195), (20, 0.0): (1.0586, 1.1430, 1.0163),
    (20, 0.5): (0.7966, 0.8598, 0.8546), (50, 0.0): (1.0224, 1.0494, 1.0074),
    (100, 0.0): (1.0099, 1.0213, 1.0021), (500, 0.0): (1.0019, 1.0033, 1.0008),
}
A5_METHODS = ("B3", "B5", "B6")


def run_a4_checks(df, c: Checks) -> None:
    """Table A4 and V4 on the wider rho grid (`12` Stage 8)."""
    N = float(df.N.iloc[0])
    for j, method in enumerate(A4_METHODS):
        diffs = {f"n={n},rho={rho}": _val(df, method, "coverage", n, rho, N=N) - exp[j]
                 for (n, rho), exp in TABLE_A4.items()}
        c.check_table("A4", f"{method} coverage vs 09 Table A4", diffs,
                      TOLERANCES["coverage_R60k"])

    for n in sorted({n for n, _ in TABLE_A4}):
        rhos = sorted({r for m, r in TABLE_A4 if m == n})
        cov = [_val(df, "B3", "coverage", n, rho, N=N) for rho in rhos]
        c.check("V4", f"B3 coverage spread across rho={rhos} at n={n}",
                max(cov) - min(cov), 0.0, TOLERANCES["rho_spread"])


def run_a5_checks(df, c: Checks) -> None:
    """V10, plus Table A5 as INFO."""
    N = float(df.N.iloc[0])
    c.check("V10", "B5 MSE ratio to B1, n=10 rho=0.0",
            _val(df, "B5", "mse_ratio_b1", 10, 0.0, N=N), 1.479, TOLERANCES["v10_abs"])

    # Table A5 is reported but NOT gated. At n = 10 the 2-fold slope is estimated on 5 points,
    # so S_ff ~ chi2_4 and E[1/S_ff^2] is infinite: the fourth moment of theta_hat_B5 does not
    # exist, the sample MSE has infinite Monte-Carlo variance, and no CLT applies. Measured
    # run-to-run relative dispersion over 8 independent seeds is ~1.1-1.4%, against a stated
    # tolerance of 2% -- too close to be a reliable gate. See 13 (2026-08-19, E1).
    for j, method in enumerate(A5_METHODS):
        diffs = {}
        for (n, rho), exp in TABLE_A5.items():
            obs = _val(df, method, "mse_ratio_b1", n, rho, N=N)
            diffs[f"n={n},rho={rho}"] = (obs - exp[j]) / exp[j]
        c.check_table("A5", f"{method} MSE ratio vs 09 Table A5 (relative)", diffs,
                      TOLERANCES["mse_ratio_rel"], info=True,
                      note="  [reported, not gated: infinite-variance MSE estimate at n=10]")


# 09 §Experiment B results (E3, first measurements by this repository, N = 1e6, R = 20,000,
# B = 2000).  Unlike Tables A1-A5 these are not preliminary numbers being reproduced -- they are
# this repository's own results, and this block is a regression test on them.
TABLE_B = {
    (10, 0.0): (0.9499, 0.8794, 0.9010, 0.9026, 0.9506),
    (10, 0.3): (0.9505, 0.8787, 0.9016, 0.8996, 0.9506),
    (10, 0.5): (0.9476, 0.8784, 0.9015, 0.9020, 0.9501),
    (10, 0.7): (0.9488, 0.8751, 0.9012, 0.8983, 0.9482),
    (10, 0.9): (0.9492, 0.8770, 0.8991, 0.9001, 0.9497),
    (20, 0.0): (0.9495, 0.9234, 0.9286, 0.9284, 0.9512),
    (20, 0.3): (0.9515, 0.9222, 0.9280, 0.9264, 0.9498),
    (20, 0.5): (0.9490, 0.9188, 0.9237, 0.9248, 0.9487),
    (20, 0.7): (0.9504, 0.9176, 0.9247, 0.9228, 0.9493),
    (20, 0.9): (0.9468, 0.9185, 0.9250, 0.9256, 0.9486),
    (50, 0.0): (0.9470, 0.9353, 0.9382, 0.9375, 0.9469),
    (50, 0.3): (0.9496, 0.9401, 0.9425, 0.9419, 0.9513),
    (50, 0.5): (0.9498, 0.9392, 0.9404, 0.9394, 0.9499),
    (50, 0.7): (0.9489, 0.9386, 0.9399, 0.9409, 0.9489),
    (50, 0.9): (0.9497, 0.9366, 0.9389, 0.9387, 0.9490),
}
B_METHODS = ("B1", "B3", "B7", "B8", "B9")


def run_b_checks(df, c: Checks, df_final=None) -> None:
    """Experiment B (`07`), the E3 results table (`09`), and the K3 verdict (`10`)."""
    N = float(df.N.iloc[0])
    for j, method in enumerate(B_METHODS):
        diffs = {f"n={n},rho={rho}": _val(df, method, "coverage", n, rho, N=N) - exp[j]
                 for (n, rho), exp in TABLE_B.items()}
        c.check_table("B", f"{method} coverage vs 09 §Experiment B", diffs,
                      TOLERANCES["coverage_R20k"])

    # 07 Experiment B failure condition: "Bootstrap coverage above 0.96 at every cell indicates
    # an implementation bug (likely resampling the wrong thing or double-counting)."
    for method in ("B7", "B8"):
        cov = [_val(df, method, "coverage", n, rho, N=N) for (n, rho) in TABLE_B]
        c.check_bound("B-bug", f"{method} min coverage over cells (not >0.96 everywhere)",
                      min(cov), 0.96, upper=True)

    # 06 §Reporting 6: failed constructions are never dropped silently.
    for method in ("B7", "B8"):
        failed = int(df[(df.method == method)].n_failed.max())
        c.check_bound("B-failures", f"{method} failed interval constructions", float(failed), 0.0)

    # --- K3, computed from the pre-registered trigger rather than asserted ---
    for label, frame, m8 in (("B=2000", df, "B8"),
                             ("B=9999", df_final, "B8-B9999")):
        if frame is None:
            continue
        Nf = float(frame.N.iloc[0])
        rhos = sorted({rho for (n, rho) in TABLE_B if n == 10})
        cov = [_val(frame, m8, "coverage", 10, rho, N=Nf) for rho in rhos]
        fires = min(cov) >= K3_COVERAGE_TRIGGER
        print(f"[INFO] K3 ({label}): B8 coverage at n=10 across rho = "
              f"{', '.join(f'{v:.4f}' for v in cov)}  min={min(cov):.4f}  "
              f"trigger>={K3_COVERAGE_TRIGGER} -> {'FIRES' if fires else 'DOES NOT FIRE'}")


def run_e_checks(df, c: Checks) -> None:
    """Experiment E (`07`), `04` §4.8.3-4.8.4 and `14` Q1's success condition."""
    cov = df[df.metric == "coverage"].pivot_table(index=["n", "rho", "N"], columns="method",
                                                  values="value")
    vr = df[df.metric == "var_ratio"].pivot_table(index=["n", "rho", "N"], columns="method",
                                                 values="value")
    sh = df[df.metric == "unlabeled_share_mean"].pivot_table(index=["n", "rho", "N"],
                                                             columns="method", values="value")
    R = int(df.R.iloc[0])
    mcse = float(np.sqrt(0.95 * 0.05 / R))

    # E1 -- (V_full) against the empirical variance. 12 Stage 11: "the first numerical test that
    # (V_full) has ever received".
    for method in ("B9-vfull", "B9"):
        diffs = {f"{k}": vr.loc[k, method] - 1.0 for k in vr.index}
        c.check_table("E-var", f"{method} mean(se^2)/Var(theta_hat) vs 1",
                      diffs, TOLERANCES["var_ratio_tol"])

    # E2 -- R4: is v482 conservative for all N, n, rho? (04 §4.8.4 R4)
    c.check_bound("E-R4", "B9 min coverage over all cells (conservativeness of v482)",
                  0.95 - 3 * mcse, float(cov["B9"].min()), upper=True)

    # E3 -- 14 Q1 success condition (a) for the repairs.
    for method in ("B9", "B9-R1", "B9-R2", "B9-R3", "B9-R5"):
        worst = float(np.abs(cov[method] - 0.95).max())
        c.check_bound("E-Q1", f"{method} worst |coverage - 0.95| over all cells "
                              f"(Q1 wants <= {TOLERANCES['q1_success']})",
                      worst, TOLERANCES["q1_success"]
                      if method in ("B9-R2", "B9-R5") else 1.0)

    # E4 -- the mechanism, as a derived prediction with no free parameters: if the statistic
    # behaves like t_{nu_eff} with nu_eff = (n-2)/(1-s)^2, applying the t_{n-2} quantile gives
    # coverage P(|t_{nu_eff}| <= t_{n-2, 0.975}).
    diffs = {}
    for k in cov.index:
        n = k[0]
        nu = (n - 2) / (1 - sh.loc[k, "B9"]) ** 2
        pred = 2 * stats.t.cdf(stats.t.ppf(0.975, n - 2), nu) - 1
        diffs[f"{k}"] = cov.loc[k, "B9"] - pred
    c.check_table("E-mech", "B9 coverage vs the Satterthwaite-mixture prediction", diffs,
                  TOLERANCES["e_mechanism_tol"])

    # E5 -- 09 Table A3 / V9 reproduced in EXPLICIT mode (the E4 brief, step 7).
    if (10, 0.9, 200.0) in cov.index:
        c.check("E-N200", "B9 coverage n=10 rho=0.9 N=200, explicit mode",
                float(cov.loc[(10, 0.9, 200.0), "B9"]), 0.9612,
                TOLERANCES["coverage_R200k"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", required=True, type=Path, help="Experiment A results CSV")
    ap.add_argument("--table-a4", type=Path, help="09 Table A4 companion CSV (extension E1)")
    ap.add_argument("--table-a5", type=Path, help="09 Table A5 companion CSV (extension E1)")
    ap.add_argument("--table-b", type=Path, help="Experiment B results CSV (extension E3)")
    ap.add_argument("--table-b-final", type=Path, help="Experiment B at B=9999 (extension E3)")
    ap.add_argument("--table-e", type=Path, nargs="*", default=None,
                    help="Experiment E results CSVs, finite-N and large-N arms (extension E4)")
    args = ap.parse_args()

    df = pd.read_csv(args.tables)
    R = int(df.R.iloc[0])
    if R != 200_000:
        print(f"[WARN] tolerances in 09 are stated for R = 200,000; this CSV has R = {R}")
    print(f"verifying {args.tables} ({len(df)} rows, R = {R})\n")

    c = run_checks(df)
    if args.table_a4:
        print(f"\n--- 09 Table A4 ({args.table_a4}) ---")
        run_a4_checks(pd.read_csv(args.table_a4), c)
    if args.table_a5:
        print(f"\n--- 09 Table A5 ({args.table_a5}) ---")
        run_a5_checks(pd.read_csv(args.table_a5), c)

    if args.table_b:
        print(f"\n--- Experiment B ({args.table_b}) ---")
        run_b_checks(pd.read_csv(args.table_b), c,
                     pd.read_csv(args.table_b_final) if args.table_b_final else None)

    if args.table_e:
        print(f"\n--- Experiment E ({', '.join(str(p) for p in args.table_e)}) ---")
        run_e_checks(pd.concat([pd.read_csv(p) for p in args.table_e]), c)

    print(f"\n{c.passed} passed, {c.failed} failed")
    if c.failed:
        print("A deviation outside tolerance is a signal, not a nuisance (09 §Tolerances): "
              "investigate and record it in 13_RESEARCH_LOG.md. Do not widen a tolerance.")
    return 1 if c.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
