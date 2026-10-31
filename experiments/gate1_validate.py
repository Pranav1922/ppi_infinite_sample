"""Gate 1 — measure Delta = coverage_act - coverage_id on the E4 grid and test Theorem G1.

Rao-Blackwellised: reduction_1 for T_act, reduction_2 for T_id, on common random numbers, so the
paired difference is far less noisy than either coverage. Reproduces the table of
docs/theory_finite_N.md section VI.9.

    PYTHONPATH=src python3 experiments/gate1_validate.py
"""
import sys
import numpy as np
from scipy import stats
from ppi_finite.theory_gate1 import coverage_bound, reduction_1, reduction_2

def measure(n, rho, N, R=2_000_000, alpha=0.05, seed=12345, chunk=500_000):
    """Rao-Blackwellised Delta: reduction_1 for T_act, reduction_2 for T_id, common randoms."""
    nu, sig_f = n - 2, 1.0
    beta, sig = rho, np.sqrt(1 - rho**2)
    c, lam = beta * sig_f / np.sqrt(N), beta * sig_f / (sig * np.sqrt(N))
    t = stats.t.ppf(1 - alpha/2, nu)
    rng = np.random.default_rng(seed)
    acc_act = acc_mid = acc_id = 0.0
    done = 0
    while done < R:
        r = min(chunk, R - done)
        f = rng.normal(0.0, sig_f, size=(r, n))
        S_ff = ((f - f.mean(1)[:, None])**2).sum(1)
        u = -f.mean(1) / np.sqrt(S_ff)
        kap = sig_f / (np.sqrt(N) * np.sqrt(S_ff))
        Z1, W = rng.normal(size=(2, r))
        Q = rng.chisquare(nu, size=r)
        sh2 = sig_f**2 * rng.chisquare(N - 1, size=r) / (N - 1)
        a = u + kap * W
        g_act, g_id = 1/n + a**2, 1/n + u**2
        s2 = sig**2 * Q / nu
        b = beta + sig * Z1 / np.sqrt(S_ff)
        V_act = s2 * g_act + b**2 * sh2 / N
        m_act = sig * a * Z1 + c * W
        acc_act += reduction_1(t, n, sig, V_act, m_act).sum()
        acc_mid += reduction_2(t, nu, Q, lam, g_act, W).sum()
        acc_id  += reduction_2(t, nu, Q, lam, g_id,  W).sum()
        done += r
    return acc_act/R, acc_mid/R, acc_id/R

grid = [(n, rho, N) for n in (5, 10, 20, 50) for rho in (0.0, 0.5, 0.9, 0.99)
        for N in (25, 100, 1000)]
print(f"{'n':>3} {'rho':>5} {'N':>5} {'cov_act':>9} {'cov_id':>9} {'Delta':>10} "
      f"{'|D_B|':>9} {'|D_PD|':>9} {'bound':>9} {'ratio':>7} {'ok':>4}")
rows = []
for n, rho, N in grid:
    ca, cm, ci = measure(n, rho, N)
    d, dB, dPD = ca - ci, ca - cm, cm - ci
    bd = coverage_bound(n, rho, N)
    ratio = abs(d) / bd.total
    ok = abs(d) <= bd.total
    rows.append((n, rho, N, d, bd.total, ratio, ok))
    print(f"{n:>3} {rho:>5} {N:>5} {ca:>9.5f} {ci:>9.5f} {d:>+10.5f} "
          f"{abs(dB):>9.5f} {abs(dPD):>9.5f} {bd.total:>9.5f} {ratio:>7.3f} {str(ok):>5}")
viol = [r for r in rows if not r[-1]]
print(f"\ncells: {len(rows)}   violations: {len(viol)}   max ratio |Delta|/bound: {max(r[5] for r in rows):.4f}")
inf = [r for r in rows if r[4] < 0.05]
print(f"cells where the bound is informative (< 0.05): {len(inf)}/{len(rows)}")
