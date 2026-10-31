r"""Gate 4 runner. Writes results/gate4_quantiles.csv and results/gate4_summary.csv.

Touches no A-I, Gate-2 or Gate-3 result.

    PYTHONPATH=src python3 experiments/gate4_run.py
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time
import tomllib

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from ppi_finite.gate4 import GATE4_CANDIDATES, quantile_berger_boos   # noqa: E402
from ppi_finite.theory_gate1 import reduction_1                        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "experiment_Gate4_optimal_quantile.toml"


def run_cell(n, rho, N, R, alpha, eta, seed, chunk=100_000):
    nu, sig = n - 2, np.sqrt(1 - rho**2)
    beta, c = rho, rho / np.sqrt(N)
    rng = np.random.default_rng(seed)
    t_n2 = stats.t.ppf(1 - alpha / 2, nu)
    t_min = stats.t.ppf(1 - alpha / 2, min(nu, N - 1))
    cov = {k: 0.0 for k in GATE4_CANDIDATES}
    wid = {k: 0.0 for k in GATE4_CANDIDATES}
    done = 0
    while done < R:
        r = min(chunk, R - done)
        f = rng.normal(0.0, 1.0, size=(r, n))
        S_ff = ((f - f.mean(1)[:, None]) ** 2).sum(1)
        u = -f.mean(1) / np.sqrt(S_ff)
        kap = 1.0 / (np.sqrt(N) * np.sqrt(S_ff))
        Z1, W = rng.normal(size=(2, r))
        Q = rng.chisquare(nu, size=r)
        sh2 = rng.chisquare(N - 1, size=r) / (N - 1)
        a = u + kap * W
        V1 = sig**2 * Q / nu * (1 / n + a**2)
        b = beta + sig * Z1 / np.sqrt(S_ff)
        V2 = b**2 * sh2 / N
        V, m = V1 + V2, sig * a * Z1 + c * W
        qs = {"B9_t_n2": np.full(r, t_n2), "B9_tmin": np.full(r, t_min),
              "C5_bb": quantile_berger_boos(V1, V2, n, N, alpha, eta)}
        for k, q in qs.items():
            cov[k] += reduction_1(q, n, sig, V, m).sum()
            wid[k] += float(np.sum(2.0 * q * np.sqrt(V)))
        done += r
    return {k: (cov[k] / R, wid[k] / R) for k in GATE4_CANDIDATES}


def main() -> int:
    cfg = tomllib.load(open(CFG, "rb"))
    R, alpha, eta = cfg["R"], cfg["alpha_level"], cfg["eta"]
    rows, t0 = [], time.time()
    todo = []
    for gname in ("grid_primary", "grid_supplementary"):
        g = cfg[gname]
        todo += [(gname, n, float(rho), float(N))
                 for n in g["n"] for rho in g["rho"] for N in g["N"]]
    print(f"Gate 4: {len(todo)} cells, R = {R:,}")
    for i, (grid, n, rho, N) in enumerate(todo, 1):
        seed = int(np.random.SeedSequence(
            entropy=cfg["seed_entropy"],
            spawn_key=(n, int(rho * 1000), int(N))).generate_state(1)[0])
        res = run_cell(n, rho, N, R, alpha, eta, seed)
        base_w = res["B9_t_n2"][1]
        tmin_w = res["B9_tmin"][1]
        for cand, (cv, wd) in res.items():
            rows.append({"grid": grid, "n": n, "rho": rho, "N": N, "candidate": cand,
                         "coverage": cv, "width": wd,
                         "width_rel_B9": wd / base_w, "width_rel_tmin": wd / tmin_w,
                         "mc_se": float(np.sqrt(0.25 / R)),
                         "N_lt_n_minus_1": bool(N < n - 1)})
        if i % 30 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  [{time.time()-t0:.0f}s]")

    out = ROOT / "results"
    with open(out / "gate4_quantiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    se_tol = cfg["criteria"]["undercoverage_tolerance_se"]
    summary = []
    for grid in ("grid_primary", "grid_supplementary"):
        for cand in GATE4_CANDIDATES:
            sub = [r for r in rows if r["candidate"] == cand and r["grid"] == grid]
            cv = np.array([r["coverage"] for r in sub])
            under = int((cv < 0.95 - se_tol * sub[0]["mc_se"]).sum())
            summary += [
                {"grid": grid, "candidate": cand, "metric": "cells", "value": len(sub)},
                {"grid": grid, "candidate": cand, "metric": "undercover_cells", "value": under},
                {"grid": grid, "candidate": cand, "metric": "min_coverage",
                 "value": round(float(cv.min()), 6)},
                {"grid": grid, "candidate": cand, "metric": "mean_width_rel_B9",
                 "value": round(float(np.mean([r["width_rel_B9"] for r in sub])), 6)},
                {"grid": grid, "candidate": cand, "metric": "mean_width_rel_tmin",
                 "value": round(float(np.mean([r["width_rel_tmin"] for r in sub])), 6)},
                {"grid": grid, "candidate": cand, "metric": "max_width_rel_tmin",
                 "value": round(float(np.max([r["width_rel_tmin"] for r in sub])), 6)},
                {"grid": grid, "candidate": cand, "metric": "min_width_rel_tmin",
                 "value": round(float(np.min([r["width_rel_tmin"] for r in sub])), 6)},
            ]
    with open(out / "gate4_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["grid", "candidate", "metric", "value"])
        w.writeheader(); w.writerows(summary)
    print(f"\nwrote gate4_quantiles.csv ({len(rows)}), gate4_summary.csv ({len(summary)})"
          f"  [{time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
