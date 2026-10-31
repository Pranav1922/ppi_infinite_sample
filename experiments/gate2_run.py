r"""Gate 2 runner — theorem validation, boundary map, quantile candidates.

Reads configs/experiment_Gate2_boundary.toml and writes
  results/gate2_boundary.csv    one row per cell: coverages, Delta, bound B, ratio
  results/gate2_quantiles.csv   long form: one row per (cell, candidate)
  results/gate2_summary.csv     the locked acceptance criteria, evaluated

Touches no existing result file and no registered estimator.

    PYTHONPATH=src python3 experiments/gate2_run.py
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time
import tomllib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from ppi_finite.gate2 import CANDIDATES, simulate_cell            # noqa: E402
from ppi_finite.theory_gate1 import coverage_bound                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "experiment_Gate2_boundary.toml"
OUT = ROOT / "results"


def cells(cfg):
    seen = set()
    for grid in ("grid_A", "grid_C"):
        g = cfg[grid]
        for n in g["n"]:
            for rho in g["rho"]:
                for N in g["N"]:
                    key = (n, float(rho), float(N))
                    # a cell appearing in both grids is run once and tagged with both
                    if key in seen:
                        continue
                    seen.add(key)
                    yield grid, n, float(rho), float(N)


def main() -> int:
    cfg = tomllib.load(open(CFG, "rb"))
    R, alpha = cfg["R"], cfg["alpha_level"]
    base = cfg["seed_entropy"]
    rows, qrows = [], []
    t0 = time.time()

    todo = list(cells(cfg))
    print(f"Gate 2: {len(todo)} distinct cells, R = {R:,} each")
    for i, (grid, n, rho, N) in enumerate(todo, 1):
        seed = int(np.random.SeedSequence(entropy=base,
                                          spawn_key=(n, int(rho * 1000), int(N))).generate_state(1)[0])
        t = time.time()
        res = simulate_cell(n, rho, N, R=R, alpha=alpha, seed=seed)
        bd = coverage_bound(n, rho, N, alpha)
        delta = res["delta_cov"]
        row = {
            "grid": grid, "n": n, "rho": rho, "N": N, "alpha": alpha, "R": R, "seed": seed,
            "N_minus_n_plus_1": N - n + 1, "N_over_n_minus_1": N / (n - 1),
            "coverage_actual": res["coverage_actual"], "coverage_ideal": res["coverage_ideal"],
            "delta_cov": delta, "bound_B": bd.total, "bound_B_B": bd.bound_B,
            "bound_B_PD": bd.bound_PD,
            "ratio_abs_delta_over_B": abs(delta) / bd.total if bd.total > 0 else np.nan,
            "slack_B_minus_abs_delta": bd.total - abs(delta),
            "bound_informative": bool(bd.total < 0.05),
            "bound_vacuous": bool(bd.total >= 1.0),
            "violation": bool(abs(delta) > bd.total),
            "mc_se": res["mc_se"], "limit_pred_rho_to_1": res["limit_pred"],
            "predicted_under_by_G8": bool(N < n - 1),
            "seconds": round(time.time() - t, 2),
        }
        rows.append(row)
        for cand in CANDIDATES:
            qrows.append({"grid": grid, "n": n, "rho": rho, "N": N, "candidate": cand,
                          "coverage": res[f"coverage_{cand}"], "width": res[f"width_{cand}"],
                          "width_rel_B9": res[f"width_{cand}"] / res["width_B9_t_n2"],
                          "mc_se": res["mc_se"]})
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  n={n} rho={rho} N={N:g}  "
                  f"cov={row['coverage_actual']:.5f}  |D|/B={row['ratio_abs_delta_over_B']:.3f}  "
                  f"[{time.time()-t0:.0f}s]")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "gate2_boundary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(OUT / "gate2_quantiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qrows[0]))
        w.writeheader(); w.writerows(qrows)

    # ---- locked acceptance criteria ----------------------------------------------------------
    viol = [r for r in rows if r["violation"]]
    A = [r for r in rows if r["grid"] == "grid_A"]
    se_tol = cfg["criteria"]["undercoverage_tolerance_se"]
    summary = [
        {"metric": "cells_total", "value": len(rows)},
        {"metric": "cells_grid_A", "value": len(A)},
        {"metric": "bound_violations", "value": len(viol)},
        {"metric": "max_ratio_abs_delta_over_B",
         "value": round(max(r["ratio_abs_delta_over_B"] for r in rows), 6)},
        {"metric": "max_abs_slack", "value": round(max(r["slack_B_minus_abs_delta"] for r in rows), 6)},
        {"metric": "min_abs_slack", "value": round(min(r["slack_B_minus_abs_delta"] for r in rows), 6)},
        {"metric": "cells_bound_informative", "value": sum(r["bound_informative"] for r in rows)},
        {"metric": "cells_bound_vacuous", "value": sum(r["bound_vacuous"] for r in rows)},
    ]
    for cand in CANDIDATES:
        qs = [q for q in qrows if q["candidate"] == cand and q["grid"] == "grid_A"]
        under = [q for q in qs if q["coverage"] < 0.95 - se_tol * q["mc_se"]]
        summary += [
            {"metric": f"{cand}__undercover_cells_gridA", "value": len(under)},
            {"metric": f"{cand}__min_coverage_gridA", "value": round(min(q["coverage"] for q in qs), 6)},
            {"metric": f"{cand}__max_coverage_gridA", "value": round(max(q["coverage"] for q in qs), 6)},
            {"metric": f"{cand}__mean_width_rel_B9_gridA",
             "value": round(float(np.mean([q["width_rel_B9"] for q in qs])), 6)},
        ]
    with open(OUT / "gate2_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "value"])
        w.writeheader(); w.writerows(summary)

    print(f"\nwrote gate2_boundary.csv ({len(rows)}), gate2_quantiles.csv ({len(qrows)}), "
          f"gate2_summary.csv ({len(summary)})   [{time.time()-t0:.0f}s]")
    if viol:
        print(f"\n*** {len(viol)} BOUND VIOLATION(S) — Gate 2 halts, investigate mathematically ***")
        for r in viol[:10]:
            print(f"    n={r['n']} rho={r['rho']} N={r['N']:g}  "
                  f"|delta|={abs(r['delta_cov']):.6f} > B={r['bound_B']:.6f}")
        return 1
    print("no bound violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
