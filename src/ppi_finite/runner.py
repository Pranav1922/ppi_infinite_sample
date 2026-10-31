"""Sweep execution and provenance (L4).  `15` §7.5.

This is the only module that holds both a method's output and `theta_true`, and the only place
common random numbers can be violated — hence the only place that has to enforce them.
"""
from __future__ import annotations

import time
from pathlib import Path

from .config import expand, load_config
from .io import provenance, write_csv, write_raw, write_sidecar
from .simulation import run_cell
from .types import CellSpec

__all__ = ["run_experiment"]


def _rows_for_cell(cell: CellSpec, metric_rows: list[dict], prov: dict,
                   realised: dict | None = None) -> list[dict]:
    common = {
        "experiment_id": cell.experiment_id, "cell_id": cell.cell_id, "dgp": cell.dgp_label,
        "n": cell.n, "N": (realised or {}).get("N", cell.N),
        "rho": (realised or {}).get("rho", cell.rho), "alpha_level": cell.alpha_level,
        "R": cell.R, "unlabeled_mode": cell.unlabeled_mode, "coef_mode": cell.coef_mode,
        "seed_spec": cell.seed_spec,
        "timestamp_utc": prov["timestamp_utc"], "python_version": prov["python_version"],
        "numpy_version": prov["numpy_version"], "scipy_version": prov["scipy_version"],
        "git_commit": prov["git_commit"],
    }
    return [{**common, **row} for row in metric_rows]


def run_experiment(config_path: str | Path, output_dir: str | Path,
                   command: str = "", progress: bool = True) -> Path:
    """Run every cell of a config and write `<experiment_id>.csv` + `.meta.json` (+ raw npz)."""
    cfg = load_config(config_path)
    cells = expand(cfg)
    prov = provenance()
    output_dir = Path(output_dir)
    started = time.perf_counter()

    rows: list[dict] = []
    for i, cell in enumerate(cells, 1):
        t0 = time.perf_counter()
        metric_rows, results, theta_true, realised = run_cell(cell)
        rows.extend(_rows_for_cell(cell, metric_rows, prov, realised))
        if cell.save_raw:
            write_raw(output_dir / f"{cell.experiment_id}_raw", cell.cell_id, results,
                      theta_true=theta_true)
        if progress:
            print(f"[{i:3d}/{len(cells)}] {cell.cell_id}  ({time.perf_counter() - t0:.1f}s)",
                  flush=True)

    csv_path = write_csv(rows, output_dir / f"{cfg['experiment_id']}.csv")
    write_sidecar(output_dir / f"{cfg['experiment_id']}.meta.json", config=cfg,
                  command=command, seconds=time.perf_counter() - started, prov=prov,
                  n_cells=len(cells))
    return csv_path
