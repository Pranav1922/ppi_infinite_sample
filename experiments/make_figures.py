#!/usr/bin/env python3
"""Regenerate figures from `results/` (`15` §2, extension E2).

    python experiments/make_figures.py --csv results/I_real_data.csv --prefix I

Consumes only `results/`; never recomputes a statistic (`15` §L5).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppi_finite.plots import make_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, help="one results CSV (E6-style figure set)")
    ap.add_argument("--all", action="store_true", help="the whole 07 figure set from results/")
    ap.add_argument("--results", default=Path("results"), type=Path)
    ap.add_argument("--out", default=Path("figures"), type=Path)
    ap.add_argument("--prefix", default="I")
    a = ap.parse_args()
    if a.all:
        from ppi_finite.plots import make_all_experiments
        for p in make_all_experiments(a.results, a.out):
            print(f"wrote {p}")
    if a.csv:
        for p in make_all(a.csv, a.out, a.prefix):
            print(f"wrote {p}")
    if not (a.all or a.csv):
        ap.error("give --all or --csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
