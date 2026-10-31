#!/usr/bin/env python3
"""Single entry point for every experiment (`15` §7.1): one runner, one config per experiment.

    python experiments/run.py --config configs/experiment_A_reproduction.toml --output results/

Nothing in `experiments/` may contain an estimator or a variance formula (`11` §Repository
structure); this script only parses arguments and calls `ppi_finite.run_experiment`.
"""
import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppi_finite import run_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", default=Path("results"), type=Path)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    command = shlex.join([sys.executable, *sys.argv])
    path = run_experiment(args.config, args.output, command=command, progress=not args.quiet)
    print(f"wrote {path}")
    print(f"wrote {path.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
