#!/usr/bin/env python3
"""Build data/mtbench_gpt35.npz: the MT-bench LLM-as-a-judge population for Experiment J.

    python3 experiments/prepare_mtbench.py

Source: `lmsys/mt_bench_human_judgments` on the Hugging Face Hub (Zheng et al., 2023), CC-BY-4.0,
pinned to one revision and checked by SHA-256 so the population cannot drift. Two splits are
joined on (question_id, model pair, turn): `human` (3,355 expert pairwise judgments) and
`gpt4_pair` (GPT-4's judgment of every one of the 2,400 (question, pair, turn) comparisons).

Population: every expert judgment of a comparison involving TARGET (the model with the most
expert judgments). For each, Y = 1 if the expert preferred TARGET, 0 if the other model, 1/2 for
a tie; Yhat is the same score from GPT-4's judgment of the identical comparison ("tie
(inconsistent)", where GPT-4's two position-swapped verdicts disagree, counts as a tie). The
estimand is therefore TARGET's expert-judged win rate on MT-bench. Documented in
docs/tier3_mtbench.md.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mtbench_gpt35.npz"
REPO = "lmsys/mt_bench_human_judgments"
REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"        # 2023-07-20, the only release
FILES = {
    "human": ("data/human-00000-of-00001-25f4910818759289.parquet",
              "4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769"),
    "gpt4_pair": ("data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet",
                  "57068327ffb5f8fddb15fedbbfe40b26fe862da2326331e5cf64a42320a00e71"),
}
TARGET = "gpt-3.5-turbo"
KEY = ["question_id", "A", "B", "turn"]


def fetch(split: str, cache: pathlib.Path) -> pd.DataFrame:
    rel, sha = FILES[split]
    path = cache / pathlib.Path(rel).name
    if not path.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{rel}"
        print(f"downloading {url}")
        # curl rather than urllib: the python.org macOS build ships no CA bundle
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(path), url], check=True)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        sys.exit(f"{path}: SHA-256 {got} != pinned {sha}")
    return pd.read_parquet(path, columns=["question_id", "model_a", "model_b", "winner", "turn"])


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Order each pair alphabetically as (A, B) and express the verdict in that order."""
    swap = df.model_a > df.model_b
    out = df.assign(A=df.model_a.where(~swap, df.model_b), B=df.model_b.where(~swap, df.model_a))
    flipped = df.winner.map({"model_a": "model_b", "model_b": "model_a"}).fillna(df.winner)
    out["verdict"] = df.winner.where(~swap, flipped)
    return out


def score(verdict: str, A: str, B: str) -> float:
    if verdict.startswith("tie"):
        return 0.5
    return 1.0 if (A if verdict == "model_a" else B) == TARGET else 0.0


def main() -> int:
    cache = ROOT / "data" / "mtbench_cache"
    cache.mkdir(parents=True, exist_ok=True)
    human = canonical(fetch("human", cache))
    gpt4 = canonical(fetch("gpt4_pair", cache)).rename(columns={"verdict": "gpt4"})
    assert not gpt4.duplicated(KEY).any(), "GPT-4 judged each comparison once"
    joined = human.merge(gpt4[KEY + ["gpt4"]], on=KEY, how="left", validate="m:1")
    assert joined.gpt4.notna().all(), "every human judgment must have a GPT-4 judgment"
    pop = joined[(joined.A == TARGET) | (joined.B == TARGET)]
    Y = np.array([score(r.verdict, r.A, r.B) for r in pop.itertuples()])
    f = np.array([score(r.gpt4, r.A, r.B) for r in pop.itertuples()])
    np.savez(OUT, Y=Y, Yhat=f)
    print(f"wrote {OUT.relative_to(ROOT)}: M={Y.size}, theta={Y.mean():.4f}, "
          f"corr={np.corrcoef(Y, f)[0, 1]:.4f}, mean(Yhat)={f.mean():.4f}, "
          f"sha256={hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
