"""Real-data producer: the Galaxies population and the gold/unlabelled subsampling protocol.

`08` Tier 2 construction recipe and `07` Experiment I. The file `data/galaxies.npz` contains a
**fully observed population** — a gold label and a model prediction for every item — and no
pre-split unlabelled arm, so the split is constructed here:

1. the full set of rows is the population `U`; `θ = mean(Y)` over it is the ground truth;
2. `n` rows are drawn **without replacement** as the gold set (both `Y` and `f` observed);
3. the remaining `M − n` rows are the unlabelled pool (**only** `f` used);
4. all methods run on that draw; repeat `R` times.

**`θ` is evaluation-only.** It is attached to `PPIData.theta_true`, which `PPIView` strips, so no
method can reach it (`15` §4.3).

**Assumption A5 is inherited, not enforced.** No predictor is fitted here — the predictions ship
with the dataset. A5 ("the predictor is a fixed function, not fitted on the labelled sample used
for inference") therefore rests on the dataset's provenance, which the `.npz` does not record.
Stated in `docs/tier2_galaxies.md` as a limitation rather than assumed away.

**The pool is not materialised.** With `M − n ≈ 16,600` and `R = 20,000` replications an
`(R, N)` array is 3·10⁸ floats. Only `f̄_N` and `σ̂_f²` enter any estimator, and both follow
exactly from population totals minus gold-set totals, in `O(n)` per replication.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import PPIData, make_data

__all__ = ["load_population", "subsample_protocol", "dgp_galaxies", "POPULATIONS"]

#: Datasets shipped as a fully observed population of `(Y, f)` pairs.
POPULATIONS = {"galaxies": "data/galaxies.npz",
               "mtbench": "data/mtbench_gpt35.npz"}   # experiments/prepare_mtbench.py


def load_population(name: str = "galaxies", path: str | Path | None = None) -> dict:
    """Load a population file and map its fields explicitly.

    The `.npz` field names are **read, not assumed**: `Y` is the gold label and `Yhat` the model
    prediction.  Raises rather than guessing if either is missing, and rejects non-finite values
    before anything can reach an estimator.
    """
    path = Path(path or POPULATIONS[name])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; E6 expects the population file to be present")
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.keys())
        missing = {"Y", "Yhat"} - set(keys)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}; found {keys}")
        Y = np.asarray(z["Y"], dtype=np.float64)
        f = np.asarray(z["Yhat"], dtype=np.float64)
    if Y.shape != f.shape or Y.ndim != 1:
        raise ValueError(f"Y {Y.shape} and Yhat {f.shape} must be 1-D and aligned")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(f))):
        raise ValueError("population contains non-finite values")
    return {
        "name": name, "path": str(path), "Y": Y, "f": f, "M": int(Y.size),
        "theta": float(Y.mean()),                       # EVALUATION ONLY
        "rho": float(np.corrcoef(Y, f)[0, 1]),
        "mu_f": float(f.mean()), "sigma2_f": float(f.var(ddof=1)),
        "binary_Y": bool(np.isin(np.unique(Y), (0.0, 1.0)).all()),
    }


def _sample_without_replacement(rng: np.random.Generator, M: int, n: int, R: int):
    """`(R, n)` indices, each row a uniform `n`-subset of `range(M)` drawn without replacement.

    Rejection on row-wise duplicates: conditioning a uniform ordered draw on distinctness is
    uniform over ordered distinct tuples, hence uniform over subsets.  Far cheaper than an
    `(R, M)` permutation when `n ≪ M`.
    """
    idx = rng.integers(0, M, (R, n))
    for _ in range(64):
        bad = np.array([len(np.unique(row)) < n for row in idx]) if n > 1 else np.zeros(R, bool)
        if not bad.any():
            return idx
        idx[bad] = rng.integers(0, M, (int(bad.sum()), n))
    # n²/M no longer small (MT-bench, n = 100 of 1,478): the rows still rejected are drawn
    # directly, which is the same uniform law over subsets. Never reached on the Galaxies grid.
    for r in np.flatnonzero(bad):
        idx[r] = rng.choice(M, n, replace=False)
    return idx


def subsample_protocol(pop: dict, *, n: int, R: int, rng: np.random.Generator) -> PPIData:
    """One batch of `R` gold/unlabelled splits (`07` Experiment I, `08` Tier 2 step 4).

    The unlabelled pool is the population complement of the gold set, so `N = M − n` and its
    summaries follow exactly from totals:

        f̄_N   = (Σ_U f − Σ_gold f) / (M − n)
        σ̂_f²  = [ (Σ_U f² − Σ_gold f²) − (M−n)·f̄_N² ] / (M − n − 1)
    """
    M, Yp, fp = pop["M"], pop["Y"], pop["f"]
    if not 3 <= n < M - 1:
        raise ValueError(f"n must satisfy 3 <= n < M-1 = {M - 1}; got {n}")
    idx = _sample_without_replacement(rng, M, n, R)
    Y, f = Yp[idx], fp[idx]

    N = float(M - n)
    tot_f, tot_ff = fp.sum(), (fp ** 2).sum()
    pool_sum = tot_f - f.sum(axis=-1)
    pool_sumsq = tot_ff - (f ** 2).sum(axis=-1)
    fbar_N = pool_sum / N
    sigma2_f = (pool_sumsq - N * fbar_N ** 2) / (N - 1.0)
    return make_data(Y, f, fbar_N=fbar_N, N=N, sigma2_f=sigma2_f,
                     theta_true=pop["theta"], mu_f=pop["mu_f"])


_POP_CACHE: dict = {}


def dgp_galaxies(rng: np.random.Generator, *, R: int, n: int, rho: float, N: float,
                 unlabeled_mode: str = "mean_only", dataset: str = "galaxies",
                 path: str | None = None) -> PPIData:
    """Producer adapter so the existing runner drives the real-data experiment unchanged.

    `rho` and `N` are **ignored**: on real data they are properties of the dataset, not controls.
    The realised values are reported by `run_cell` instead of the configured ones.
    """
    key = (dataset, path)
    if key not in _POP_CACHE:
        _POP_CACHE[key] = load_population(dataset, path)
    return subsample_protocol(_POP_CACHE[key], n=n, R=R, rng=rng)
