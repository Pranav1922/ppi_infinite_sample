"""Config parsing and validation (L4).  TOML via the stdlib `tomllib` (`15` §7.2).

Unknown keys **raise**: a silently-ignored typo in a config is a silently-wrong experiment.
"""
from __future__ import annotations

import tomllib
import zlib
from pathlib import Path

from .registry import METHODS
from .types import CellSpec

__all__ = ["load_config", "expand"]

TOP_LEVEL_KEYS = {
    "experiment_id", "description", "seed_entropy", "R", "alpha_levels", "unlabeled_mode",
    "coef_mode", "methods", "allow_mean_only_small_N", "dgp", "sweep", "outputs",
}
SWEEP_KEYS = {"n", "rho", "N"}
OUTPUT_KEYS = {"save_raw", "figures"}

#: Parameters each DGP family accepts, beyond the universal sweep keys.  Validated strictly:
#: a key that belongs to another family is a typo, and a silently-ignored typo is a silently-wrong
#: experiment (`15` §7.2).
FAMILY_KEYS = {
    "gaussian": {"var_Y", "var_f", "mean_Y", "mean_f"},
    "heavy_tail": {"nu", "heavy_f"},
    "skew": {"skew_family", "prevalence"},
    "heteroskedastic": {"gamma", "form"},
    "galaxies": {"dataset", "path"},
}

#: Sweep keys each family requires.  On real data `rho` and `N` are **observed properties of the
#: dataset, not controls**, so they are omitted from the config and the realised values are
#: written into the results CSV by the runner instead (`07` Experiment I: "the realised `rho` of
#: the dataset (reported, not controlled)").
FAMILY_SWEEPS = {"galaxies": {"n"}}


def _reject_unknown(got, allowed, where: str) -> None:
    unknown = set(got) - set(allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}; allowed: {sorted(allowed)}")


def load_config(path: str | Path) -> dict:
    """Parse and validate a config.  Returns the resolved dict (goes into the sidecar)."""
    path = Path(path)
    with path.open("rb") as fh:
        cfg = tomllib.load(fh)
    _reject_unknown(cfg, TOP_LEVEL_KEYS, str(path))

    for key in ("experiment_id", "seed_entropy", "R", "alpha_levels", "unlabeled_mode",
                "coef_mode", "methods", "dgp", "sweep"):
        if key not in cfg:
            raise ValueError(f"missing required key {key!r} in {path}")

    family = cfg["dgp"].get("family")
    if family not in FAMILY_KEYS:
        raise ValueError(f"unknown dgp family {family!r}; available: {sorted(FAMILY_KEYS)}")
    _reject_unknown(cfg["dgp"], {"family"} | FAMILY_KEYS[family], f"[dgp] for family {family!r}")
    # [sweep] may carry at most one DGP parameter in addition to n / rho / N; it becomes the
    # dgp_index of the seed spawn key (`15` §7.4 reserves that slot for exactly this).
    swept_dgp = sorted(set(cfg["sweep"]) - SWEEP_KEYS)
    if len(swept_dgp) > 1:
        raise ValueError(f"[sweep] may contain at most one DGP parameter; got {swept_dgp}")
    if swept_dgp and swept_dgp[0] not in FAMILY_KEYS[family]:
        raise ValueError(f"[sweep].{swept_dgp[0]} is not a parameter of dgp family {family!r}; "
                         f"available: {sorted(FAMILY_KEYS[family])}")
    _reject_unknown(cfg["sweep"], SWEEP_KEYS | set(swept_dgp), "[sweep]")
    _reject_unknown(cfg.get("outputs", {}), OUTPUT_KEYS, "[outputs]")

    if not (isinstance(cfg["R"], int) and cfg["R"] > 0):
        raise ValueError(f"R must be a positive integer; got {cfg['R']!r}")
    if not isinstance(cfg["seed_entropy"], int):
        raise ValueError("seed_entropy must be an integer")
    for a in cfg["alpha_levels"]:
        if not 0.0 < a < 1.0:
            raise ValueError(f"alpha_level must lie in (0,1); got {a}")
    for m in cfg["methods"]:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; registered: {sorted(METHODS)}")
    for key in FAMILY_SWEEPS.get(family, SWEEP_KEYS):
        if key not in cfg["sweep"] or not cfg["sweep"][key]:
            raise ValueError(f"[sweep].{key} must be a non-empty list")
    forbidden = (SWEEP_KEYS - FAMILY_SWEEPS[family]) & set(cfg["sweep"]) \
        if family in FAMILY_SWEEPS else set()
    if forbidden:
        raise ValueError(f"[sweep] must not set {sorted(forbidden)} for dgp family {family!r}: "
                         "on real data these are observed, not controlled")
    if not cfg["alpha_levels"]:
        raise ValueError("alpha_levels must be non-empty")
    if cfg["unlabeled_mode"] not in ("mean_only", "explicit"):
        raise ValueError(f"unlabeled_mode must be 'mean_only' or 'explicit'; "
                         f"got {cfg['unlabeled_mode']!r}")
    if cfg["coef_mode"] not in ("ols", "ppi_plus"):
        raise ValueError(f"coef_mode must be 'ols' or 'ppi_plus'; got {cfg['coef_mode']!r}")
    if cfg["coef_mode"] != "ols":
        raise NotImplementedError("coef_mode='ppi_plus' is not wired through the runner in the "
                                  "MVP; it is an OPEN research parameter (05 step 4).")

    # 15 §4.4 / 08 §T1.1: mean-only mode treats sigma_f^2 as known.  Harmless at large N, not
    # at small N — so it has to be asked for explicitly.
    if (cfg["unlabeled_mode"] == "mean_only" and "N" in cfg["sweep"]
            and not cfg.get("allow_mean_only_small_N", False)):
        small = [N for N in cfg["sweep"]["N"] if N <= 5000]
        if small:
            raise ValueError(
                f"unlabeled_mode='mean_only' with N={small} (<= 5000) treats sigma_f^2 as known "
                "where it should be estimated (08 §T1.1). Set allow_mean_only_small_N = true to "
                "do it deliberately, or use unlabeled_mode='explicit'.")
    return cfg


def expand(cfg: dict) -> list[CellSpec]:
    """Cartesian product of `[sweep]` x `alpha_levels` -> cells, with deterministic seeds.

    Seed indices come from the **sorted** sweep values (`15` §7.4), so a cell's draws do not
    depend on the order values appear in the config, nor on which other values are present.
    """
    ns = sorted(cfg["sweep"]["n"])
    Ns = sorted(cfg["sweep"].get("N", [float("nan")]))
    rhos = sorted(cfg["sweep"].get("rho", [float("nan")]))
    alphas = sorted(cfg["alpha_levels"])
    exp_index = int(zlib.crc32(cfg["experiment_id"].encode()) & 0x7FFFFFFF)

    dgp = dict(cfg["dgp"])
    family = dgp.pop("family")
    swept = sorted(set(cfg["sweep"]) - SWEEP_KEYS)
    swept_key = swept[0] if swept else None
    swept_vals = sorted(cfg["sweep"][swept_key]) if swept_key else [None]
    outputs = cfg.get("outputs", {})

    cells = []
    for i_d, dval in enumerate(swept_vals):
        params = dict(dgp) if swept_key is None else {**dgp, swept_key: dval}
        for i_n, n in enumerate(ns):
            for i_N, N in enumerate(Ns):
                for i_rho, rho in enumerate(rhos):
                    for i_a, alpha in enumerate(alphas):
                        cells.append(CellSpec(
                            experiment_id=cfg["experiment_id"], dgp=family, dgp_params=params,
                            n=int(n), N=float(N), rho=float(rho), alpha_level=float(alpha),
                            R=int(cfg["R"]), methods=tuple(cfg["methods"]),
                            unlabeled_mode=cfg["unlabeled_mode"], coef_mode=cfg["coef_mode"],
                            seed_spawn_key=(exp_index, i_n, i_N, i_rho, i_a, i_d),
                            seed_entropy=int(cfg["seed_entropy"]),
                            save_raw=bool(outputs.get("save_raw", False)),
                            dgp_swept_key=swept_key,
                        ))
    return cells
