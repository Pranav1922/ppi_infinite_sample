"""Regression: same seed spec -> identical CSV modulo timestamps (`15` §9, `12` §Reproducibility)."""
import pandas as pd
import pytest

from ppi_finite import run_experiment
from ppi_finite.config import expand, load_config

VOLATILE = ["timestamp_utc", "python_version", "numpy_version", "scipy_version", "git_commit"]
#: `seconds_per_interval` is wall-clock: it is the one metric that cannot be bit-reproducible.
VOLATILE_METRICS = ["seconds_per_interval"]


def _run(tmp_path, name):
    out = tmp_path / name
    df = pd.read_csv(run_experiment("configs/smoke.toml", out, progress=False))
    return df[~df.metric.isin(VOLATILE_METRICS)].reset_index(drop=True)


def test_two_runs_agree_exactly(tmp_path):
    a = _run(tmp_path, "a").drop(columns=VOLATILE)
    b = _run(tmp_path, "b").drop(columns=VOLATILE)
    pd.testing.assert_frame_equal(a, b)


def test_sidecar_is_written(tmp_path):
    _run(tmp_path, "c")
    assert (tmp_path / "c" / "smoke.meta.json").exists()


def test_seed_indices_come_from_sorted_values_not_config_order():
    cfg = load_config("configs/smoke.toml")
    shuffled = {**cfg, "sweep": {**cfg["sweep"], "rho": list(reversed(cfg["sweep"]["rho"]))}}
    keys = {c.cell_id: c.seed_spawn_key for c in expand(cfg)}
    keys_shuffled = {c.cell_id: c.seed_spawn_key for c in expand(shuffled)}
    assert keys == keys_shuffled


def test_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(open("configs/smoke.toml").read() + '\nnot_a_key = 1\n')
    with pytest.raises(ValueError, match="unknown key"):
        load_config(bad)


def test_config_rejects_mean_only_at_small_N(tmp_path):
    bad = tmp_path / "small.toml"
    bad.write_text(open("configs/smoke.toml").read().replace("N   = [1_000_000]", "N   = [200]"))
    with pytest.raises(ValueError, match="allow_mean_only_small_N"):
        load_config(bad)


def test_config_rejects_unknown_method(tmp_path):
    bad = tmp_path / "m.toml"
    bad.write_text(open("configs/smoke.toml").read().replace('"B9"]', '"B42"]'))
    with pytest.raises(ValueError, match="unknown method"):
        load_config(bad)
