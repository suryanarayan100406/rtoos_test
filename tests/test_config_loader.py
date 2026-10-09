"""Content hashing + the layered config loader precedence."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from drishti.bundle import combine, sha256_bytes, sha256_obj
from drishti.config import DrishtiConfig, load_config
from drishti.config.loader import ConfigError, deep_merge, parse_set_overrides

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------- hashing
def test_sha256_obj_key_order_independent():
    assert sha256_obj({"a": 1, "b": 2}) == sha256_obj({"b": 2, "a": 1})
    assert sha256_obj({"a": 1}) != sha256_obj({"a": 2})


def test_combine_order_independent():
    assert combine("x", "y") == combine("y", "x")
    assert combine("x", "y") != combine("x", "z")
    assert combine() == sha256_bytes(b"")
    assert combine("x", "", "y") == combine("x", "y")   # empties ignored


# --------------------------------------------------------------------- merge
def test_deep_merge_is_recursive_and_pure():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = deep_merge(base, {"a": {"y": 20, "z": 30}})
    assert out == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}
    assert base["a"]["y"] == 2                          # original untouched


# --------------------------------------------------------------------- precedence
def test_default_profile_loads():
    cfg, ds = load_config(config_dir=CONFIGS)
    assert isinstance(cfg, DrishtiConfig)
    assert ds is None
    assert cfg.ingest.target_fps == 4.0


def test_profile_overrides_default():
    cfg, _ = load_config(profile="fast", config_dir=CONFIGS)
    assert cfg.ingest.target_fps == 2.0
    assert cfg.poses.use_glomap is False
    assert cfg.dense.tsdf_voxel_m == 0.10


def test_set_overrides_beat_profile():
    cfg, _ = load_config(
        profile="fast",
        set_overrides=["ingest.target_fps=9", "dense.tsdf_voxel_m=0.03"],
        config_dir=CONFIGS,
    )
    assert cfg.ingest.target_fps == 9.0
    assert cfg.dense.tsdf_voxel_m == 0.03


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DRISHTI__ingest__target_fps", "7")
    cfg, _ = load_config(config_dir=CONFIGS)
    assert cfg.ingest.target_fps == 7.0


def test_set_beats_env(monkeypatch):
    monkeypatch.setenv("DRISHTI__run__seed", "111")
    cfg, _ = load_config(set_overrides=["run.seed=222"], config_dir=CONFIGS)
    assert cfg.run.seed == 222


def test_dataset_descriptor_crs_override(tmp_path):
    ds_file = tmp_path / "ds.yaml"
    ds_file.write_text(
        "name: t\nvideo: v.MP4\n"
        "telemetry:\n  format: dji_srt\n  path: v.SRT\n"
        "crs:\n  mode: epsg\n  epsg: 32643\n",
        encoding="utf-8",
    )
    cfg, ds = load_config(dataset=ds_file, config_dir=CONFIGS)
    assert ds is not None and ds.name == "t"
    assert cfg.crs.mode == "epsg"
    assert cfg.crs.epsg == 32643


def test_unknown_key_fails_loudly():
    with pytest.raises(ValidationError):
        load_config(set_overrides=["ingest.nonexistent_key=5"], config_dir=CONFIGS)


def test_set_without_equals_raises():
    with pytest.raises(ConfigError):
        parse_set_overrides(["noequalshere"])


def test_missing_config_dir_raises():
    with pytest.raises(ConfigError):
        load_config(config_dir="/definitely/not/here")
