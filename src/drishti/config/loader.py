"""Layered configuration loader.

Precedence (highest wins): --set overrides > env (DRISHTI__section__key) > dataset descriptor
(crs/report) > profile > configs/default.yaml. The fully resolved config is what gets snapshotted
into the run manifest, so a run always records the exact values it used.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .models import DatasetDescriptor, DrishtiConfig

ENV_PREFIX = "DRISHTI__"


class ConfigError(RuntimeError):
    """Raised on a missing/invalid config layer. Fails loudly (no silent defaults)."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file is not a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins). Returns a new dict."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _coerce(value: str) -> Any:
    """Parse a scalar override string with YAML rules (so 'true'->bool, '0.05'->float)."""
    return yaml.safe_load(value)


def _nested(dotted: str, value: Any) -> dict[str, Any]:
    parts = dotted.split(".")
    node: dict[str, Any] = {}
    cur = node
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node


def parse_set_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ['poses.use_glomap=false', 'run.seed=7'] into a nested dict."""
    result: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects key.path=value, got: {pair!r}")
        key, _, raw = pair.partition("=")
        result = deep_merge(result, _nested(key.strip(), _coerce(raw.strip())))
    return result


def parse_env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Read DRISHTI__section__key=value env vars into a nested dict."""
    environ = environ if environ is not None else dict(os.environ)
    result: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        dotted = name[len(ENV_PREFIX):].replace("__", ".").lower()
        result = deep_merge(result, _nested(dotted, _coerce(raw)))
    return result


def load_dataset(path: str | Path) -> DatasetDescriptor:
    data = _read_yaml(Path(path))
    return DatasetDescriptor.model_validate(data)


def load_config(
    profile: str | None = None,
    dataset: str | Path | None = None,
    set_overrides: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> tuple[DrishtiConfig, DatasetDescriptor | None]:
    """Assemble the effective config and (optionally) the dataset descriptor.

    Raises ConfigError / pydantic ValidationError on any malformed layer — never falls back to a
    silent default for a bad file.
    """
    cfg_dir = Path(config_dir or os.environ.get("DRISHTI_CONFIG_DIR", "configs"))
    merged = _read_yaml(cfg_dir / "default.yaml")

    # profile: explicit arg, else whatever default.yaml declares under run.profile
    chosen_profile = profile or merged.get("run", {}).get("profile", "balanced")
    if chosen_profile:
        merged = deep_merge(merged, _read_yaml(cfg_dir / "profiles" / f"{chosen_profile}.yaml"))
        merged = deep_merge(merged, {"run": {"profile": chosen_profile}})

    descriptor: DatasetDescriptor | None = None
    if dataset is not None:
        descriptor = load_dataset(dataset)
        # dataset descriptor may override crs/report of the main config
        ds_overrides: dict[str, Any] = {}
        if descriptor.crs is not None:
            ds_overrides["crs"] = descriptor.crs.model_dump()
        if descriptor.report is not None:
            ds_overrides["report"] = descriptor.report.model_dump()
        merged = deep_merge(merged, ds_overrides)

    merged = deep_merge(merged, parse_env_overrides())
    merged = deep_merge(merged, parse_set_overrides(set_overrides))

    cfg = DrishtiConfig.model_validate(merged)
    return cfg, descriptor
