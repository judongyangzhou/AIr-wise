"""Load project configuration from YAML.

``configs/default.yaml`` holds portable defaults. ``configs/local.yaml`` in the
same directory, when present, overrides those defaults and is not committed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from airwise.config.models import AppSettings, AqiSettings, PathSettings


class ConfigError(KeyError):
    """A required configuration key is missing."""


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the directory that contains both pyproject.toml and configs/default.yaml."""
    current = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "configs" / "default.yaml").is_file():
            return parent
    raise FileNotFoundError(
        "Could not find configs/default.yaml next to pyproject.toml. "
        "Set AIRWISE_CONFIG to an explicit config path."
    )


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a config path against the repository root when it is relative."""
    path = Path(value)
    if path.is_absolute():
        return path
    return find_repo_root() / path


@lru_cache(maxsize=8)
def resolve_config_path(path: str | Path | None = None) -> Path:
    """Return the base YAML path, before the local override is applied."""
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    env_path = os.environ.get("AIRWISE_CONFIG")
    if env_path:
        config_path = Path(env_path)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"AIRWISE_CONFIG points to a missing file: {config_path}"
            )
        return config_path

    return find_repo_root() / "configs" / "default.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Config file must be a mapping: {path}")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively. Lists and scalars in ``override`` replace the base."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _local_override_path(config_path: Path) -> Path | None:
    local_path = config_path.parent / "local.yaml"
    if local_path.is_file() and local_path.resolve() != config_path.resolve():
        return local_path
    return None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the base config and deep-merge ``local.yaml`` from the same directory."""
    config_path = resolve_config_path(path)
    config = _load_yaml(config_path)
    local_path = _local_override_path(config_path)
    if local_path is not None:
        config = deep_merge(config, _load_yaml(local_path))
    return config


def require_config(config: Mapping[str, Any], *keys: str) -> Any:
    """Return a nested value or raise ``ConfigError`` naming ``configs/local.yaml``."""
    node: Any = config
    for index, key in enumerate(keys):
        if not isinstance(node, Mapping) or key not in node or node[key] is None:
            dotted = ".".join(keys[: index + 1])
            raise ConfigError(
                f"Missing config key {dotted!r}. "
                "Set it in configs/default.yaml or override it in configs/local.yaml."
            )
        node = node[key]
    return node


def pollutant_variables(config: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Map report pollutant ids (``pm25``) to CAMS variable names (``pm2p5_conc``)."""
    cfg = config if config is not None else load_config()
    variables = require_config(cfg, "cams_europe", "variables")
    if not isinstance(variables, Mapping):
        raise ConfigError(
            "Config key 'cams_europe.variables' must be a mapping. "
            "Set it in configs/default.yaml or override it in configs/local.yaml."
        )
    return {str(key): str(value) for key, value in variables.items()}


def europe_bounds(config: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return the Europe crop from ``era5_euro.area``."""
    cfg = config if config is not None else load_config()
    area = require_config(cfg, "era5_euro", "area")
    return {
        name: float(require_config(area, name))
        for name in ("south", "north", "west", "east")
    }


def cams_download_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the shared CAMS download request fields."""
    cfg = config if config is not None else load_config()
    cams = require_config(cfg, "cams_europe")
    variables = require_config(cams, "download_variables")
    return {
        "dataset": str(require_config(cams, "dataset")),
        "variables": [str(value) for value in variables],
        "model": str(require_config(cams, "model")),
        "level": str(require_config(cams, "level")),
        "data_format": str(require_config(cams, "data_format")),
    }


def load_settings(path: str | Path | None = None) -> AppSettings:
    """Load typed settings once at a pipeline boundary."""
    config = load_config(path)
    paths = require_config(config, "paths")

    def configured_path(key: str) -> Path:
        return resolve_repo_path(require_config(paths, key))

    aqi = config.get("aqi") or {}
    threshold = aqi.get("threshold_config")
    return AppSettings(
        paths=PathSettings(
            cams_data_raw=configured_path("cams_data_raw"),
            cams_forecast_daily=configured_path("cams_forecast_daily"),
            cams_aqi_daily=configured_path("cams_aqi_daily"),
            cams_policy_forecast=configured_path("cams_policy_forecast"),
            open_ifs_data=configured_path("open_ifs_data"),
            bulletin_uq=configured_path("bulletin_uq"),
            era5_data_raw=configured_path("era5_data_raw"),
            era5_data_cams_grid=configured_path("era5_data_cams_grid"),
            training_zarr=configured_path("training_zarr"),
            checkpoints=configured_path("checkpoints"),
            reports=configured_path("reports"),
            nuts_shapefile=configured_path("nuts_shapefile"),
            land_sea_mask=configured_path("land_sea_mask"),
        ),
        aqi=AqiSettings(
            standard=str(aqi.get("standard", "European_AQI")),
            threshold_config=resolve_repo_path(threshold) if threshold else None,
        ),
        raw=config,
    )
