"""Configuration loading and typed application settings."""

from airwise.config.loader import (
    ConfigError,
    cams_download_settings,
    deep_merge,
    europe_bounds,
    find_repo_root,
    load_config,
    load_settings,
    pollutant_variables,
    require_config,
    resolve_config_path,
    resolve_repo_path,
)
from airwise.config.models import AppSettings, AqiSettings, PathSettings

__all__ = [
    "AppSettings",
    "AqiSettings",
    "ConfigError",
    "PathSettings",
    "cams_download_settings",
    "deep_merge",
    "europe_bounds",
    "find_repo_root",
    "load_config",
    "load_settings",
    "pollutant_variables",
    "require_config",
    "resolve_config_path",
    "resolve_repo_path",
]
