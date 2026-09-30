"""Typed application settings built from the external YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PathSettings:
    cams_data_raw: Path
    cams_forecast_daily: Path
    cams_aqi_daily: Path
    cams_policy_forecast: Path
    open_ifs_data: Path
    bulletin_uq: Path
    era5_data_raw: Path
    era5_data_cams_grid: Path
    training_zarr: Path
    checkpoints: Path
    reports: Path
    nuts_shapefile: Path
    land_sea_mask: Path


@dataclass(frozen=True)
class AqiSettings:
    standard: str = "European_AQI"
    threshold_config: Path | None = None


@dataclass(frozen=True)
class AppSettings:
    """Typed settings passed into pipelines instead of reloading YAML."""

    paths: PathSettings
    aqi: AqiSettings
    raw: Mapping[str, Any]
