"""Pipeline for producing a daily AQI NetCDF from local CAMS input."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import xarray as xr

from airwise.config import AppSettings, load_settings
from airwise.data.io.aqi_product import save_aqi_dataset
from airwise.data.io.cams import (
    load_cams_dataset,
    resolve_daily_nc_path,
)
from airwise.domain.aqi import classify_aqi, load_pollutant_bins


def compute_aqi_from_path(
    forecast_path: str | Path,
    *,
    threshold_config: str | Path | None = None,
) -> xr.Dataset:
    """Load a local concentration product and classify it into AQI levels."""
    with load_cams_dataset(forecast_path) as dataset:
        return classify_aqi(
            dataset,
            load_pollutant_bins(threshold_config),
        ).load()


def compute_daily_forecast_aqi(
    init_date: date,
    *,
    settings: AppSettings | None = None,
    forecast_dir: str | Path | None = None,
    aqi_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> Path:
    """Compute a daily AQI file without performing any network access."""
    resolved = settings or load_settings()
    forecast_root = Path(forecast_dir or resolved.paths.cams_forecast_daily)
    output_root = Path(aqi_dir or resolved.paths.cams_aqi_daily)
    forecast_path = resolve_daily_nc_path(forecast_root, init_date)
    output_path = resolve_daily_nc_path(output_root, init_date)
    if skip_existing and output_path.exists():
        return output_path
    if not forecast_path.exists():
        raise FileNotFoundError(f"CAMS forecast file not found: {forecast_path}")

    aqi = compute_aqi_from_path(
        forecast_path,
        threshold_config=resolved.aqi.threshold_config,
    )
    try:
        return save_aqi_dataset(aqi, output_path)
    finally:
        aqi.close()
