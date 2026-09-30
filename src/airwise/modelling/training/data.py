"""Load and align CAMS/ERA5 inputs for error-modelling training."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from airwise.config import load_config
from airwise.data.io.training_store import (
    default_output_root,
    open_training_zarr_cache,
)
from airwise.modelling.config import get_error_modelling_settings
from airwise.modelling.features import (
    _apply_log_transform,
    load_processed_cams_features,
    load_training_era5_features,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ErrorModellingData:
    cams_forecast: xr.Dataset
    cams_analysis: xr.Dataset
    era5: xr.Dataset
    pollutant: str = "pm10_conc"
    source: str = "netcdf"
    pollutant_log_transform: str | None = "log1p"

    def close(self) -> None:
        for dataset in (self.cams_forecast, self.cams_analysis, self.era5):
            try:
                dataset.close()
            except RuntimeError as exc:
                logger.debug("Ignoring xarray dataset close error: %s", exc)


def _period_for_years(years: Sequence[int]) -> tuple[str, str]:
    first_year = min(years)
    last_year = max(years)
    return f"{first_year}-01-01 00:00:00", f"{last_year}-12-31 23:00:00"


def _sort_and_deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds

    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    if not time_index.is_monotonic_increasing:
        logger.info("Sorting dataset by non-monotonic time coordinate before time selection")
        ds = ds.sortby("time")
        time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))

    if time_index.has_duplicates:
        duplicate_times = time_index[time_index.duplicated(keep=False)].unique()
        preview = ", ".join(str(timestamp) for timestamp in duplicate_times[:5])
        raise ValueError(
            f"Dataset contains {len(duplicate_times)} duplicated timestamps after sorting. "
            f"First duplicates: {preview}. Refusing to drop data silently."
        )

    return ds


def _select_period(ds: xr.Dataset, period: tuple[str, str] | None) -> xr.Dataset:
    if period is None:
        return _sort_and_deduplicate_time(ds)

    ds = _sort_and_deduplicate_time(ds)
    start = pd.Timestamp(period[0])
    end = pd.Timestamp(period[1])
    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    time_mask = (time_index >= start) & (time_index <= end)
    return ds.isel(time=np.flatnonzero(time_mask))


def _log_time_summary(name: str, ds: xr.Dataset) -> None:
    if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
        logger.info("%s time summary: no time coordinate", name)
        return
    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    logger.info(
        "%s time summary: n=%d unique=%d start=%s end=%s",
        name,
        len(time_index),
        time_index.nunique(),
        time_index.min(),
        time_index.max(),
    )


def _select_common_times(
    cams_forecast: xr.Dataset,
    cams_analysis: xr.Dataset,
    era5: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    cams_forecast = _sort_and_deduplicate_time(cams_forecast)
    cams_analysis = _sort_and_deduplicate_time(cams_analysis)
    era5 = _sort_and_deduplicate_time(era5)

    forecast_times = cams_forecast["time"].values.astype("datetime64[ns]")
    analysis_times = cams_analysis["time"].values.astype("datetime64[ns]")
    era5_times = era5["time"].values.astype("datetime64[ns]")

    common_times = np.intersect1d(np.intersect1d(forecast_times, analysis_times), era5_times)
    if common_times.size == 0:
        raise ValueError("CAMS forecast, CAMS analysis, and ERA5 have no overlapping timestamps.")

    logger.info(
        "Aligned CAMS/ERA5 on %d common timestamps: %s to %s",
        common_times.size,
        pd.Timestamp(common_times[0]),
        pd.Timestamp(common_times[-1]),
    )
    return (
        cams_forecast.sel(time=common_times),
        cams_analysis.sel(time=common_times),
        era5.sel(time=common_times),
    )


def _resolve_pollutant_log_transform(
    pollutant: str,
    pollutant_log_transform: str | None | bool | Mapping[str, str | None] | object = ...,
) -> str | None:
    settings = get_error_modelling_settings()
    if pollutant_log_transform is ...:
        return settings.pollutant_log_transform_for(pollutant)
    if pollutant_log_transform is False or pollutant_log_transform is None:
        return None
    if isinstance(pollutant_log_transform, Mapping):
        if pollutant in pollutant_log_transform:
            value = pollutant_log_transform[pollutant]
            return None if value in (None, False) else str(value)
        return settings.pollutant_log_transform_for(pollutant)
    return str(pollutant_log_transform)


def _apply_pollutant_log_transform(
    cams_forecast: xr.Dataset,
    cams_analysis: xr.Dataset,
    pollutant: str,
    pollutant_log_transform: str | None,
) -> tuple[xr.Dataset, xr.Dataset]:
    if pollutant_log_transform is None:
        return cams_forecast, cams_analysis
    transform = {pollutant: pollutant_log_transform}
    return (
        _apply_log_transform(cams_forecast, transform),
        _apply_log_transform(cams_analysis, transform),
    )


def _select_pollutant_datasets(
    cams_forecast: xr.Dataset,
    cams_analysis: xr.Dataset,
    pollutant: str,
) -> tuple[xr.Dataset, xr.Dataset]:
    if pollutant not in cams_forecast.data_vars:
        raise KeyError(
            f"Pollutant {pollutant!r} missing from CAMS forecast cache. "
            f"Available: {list(cams_forecast.data_vars)}"
        )
    if pollutant not in cams_analysis.data_vars:
        raise KeyError(
            f"Pollutant {pollutant!r} missing from CAMS analysis cache. "
            f"Available: {list(cams_analysis.data_vars)}"
        )
    return cams_forecast[[pollutant]], cams_analysis[[pollutant]]


def _select_era5_variables(era5: xr.Dataset, era5_variables: Sequence[str]) -> xr.Dataset:
    # Keep rain_mask when present even if not explicitly requested.
    selected = list(era5_variables)
    if "rain_mask" in era5.data_vars and "rain_mask" not in selected:
        selected.append("rain_mask")
    missing = [var_name for var_name in selected if var_name not in era5.data_vars]
    if missing:
        raise KeyError(f"ERA5 variables missing from cache: {missing}. Available: {list(era5.data_vars)}")
    return era5[selected]


def load_error_modelling_data(
    years: Sequence[int],
    pollutant: str | None = None,
    era5_variables: Sequence[str] | None = None,
    period: tuple[str, str] | None = None,
    chunks: dict | None = None,
    era5_grid: str | None = None,
    pollutant_log_transform: str | None | bool | Mapping[str, str | None] | object = ...,
) -> ErrorModellingData:
    """
    Load aligned CAMS and ERA5 tensors from processed NetCDF sources.

    CAMS processed files are expected without pollutant log transforms. The
    training transform is applied after loading using YAML/CLI settings.
    """
    settings = get_error_modelling_settings()
    pollutant = pollutant or settings.pollutant
    era5_variables = list(era5_variables or settings.era5_variables)
    era5_grid = era5_grid or settings.era5_grid
    period = period or _period_for_years(years)
    resolved_transform = _resolve_pollutant_log_transform(pollutant, pollutant_log_transform)
    temporal_resolution = settings.cams_training_temporal_resolution

    logger.info(
        "Loading CAMS forecast for years=%s, pollutant=%s, period=%s, log_transform=%s",
        list(years),
        pollutant,
        period,
        resolved_transform,
    )
    cams_forecast = load_processed_cams_features(
        years=years,
        data_type="forecast",
        variables=[pollutant],
        processed_log_transform=None,
        temporal_resolution=temporal_resolution,
        log_transform=False,
        chunks=chunks,
    )
    logger.info("Loaded CAMS forecast dims=%s", dict(cams_forecast.sizes))

    logger.info(
        "Loading CAMS analysis for years=%s, pollutant=%s, period=%s",
        list(years),
        pollutant,
        period,
    )
    cams_analysis = load_processed_cams_features(
        years=years,
        data_type="analysis",
        variables=[pollutant],
        processed_log_transform=None,
        temporal_resolution=temporal_resolution,
        log_transform=False,
        chunks=chunks,
    )
    logger.info("Loaded CAMS analysis dims=%s", dict(cams_analysis.sizes))

    logger.info(
        "Loading training ERA5 variables=%s grid=%s log_transform=%s",
        list(era5_variables),
        era5_grid,
        settings.era5_log_transform,
    )
    era5 = load_training_era5_features(
        period=period,
        variables=list(era5_variables),
        temporal_resolution=settings.era5_temporal_resolution,
        grid=era5_grid,
        chunks=chunks,
    )
    logger.info("Loaded ERA5 dims=%s variables=%s", dict(era5.sizes), list(era5.data_vars))

    cams_forecast = _select_period(cams_forecast, period)
    cams_analysis = _select_period(cams_analysis, period)
    _log_time_summary("CAMS forecast after period selection", cams_forecast)
    _log_time_summary("CAMS analysis after period selection", cams_analysis)
    _log_time_summary("ERA5 after period selection", era5)
    cams_forecast, cams_analysis, era5 = _select_common_times(cams_forecast, cams_analysis, era5)
    cams_forecast, cams_analysis = _apply_pollutant_log_transform(
        cams_forecast,
        cams_analysis,
        pollutant,
        resolved_transform,
    )

    return ErrorModellingData(
        cams_forecast=cams_forecast,
        cams_analysis=cams_analysis,
        era5=era5,
        pollutant=pollutant,
        source="netcdf",
        pollutant_log_transform=resolved_transform,
    )


def load_error_modelling_data_from_zarr(
    pollutant: str | None = None,
    era5_variables: Sequence[str] | None = None,
    period: tuple[str, str] | None = None,
    years: Sequence[int] | None = None,
    zarr_root: str | Path | None = None,
    chunks: dict | None = None,
    pollutant_log_transform: str | None | bool | Mapping[str, str | None] | object = ...,
    config: Mapping | None = None,
) -> ErrorModellingData:
    """
    Load aligned CAMS/ERA5 tensors from the shared training Zarr cache.

    The cache stores physical CAMS concentrations (no pollutant log1p) and
    recipe-processed ERA5. Pollutant log transforms are applied here.
    """
    settings = get_error_modelling_settings()
    cfg = config or load_config()
    root = Path(zarr_root) if zarr_root is not None else default_output_root(cfg)
    pollutant = pollutant or settings.pollutant
    era5_variables = list(era5_variables or settings.era5_variables)
    resolved_transform = _resolve_pollutant_log_transform(pollutant, pollutant_log_transform)
    period = period or (_period_for_years(years) if years is not None else None)

    logger.info(
        "Loading training Zarr cache from %s pollutant=%s log_transform=%s period=%s",
        root,
        pollutant,
        resolved_transform,
        period,
    )
    cams_forecast, cams_analysis, era5, meta = open_training_zarr_cache(
        output_root=root,
        chunks=chunks,
        config=cfg,
    )
    logger.info(
        "Opened Zarr cache meta years=%s time_n=%s era5_grid=%s",
        meta.get("years"),
        meta.get("time", {}).get("n"),
        meta.get("era5", {}).get("grid"),
    )

    cams_forecast, cams_analysis = _select_pollutant_datasets(cams_forecast, cams_analysis, pollutant)
    era5 = _select_era5_variables(era5, era5_variables)

    if period is not None:
        cams_forecast = _select_period(cams_forecast, period)
        cams_analysis = _select_period(cams_analysis, period)
        era5 = _select_period(era5, period)
        cams_forecast, cams_analysis, era5 = _select_common_times(cams_forecast, cams_analysis, era5)

    cams_forecast, cams_analysis = _apply_pollutant_log_transform(
        cams_forecast,
        cams_analysis,
        pollutant,
        resolved_transform,
    )
    _log_time_summary("CAMS forecast from Zarr", cams_forecast)
    _log_time_summary("CAMS analysis from Zarr", cams_analysis)
    _log_time_summary("ERA5 from Zarr", era5)

    return ErrorModellingData(
        cams_forecast=cams_forecast,
        cams_analysis=cams_analysis,
        era5=era5,
        pollutant=pollutant,
        source="zarr",
        pollutant_log_transform=resolved_transform,
    )


def resolve_training_zarr_root(
    zarr_root: str | Path | None = None,
    config: Mapping | None = None,
) -> Path:
    cfg = config or load_config()
    return Path(zarr_root) if zarr_root is not None else default_output_root(cfg)


def training_zarr_available(
    zarr_root: str | Path | None = None,
    config: Mapping | None = None,
) -> bool:
    root = resolve_training_zarr_root(zarr_root=zarr_root, config=config)
    return (root / "meta.json").is_file()
