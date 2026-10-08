"""Build daily air quality bulletin JSON from CAMS forecast data."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from airwise.config import load_config, pollutant_variables, require_config, resolve_repo_path
from airwise.data.io.aqi_product import save_aqi_dataset
from airwise.data.io.cams import (
    load_cams_dataset,
    resolve_cams_nc_path,
    resolve_daily_nc_path,
    select_report_day,
)
from airwise.data.preprocessing.masks import load_cams_land_mask
from airwise.domain.aqi import classify_aqi
from airwise.domain.pollutants import POLLUTANT_IDS
from airwise.domain.regions import RegionDef
from airwise.pipelines.compute_confidence import ensure_daily_uq_dataset
from airwise.reporting.confidence_cells import attach_confidence_to_report
from airwise.reporting.json import write_report_json as _write_report_json
from airwise.reporting.region_masks import build_region_land_masks, default_nuts_shapefile
from airwise.reporting.source_receptor import attach_transboundary_pollution

logger = logging.getLogger(__name__)

POLLUTANT_LABELS: dict[str, dict[str, str]] = {
    "no2": {"label": "NO2", "unit": "ug/m3"},
    "o3": {"label": "O3", "unit": "ug/m3"},
    "pm25": {"label": "PM2.5", "unit": "ug/m3"},
    "pm10": {"label": "PM10", "unit": "ug/m3"},
}


def pollutant_meta() -> dict[str, dict[str, str]]:
    """Report pollutant metadata, with variable names taken from config."""
    variables = pollutant_variables()
    return {
        pollutant_id: {**labels, "variable": variables[pollutant_id]}
        for pollutant_id, labels in POLLUTANT_LABELS.items()
    }


def load_region_config(path: str | Path) -> tuple[str, list[RegionDef]]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    country = data["country"]
    regions = [
        RegionDef(
            id=region["id"],
            name=region["name"],
            representative_city=region["representative_city"],
            lat=float(region["lat"]),
            lon=float(region["lon"]),
            nuts_id=region.get("nuts_id"),
        )
        for region in data["regions"]
    ]
    return country, regions


def _format_iso_time(timestamp: pd.Timestamp) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_point_series(
    day_ds: xr.Dataset,
    variable: str,
    lat: float,
    lon: float,
) -> np.ndarray:
    series = day_ds[variable].sel(latitude=lat, longitude=lon, method="nearest")
    values = np.asarray(series.values, dtype=float)
    if values.shape != (24,):
        raise ValueError(f"Expected 24 values for {variable}, got shape {values.shape}")
    return values


def _round_series(values: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in np.asarray(values, dtype=float).tolist()]


def _apply_region_mask(field: xr.DataArray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(field.values, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected (time, lat, lon) field, got shape {values.shape}")
    if mask.shape != values.shape[1:]:
        raise ValueError(
            f"Region mask shape {mask.shape} does not match field spatial shape {values.shape[1:]}"
        )
    masked = np.where(np.asarray(mask, dtype=bool), values, np.nan)
    if not np.isfinite(masked).any():
        raise ValueError("Region mask contains no finite land-grid values")
    return masked


def _spatial_max_series(masked: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        hourly = np.nanmax(masked, axis=(1, 2))
    if hourly.shape != (24,):
        raise ValueError(f"Expected 24 hourly maxima, got shape {hourly.shape}")
    if not np.isfinite(hourly).all():
        raise ValueError("Hourly spatial maximum contains non-finite values")
    return hourly.astype(float)


def _global_max_location(masked: np.ndarray) -> tuple[int, int, int, float]:
    flat = int(np.nanargmax(masked))
    time_index, lat_index, lon_index = np.unravel_index(flat, masked.shape)
    return int(time_index), int(lat_index), int(lon_index), float(masked[time_index, lat_index, lon_index])


def _build_pollutant_cell(
    conc_values: np.ndarray,
    aqi_values: np.ndarray,
    *,
    max_lat: float | None = None,
    max_lon: float | None = None,
) -> dict[str, Any]:
    max_idx = int(np.nanargmax(conc_values))
    max_concentration = round(float(conc_values[max_idx]), 2)
    aqi_level = int(aqi_values[max_idx])
    if not 1 <= aqi_level <= 6:
        raise ValueError(f"AQI level must be between 1 and 6, got {aqi_level}")
    cell: dict[str, Any] = {
        "max_concentration": max_concentration,
        "max_time": f"{max_idx:02d}:00",
        "aqi_level": aqi_level,
        "confidence_score": None,
    }
    if max_lat is not None and max_lon is not None:
        cell["max_lat"] = float(max_lat)
        cell["max_lon"] = float(max_lon)
    return cell


def extract_region_data(
    conc_day: xr.Dataset,
    aqi_day: xr.Dataset,
    region: RegionDef,
    mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cells: dict[str, Any] = {}
    series: dict[str, Any] = {}
    latitudes = np.asarray(conc_day["latitude"].values, dtype=float)
    longitudes = np.asarray(conc_day["longitude"].values, dtype=float)

    variables = pollutant_variables()
    for pollutant_id in POLLUTANT_IDS:
        variable = variables[pollutant_id]
        if mask is None:
            conc_values = _extract_point_series(conc_day, variable, region.lat, region.lon)
            aqi_values = _extract_point_series(aqi_day, variable, region.lat, region.lon)
            cells[pollutant_id] = _build_pollutant_cell(conc_values, aqi_values)
            series[pollutant_id] = {"values": _round_series(conc_values)}
            continue

        conc_masked = _apply_region_mask(conc_day[variable], mask)
        aqi_masked = _apply_region_mask(aqi_day[variable], mask)
        conc_values = _spatial_max_series(conc_masked)
        time_index, lat_index, lon_index, peak = _global_max_location(conc_masked)
        aqi_at_peak = int(aqi_masked[time_index, lat_index, lon_index])
        if not 1 <= aqi_at_peak <= 6:
            raise ValueError(f"AQI level must be between 1 and 6, got {aqi_at_peak}")
        cells[pollutant_id] = {
            "max_concentration": round(float(peak), 2),
            "max_time": f"{time_index:02d}:00",
            "aqi_level": aqi_at_peak,
            "confidence_score": None,
            "max_lat": float(latitudes[lat_index]),
            "max_lon": float(longitudes[lon_index]),
        }
        series[pollutant_id] = {
            "values": _round_series(conc_values),
            "distribution": _round_series(conc_masked[np.isfinite(conc_masked)]),
        }

    region_summary = {
        "id": region.id,
        "name": region.name,
        "representative_city": region.representative_city,
        "lat": region.lat,
        "lon": region.lon,
        "cells": cells,
    }
    if region.nuts_id:
        region_summary["nuts_id"] = region.nuts_id
    region_time_series = {
        "region_id": region.id,
        "series": series,
    }
    return region_summary, region_time_series


def build_daily_bulletin_json(
    *,
    country: str,
    regions: list[RegionDef],
    report_date: date,
    conc_day: xr.Dataset,
    aqi_day: xr.Dataset,
    uq_ds: xr.Dataset,
    project_ref: str = "AIr-wise / Code for Earth 2026",
    project_title: str = "AI-Based Uncertainty-Aware Air Quality Assessment",
    region_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    region_summaries: list[dict[str, Any]] = []
    region_series: list[dict[str, Any]] = []

    for region in regions:
        mask = None if region_masks is None else region_masks.get(region.id)
        summary, time_series = extract_region_data(conc_day, aqi_day, region, mask=mask)
        region_summaries.append(summary)
        region_series.append(time_series)

    times = [_format_iso_time(ts) for ts in pd.to_datetime(conc_day.time.values)]
    uses_shapefile = region_masks is not None
    table_interpretation = (
        "Table Interpretation (to be included on the website as information): "
        "[Max concentration | timestamp of max value | AQI confidence score]"
    )
    heatmap_interpretation = (
        "Each pollutant panel shows hourly concentration by region, "
        "coloured with European AQI thresholds. Box plots show the daily "
        "distribution for each region."
    )
    if uses_shapefile:
        table_interpretation += (
            ". Maxima are the highest land-grid concentration inside the NUTS region polygon."
        )
        heatmap_interpretation = (
            "Each pollutant panel shows the hourly spatial-maximum concentration "
            "among land grid points in the region, coloured with European AQI "
            "thresholds. Box plots show the daily distribution of all land-grid "
            "values in each region."
        )

    report_data = {
        "schema_version": "1.0",
        "report_type": "daily_air_quality_bulletin",
        "metadata": {
            "title": f"Daily Air Quality bulletin for {country}",
            "country": country,
            "report_date": report_date.isoformat(),
            "project_ref": project_ref,
            "project_title": project_title,
        },
        "pollutant_summary_by_region": {
            "section_title": "2 Pollutant Summary per region",
            "interpretation": table_interpretation,
            "pollutant_order": list(POLLUTANT_IDS),
            "pollutants": pollutant_meta(),
            "regions": region_summaries,
        },
        "region_leadtime_heatmap": {
            "section_title": "3 Regional concentration by lead time",
            "interpretation": heatmap_interpretation,
        },
        "forecast_time_series": {
            "section_title": "4 Forecast concentration time series",
            "times": times,
            "regions": region_series,
        },
    }
    attach_confidence_to_report(report_data, uq_ds)
    return report_data


def _resolve_aqi_nc_path(
    aqi_dir: str | Path,
    report_date: date,
    *,
    legacy_monthly_dir: str | Path | None = None,
) -> Path | None:
    """Resolve a daily AQI NetCDF, with optional legacy monthly fallback."""
    daily = resolve_daily_nc_path(Path(aqi_dir), report_date)
    if daily.is_file():
        return daily
    if legacy_monthly_dir is not None:
        try:
            return resolve_cams_nc_path(legacy_monthly_dir, report_date)
        except FileNotFoundError:
            return None
    return None


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    candidate = default if path is None else Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        repo_candidate = resolve_repo_path(candidate)
        if repo_candidate.is_file():
            return repo_candidate
    return candidate


def _build_masks_for_bulletin(
    *,
    conc_day: xr.Dataset,
    regions: list[RegionDef],
    nuts_shapefile: str | Path | None,
    land_sea_mask_path: str | Path | None,
) -> dict[str, np.ndarray] | None:
    nuts_id_by_region = {
        region.id: region.nuts_id for region in regions if region.nuts_id
    }
    if not nuts_id_by_region:
        return None
    shapefile = _resolve_path(nuts_shapefile, default_nuts_shapefile())
    land_path = _resolve_path(
        land_sea_mask_path,
        resolve_repo_path(require_config(load_config(), "paths", "land_sea_mask")),
    )
    logger.info("Rasterising NUTS land masks from %s", shapefile)
    land_mask = None
    if land_path.is_file():
        land_mask = load_cams_land_mask(
            land_path,
            latitude=conc_day["latitude"].values,
            longitude=conc_day["longitude"].values,
        )
        logger.info(
            "Applied CAMS land mask from %s (%d land cells)",
            land_path,
            int(np.asarray(land_mask).sum()),
        )
    else:
        logger.warning(
            "Land-sea mask not found at %s; using polygon cell centres only",
            land_path,
        )
    masks = build_region_land_masks(
        latitude=conc_day["latitude"].values,
        longitude=conc_day["longitude"].values,
        nuts_id_by_region=nuts_id_by_region,
        shapefile=shapefile,
        land_mask=land_mask,
    )
    for region_id, mask in masks.items():
        logger.info("Region %s: %d land grid points", region_id, int(mask.sum()))
    return masks


def build_daily_bulletin_from_paths(
    *,
    report_date: date,
    region_config_path: str | Path,
    raw_forecast_dir: str | Path,
    aqi_forecast_dir: str | Path | None = None,
    daily_forecast_dir: str | Path | None = None,
    daily_aqi_dir: str | Path | None = None,
    uq_dir: str | Path | None = None,
    open_ifs_root: str | Path | None = None,
    force_confidence: bool = False,
    device: str | None = None,
    nuts_shapefile: str | Path | None = None,
    land_sea_mask_path: str | Path | None = None,
) -> dict[str, Any]:
    country, regions = load_region_config(region_config_path)
    logger.info(
        "Building bulletin for %s (%s, %d regions)",
        report_date.isoformat(),
        country,
        len(regions),
    )

    raw_path = resolve_cams_nc_path(
        raw_forecast_dir,
        report_date,
        daily_dir=daily_forecast_dir,
    )
    logger.info("Using CAMS concentration file %s", raw_path)
    aqi_dir = Path(
        daily_aqi_dir
        or aqi_forecast_dir
        or resolve_repo_path(require_config(load_config(), "paths", "cams_aqi_daily"))
    )
    aqi_path = _resolve_aqi_nc_path(
        aqi_dir,
        report_date,
        legacy_monthly_dir=aqi_forecast_dir,
    )

    conc_ds = load_cams_dataset(raw_path)
    aqi_ds = None
    uq_ds = None
    try:
        conc_day = select_report_day(conc_ds, report_date)
        if aqi_path is not None:
            logger.info("Using AQI file %s", aqi_path)
            aqi_ds = load_cams_dataset(aqi_path)
            aqi_day = select_report_day(aqi_ds, report_date)
        else:
            logger.warning(
                "AQI file not found under %s; classifying AQI from concentrations",
                aqi_dir,
            )
            aqi_day = classify_aqi(conc_day)
            if raw_path.name == f"{report_date:%Y%m%d}.nc":
                saved = save_aqi_dataset(
                    aqi_day,
                    resolve_daily_nc_path(aqi_dir, report_date),
                )
                logger.info("Wrote computed AQI file to %s", saved)
        logger.info("Attaching required AQI confidence scores (force=%s)", force_confidence)
        uq_ds = ensure_daily_uq_dataset(
            report_date=report_date,
            conc_day=conc_day,
            aqi_day=aqi_day,
            uq_dir=uq_dir,
            open_ifs_root=open_ifs_root,
            force=force_confidence,
            device=device,
        )
        logger.info("Confidence product loaded for region-max sampling")
        region_masks = _build_masks_for_bulletin(
            conc_day=conc_day,
            regions=regions,
            nuts_shapefile=nuts_shapefile,
            land_sea_mask_path=land_sea_mask_path,
        )
        return build_daily_bulletin_json(
            country=country,
            regions=regions,
            report_date=report_date,
            conc_day=conc_day,
            aqi_day=aqi_day,
            uq_ds=uq_ds,
            region_masks=region_masks,
        )
    finally:
        conc_ds.close()
        if aqi_ds is not None:
            aqi_ds.close()
        if uq_ds is not None:
            uq_ds.close()


def write_report_json(
    output_path: str | Path,
    report_data: dict[str, Any],
    *,
    validate: bool = True,
) -> Path:
    output = _write_report_json(output_path, report_data, validate=validate)
    logger.info("Wrote bulletin JSON to %s", output)
    return output


def generate_report_json(
    *,
    report_date: date,
    region_config_path: str | Path,
    output_path: str | Path,
    raw_forecast_dir: str | Path | None = None,
    aqi_forecast_dir: str | Path | None = None,
    daily_forecast_dir: str | Path | None = None,
    daily_aqi_dir: str | Path | None = None,
    uq_dir: str | Path | None = None,
    open_ifs_root: str | Path | None = None,
    force_confidence: bool = False,
    device: str | None = None,
    nuts_shapefile: str | Path | None = None,
    land_sea_mask_path: str | Path | None = None,
    policy_dir: str | Path | None = None,
) -> Path:
    cfg = load_config()
    paths = cfg["paths"]
    raw_root = (
        Path(raw_forecast_dir)
        if raw_forecast_dir is not None
        else resolve_repo_path(require_config(paths, "cams_data_raw"))
    )
    raw_dir = raw_root / "forecast"
    daily_raw = Path(daily_forecast_dir) if daily_forecast_dir is not None else resolve_repo_path(
        require_config(paths, "cams_forecast_daily")
    )
    daily_aqi = Path(
        daily_aqi_dir
        or aqi_forecast_dir
        or resolve_repo_path(require_config(paths, "cams_aqi_daily"))
    )
    shapefile = nuts_shapefile or require_config(paths, "nuts_shapefile")
    land_mask = resolve_repo_path(land_sea_mask_path or require_config(paths, "land_sea_mask"))
    logger.info(
        "Resolved data roots: forecast_monthly=%s forecast_daily=%s aqi_daily=%s shapefile=%s",
        raw_dir,
        daily_raw,
        daily_aqi,
        shapefile,
    )

    report_data = build_daily_bulletin_from_paths(
        report_date=report_date,
        region_config_path=region_config_path,
        raw_forecast_dir=raw_dir,
        aqi_forecast_dir=None,
        daily_forecast_dir=daily_raw,
        daily_aqi_dir=daily_aqi,
        uq_dir=uq_dir,
        open_ifs_root=open_ifs_root,
        force_confidence=force_confidence,
        device=device,
        nuts_shapefile=shapefile,
        land_sea_mask_path=land_mask,
    )
    attach_transboundary_pollution(report_data, report_date, policy_dir=policy_dir)
    if report_data.get("transboundary_pollution") is None:
        logger.info("Transboundary pollution table omitted")
    else:
        logger.info("Attached transboundary pollution table data")
    return write_report_json(output_path, report_data)
