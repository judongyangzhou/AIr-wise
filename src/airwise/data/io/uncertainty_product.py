"""Build, read, and write daily gridded uncertainty products."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from airwise.data.io.netcdf import load_netcdf, write_netcdf_atomic
from airwise.domain.pollutants import POLLUTANT_IDS

DEFAULT_LEAD_HOURS = tuple(range(0, 24, 3))


def bulletin_uq_path(directory: str | Path, report_date: date) -> Path:
    return Path(directory) / f"bulletin_uq_{report_date.isoformat()}.nc"


def build_uq_dataset(
    *,
    report_date: date,
    times: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    confidence: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    cams_aqi: np.ndarray,
    land_mask: np.ndarray,
    transforms: Sequence[str],
    is_native_lead: np.ndarray,
    lead_hours: Sequence[int] | None = None,
    pollutants: Sequence[str] = POLLUTANT_IDS,
    attrs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    resolved_leads = tuple(
        int(hour) for hour in (lead_hours or DEFAULT_LEAD_HOURS)
    )
    dataset = xr.Dataset(
        data_vars={
            "confidence_score": (
                ("pollutant", "time", "latitude", "longitude"),
                np.asarray(confidence, dtype=np.float32),
            ),
            "cams_aqi": (
                ("pollutant", "time", "latitude", "longitude"),
                np.asarray(cams_aqi, dtype=np.int8),
            ),
            "mu": (
                ("pollutant", "lead_time", "latitude", "longitude"),
                np.asarray(mu, dtype=np.float32),
            ),
            "sigma": (
                ("pollutant", "lead_time", "latitude", "longitude"),
                np.asarray(sigma, dtype=np.float32),
            ),
            "land_mask": (
                ("latitude", "longitude"),
                np.asarray(land_mask, dtype=np.int8),
            ),
            "is_native_lead": (
                ("time",),
                np.asarray(is_native_lead, dtype=np.int8),
            ),
            "mu_transform": (("pollutant",), np.asarray(list(transforms))),
        },
        coords={
            "pollutant": list(pollutants),
            "time": np.asarray(times),
            "lead_time": np.arange(len(resolved_leads), dtype=np.int16),
            "lead_hour": (
                "lead_time",
                np.asarray(resolved_leads, dtype=np.int16),
            ),
            "latitude": np.asarray(latitude),
            "longitude": np.asarray(longitude),
        },
        attrs={
            "title": "Daily bulletin AQI confidence product",
            "report_date": report_date.isoformat(),
            "confidence_definition": (
                "100 * P(CAMS forecast AQI bin | interpolated mu, sigma). "
                "Hours after the final native lead clamp to that lead."
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            **dict(attrs or {}),
        },
    )
    dataset["confidence_score"].attrs.update(
        {
            "units": "%",
            "long_name": "AQI confidence score",
            "valid_min": np.float32(0),
            "valid_max": np.float32(100),
        }
    )
    dataset["mu"].attrs["long_name"] = "Predictive mean in pollutant model space"
    dataset["sigma"].attrs["long_name"] = "Predictive std in pollutant model space"
    dataset["cams_aqi"].attrs["long_name"] = "European AQI level of CAMS forecast"
    return dataset


def _encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    latitude = int(dataset.sizes["latitude"])
    longitude = int(dataset.sizes["longitude"])
    spatial = (min(64, latitude), min(128, longitude))
    encoding: dict[str, dict[str, Any]] = {}
    for name in ("confidence_score", "mu", "sigma"):
        if name in dataset:
            encoding[name] = {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
                "chunksizes": (1,) * (dataset[name].ndim - 2) + spatial,
            }
    if "cams_aqi" in dataset:
        encoding["cams_aqi"] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "i1",
            "_FillValue": np.int8(-1),
            "chunksizes": (1, 1) + spatial,
        }
    if "land_mask" in dataset:
        encoding["land_mask"] = {"zlib": True, "complevel": 4, "dtype": "i1"}
    return encoding


def write_uq_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    return write_netcdf_atomic(dataset, path, encoding=_encoding(dataset))


def load_uq_netcdf(path: str | Path) -> xr.Dataset:
    return load_netcdf(path)
