"""CAMS NetCDF loading and preprocessing utilities."""

from __future__ import annotations

import calendar
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import xarray as xr


def resolve_monthly_nc_path(base_dir: Path, report_date: date) -> Path:
    """Return monthly CAMS file path for a given report date."""
    start = report_date.replace(day=1)
    last_day = calendar.monthrange(report_date.year, report_date.month)[1]
    end = report_date.replace(day=last_day)
    return Path(base_dir) / f"{start:%Y-%m-%d}_{end:%Y-%m-%d}.nc"


def resolve_daily_nc_path(base_dir: Path, report_date: date) -> Path:
    """Return daily CAMS file path ``YYYYMMDD.nc`` for a given report date."""
    return Path(base_dir) / f"{report_date:%Y%m%d}.nc"


def resolve_cams_nc_path(
    monthly_dir: str | Path,
    report_date: date,
    *,
    daily_dir: str | Path | None = None,
) -> Path:
    """Prefer a monthly CAMS file, then a daily ``YYYYMMDD.nc`` file."""
    monthly = resolve_monthly_nc_path(Path(monthly_dir), report_date)
    if monthly.is_file():
        return monthly
    if daily_dir is not None:
        daily = resolve_daily_nc_path(Path(daily_dir), report_date)
        if daily.is_file():
            return daily
        raise FileNotFoundError(
            f"CAMS NetCDF not found. Tried monthly path {monthly} and daily path {daily}"
        )
    raise FileNotFoundError(f"CAMS NetCDF not found: {monthly}")


def reassgin_time_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rebuild hourly datetime coordinates from CAMS time.long_name metadata."""
    timestamp = ds.time.long_name[19:27]
    timestamp_init = datetime.strptime(timestamp, "%Y%m%d")
    time_coords = (
        pd.date_range(timestamp_init, periods=len(ds.time), freq="1h")
        .strftime("%Y-%m-%d %H:%M:%S")
        .astype("datetime64[ns]")
    )
    return ds.assign_coords(time=time_coords)


def longitude_reassign(ds: xr.Dataset) -> xr.Dataset:
    """Normalise longitude from 0-360 to -180-180 and sort."""
    return ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")


def squeeze_dimensions(ds: xr.Dataset) -> xr.Dataset:
    """Drop singleton dimensions such as surface level."""
    return ds.squeeze(drop=True)


def _needs_time_reassignment(ds: xr.Dataset) -> bool:
    return not pd.api.types.is_datetime64_any_dtype(ds.time.dtype)


def _needs_longitude_reassignment(ds: xr.Dataset) -> bool:
    return float(ds.longitude.max()) > 180.0


def preprocess_cams_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Apply standard CAMS preprocessing (time, longitude, squeeze)."""
    if _needs_time_reassignment(ds):
        ds = reassgin_time_coords(ds)
    if _needs_longitude_reassignment(ds):
        ds = longitude_reassign(ds)
    return squeeze_dimensions(ds)


def load_cams_dataset(path: str | Path) -> xr.Dataset:
    """Open a CAMS NetCDF file and apply preprocessing."""
    ds = xr.open_dataset(path)
    return preprocess_cams_dataset(ds)


def select_report_day(ds: xr.Dataset, report_date: date) -> xr.Dataset:
    """Select 24 hourly timesteps for a single report date."""
    day_start = pd.Timestamp(report_date)
    day_end = day_start + pd.Timedelta(hours=23, minutes=59, seconds=59)
    day = ds.sel(time=slice(day_start, day_end))
    if day.sizes.get("time", 0) != 24:
        raise ValueError(
            f"Expected 24 hourly timesteps for {report_date:%Y-%m-%d}, "
            f"got {day.sizes.get('time', 0)}"
        )
    return day
