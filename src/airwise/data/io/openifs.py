"""Load and preprocess the daily OpenIFS control forecast for inference."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from airwise.data.preprocessing.meteorology import relative_humidity_from_dewpoint
from airwise.data.preprocessing.regridding import get_regrid_methods

OPEN_IFS_LEAD_HOURS = tuple(range(0, 25, 3))
MODEL_LEAD_HOURS = tuple(range(0, 24, 3))
CONTROL_MEMBER = 0
OPEN_IFS_CONTROL_NUMBERS = (CONTROL_MEMBER,)
OPEN_IFS_REQUIRED_VARIABLES = ("u10", "v10", "t2m", "d2m", "tp", "sp", "ssrd")
OPEN_IFS_MODEL_VARIABLES = ("u10", "v10", "t2m", "tp", "sp", "ssrd", "rh", "rain_mask")
OPEN_IFS_ACCUMULATED_VARIABLES = ("tp", "ssrd")
RAIN_THRESHOLD_MM = 1e-2


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def open_ifs_path(root: str | Path, init_date: date | str, lead_hour: int) -> Path:
    """Compatibility alias for :func:`open_ifs_control_path`."""
    return open_ifs_control_path(root, init_date, lead_hour)


def open_ifs_control_path(root: str | Path, init_date: date | str, lead_hour: int) -> Path:
    """Return the NetCDF path for one downloaded IFS control forecast lead."""
    day = _as_date(init_date)
    return (
        Path(root)
        / day.strftime("%Y%m%d")
        / f"{day:%Y%m%d}000000-{lead_hour}h-oper-fc.nc"
    )


def resolve_open_ifs_path(root: str | Path, init_date: date | str, lead_hour: int) -> Path:
    """Return the only supported production input: the control forecast."""
    return open_ifs_control_path(root, init_date, lead_hour)


def _open_open_ifs_dataset(path: Path) -> xr.Dataset:
    """Open current files and older files with a conflicting ``dtype`` attribute."""
    try:
        return xr.open_dataset(path)
    except ValueError as exc:
        if "overwriting existing key dtype" not in str(exc):
            raise

    raw = xr.open_dataset(path, decode_cf=False)
    try:
        for variable in raw.variables.values():
            variable.attrs.pop("dtype", None)
        decoded = xr.decode_cf(raw, decode_timedelta=True)
        decoded.set_close(raw.close)
        return decoded
    except Exception:
        raw.close()
        raise


def _validate_open_ifs_file(
    ds: xr.Dataset,
    path: Path,
    *,
    init_date: date,
    lead_hour: int,
) -> pd.Timestamp:
    missing = [name for name in OPEN_IFS_REQUIRED_VARIABLES if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Open IFS file {path} is missing variables {missing}")

    numbers = tuple(int(value) for value in np.atleast_1d(ds["number"].values))
    if numbers != OPEN_IFS_CONTROL_NUMBERS:
        raise ValueError(
            f"Expected OpenIFS control forecast number=0 in {path}, got {numbers}"
        )

    valid_time = pd.Timestamp(ds["valid_time"].item())
    expected = pd.Timestamp(init_date) + pd.Timedelta(hours=lead_hour)
    if valid_time != expected:
        raise ValueError(
            f"Unexpected valid_time in {path}: expected {expected}, got {valid_time}"
        )
    if "lead_hour" in ds.coords and int(ds["lead_hour"].item()) != lead_hour:
        raise ValueError(
            f"Unexpected lead_hour in {path}: expected {lead_hour}, "
            f"got {int(ds['lead_hour'].item())}"
        )
    return valid_time


def validate_open_ifs_control_file(
    path: str | Path,
    init_date: date | str,
    lead_hour: int,
) -> None:
    """Raise when an existing OpenIFS lead cannot be used for inference."""
    resolved_path = Path(path)
    day = _as_date(init_date)
    with _open_open_ifs_dataset(resolved_path) as source:
        _validate_open_ifs_file(
            source,
            resolved_path,
            init_date=day,
            lead_hour=int(lead_hour),
        )


def load_open_ifs_day(
    root: str | Path,
    init_date: date | str,
    *,
    lead_hours: Sequence[int] = OPEN_IFS_LEAD_HOURS,
) -> xr.Dataset:
    """
    Lazily combine one 00Z OpenIFS control forecast along ``time``.

    Every input must contain exactly ``number=0``. Files remain lazily backed
    by their NetCDF sources until values are selected and materialised.
    """
    day = _as_date(init_date)
    datasets: list[xr.Dataset] = []
    sources: list[xr.Dataset] = []
    try:
        for lead_hour in lead_hours:
            path = resolve_open_ifs_path(root, day, int(lead_hour))
            if not path.is_file():
                raise FileNotFoundError(f"Missing Open IFS lead file: {path}")

            source = _open_open_ifs_dataset(path)
            sources.append(source)
            valid_time = _validate_open_ifs_file(
                source,
                path,
                init_date=day,
                lead_hour=int(lead_hour),
            )
            selected = source[list(OPEN_IFS_REQUIRED_VARIABLES)]
            selected = selected.drop_vars(
                [
                    name
                    for name in ("time", "step", "valid_time", "lead_hour")
                    if name in selected.coords
                ],
                errors="ignore",
            )
            selected = selected.expand_dims(time=[valid_time])
            selected = selected.assign_coords(lead_hour=("time", [int(lead_hour)]))
            datasets.append(selected)

        combined = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
        combined.attrs.update(
            {
                "source": "ECMWF IFS control forecast (oper/fc)",
                "init_date": day.isoformat(),
                "init_time": "00Z",
                "forecast_member": CONTROL_MEMBER,
            }
        )
        combined.set_close(lambda: [source.close() for source in sources])
        return combined
    except Exception:
        for source in sources:
            source.close()
        raise


def deaccumulate_open_ifs(
    ds: xr.Dataset,
    *,
    accumulated_variables: Sequence[str] = OPEN_IFS_ACCUMULATED_VARIABLES,
) -> xr.Dataset:
    """
    Convert forecast accumulations to average hourly amounts.

    Open IFS ``tp`` and ``ssrd`` are accumulated from 00Z. Consecutive
    three-hour leads are differenced and divided by three to approximate the
    hourly ERA5 accumulation fields used for training. The 0h value is zero.
    """
    if ds.sizes.get("time", 0) < 2:
        raise ValueError("At least two Open IFS leads are required for deaccumulation.")

    lead_hours = np.asarray(ds["lead_hour"].values, dtype=np.int64)
    intervals = np.diff(lead_hours)
    if np.any(intervals <= 0):
        raise ValueError(f"Open IFS lead hours must be strictly increasing, got {lead_hours}")

    out = ds.copy()
    for name in accumulated_variables:
        if name not in out:
            raise KeyError(f"Cannot deaccumulate missing Open IFS variable {name!r}")
        cumulative = out[name]
        first = xr.zeros_like(cumulative.isel(time=0))
        increments = cumulative.diff("time").clip(min=0)
        interval_hours = xr.DataArray(
            intervals.astype("float32"),
            dims=("time",),
            coords={"time": increments["time"]},
        )
        hourly = increments / interval_hours
        out[name] = xr.concat([first, hourly], dim="time").assign_coords(time=ds["time"])
        out[name].attrs = dict(cumulative.attrs)
        out[name].attrs["open_ifs_accumulation_processing"] = (
            "difference consecutive forecast accumulations and divide by interval hours"
        )

    out["tp"] = out["tp"] * 1000.0
    out["tp"].attrs.update({"units": "mm", "unit_conversion": "m_to_mm"})
    return out


def regrid_open_ifs_to_cams(
    ds: xr.Dataset,
    target_grid: xr.Dataset,
    *,
    methods: dict[str, str] | None = None,
) -> xr.Dataset:
    """Interpolate Open IFS fields onto an exact CAMS latitude/longitude grid."""
    resolved_methods = methods or get_regrid_methods()
    regridded: list[xr.Dataset] = []
    for name in OPEN_IFS_REQUIRED_VARIABLES:
        method = resolved_methods.get(name, "linear")
        if method == "bilinear":
            method = "linear"
        da = ds[name].interp(
            latitude=target_grid["latitude"],
            longitude=target_grid["longitude"],
            method=method,
        )
        da.attrs = dict(ds[name].attrs)
        da.attrs["regrid_method"] = method
        regridded.append(da.to_dataset(name=name))
    return xr.merge(regridded, compat="override", join="exact")
