"""Shared NetCDF read/write policy for AIr-wise products."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import xarray as xr


NETCDF_ENGINE = "netcdf4"
NETCDF_FORMAT = "NETCDF4"


def write_netcdf_atomic(
    dataset: xr.Dataset,
    path: str | Path,
    *,
    encoding: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write a compressed NetCDF product atomically using the project engine."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(
        temporary,
        mode="w",
        format=NETCDF_FORMAT,
        engine=NETCDF_ENGINE,
        encoding=dict(encoding or {}),
    )
    temporary.replace(output)
    return output


def load_netcdf(path: str | Path) -> xr.Dataset:
    """Load a NetCDF product eagerly so callers do not retain file handles."""
    return xr.load_dataset(path, engine=NETCDF_ENGINE)
