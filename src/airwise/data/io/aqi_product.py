"""Read and write daily AQI NetCDF products."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from airwise.data.io.netcdf import write_netcdf_atomic


AQI_ENCODING = {
    "dtype": "int16",
    "zlib": True,
    "complevel": 5,
    "_FillValue": -9999,
}


def save_aqi_dataset(dataset: xr.Dataset, path: str | Path) -> Path:
    encoding = {name: dict(AQI_ENCODING) for name in dataset.data_vars}
    return write_netcdf_atomic(dataset, path, encoding=encoding)
