"""Align the CAMS land-sea mask to a target latitude/longitude grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_LAND_SEA_MASK_PATH = Path("data/land_sea_mask/land_sea_mask.nc")


def _wrap_longitude_to_minus180_180(longitude) -> np.ndarray:
    values = np.asarray(longitude, dtype=np.float64)
    return ((values + 180.0) % 360.0) - 180.0


def load_cams_land_mask(
    path: str | Path = DEFAULT_LAND_SEA_MASK_PATH,
    *,
    latitude=None,
    longitude=None,
    land_threshold: float = 0.5,
) -> np.ndarray:
    """
    Load the CAMS regional land-sea mask aligned to a target lat/lon grid.

    The source file stores ``lsm`` as 0 (sea) / 1 (land) on a 0-360 longitude
    axis with ascending latitude. Target CAMS/Zarr grids commonly use
    [-25, 45] longitude and descending latitude.
    """
    import xarray as xr

    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Land-sea mask not found: {mask_path}")

    with xr.open_dataset(mask_path) as ds:
        if "lsm" not in ds.data_vars:
            raise KeyError(f"Expected variable 'lsm' in {mask_path}, found {list(ds.data_vars)}")
        land = ds["lsm"].load()
        land = land.assign_coords(longitude=_wrap_longitude_to_minus180_180(land["longitude"]))
        land = land.sortby("longitude")
        if latitude is not None or longitude is not None:
            reindex_kwargs = {}
            if latitude is not None:
                reindex_kwargs["latitude"] = np.asarray(latitude)
            if longitude is not None:
                reindex_kwargs["longitude"] = np.asarray(longitude)
            land = land.reindex(**reindex_kwargs, method="nearest", tolerance=1e-4)
        mask = (land.values > land_threshold).astype(bool)

    if mask.ndim != 2:
        raise ValueError(f"Land mask must be 2D [latitude, longitude], got shape {mask.shape}")
    if not np.any(mask):
        raise ValueError(f"Land mask from {mask_path} contains no land points after alignment.")
    return mask
