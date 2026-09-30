"""Low-level access to the local training Zarr stores."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import xarray as xr

from airwise.config import load_config, require_config, resolve_repo_path

logger = logging.getLogger(__name__)

CAMS_FORECAST_NAME = "cams_forecast.zarr"
CAMS_ANALYSIS_NAME = "cams_analysis.zarr"
ERA5_NAME = "era5.zarr"
META_NAME = "meta.json"


def _require_zarr() -> None:
    try:
        import zarr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading the training cache requires the `zarr` package. "
            "Install it with `pip install 'airwise[zarr]'`."
        ) from exc


def default_output_root(config: Mapping | None = None) -> Path:
    cfg = config or load_config()
    return resolve_repo_path(require_config(cfg, "paths", "training_zarr"))


def _resolve_open_chunks(chunks: dict | None) -> dict | None:
    if chunks is None:
        return None
    try:
        import dask  # noqa: F401
    except ImportError:
        logger.warning(
            "dask is not installed; opening training Zarr without chunks=%s",
            chunks,
        )
        return None
    return chunks


def open_training_zarr_cache(
    output_root: str | Path | None = None,
    chunks: dict | None = None,
    config: Mapping | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, dict[str, Any]]:
    """Open a previously built training Zarr cache."""
    _require_zarr()
    root = Path(output_root) if output_root is not None else default_output_root(config)
    meta_path = root / META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Training Zarr metadata not found: {meta_path}")

    open_chunks = _resolve_open_chunks(chunks)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cams_forecast = xr.open_zarr(
        root / CAMS_FORECAST_NAME,
        chunks=open_chunks,
        consolidated=True,
    )
    cams_analysis = xr.open_zarr(
        root / CAMS_ANALYSIS_NAME,
        chunks=open_chunks,
        consolidated=True,
    )
    era5 = xr.open_zarr(
        root / ERA5_NAME,
        chunks=open_chunks,
        consolidated=True,
    )
    return cams_forecast, cams_analysis, era5, meta
