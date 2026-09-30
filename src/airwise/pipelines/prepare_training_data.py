"""Build a Zarr cache so training can read CAMS and ERA5 quickly.

Downloads stay in NetCDF. Training opens the same years many times and slices
pollutants and meteorology together, so this module writes chunked Zarr stores
that the error model reads directly.

The cache keeps physical CAMS concentrations. Pollutant transforms such as
PM10 log1p stay in the training config. ERA5 is stored once with the shared
recipe (tp/blh log1p, rh, rain_mask). Z-score statistics stay in external JSON
files. Several pollutants share one time axis as parallel variables.

Example::

    airwise-build-training-zarr --years 2023 2024 2025 --overwrite
    airwise-build-training-zarr --stores era5 --overwrite
    airwise-build-training-zarr --years 2026 --output-root data/training_zarr_2026 --overwrite
    airwise-build-training-zarr --years 2026 --output-root data/training_zarr_2026 --stores era5 --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from airwise.config import load_config
from airwise.data.io.training_store import (
    CAMS_ANALYSIS_NAME,
    CAMS_FORECAST_NAME,
    ERA5_NAME,
    META_NAME,
    _require_zarr,
    default_output_root,
    open_training_zarr_cache,
)
from airwise.modelling.config import get_error_modelling_settings
from airwise.modelling.features import (
    load_processed_cams_features,
    load_training_era5_features,
)


logger = logging.getLogger(__name__)

DEFAULT_TIME_CHUNK = 8


def _period_for_years(years: Sequence[int]) -> tuple[str, str]:
    return f"{min(years)}-01-01 00:00:00", f"{max(years)}-12-31 23:00:00"


def _sort_and_deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds

    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    if not time_index.is_monotonic_increasing:
        ds = ds.sortby("time")
        time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))

    if time_index.has_duplicates:
        duplicate_times = time_index[time_index.duplicated(keep=False)].unique()
        preview = ", ".join(str(timestamp) for timestamp in duplicate_times[:5])
        raise ValueError(
            f"Dataset contains {len(duplicate_times)} duplicated timestamps after sorting. "
            f"First duplicates: {preview}."
        )
    return ds


def _select_period(ds: xr.Dataset, period: tuple[str, str] | None) -> xr.Dataset:
    ds = _sort_and_deduplicate_time(ds)
    if period is None:
        return ds
    start = pd.Timestamp(period[0])
    end = pd.Timestamp(period[1])
    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    time_mask = (time_index >= start) & (time_index <= end)
    return ds.isel(time=np.flatnonzero(time_mask))


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


STORE_NAMES = ("cams_forecast", "cams_analysis", "era5")


def _encoding_for_dataset(ds: xr.Dataset, time_chunk: int) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for var_name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(min(time_chunk, int(da.sizes["time"])))
            else:
                chunks.append(int(da.sizes[dim]))
        encoding[var_name] = {
            "chunks": tuple(chunks),
            "dtype": "float32",
        }
    return encoding


def _rechunk_for_zarr_write(ds: xr.Dataset, time_chunk: int) -> xr.Dataset:
    """Align dask chunks with the Zarr encoding to avoid safe_chunks errors."""
    try:
        import dask  # noqa: F401
    except ImportError:
        return ds

    # Numpy-backed datasets do not need rechunking for to_zarr.
    if not any(hasattr(da.data, "chunks") for da in ds.data_vars.values()):
        return ds

    chunk_sizes: dict[str, int] = {}
    if "time" in ds.dims:
        chunk_sizes["time"] = min(time_chunk, int(ds.sizes["time"]))
    for dim in ("latitude", "longitude"):
        if dim in ds.dims:
            chunk_sizes[dim] = -1
    if not chunk_sizes:
        return ds
    return ds.chunk(chunk_sizes)


def _normalize_stores(stores: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(STORE_NAMES if stores is None else stores)
    unknown = [name for name in selected if name not in STORE_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown store name(s): {unknown}. Expected one or more of: {list(STORE_NAMES)}"
        )
    if not selected:
        raise ValueError("At least one store must be selected.")
    # Preserve canonical order.
    return tuple(name for name in STORE_NAMES if name in selected)


def _cast_float32(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy()
    for var_name in out.data_vars:
        out[var_name] = out[var_name].astype("float32")
    return out


def _annotate_cams_dataset(ds: xr.Dataset, data_type: str) -> xr.Dataset:
    out = ds.copy()
    out.attrs = dict(ds.attrs)
    out.attrs.update(
        {
            "airwise_role": f"cams_{data_type}",
            "airwise_pollutant_log_transform": "none",
            "airwise_note": (
                "Physical concentrations without pollutant log/z-score. "
                "Apply per-pollutant transforms at training time."
            ),
        }
    )
    for var_name in out.data_vars:
        out[var_name].attrs = dict(ds[var_name].attrs)
        out[var_name].attrs.setdefault("pollutant_log_transform", "none")
    return out


def _annotate_era5_dataset(ds: xr.Dataset, era5_grid: str, era5_log_transform: Mapping[str, str]) -> xr.Dataset:
    out = ds.copy()
    out.attrs = dict(ds.attrs)
    out.attrs.update(
        {
            "airwise_role": "era5",
            "airwise_era5_grid": era5_grid,
            "airwise_log_transform": json.dumps(dict(era5_log_transform)),
            "airwise_note": (
                "Shared meteorology cache. Includes training recipe transforms "
                "(configured ERA5 log transforms, derived rh, rain_mask). No z-score."
            ),
        }
    )
    return out


def build_training_cache_meta(
    *,
    years: Sequence[int],
    pollutants: Sequence[str],
    temporal_resolution: int,
    era5_grid: str,
    era5_variables: Sequence[str],
    era5_log_transform: Mapping[str, str],
    pollutant_log_transform: Mapping[str, str | None],
    time_chunk: int,
    common_times: Sequence[np.datetime64] | pd.DatetimeIndex,
    cams_forecast: xr.Dataset,
    cams_analysis: xr.Dataset,
    era5: xr.Dataset,
    output_root: Path,
) -> dict[str, Any]:
    time_index = pd.DatetimeIndex(pd.to_datetime(common_times))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
        "output_root": str(output_root),
        "years": list(years),
        "temporal_resolution_hours": temporal_resolution,
        "time": {
            "n": int(len(time_index)),
            "start": time_index.min().isoformat() if len(time_index) else None,
            "end": time_index.max().isoformat() if len(time_index) else None,
        },
        "chunks": {"time": time_chunk},
        "cams": {
            "pollutants": list(pollutants),
            "pollutant_log_transform": None,
            "zscore": False,
            "forecast_store": CAMS_FORECAST_NAME,
            "analysis_store": CAMS_ANALYSIS_NAME,
            "dims": dict(cams_forecast.sizes),
        },
        "era5": {
            "grid": era5_grid,
            "variables": list(era5.data_vars),
            "requested_variables": list(era5_variables),
            "log_transform": dict(era5_log_transform),
            "zscore": False,
            "store": ERA5_NAME,
            "dims": dict(era5.sizes),
        },
        "training_notes": {
            "apply_pollutant_log_at_train_time": True,
            "recommended_pollutant_log_transform": dict(pollutant_log_transform),
            "target_error": "analysis - forecast in the chosen pollutant transform space",
            "zscore_stats": "external JSON; not stored in this cache",
        },
        "stores": {
            "cams_forecast": CAMS_FORECAST_NAME,
            "cams_analysis": CAMS_ANALYSIS_NAME,
            "era5": ERA5_NAME,
            "meta": META_NAME,
        },
        "analysis_dims": dict(cams_analysis.sizes),
    }


def write_training_zarr_datasets(
    cams_forecast: xr.Dataset | None,
    cams_analysis: xr.Dataset | None,
    era5: xr.Dataset | None,
    output_root: str | Path,
    meta: Mapping[str, Any],
    time_chunk: int = DEFAULT_TIME_CHUNK,
    overwrite: bool = False,
    stores: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write already-aligned datasets to the training Zarr layout."""
    _require_zarr()
    selected = _normalize_stores(stores)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "cams_forecast": output_root / CAMS_FORECAST_NAME,
        "cams_analysis": output_root / CAMS_ANALYSIS_NAME,
        "era5": output_root / ERA5_NAME,
        "meta": output_root / META_NAME,
    }
    if not overwrite:
        existing = [paths[name] for name in selected if paths[name].exists()]
        if existing:
            raise FileExistsError(
                "Training Zarr outputs already exist. Pass overwrite=True to replace them:\n"
                + "\n".join(str(path) for path in existing)
            )

    mode = "w" if overwrite else "w-"
    written: dict[str, Path] = {"meta": paths["meta"]}

    if "cams_forecast" in selected:
        if cams_forecast is None:
            raise ValueError("cams_forecast dataset is required when writing cams_forecast store.")
        cams_forecast = _rechunk_for_zarr_write(
            _annotate_cams_dataset(_cast_float32(cams_forecast), "forecast"),
            time_chunk,
        )
        logger.info("Writing CAMS forecast Zarr to %s", paths["cams_forecast"])
        cams_forecast.to_zarr(
            paths["cams_forecast"],
            mode=mode,
            encoding=_encoding_for_dataset(cams_forecast, time_chunk),
            consolidated=True,
        )
        written["cams_forecast"] = paths["cams_forecast"]

    if "cams_analysis" in selected:
        if cams_analysis is None:
            raise ValueError("cams_analysis dataset is required when writing cams_analysis store.")
        cams_analysis = _rechunk_for_zarr_write(
            _annotate_cams_dataset(_cast_float32(cams_analysis), "analysis"),
            time_chunk,
        )
        logger.info("Writing CAMS analysis Zarr to %s", paths["cams_analysis"])
        cams_analysis.to_zarr(
            paths["cams_analysis"],
            mode=mode,
            encoding=_encoding_for_dataset(cams_analysis, time_chunk),
            consolidated=True,
        )
        written["cams_analysis"] = paths["cams_analysis"]

    if "era5" in selected:
        if era5 is None:
            raise ValueError("era5 dataset is required when writing era5 store.")
        era5_meta = meta.get("era5", {})
        era5 = _rechunk_for_zarr_write(
            _annotate_era5_dataset(
                _cast_float32(era5),
                str(era5_meta.get("grid", "cams")),
                dict(era5_meta.get("log_transform", {})),
            ),
            time_chunk,
        )
        logger.info("Writing ERA5 Zarr to %s", paths["era5"])
        era5.to_zarr(
            paths["era5"],
            mode=mode,
            encoding=_encoding_for_dataset(era5, time_chunk),
            consolidated=True,
        )
        written["era5"] = paths["era5"]

    paths["meta"].write_text(json.dumps(dict(meta), indent=2), encoding="utf-8")
    logger.info("Wrote training cache metadata to %s", paths["meta"])
    return written


def build_training_zarr_cache(
    years: Sequence[int] | None = None,
    pollutants: Sequence[str] | None = None,
    output_root: str | Path | None = None,
    temporal_resolution: int | None = None,
    era5_grid: str | None = None,
    era5_variables: Sequence[str] | None = None,
    period: tuple[str, str] | None = None,
    time_chunk: int = DEFAULT_TIME_CHUNK,
    chunks: dict | None = None,
    overwrite: bool = False,
    stores: Sequence[str] | None = None,
    config: Mapping | None = None,
) -> dict[str, Path]:
    """
    Build the shared training Zarr cache.

    CAMS variables are stored without pollutant log1p/z-score.
    ERA5 uses ``load_training_era5_features`` (shared meteo recipe, no z-score).

    ``stores`` can be used to rewrite only selected outputs, e.g. ``("era5",)``.
    When rewriting ERA5 only, existing CAMS Zarr stores are reused for time alignment.
    """
    cfg = config or load_config()
    settings = get_error_modelling_settings()
    selected = _normalize_stores(stores)
    years = tuple(years or (*settings.train_val_years, *settings.test_years))
    pollutants = list(pollutants or cfg["cams_europe"]["pollutant_variables"])
    era5_variables = list(era5_variables or settings.era5_variables)
    era5_grid = era5_grid or settings.era5_grid
    temporal_resolution = (
        settings.cams_training_temporal_resolution
        if temporal_resolution is None
        else temporal_resolution
    )
    output_root = Path(output_root) if output_root is not None else default_output_root(cfg)
    period = period or _period_for_years(years)
    chunks = chunks if chunks is not None else {"time": max(time_chunk, 24)}

    logger.info(
        "Building training Zarr cache: years=%s pollutants=%s era5_grid=%s stores=%s output=%s",
        list(years),
        pollutants,
        era5_grid,
        list(selected),
        output_root,
    )

    need_cams = "cams_forecast" in selected or "cams_analysis" in selected
    need_era5 = "era5" in selected
    reuse_cams_for_alignment = need_era5 and not need_cams

    cams_forecast = None
    cams_analysis = None
    era5 = None
    existing_meta: dict[str, Any] | None = None

    if reuse_cams_for_alignment:
        forecast_path = output_root / CAMS_FORECAST_NAME
        analysis_path = output_root / CAMS_ANALYSIS_NAME
        meta_path = output_root / META_NAME
        missing = [str(path) for path in (forecast_path, analysis_path, meta_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Rewriting ERA5 only requires existing CAMS Zarr stores and meta.json. Missing:\n"
                + "\n".join(missing)
            )
        logger.info("Reusing existing CAMS Zarr stores for ERA5 time alignment")
        open_chunks = _resolve_open_chunks(chunks)
        cams_forecast = xr.open_zarr(forecast_path, chunks=open_chunks, consolidated=True)
        cams_analysis = xr.open_zarr(analysis_path, chunks=open_chunks, consolidated=True)
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    elif need_cams:
        cams_forecast = load_processed_cams_features(
            years=years,
            data_type="forecast",
            variables=pollutants,
            processed_log_transform=None,
            temporal_resolution=temporal_resolution,
            log_transform=False,
            chunks=chunks,
        )
        cams_analysis = load_processed_cams_features(
            years=years,
            data_type="analysis",
            variables=pollutants,
            processed_log_transform=None,
            temporal_resolution=temporal_resolution,
            log_transform=False,
            chunks=chunks,
        )

    if need_era5:
        era5 = load_training_era5_features(
            period=period,
            variables=era5_variables,
            temporal_resolution=temporal_resolution,
            grid=era5_grid,
            chunks=chunks,
        )

    if cams_forecast is not None and cams_analysis is not None and era5 is not None:
        if not reuse_cams_for_alignment:
            cams_forecast = _select_period(cams_forecast, period)
            cams_analysis = _select_period(cams_analysis, period)
        cams_forecast, cams_analysis, era5 = _select_common_times(cams_forecast, cams_analysis, era5)
    elif cams_forecast is not None and cams_analysis is not None:
        cams_forecast = _select_period(cams_forecast, period)
        cams_analysis = _select_period(cams_analysis, period)
        common_times = np.intersect1d(
            cams_forecast["time"].values.astype("datetime64[ns]"),
            cams_analysis["time"].values.astype("datetime64[ns]"),
        )
        if common_times.size == 0:
            raise ValueError("CAMS forecast and analysis have no overlapping timestamps.")
        cams_forecast = cams_forecast.sel(time=common_times)
        cams_analysis = cams_analysis.sel(time=common_times)

    if cams_forecast is None or cams_analysis is None:
        raise ValueError("CAMS forecast/analysis are required to build training cache metadata.")
    meta_era5 = era5 if era5 is not None else cams_forecast

    meta = build_training_cache_meta(
        years=years,
        pollutants=pollutants if need_cams or existing_meta is None else list(
            existing_meta.get("cams", {}).get("pollutants", pollutants)
        ),
        temporal_resolution=temporal_resolution,
        era5_grid=era5_grid,
        era5_variables=era5_variables,
        era5_log_transform=settings.era5_log_transform,
        pollutant_log_transform=settings.pollutant_log_transform,
        time_chunk=time_chunk,
        common_times=cams_forecast["time"].values,
        cams_forecast=cams_forecast,
        cams_analysis=cams_analysis,
        era5=meta_era5,
        output_root=output_root,
    )
    if existing_meta is not None and not need_cams:
        # Keep previously recorded CAMS dims when only ERA5 is rewritten.
        meta["cams"] = existing_meta.get("cams", meta["cams"])
        meta["analysis_dims"] = existing_meta.get("analysis_dims", meta["analysis_dims"])

    try:
        return write_training_zarr_datasets(
            cams_forecast=cams_forecast if "cams_forecast" in selected else None,
            cams_analysis=cams_analysis if "cams_analysis" in selected else None,
            era5=era5 if "era5" in selected else None,
            output_root=output_root,
            meta=meta,
            time_chunk=time_chunk,
            overwrite=overwrite,
            stores=selected,
        )
    finally:
        if reuse_cams_for_alignment:
            for dataset in (cams_forecast, cams_analysis):
                if dataset is not None:
                    try:
                        dataset.close()
                    except RuntimeError:
                        pass


def build_parser() -> argparse.ArgumentParser:
    settings = get_error_modelling_settings()
    default_years = list((*settings.train_val_years, *settings.test_years))
    parser = argparse.ArgumentParser(
        description=(
            "Build a shared CAMS/ERA5 training Zarr cache. "
            "CAMS pollutants are stored without log1p; ERA5 uses the shared training recipe; "
            "z-score is not applied."
        ),
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=default_years,
        help=f"Calendar years to include (default: {' '.join(str(y) for y in default_years)}).",
    )
    parser.add_argument(
        "--pollutants",
        nargs="+",
        help="CAMS pollutant variables to store. Defaults to cams_europe.pollutant_variables.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output directory for the Zarr stores. Defaults to paths.training_zarr.",
    )
    parser.add_argument(
        "--temporal-resolution",
        type=int,
        default=settings.cams_training_temporal_resolution,
        help="UTC hour interval to keep.",
    )
    parser.add_argument(
        "--era5-grid",
        choices=("cams", "raw"),
        default=settings.era5_grid,
        help="ERA5 spatial source grid.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=DEFAULT_TIME_CHUNK,
        help="Zarr chunk size along time (default: 8, one daily sample).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Zarr stores selected by --stores.",
    )
    parser.add_argument(
        "--stores",
        nargs="+",
        choices=list(STORE_NAMES),
        default=list(STORE_NAMES),
        help=(
            "Which stores to write. Use 'era5' alone to rewrite only ERA5 while "
            "reusing existing CAMS Zarr stores for time alignment."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional airwise config YAML.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = load_config(args.config)
    paths = build_training_zarr_cache(
        years=args.years,
        pollutants=args.pollutants,
        output_root=args.output_root,
        temporal_resolution=args.temporal_resolution,
        era5_grid=args.era5_grid,
        time_chunk=args.time_chunk,
        overwrite=args.overwrite,
        stores=args.stores,
        config=cfg,
    )
    print("Wrote training Zarr cache:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
