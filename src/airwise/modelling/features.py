from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from airwise.config import load_config
from airwise.data.preprocessing.regridding import resolve_era5_data_root
from airwise.data.preprocessing.meteorology import relative_humidity_from_dewpoint

if TYPE_CHECKING:
    from airwise.modelling.stats import ErrorModellingStats


def _normalise_time_coord(ds: xr.Dataset, time_name: str) -> xr.Dataset:
    if time_name != "time" and time_name in ds.coords:
        ds = ds.rename({time_name: "time"})
    return ds


def _select_period(ds: xr.Dataset, period: tuple[str, str] | None) -> xr.Dataset:
    if period is None:
        return ds
    start, end = period
    return ds.sel(time=slice(start, end))


def _parse_period_bounds(period: tuple[str, str] | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if period is None:
        return None

    start, end = period
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"Period end {end!r} is earlier than start {start!r}.")

    return start_ts, end_ts


def _select_temporal_resolution(ds: xr.Dataset, temporal_resolution: int | None) -> xr.Dataset:
    if temporal_resolution is None:
        return ds

    if "time" not in ds.coords:
        raise KeyError("Dataset must have a 'time' coordinate before temporal selection.")

    if not isinstance(temporal_resolution, int):
        raise TypeError("temporal_resolution must be an integer hour interval, e.g. 1, 3, 5, 6.")

    if temporal_resolution < 1 or temporal_resolution > 24:
        raise ValueError("temporal_resolution must be between 1 and 24.")

    selected_hours = list(range(0, 24, temporal_resolution))
    return ds.sel(time=ds["time"].dt.hour.isin(selected_hours))


def _convert_units(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy()
    if "tp" in out:
        out["tp"] = out["tp"] * 1000
        out["tp"].attrs.update(ds["tp"].attrs)
        out["tp"].attrs["units"] = "mm"
        out["tp"].attrs["unit_conversion"] = "m_to_mm"
    return out


def _is_lazy_dataarray(da: xr.DataArray) -> bool:
    return getattr(da.data, "chunks", None) is not None


def _apply_log_transform(ds: xr.Dataset, transform=None) -> xr.Dataset:
    if transform is None or transform is False:
        return ds

    if isinstance(transform, str):
        transform = {var_name: transform for var_name in ds.data_vars}

    out = ds.copy()

    for var_name, method in transform.items():
        if var_name not in out:
            continue

        da = out[var_name]

        if method == "log1p":
            if not _is_lazy_dataarray(da) and bool((da <= -1).any()):
                raise ValueError(f"{var_name} contains values <= -1; log1p is invalid.")
            out[var_name] = xr.apply_ufunc(np.log1p, da, dask="allowed")
        elif method == "log":
            if not _is_lazy_dataarray(da) and bool((da <= 0).any()):
                raise ValueError(f"{var_name} contains values <= 0; log is invalid.")
            out[var_name] = xr.apply_ufunc(np.log, da, dask="allowed")
        elif method in (None, False):
            continue
        else:
            raise ValueError(f"Unknown log transform for {var_name}: {method}")

    return out

def _reassign_cams_time_coords(ds: xr.Dataset, time_name: str) -> xr.Dataset:
    """
    Reassign CAMS time coordinates from the timestamp encoded in `time.long_name`.

    Some raw CAMS files store hourly steps on the `time` coordinate, while the
    actual initial date is encoded in `ds.time.long_name`. This function rebuilds
    an hourly datetime64 time coordinate before multiple files are concatenated.
    """
    if time_name not in ds:
        raise KeyError(f"CAMS dataset must contain a {time_name!r} coordinate.")

    long_name = ds[time_name].attrs.get("long_name", "")
    try:
        timestamp = long_name[19:27]
        timestamp_init = datetime.strptime(timestamp, "%Y%m%d")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot parse initial timestamp from ds[{time_name!r}].long_name: {long_name!r}"
        ) from exc
    time_coords = pd.date_range(
        timestamp_init,
        periods=ds.sizes[time_name],
        freq="1h",
    ).strftime("%Y-%m-%d %H:%M:%S").astype('datetime64[ns]')

    return ds.assign_coords({time_name: time_coords})


def _reassign_longitude(ds: xr.Dataset, longitude_name: str) -> xr.Dataset:
    if longitude_name not in ds.coords:
       raise KeyError(f"Dataset must contain a '{longitude_name}' coordinate.")

    return ds.assign_coords({
        longitude_name: (((ds[longitude_name] + 180) % 360) - 180)
    }).sortby(longitude_name)


def _squeeze_dimensions(ds: xr.Dataset) -> xr.Dataset:
    return ds.squeeze(drop=True)


def _parse_cams_file_period(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Parse CAMS file coverage from names like YYYY-MM-DD_YYYY-MM-DD.nc.
    """
    parts = path.stem.split("_")
    if len(parts) < 2:
        return None

    try:
        return pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
    except ValueError:
        return None


def _filter_cams_paths_by_period(
    paths: Sequence[Path],
    period: tuple[str, str] | None,
) -> list[Path]:
    """
    Keep only CAMS files whose filename date range overlaps the requested period.
    """
    bounds = _parse_period_bounds(period)
    if bounds is None:
        return list(paths)

    period_start, period_end = bounds
    selected_paths = []
    for path in paths:
        file_bounds = _parse_cams_file_period(path)
        if file_bounds is None:
            continue

        file_start, file_end = file_bounds
        if file_start <= period_end and file_end >= period_start:
            selected_paths.append(path)

    return selected_paths


def _open_cams_file(
    path: Path,
    coordinate_names: Mapping[str, str],
    variables: Sequence[str],
    period: tuple[str, str] | None = None,
    chunks: dict | None = None,
) -> xr.Dataset:
    ds = xr.open_dataset(path, chunks=chunks)
    ds = _reassign_cams_time_coords(ds, coordinate_names["time"])
    ds = _reassign_longitude(ds, coordinate_names["longitude"])
    ds = _squeeze_dimensions(ds)
    missing = [var_name for var_name in variables if var_name not in ds.data_vars]
    if missing:
        raise KeyError(f"CAMS variables not found in {path}: {missing}. Available: {list(ds.data_vars)}")

    ds = ds[list(variables)]
    ds = _select_period(ds, period)
    return ds


def _add_rain_mask(
    ds: xr.Dataset,
    threshold: float = 1e-2,
    variable: str = "tp",
    mask_name: str = "rain_mask",
) -> xr.Dataset:
    """
    Add a binary rain occurrence mask as an additional model input.
    The mask is generated from the precipitation variable before any log
    transform is applied. It is intended to help neural networks distinguish
    between dry and wet conditions, especially because precipitation is often
    zero-inflated.
    Parameters
    ----------
    ds:
        Input dataset.
    threshold:
        Rain/no-rain threshold. If `tp` has already been converted from metres
        to millimetres, use `1e-2`, which is equivalent to `1e-5 m`.
    variable:
        Name of the precipitation variable used to derive the mask.
    mask_name:
        Name of the new mask variable added to the dataset.
    Returns
    -------
    xr.Dataset
        Dataset with an additional float32 variable:
        - 1.0 means precipitation is greater than `threshold`.
        - 0.0 means precipitation is less than or equal to `threshold`.
        If `variable` is not present, the original dataset is returned unchanged.
    """
    if variable not in ds:
        return ds

    out = ds.copy()
    out[mask_name] = (out[variable] > threshold).astype("float32")
    out[mask_name].attrs = {
        "long_name": "binary rain occurrence mask",
        "description": f"1 if {variable} > {threshold}, else 0",
        "source_variable": variable,
        "threshold": threshold,
    }
    return out


_RH_HELPER_VARS = ("t2m", "d2m")


def _add_relative_humidity(
    ds: xr.Dataset,
    temperature_var: str = "t2m",
    dewpoint_var: str = "d2m",
    rh_name: str = "rh",
) -> xr.Dataset:
    """
    Add relative humidity (%) derived from 2 m temperature and dewpoint.

    Requires both ``t2m`` and ``d2m`` (Kelvin) to be present. The result is a
    float32 field clipped to ``[0, 100]``.
    """
    missing = [name for name in (temperature_var, dewpoint_var) if name not in ds]
    if missing:
        raise KeyError(
            f"Cannot compute relative humidity; missing variables: {missing}. "
            f"Available: {list(ds.data_vars)}"
        )

    out = ds.copy()
    rh = relative_humidity_from_dewpoint(out[temperature_var], out[dewpoint_var])
    out[rh_name] = rh.astype("float32")
    out[rh_name].attrs = {
        "long_name": "2m relative humidity",
        "units": "%",
        "description": (
            "Relative humidity derived from t2m and d2m via the "
            "August-Roche-Magnus approximation; clipped to [0, 100]."
        ),
        "source_variables": [temperature_var, dewpoint_var],
        "formula": "August-Roche-Magnus",
    }
    return out


def _resolve_era5_load_variables(
    variables: Sequence[str],
    add_relative_humidity: bool,
) -> tuple[list[str], bool, set[str]]:
    """
    Expand the requested ERA5 variable list with helpers needed for RH.

    Returns
    -------
    load_variables:
        On-disk variable folders to open (never includes derived ``rh``).
    compute_rh:
        Whether relative humidity should be derived after loading.
    drop_after_rh:
        Helper variables that were loaded only to compute RH and should be
        dropped afterwards if the caller did not request them.
    """
    requested = list(variables)
    compute_rh = add_relative_humidity or ("rh" in requested)
    load_variables = [var_name for var_name in requested if var_name != "rh"]
    drop_after_rh: set[str] = set()

    if compute_rh:
        for helper in _RH_HELPER_VARS:
            if helper not in load_variables:
                load_variables.append(helper)
                if helper not in requested:
                    drop_after_rh.add(helper)

    return load_variables, compute_rh, drop_after_rh


def load_era5_features(
    variables: Sequence[str] | None = None,
    period: tuple[str, str] | None = None,
    temporal_resolution: int | None = None,
    log_transform=None,
    add_rain_mask=True,
    rain_threshold_mm=1e-2,
    add_relative_humidity: bool | None = None,
    grid: str = "cams",
    data_dir: str | Path | None = None,
    chunks: dict | None = None,
) -> xr.Dataset:
    """
    Load ERA5 meteorological features for error modelling.

    Parameters
    ----------
    variables:
        ERA5 short variable folders to load, e.g. ["u10", "v10", "t2m", "tp"].
        Include ``"rh"`` to derive relative humidity from ``t2m`` and ``d2m``
        (``d2m`` is loaded automatically and dropped unless also requested).
        Defaults to all variables defined in configs/default.yaml.
    period:
        Optional inclusive time range, e.g. ("2023-06-01", "2024-12-31").
    temporal_resolution:
        Optional UTC hour interval for direct selection from hourly ERA5 data.
        For example, 3 keeps UTC hours 0, 3, 6, ..., 21;
        5 keeps UTC hours 0, 5, 10, 15, 20.
        If None or 1, keeps all hourly data.
    log_transform:
        None, "log1p", "log", or a mapping such as {"tp": "log1p", "ssrd": "log1p"}.
    add_rain_mask:
        If True, add a float32 binary `rain_mask` variable derived from precipitation.
        Values are 1.0 for rainy grid cells/timesteps and 0.0 otherwise.
        If False, no rain mask variable is added.
    rain_threshold_mm:
        Threshold for precipitation to be considered rainy, in millimeters.
        Defaults to 1e-2 mm.
    add_relative_humidity:
        If True, always derive ``rh`` from ``t2m`` and ``d2m``.
        If None (default), derive ``rh`` only when ``"rh"`` is in ``variables``.
        If False, never derive ``rh`` even if listed in ``variables``.
    grid:
        ERA5 spatial grid to load:
        - ``"cams"`` (default): resampled ERA5 on the CAMS Europe 0.1deg grid
          (``paths.era5_data_cams_grid``, e.g. ``/data/jz1618/ERA5/euro_cams_grid``).
        - ``"raw"``: native ERA5 0.25deg files
          (``paths.era5_data_raw``, e.g. ``/data/jz1618/ERA5/euro``).
    data_dir:
        Optional explicit ERA5 root directory. Overrides ``grid`` when provided.
        Defaults to paths.era5_data_raw or paths.era5_data_cams_grid.
    chunks:
        Optional dask chunks passed to xarray.open_mfdataset.
    """
    if grid not in {"raw", "cams"}:
        raise ValueError("grid must be either 'raw' or 'cams'.")

    cfg = load_config()
    era5_cfg = cfg["era5_euro"]
    data_root = resolve_era5_data_root(grid=grid, data_dir=data_dir, config=cfg)
    time_name = era5_cfg["coordinate_names"]["time"]

    requested_variables = list(variables or era5_cfg["variables"].keys())
    if add_relative_humidity is False:
        compute_rh = False
        load_variables = [var_name for var_name in requested_variables if var_name != "rh"]
        drop_after_rh: set[str] = set()
        if "rh" in requested_variables:
            raise ValueError(
                "variables includes 'rh' but add_relative_humidity=False. "
                "Pass add_relative_humidity=True/None or remove 'rh'."
            )
    else:
        load_variables, compute_rh, drop_after_rh = _resolve_era5_load_variables(
            requested_variables,
            add_relative_humidity=bool(add_relative_humidity),
        )

    datasets = []
    for var_name in load_variables:
        files = sorted((data_root / var_name).glob("*.nc"))
        if not files:
            raise FileNotFoundError(
                f"No ERA5 NetCDF files found for {var_name} under {data_root / var_name} "
                f"(grid={grid!r})."
            )

        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks=chunks,
        )
        ds = _normalise_time_coord(ds, time_name)

        if var_name in ds.data_vars:
            ds = ds[[var_name]]
        elif len(ds.data_vars) == 1:
            only_var = next(iter(ds.data_vars))
            ds = ds.rename({only_var: var_name})[[var_name]]
        else:
            raise KeyError(f"Cannot identify data variable for {var_name}. Found: {list(ds.data_vars)}")

        datasets.append(ds)

    ds = xr.merge(datasets, compat="override", join="inner")
    ds = _select_period(ds, period)
    ds = _select_temporal_resolution(ds, temporal_resolution)
    ds = _convert_units(ds)
    if add_rain_mask:
        ds = _add_rain_mask(ds, threshold=rain_threshold_mm)
    if compute_rh:
        ds = _add_relative_humidity(ds)
        helpers_to_drop = [name for name in drop_after_rh if name in ds]
        if helpers_to_drop:
            ds = ds.drop_vars(helpers_to_drop)
    ds = _apply_log_transform(ds, log_transform)

    return ds


def _training_era5_defaults():
    from airwise.modelling.config import get_error_modelling_settings

    return get_error_modelling_settings()


def __getattr__(name: str):
    """Lazy YAML-backed aliases for older TRAINING_ERA5_* imports."""
    settings = _training_era5_defaults()
    aliases = {
        "TRAINING_ERA5_VARIABLES": list(settings.era5_variables),
        "TRAINING_ERA5_LOG_TRANSFORM": dict(settings.era5_log_transform),
        "TRAINING_ERA5_SKIP_ZSCORE": frozenset(settings.era5_skip_zscore),
        "TRAINING_ERA5_GRID_DEFAULT": settings.era5_grid,
    }
    if name in aliases:
        return aliases[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_training_era5_features(
    period: tuple[str, str] | None = None,
    variables: Sequence[str] | None = None,
    temporal_resolution: int | None = None,
    grid: str | None = None,
    data_dir: str | Path | None = None,
    chunks: dict | None = None,
    add_rain_mask: bool = True,
    rain_threshold_mm: float = 1e-2,
) -> xr.Dataset:
    """
    Load ERA5 meteorology for neural-network error-model training.

    Defaults come from ``configs/default.yaml`` ``error_modelling.era5``.

    - Variables: configured ERA5 list (+ ``rain_mask`` by default)
    - ``d2m`` is loaded only to derive ``rh`` (August-Roche-Magnus), then dropped
    - ``tp`` / ``blh``: log transforms from config
    - ``rh``: percent in ``[0, 100]``, **no** log / log1p (z-score later in Dataset)
    - ``rain_mask``: binary 0/1 from ``tp > threshold``, **no** z-score later
    - Spatial source selectable via ``grid``:
      - ``"cams"`` (default): ``paths.era5_data_cams_grid`` (resampled 0.1deg)
      - ``"raw"``: ``paths.era5_data_raw`` (native 0.25deg)

    Z-score normalisation is **not** applied here; it happens in
    ``ErrorModellingDataset`` using training-set statistics, skipping
    configured ``skip_zscore`` variables.
    """
    settings = _training_era5_defaults()
    grid = grid or settings.era5_grid
    temporal_resolution = (
        settings.era5_temporal_resolution if temporal_resolution is None else temporal_resolution
    )
    if grid not in {"raw", "cams"}:
        raise ValueError("grid must be either 'raw' or 'cams'.")

    selected = list(variables or settings.era5_variables)
    return load_era5_features(
        variables=selected,
        period=period,
        temporal_resolution=temporal_resolution,
        log_transform=dict(settings.era5_log_transform),
        add_rain_mask=add_rain_mask,
        rain_threshold_mm=rain_threshold_mm,
        add_relative_humidity=None,  # derive rh iff "rh" is in variables
        grid=grid,
        data_dir=data_dir,
        chunks=chunks,
    )

def load_cams_features(
    data_type: str = "analysis",
    variables: Sequence[str] | None = None,
    period: tuple[str, str] | None = None,
    temporal_resolution: int | None = None,
    log_transform=None,
    data_dir: str | Path | None = None,
    files: Sequence[str | Path] | None = None,
    chunks: dict | None = None,
) -> xr.Dataset:
    """
    Load and preprocess raw CAMS Europe air quality data.

    Parameters
    ----------
    data_type:
        CAMS subset to read. Must be either "analysis" or "forecast".
        Files are read from paths.cams_data_raw / data_type.
    variables:
        CAMS variables to keep, e.g. ["no2_conc", "o3_conc"].
        Defaults to configs/default.yaml `cams_europe.pollutant_variables`.
    period:
        Optional inclusive time range, e.g. ("2023-06-01", "2024-12-31").
    temporal_resolution:
        Optional UTC hour interval for direct selection from hourly CAMS data.
        For example, 3 keeps UTC hours 0, 3, 6, ..., 21.
    log_transform:
        None, "log1p", "log", or a mapping such as {"no2_conc": "log1p"}.
    data_dir:
        Defaults to config paths.cams_data_raw.
    files:
        Optional explicit file list. If None, candidate .nc files are discovered
        under the selected data_type folder. When period is provided, filenames
        are filtered before opening so unrelated files are not read.
    chunks:
        Optional dask chunks passed to xarray.open_dataset.

    Returns
    -------
    xr.Dataset
        Preprocessed CAMS dataset with corrected time coordinate, reassigned
        longitude, squeezed singleton dimensions, optional period selection,
        optional temporal selection, and optional log transform.
    """
    if data_type not in {"analysis", "forecast"}:
        raise ValueError("data_type must be either 'analysis' or 'forecast'.")

    cfg = load_config()
    cams_cfg = cfg["cams_europe"]
    data_root = Path(data_dir or cfg["paths"]["cams_data_raw"]) / data_type
    coordinate_names = cams_cfg["coordinate_names"]

    if files is None:
        paths = sorted(data_root.glob("*.nc"))
    else:
        paths = [
            Path(file) if Path(file).is_absolute() else data_root / file
            for file in files
        ]
    paths = _filter_cams_paths_by_period(paths, period)

    if not paths:
        raise FileNotFoundError(f"No CAMS NetCDF files found under {data_root} for period {period}")

    variables = list(variables or cams_cfg["pollutant_variables"])
    datasets = [
        _open_cams_file(
            path,
            coordinate_names=coordinate_names,
            variables=variables,
            period=period,
            chunks=chunks,
        )
        for path in paths
    ]

    ds = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ).sortby("time")

    ds = _select_temporal_resolution(ds, temporal_resolution)
    ds = _apply_log_transform(ds, log_transform)

    return ds


def _log_transform_suffix(log_transform) -> str:
    if log_transform is None or log_transform is False:
        return "raw"
    if isinstance(log_transform, str):
        return log_transform
    if isinstance(log_transform, Mapping):
        methods = sorted({method for method in log_transform.values() if method})
        return "raw" if not methods else "_".join(methods)
    return "transformed"


def _cams_processed_output_path(
    output_root: Path,
    data_type: str,
    year: int,
    temporal_resolution: int | None,
    log_transform,
    output_format: str,
) -> Path:
    temporal_suffix = "hourly" if temporal_resolution in (None, 1) else f"{temporal_resolution}hourly"
    transform_suffix = _log_transform_suffix(log_transform)
    suffix = ".zarr" if output_format == "zarr" else ".nc"
    filename = f"{year}_{temporal_suffix}_{transform_suffix}{suffix}"
    return output_root / data_type / filename


def _cams_error_modelling_config(cfg: Mapping) -> Mapping:
    return cfg.get("error_modelling", {}).get("cams", {})


def _open_processed_cams_dataset(
    paths: Sequence[Path],
    variables: Sequence[str],
    output_format: str,
    chunks: dict | None = None,
) -> xr.Dataset:
    """
    Open yearly processed CAMS files with an ordered concat.

    Avoid ``open_mfdataset(..., coords="minimal", join="override")`` here:
    with yearly files, that combination can incorrectly reuse one file's
    ``time`` coordinate for all concatenated years. Opening files explicitly
    keeps each file's own time coordinate before concatenation.
    """
    def _select_variables(ds: xr.Dataset) -> xr.Dataset:
        missing_vars = [var_name for var_name in variables if var_name not in ds.data_vars]
        if missing_vars:
            raise KeyError(
                f"Processed CAMS variables not found: {missing_vars}. "
                f"Available: {list(ds.data_vars)}"
            )
        return ds[list(variables)]

    datasets = []
    for path in paths:
        if output_format == "zarr":
            ds = xr.open_zarr(path, chunks=chunks)
        else:
            ds = xr.open_dataset(path, chunks=chunks)
        datasets.append(_select_variables(ds))

    return xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
        combine_attrs="override",
    )


def preprocess_cams_year(
    year: int,
    data_type: str = "analysis",
    variables: Sequence[str] | None = None,
    temporal_resolution: int | None = None,
    log_transform=None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_format: str | None = None,
    chunks: dict | None = None,
    skip_existing: bool = True,
) -> Path:
    """
    Preprocess one calendar year of raw CAMS data and save it for training.

    The output path defaults to configs/default.yaml `paths.cams_data_processed`
    and is organised by data_type, for example:
    `/data/.../processed/analysis/2023_3hourly_raw.nc`.

    Parameters
    ----------
    year:
        Calendar year to preprocess.
    data_type:
        CAMS subset to process. Must be either "analysis" or "forecast".
    variables:
        CAMS variables to keep. Defaults to configs/default.yaml
        `cams_europe.pollutant_variables`.
    temporal_resolution:
        Optional UTC hour interval for direct temporal selection.
    log_transform:
        Optional log transform passed to load_cams_features.
    data_dir:
        Raw CAMS root directory. Defaults to paths.cams_data_raw.
    output_dir:
        Processed CAMS root directory. Defaults to paths.cams_data_processed.
    output_format:
        Either "netcdf" or "zarr".
    chunks:
        Optional dask chunks passed to xarray.open_dataset.
    skip_existing:
        If True, return the existing output path without recomputing.

    Returns
    -------
    Path
        Path to the processed yearly dataset.
    """
    cfg = load_config()
    cams_defaults = _cams_error_modelling_config(cfg).get("processed", {})
    temporal_resolution = (
        cams_defaults.get("temporal_resolution", 1)
        if temporal_resolution is None
        else temporal_resolution
    )
    log_transform = cams_defaults.get("log_transform") if log_transform is None else log_transform
    output_format = output_format or cams_defaults.get("output_format", "netcdf")

    if output_format not in {"netcdf", "zarr"}:
        raise ValueError("output_format must be either 'netcdf' or 'zarr'.")

    output_root = Path(output_dir or cfg["paths"]["cams_data_processed"])
    output_path = _cams_processed_output_path(
        output_root=output_root,
        data_type=data_type,
        year=year,
        temporal_resolution=temporal_resolution,
        log_transform=log_transform,
        output_format=output_format,
    )

    if skip_existing and output_path.exists():
        return output_path

    period = (f"{year}-01-01", f"{year}-12-31 23:00:00")
    ds = load_cams_features(
        data_type=data_type,
        variables=variables,
        period=period,
        temporal_resolution=temporal_resolution,
        log_transform=log_transform,
        data_dir=data_dir,
        chunks=chunks,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "zarr":
        ds.to_zarr(output_path, mode="w")
    else:
        ds.to_netcdf(output_path)

    ds.close()
    return output_path


def preprocess_cams_years(
    years: Sequence[int],
    data_type: str = "analysis",
    variables: Sequence[str] | None = None,
    temporal_resolution: int | None = None,
    log_transform=None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_format: str | None = None,
    chunks: dict | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Preprocess multiple CAMS years and save each year as a separate dataset.
    """
    return [
        preprocess_cams_year(
            year=year,
            data_type=data_type,
            variables=variables,
            temporal_resolution=temporal_resolution,
            log_transform=log_transform,
            data_dir=data_dir,
            output_dir=output_dir,
            output_format=output_format,
            chunks=chunks,
            skip_existing=skip_existing,
        )
        for year in years
    ]

def load_processed_cams_features(
    years: Sequence[int],
    data_type: str = "analysis",
    variables: Sequence[str] | None = None,
    processed_temporal_resolution: int | None = None,
    processed_log_transform=None,
    temporal_resolution: int | None = None,
    log_transform=None,
    data_dir: str | Path | None = None,
    output_format: str | None = None,
    chunks: dict | None = None,
) -> xr.Dataset:
    """
    Load yearly preprocessed CAMS datasets for model training.

    This function reads files produced by `preprocess_cams_year` or
    `preprocess_cams_years`, instead of reprocessing raw CAMS files.

    Parameters
    ----------
    years:
        Calendar years to load.
    data_type:
        CAMS subset to read. Must be either "analysis" or "forecast".
    variables:
        CAMS variables to keep. Defaults to configs/default.yaml
        `cams_europe.pollutant_variables`.
    processed_temporal_resolution:
        Temporal resolution used when the processed yearly files were saved.
        This is used only to locate files on disk. For example, 1 or None reads
        files named like `2023_hourly_raw.nc`.
    processed_log_transform:
        Log transform used when the processed yearly files were saved. This is
        used only to locate files on disk. Use None for files named with the
        `raw` suffix.
    temporal_resolution:
        Optional UTC hour interval to select after loading the processed data.
        For example, if the processed files are hourly, 3 keeps UTC hours
        0, 3, 6, ..., 21 for model training.
    log_transform:
        Optional log transform to apply after loading the processed data.
        For example, use "log1p" to transform all selected CAMS variables, or a
        mapping such as {"no2_conc": "log1p"}. If None, the default is read
        from configs/default.yaml. Pass False to disable log transformation.
    data_dir:
        Processed CAMS root directory. Defaults to paths.cams_data_processed.
    output_format:
        Either "netcdf" or "zarr".
    chunks:
        Optional dask chunks passed to xarray open functions.
    """
    cfg = load_config()
    cams_cfg = cfg["cams_europe"]
    cams_defaults = _cams_error_modelling_config(cfg)
    processed_defaults = cams_defaults.get("processed", {})
    training_defaults = cams_defaults.get("training_input", {})

    processed_temporal_resolution = (
        processed_defaults.get("temporal_resolution", 1)
        if processed_temporal_resolution is None
        else processed_temporal_resolution
    )
    processed_log_transform = (
        processed_defaults.get("log_transform")
        if processed_log_transform is None
        else processed_log_transform
    )
    temporal_resolution = (
        training_defaults.get("temporal_resolution")
        if temporal_resolution is None
        else temporal_resolution
    )
    log_transform = training_defaults.get("log_transform") if log_transform is None else log_transform
    output_format = output_format or processed_defaults.get("output_format", "netcdf")

    if output_format not in {"netcdf", "zarr"}:
        raise ValueError("output_format must be either 'netcdf' or 'zarr'.")

    data_root = Path(data_dir or cfg["paths"]["cams_data_processed"])

    variables = list(variables or cams_cfg["pollutant_variables"])
    paths = [
        _cams_processed_output_path(
            output_root=data_root,
            data_type=data_type,
            year=year,
            temporal_resolution=processed_temporal_resolution,
            log_transform=processed_log_transform,
            output_format=output_format,
        )
        for year in sorted(years)
    ]
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing processed CAMS files:\n"
            + "\n".join(str(path) for path in missing_paths)
        )

    ds = _open_processed_cams_dataset(
        paths=paths,
        variables=variables,
        output_format=output_format,
        chunks=chunks,
    )
    ds = _select_temporal_resolution(ds, temporal_resolution)
    ds = _apply_log_transform(ds, log_transform)
    return ds


def prepare_open_ifs_features(
    dataset: xr.Dataset,
    target_grid: xr.Dataset,
    stats: "ErrorModellingStats",
    *,
    numbers: Sequence[int] = (0,),
    rain_threshold_mm: float = 1e-2,
    methods: dict[str, str] | None = None,
) -> xr.DataArray:
    """Build model-ready features from the OpenIFS control forecast."""
    from airwise.data.io.openifs import (
        MODEL_LEAD_HOURS,
        OPEN_IFS_MODEL_VARIABLES,
        deaccumulate_open_ifs,
        regrid_open_ifs_to_cams,
    )

    selected = dataset.sel(number=list(numbers))
    selected = selected.sel(time=selected["lead_hour"].isin(MODEL_LEAD_HOURS))
    processed = deaccumulate_open_ifs(selected)
    processed = regrid_open_ifs_to_cams(processed, target_grid, methods=methods)
    processed["rh"] = relative_humidity_from_dewpoint(
        processed["t2m"], processed["d2m"]
    ).astype("float32")
    processed["rain_mask"] = (processed["tp"] > rain_threshold_mm).astype("float32")
    processed["tp"] = np.log1p(processed["tp"])

    missing = [
        name
        for name in OPEN_IFS_MODEL_VARIABLES
        if name != "rain_mask" and name not in stats.era5
    ]
    if missing:
        raise KeyError(f"Training stats are missing OpenIFS variables: {missing}")
    for name in OPEN_IFS_MODEL_VARIABLES:
        if name != "rain_mask":
            variable_stats = stats.era5[name]
            processed[name] = (
                processed[name] - variable_stats.mean
            ) / variable_stats.std

    return (
        processed[list(OPEN_IFS_MODEL_VARIABLES)]
        .to_array("variable")
        .transpose("number", "variable", "time", "latitude", "longitude")
        .astype("float32")
        .assign_coords(variable=list(OPEN_IFS_MODEL_VARIABLES))
    )