"""Resample ERA5 single-level fields onto the CAMS Europe grid.

Example::

    airwise-regrid-era5 --years 2023 2024 --variables u10 t2m tp --chunk-time 2190 --overwrite
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
import yaml

from airwise.config import load_config, require_config, resolve_repo_path

XARRAY_METHOD_ALIASES = {
    "bilinear": "linear",
    "nearest": "nearest",
    "linear": "linear",
}


def _resolve_method(method: str) -> str:
    try:
        return XARRAY_METHOD_ALIASES[method]
    except KeyError as exc:
        supported = ", ".join(sorted(XARRAY_METHOD_ALIASES))
        raise ValueError(f"Unsupported regrid method {method!r}. Expected one of: {supported}.") from exc


def get_regrid_methods(config: dict | None = None) -> dict[str, str]:
    """Return per-variable regrid methods from ``era5_euro.regrid.methods``."""
    cfg = config or load_config()
    configured = require_config(cfg, "era5_euro", "regrid", "methods")
    if not isinstance(configured, dict) or not configured:
        raise ValueError(
            "era5_euro.regrid.methods must be a non-empty mapping. "
            "Set it in configs/default.yaml or configs/local.yaml."
        )
    return {str(name): str(method) for name, method in configured.items()}


def resolve_era5_data_root(
    grid: str = "raw",
    data_dir: str | Path | None = None,
    config: dict | None = None,
) -> Path:
    """Resolve the ERA5 data root for raw or CAMS-grid resampled files."""
    if data_dir is not None:
        return Path(data_dir)

    cfg = config or load_config()
    paths = cfg["paths"]
    if grid == "raw":
        return resolve_repo_path(require_config(paths, "era5_data_raw"))
    if grid == "cams":
        return resolve_repo_path(require_config(paths, "era5_data_cams_grid"))
    raise ValueError("grid must be either 'raw' or 'cams'.")


def _monthly_cams_reference_path(config: dict) -> Path:
    reference = config["era5_euro"]["regrid"]["reference_cams"]
    reference_path = Path(reference)
    if reference_path.is_absolute():
        return reference_path
    return resolve_repo_path(require_config(config, "paths", "cams_data_raw")) / "forecast" / reference


def _grid_cache_dir(output_root: Path) -> Path:
    return output_root / "_grid"


def _manifest_path(output_root: Path) -> Path:
    return _grid_cache_dir(output_root) / "regrid_manifest.yaml"


def _target_grid_path(output_root: Path) -> Path:
    return _grid_cache_dir(output_root) / "cams_target_grid.nc"


def normalize_cams_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Convert CAMS longitude from 0-360 to -180-180 and sort."""
    if float(ds.longitude.max()) > 180.0:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")
    return ds


def load_cams_target_grid(
    reference_path: str | Path | None = None,
    config: dict | None = None,
) -> xr.Dataset:
    """Load CAMS latitude/longitude coordinates used as the regrid target."""
    cfg = config or load_config()
    path = Path(reference_path) if reference_path is not None else _monthly_cams_reference_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"CAMS reference grid file not found: {path}")

    with xr.open_dataset(path) as ds:
        ds = normalize_cams_longitude(ds)
        return xr.Dataset(
            {
                "latitude": ds.latitude.copy(deep=True),
                "longitude": ds.longitude.copy(deep=True),
            }
        )


def save_cams_target_grid(
    target_grid: xr.Dataset,
    output_root: str | Path,
    *,
    reference_path: str | Path,
    methods: dict[str, str],
    source_root: str | Path,
    config: dict | None = None,
) -> tuple[Path, Path]:
    """Persist cached target grid coordinates and manifest metadata."""
    cfg = config or load_config()
    output_root = Path(output_root)
    grid_dir = _grid_cache_dir(output_root)
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_path = _target_grid_path(output_root)
    target_grid.to_netcdf(grid_path)

    era5_cfg = cfg["era5_euro"]
    area = era5_cfg["area"]
    manifest = {
        "version": 1,
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "reference_cams": str(reference_path),
        "source": {
            "path": str(source_root),
            "resolution": 0.25,
            "extent": {
                "north": area["north"],
                "south": area["south"],
                "west": area["west"],
                "east": area["east"],
            },
        },
        "target": {
            "resolution": 0.1,
            "shape": [int(target_grid.sizes["latitude"]), int(target_grid.sizes["longitude"])],
            "latitude": [float(target_grid.latitude.max()), float(target_grid.latitude.min())],
            "longitude": [float(target_grid.longitude.min()), float(target_grid.longitude.max())],
            "longitude_convention": "-180..180",
        },
        "methods": methods,
        "time_coord": era5_cfg["coordinate_names"]["time"],
    }
    manifest_path = _manifest_path(output_root)
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    return grid_path, manifest_path


def ensure_cams_target_grid(
    output_root: str | Path | None = None,
    reference_path: str | Path | None = None,
    config: dict | None = None,
    *,
    overwrite: bool = False,
) -> xr.Dataset:
    """Load or create the cached CAMS target grid under the resampled ERA5 root."""
    cfg = config or load_config()
    if output_root is None:
        output_root = resolve_repo_path(require_config(cfg, "paths", "era5_data_cams_grid"))
    else:
        output_root = Path(output_root)
    reference = Path(reference_path) if reference_path is not None else _monthly_cams_reference_path(cfg)
    grid_path = _target_grid_path(output_root)

    if grid_path.exists() and not overwrite:
        with xr.open_dataset(grid_path) as cached:
            return xr.Dataset(
                {
                    "latitude": cached.latitude.copy(deep=True),
                    "longitude": cached.longitude.copy(deep=True),
                }
            )

    methods = get_regrid_methods(cfg)
    target_grid = load_cams_target_grid(reference, cfg)
    save_cams_target_grid(
        target_grid,
        output_root,
        reference_path=reference,
        methods=methods,
        source_root=resolve_repo_path(require_config(cfg, "paths", "era5_data_raw")),
        config=cfg,
    )
    return target_grid


def _resolve_dataset_variable(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    if var_name in ds.data_vars:
        return ds
    if len(ds.data_vars) == 1:
        only_var = next(iter(ds.data_vars))
        if only_var != var_name:
            return ds.rename({only_var: var_name})
        return ds
    raise KeyError(f"Cannot identify data variable for {var_name}. Found: {list(ds.data_vars)}")


def regrid_era5_dataset(
    ds: xr.Dataset,
    var_name: str,
    target_grid: xr.Dataset,
    method: str,
) -> xr.Dataset:
    """Spatially resample one ERA5 variable onto the CAMS target grid."""
    xarray_method = _resolve_method(method)
    ds = _resolve_dataset_variable(ds, var_name)
    regridded = ds.interp(
        latitude=target_grid.latitude,
        longitude=target_grid.longitude,
        method=xarray_method,
    )
    regridded = regridded[[var_name]]
    attrs = dict(ds[var_name].attrs)
    attrs["regrid_method"] = method
    attrs["regrid_source_grid"] = "ERA5 0.25deg regular_ll"
    attrs["regrid_target_grid"] = "CAMS Europe 0.1deg"
    regridded[var_name].attrs.update(attrs)
    return regridded


def _netcdf_encoding(ds: xr.Dataset, var_name: str) -> dict[str, dict]:
    return {
        var_name: {
            "zlib": True,
            "complevel": 1,
            "dtype": "float32",
        }
    }


def regrid_era5_file(
    input_path: str | Path,
    output_path: str | Path,
    var_name: str,
    target_grid: xr.Dataset,
    method: str,
    *,
    chunks: dict | None = None,
    overwrite: bool = False,
) -> Path:
    """Resample one ERA5 yearly NetCDF file and write the result."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    open_kwargs: dict = {}
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    with xr.open_dataset(input_path, **open_kwargs) as ds:
        regridded = regrid_era5_dataset(ds, var_name, target_grid, method)
        regridded.to_netcdf(
            output_path,
            encoding=_netcdf_encoding(regridded, var_name),
        )
    return output_path


def _discover_years(source_dir: Path) -> list[int]:
    years = []
    for path in sorted(source_dir.glob("*.nc")):
        if path.stem.isdigit():
            years.append(int(path.stem))
    return years


def regrid_era5_years(
    years: Sequence[int] | None = None,
    variables: Sequence[str] | None = None,
    *,
    source_root: str | Path | None = None,
    output_root: str | Path | None = None,
    reference_path: str | Path | None = None,
    overwrite: bool = False,
    chunks: dict | None = None,
    config: dict | None = None,
) -> list[Path]:
    """Batch-resample ERA5 yearly files for the selected variables."""
    cfg = config or load_config()
    if source_root is None:
        source_root = resolve_repo_path(require_config(cfg, "paths", "era5_data_raw"))
    else:
        source_root = Path(source_root)
    if output_root is None:
        output_root = resolve_repo_path(require_config(cfg, "paths", "era5_data_cams_grid"))
    else:
        output_root = Path(output_root)
    methods = get_regrid_methods(cfg)
    variables = list(variables or methods.keys())

    missing_methods = [var_name for var_name in variables if var_name not in methods]
    if missing_methods:
        raise KeyError(f"Missing regrid methods for variables: {missing_methods}")

    target_grid = ensure_cams_target_grid(
        output_root=output_root,
        reference_path=reference_path,
        config=cfg,
        overwrite=overwrite,
    )

    written: list[Path] = []
    for var_name in variables:
        var_dir = source_root / var_name
        if not var_dir.exists():
            raise FileNotFoundError(f"ERA5 variable directory not found: {var_dir}")

        var_years = years or _discover_years(var_dir)
        for year in var_years:
            input_path = var_dir / f"{year}.nc"
            if not input_path.exists():
                raise FileNotFoundError(f"ERA5 yearly file not found: {input_path}")

            output_path = output_root / var_name / f"{year}.nc"
            written.append(
                regrid_era5_file(
                    input_path,
                    output_path,
                    var_name,
                    target_grid,
                    methods[var_name],
                    chunks=chunks,
                    overwrite=overwrite,
                )
            )
    return written


def _build_time_chunks(time_name: str, chunk_time: int) -> dict | None:
    try:
        import dask  # noqa: F401
    except ImportError:
        return None

    chunks = {time_name: chunk_time}
    if time_name != "time":
        chunks["time"] = chunk_time
    return chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resample ERA5 single-level fields onto the CAMS Europe grid.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Years to process. Defaults to all *.nc files under each variable directory.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        help="ERA5 variable folders to process. Defaults to all configured variables.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Raw ERA5 root directory. Defaults to paths.era5_data_raw.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Resampled ERA5 root directory. Defaults to paths.era5_data_cams_grid.",
    )
    parser.add_argument(
        "--reference-cams",
        type=Path,
        help="CAMS monthly NetCDF used to define the target grid.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached target grid and existing resampled yearly files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional config YAML. Defaults to configs/default.yaml, then configs/local.yaml when it exists.",
    )
    parser.add_argument(
        "--chunk-time",
        type=int,
        default=24,
        help="Dask chunk size along the ERA5 time dimension.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    time_name = cfg["era5_euro"]["coordinate_names"]["time"]
    chunks = _build_time_chunks(time_name, args.chunk_time)
    if chunks is None:
        print("dask is not installed; processing yearly files without chunking.")

    written = regrid_era5_years(
        years=args.years,
        variables=args.variables,
        source_root=args.source_root,
        output_root=args.output_root,
        reference_path=args.reference_cams,
        overwrite=args.overwrite,
        chunks=chunks,
        config=cfg,
    )
    print(f"Wrote {len(written)} resampled ERA5 file(s).")


if __name__ == "__main__":
    main()