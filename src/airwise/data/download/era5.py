"""Download one year of ERA5 single-level reanalysis for the training domain.

The product is the Copernicus CDS dataset ``reanalysis-era5-single-levels``.
Each request is one short name from ``era5_euro.variables`` (``u10``, ``v10``,
``t2m``, ``d2m``, ``sp``, ``ssrd``, ``tp``, ``blh``), hourly, cropped to Europe.
The file is written to ``paths.era5_data_raw/<short_name>/<year>.nc``.

Example::

    airwise-download-era5 --year 2024 --variable ssrd
    airwise-download-era5 --year 2024 --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi

from airwise.config import europe_bounds, load_config, require_config, resolve_repo_path

MONTHS = [f"{month:02d}" for month in range(1, 13)]
DAYS = [f"{day:02d}" for day in range(1, 32)]
HOURS = [f"{hour:02d}:00" for hour in range(24)]


def era5_output_path(year: int, short_name: str, cfg: dict | None = None) -> Path:
    """Return ``paths.era5_data_raw/<short_name>/<year>.nc``."""
    cfg = load_config() if cfg is None else cfg
    return resolve_repo_path(require_config(cfg, "paths", "era5_data_raw")) / short_name / f"{year}.nc"


def download_era5(
    year: int,
    short_name: str,
    variable: str | None = None,
    *,
    months: list[str] | None = None,
    skip_existing: bool = True,
) -> Path:
    """Download one ERA5 variable for ``year`` into ``paths.era5_data_raw``.

    ``short_name`` is the key in ``era5_euro.variables``. ``variable`` overrides
    the CDS name from that table when set. ``months`` defaults to the full year.
    """
    cfg = load_config()
    era = require_config(cfg, "era5_euro")
    variables = require_config(era, "variables")
    cds_name = variable if variable is not None else require_config(variables, short_name)
    bounds = europe_bounds(cfg)
    out = era5_output_path(year, short_name, cfg)
    if skip_existing and out.exists():
        print(f"Skip existing ERA5 file: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": [require_config(era, "product_type")],
        "variable": [cds_name],
        "year": [str(year)],
        "month": list(months or MONTHS),
        "day": DAYS,
        "time": HOURS,
        "data_format": require_config(era, "data_format"),
        "download_format": require_config(era, "download_format"),
        "area": [bounds["north"], bounds["west"], bounds["south"], bounds["east"]],
    }
    print(f"Downloading ERA5 {short_name} ({cds_name}) {year} -> {out}")
    cdsapi.Client().retrieve(require_config(era, "dataset"), request).download(str(out))
    return out


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    known = list(require_config(cfg, "era5_euro", "variables"))
    parser = argparse.ArgumentParser(
        description="Download ERA5 single-level fields for one calendar year.",
    )
    parser.add_argument("--year", "-y", type=int, required=True, help="Calendar year, e.g. 2024.")
    parser.add_argument(
        "--variable",
        "-v",
        choices=known,
        help=f"Short name from era5_euro.variables ({', '.join(known)}).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every variable in era5_euro.variables.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=None,
        help="Months to request, 1-12. Default: the full year.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing file.",
    )
    args = parser.parse_args(argv)
    if args.all == (args.variable is not None):
        parser.error("Pass either --variable or --all.")
    if args.months is not None and any(month < 1 or month > 12 for month in args.months):
        parser.error("--months must be between 1 and 12.")
    months = None if args.months is None else [f"{month:02d}" for month in args.months]
    names = known if args.all else [args.variable]
    for short_name in names:
        download_era5(
            args.year,
            short_name,
            months=months,
            skip_existing=not args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
