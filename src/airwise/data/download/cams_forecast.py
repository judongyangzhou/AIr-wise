"""Download one day of the CAMS Europe air-quality forecast.

The product is the Copernicus ADS dataset ``cams-europe-air-quality-forecasts``,
the 00:00 UTC ensemble forecast at surface level. One request covers lead
hours 0–23 and the variables in ``cams_europe.download_variables``
(NO2, O3, PM2.5, PM10, SO2). The file is written to
``paths.cams_forecast_daily/YYYYMMDD.nc``.

Example::

    airwise-download-cams-forecast --date 2026-08-01
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import cdsapi

from airwise.config import cams_download_settings, load_config, require_config, resolve_repo_path

LEADTIME = [str(hour) for hour in range(24)]


def parse_date(value: str) -> date:
    """Parse YYYY-MM-DD or YYYYMMDD."""
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date {value!r}; use YYYY-MM-DD or YYYYMMDD"
    )


def default_out_dir() -> Path:
    return resolve_repo_path(require_config(load_config(), "paths", "cams_forecast_daily"))


def forecast_path(init_date: date, out_dir: str | Path | None = None) -> Path:
    directory = Path(out_dir) if out_dir is not None else default_out_dir()
    return directory / f"{init_date:%Y%m%d}.nc"


def download_cams_forecast_day(
    init_date: date | None = None,
    *,
    out_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> Path:
    """Download the 00:00 UTC CAMS ensemble forecast (lead 0-23 h).

    If ``init_date`` is omitted, use today (UTC).
    """
    init_date = init_date or datetime.now(timezone.utc).date()
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = forecast_path(init_date, out_dir)

    if skip_existing and out.exists():
        print(f"Skip existing forecast: {out}")
        return out

    cams = cams_download_settings()
    iso = init_date.isoformat()
    request = {
        "variable": cams["variables"],
        "model": [cams["model"]],
        "level": [cams["level"]],
        "date": [f"{iso}/{iso}"],
        "type": ["forecast"],
        "time": ["00:00"],
        "leadtime_hour": LEADTIME,
        "data_format": cams["data_format"],
    }
    print(f"Downloading CAMS forecast: {iso} -> {out}")
    cdsapi.Client().retrieve(cams["dataset"], request).download(str(out))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a CAMS Europe air-quality forecast for one day (default: today UTC).",
    )
    parser.add_argument(
        "--date",
        "-d",
        type=parse_date,
        default=None,
        help="Initialisation date YYYY-MM-DD or YYYYMMDD (default: today UTC).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: paths.cams_forecast_daily).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing file.",
    )
    args = parser.parse_args(argv)
    output = download_cams_forecast_day(
        args.date,
        out_dir=args.out_dir or default_out_dir(),
        skip_existing=not args.overwrite,
    )
    print(f"Wrote CAMS forecast to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
