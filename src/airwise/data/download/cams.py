"""Download historical CAMS Europe air quality, one month at a time.

The product is the Copernicus ADS dataset ``cams-europe-air-quality-forecasts``.
Each month is two files: surface analysis at lead 0 for every hour, and the
00:00 UTC ensemble forecast for lead hours 0–23. Both use
``cams_europe.download_variables`` (NO2, O3, PM2.5, PM10, SO2). Files are
written to ``paths.cams_data_raw/analysis`` and ``paths.cams_data_raw/forecast``
as ``YYYY-MM-DD_YYYY-MM-DD.nc``.

Example::

    airwise-download-cams-archive --start 2023-06 --end 2026-04
"""

from __future__ import annotations

import argparse
import calendar
from datetime import datetime
from pathlib import Path

import cdsapi

from airwise.config import cams_download_settings, load_config, require_config, resolve_repo_path

HOURLY = [f"{hour:02d}:00" for hour in range(24)]
LEADTIME = [str(hour) for hour in range(24)]


def parse_year_month(value: str) -> tuple[int, int]:
    """Parse ``YYYY-MM`` into ``(year, month)``."""
    try:
        parsed = datetime.strptime(value.strip(), "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid month {value!r}; use YYYY-MM.") from exc
    return parsed.year, parsed.month


def month_bounds(year: int, month: int) -> tuple[str, str]:
    """Return the first and last day of a month as ``YYYY-MM-DD``."""
    _, last_day = calendar.monthrange(year, month)
    prefix = f"{year}-{month:02d}"
    return f"{prefix}-01", f"{prefix}-{last_day:02d}"


def iter_months(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
):
    """Yield ``(year, month)`` from the start month through the end month."""
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def _cams_data_dir(data_dir=None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return resolve_repo_path(require_config(load_config(), "paths", "cams_data_raw"))


def download_analysis(date, end_date=None, data_dir=None):
    data_dir = _cams_data_dir(data_dir)
    end = end_date or date
    cams = cams_download_settings()

    out_dir = data_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{date}_{end}.nc"
    request = {
        "variable": cams["variables"],
        "model": [cams["model"]],
        "level": [cams["level"]],
        "date": [f"{date}/{end}"],
        "type": ["analysis"],
        "time": HOURLY,
        "leadtime_hour": ["0"],
        "data_format": cams["data_format"],
    }
    cdsapi.Client().retrieve(cams["dataset"], request).download(str(out))
    return out


def download_forecast(date, end_date=None, data_dir=None):
    data_dir = _cams_data_dir(data_dir)
    end = end_date or date
    cams = cams_download_settings()

    out_dir = data_dir / "forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{date}_{end}.nc"
    request = {
        "variable": cams["variables"],
        "model": [cams["model"]],
        "level": [cams["level"]],
        "date": [f"{date}/{end}"],
        "type": ["forecast"],
        "time": ["00:00"],
        "leadtime_hour": LEADTIME,
        "data_format": cams["data_format"],
    }
    cdsapi.Client().retrieve(cams["dataset"], request).download(str(out))
    return out


def download_month(
    year: int,
    month: int,
    *,
    data_dir=None,
    skip_existing: bool = True,
) -> tuple[Path | None, Path | None]:
    """Download one month of analysis and forecast."""
    start, end = month_bounds(year, month)
    data_dir = _cams_data_dir(data_dir)
    analysis_out = data_dir / "analysis" / f"{start}_{end}.nc"
    forecast_out = data_dir / "forecast" / f"{start}_{end}.nc"

    analysis_path = None
    forecast_path = None

    if not (skip_existing and analysis_out.exists()):
        print(f"Downloading analysis: {start} -> {end}")
        analysis_path = download_analysis(start, end, data_dir=data_dir)
    else:
        print(f"Skip existing analysis: {analysis_out}")

    if not (skip_existing and forecast_out.exists()):
        print(f"Downloading forecast: {start} -> {end}")
        forecast_path = download_forecast(start, end, data_dir=data_dir)
    else:
        print(f"Skip existing forecast: {forecast_out}")

    return analysis_path, forecast_path


def download_range(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    *,
    data_dir=None,
    skip_existing: bool = True,
):
    """Download each month separately, requesting analysis and forecast once per month."""
    for year, month in iter_months(start_year, start_month, end_year, end_month):
        print(f"\n=== {year}-{month:02d} ===")
        download_month(
            year,
            month,
            data_dir=data_dir,
            skip_existing=skip_existing,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download historical CAMS Europe analysis and forecast, one month at a time."
        ),
    )
    parser.add_argument(
        "--start",
        required=True,
        type=parse_year_month,
        help="First month, YYYY-MM.",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=parse_year_month,
        help="Last month, YYYY-MM, inclusive.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: paths.cams_data_raw).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing monthly files.",
    )
    args = parser.parse_args(argv)
    start_year, start_month = args.start
    end_year, end_month = args.end
    if (end_year, end_month) < (start_year, start_month):
        parser.error("--end must be the same month as --start, or a later month.")
    download_range(
        start_year,
        start_month,
        end_year,
        end_month,
        data_dir=args.out_dir,
        skip_existing=not args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
