"""CLI for local AQI product generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from airwise.cli._common import configure_logging, parse_date
from airwise.pipelines.compute_aqi import compute_daily_forecast_aqi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a daily AQI NetCDF from an already-downloaded CAMS forecast."
    )
    parser.add_argument("--date", "-d", type=parse_date)
    parser.add_argument("--forecast-dir", type=Path)
    parser.add_argument("--aqi-dir", "--output-dir", dest="aqi_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    report_date = args.date or datetime.now(timezone.utc).date()
    try:
        output = compute_daily_forecast_aqi(
            report_date,
            forecast_dir=args.forecast_dir,
            aqi_dir=args.aqi_dir,
            skip_existing=not args.overwrite,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
