"""CLI for the daily uncertainty/confidence product."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from airwise.cli._common import configure_logging, parse_date
from airwise.config import load_settings
from airwise.data.io.cams import load_cams_dataset, resolve_daily_nc_path
from airwise.data.io.uncertainty_product import bulletin_uq_path
from airwise.pipelines.compute_confidence import ensure_daily_uq_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute daily AQI confidence from local CAMS and OpenIFS control inputs."
    )
    parser.add_argument("--date", "-d", required=True, type=parse_date)
    parser.add_argument("--forecast-dir", type=Path)
    parser.add_argument("--aqi-dir", type=Path)
    parser.add_argument("--uq-dir", type=Path)
    parser.add_argument("--open-ifs-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    settings = load_settings()
    forecast_dir = args.forecast_dir or settings.paths.cams_forecast_daily
    aqi_dir = args.aqi_dir or settings.paths.cams_aqi_daily
    uq_dir = args.uq_dir or settings.paths.bulletin_uq
    forecast_path = resolve_daily_nc_path(forecast_dir, args.date)
    aqi_path = resolve_daily_nc_path(aqi_dir, args.date)
    with load_cams_dataset(forecast_path) as concentrations, load_cams_dataset(
        aqi_path
    ) as aqi:
        product = ensure_daily_uq_dataset(
            report_date=args.date,
            conc_day=concentrations,
            aqi_day=aqi,
            uq_dir=uq_dir,
            open_ifs_root=args.open_ifs_root or settings.paths.open_ifs_data,
            force=args.force,
            device=args.device,
        )
        product.close()
    print(bulletin_uq_path(uq_dir, args.date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
