"""CLI for explicit acquisition of daily remote inputs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from airwise.cli._common import (
    SUPPORTED_COUNTRY,
    configure_logging,
    parse_date,
    parse_supported_country,
)
from airwise.data.download.openifs import OpenIFSDownloadError
from airwise.pipelines.acquire_daily_inputs import acquire_daily_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download CAMS, OpenIFS control, and Policy inputs for one day. "
            "Only Germany is currently supported."
        )
    )
    parser.add_argument("--date", "-d", required=True, type=parse_date)
    parser.add_argument(
        "--country",
        default=SUPPORTED_COUNTRY,
        type=parse_supported_country,
        help="Policy download country. Only Germany is currently supported.",
    )
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
    try:
        paths = acquire_daily_inputs(
            args.date,
            country=args.country,
            skip_existing=not args.overwrite,
        )
    except OpenIFSDownloadError as exc:
        print(exc, file=sys.stderr)
        return 1
    locations = [paths.cams_forecast]
    if paths.openifs_control:
        locations.append(paths.openifs_control[0].parent)
    locations.append(paths.policy_forecasts[0].parent)
    print("===== Download finished. All data saved to: =====")
    for path in locations:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
