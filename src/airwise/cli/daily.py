"""Run the published daily report from download through PDF."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from airwise.cli._common import SUPPORTED_COUNTRY, parse_date, parse_supported_country
from airwise.cli.acquire import main as acquire_main
from airwise.cli.aqi import main as aqi_main
from airwise.cli.confidence import main as confidence_main
from airwise.cli.report import default_region_config
from airwise.cli.report import main as report_main
from airwise.config import find_repo_root

Step = tuple[str, Callable[[Sequence[str]], int], list[str]]


def default_report_json(report_day: str) -> Path:
    return find_repo_root() / "reports" / f"germany_{report_day}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download one day's inputs, compute AQI and neural confidence, "
            "and write the JSON and PDF bulletin. Only Germany is currently supported."
        )
    )
    parser.add_argument("--date", "-d", required=True, type=parse_date)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for neural confidence and the bulletin (default: cpu).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="JSON path. Default: reports/germany_YYYYMMDD.json. The PDF uses a .pdf suffix.",
    )
    parser.add_argument(
        "--region-config",
        "-r",
        default=str(default_region_config()),
        help="Region YAML. Default: configs/regions/germany.yaml.",
    )
    parser.add_argument(
        "--country",
        default=SUPPORTED_COUNTRY,
        type=parse_supported_country,
        help="Policy download country. Only Germany is currently supported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download inputs and recompute AQI even when files already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def daily_steps(args: argparse.Namespace) -> list[Step]:
    """Arguments for the four published commands, in execution order."""
    day = args.date.strftime("%Y%m%d")
    output = args.output or str(default_report_json(day))
    logging_args = ["--log-level", args.log_level]
    overwrite = ["--overwrite"] if args.overwrite else []
    return [
        (
            "acquire",
            acquire_main,
            ["--date", day, "--country", args.country, *overwrite, *logging_args],
        ),
        (
            "aqi",
            aqi_main,
            ["--date", day, *overwrite, *logging_args],
        ),
        (
            "confidence",
            confidence_main,
            ["--date", day, "--device", args.device, *logging_args],
        ),
        (
            "report",
            report_main,
            [
                "--date",
                day,
                "--region-config",
                args.region_config,
                "--output",
                output,
                "--device",
                args.device,
                *logging_args,
            ],
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name, step, step_argv in daily_steps(args):
        status = step(step_argv)
        if status != 0:
            print(f"Daily report stopped during {name} (status {status}).")
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
