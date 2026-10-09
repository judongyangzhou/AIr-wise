"""Command-line adapter for daily bulletin JSON and PDF generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.resources import as_file
from pathlib import Path

from airwise.cli._common import configure_logging, parse_date
from airwise.config import find_repo_root
from airwise.domain.bulletin import load_report
from airwise.pipelines.daily_bulletin import generate_report_json
from airwise.reporting.pdf_generator import generate_pdf_from_json
from airwise.reporting.source_receptor import (
    DEFAULT_COUNTRY,
    DEFAULT_INVENTORY,
    generate_tbi_report,
    log_previous_run_reminder,
)
from airwise.resources import resource


def default_region_config() -> Path:
    return find_repo_root() / "configs" / "regions" / "germany.yaml"


def build_json_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a daily bulletin JSON and PDF from local products."
    )
    parser.add_argument("--date", "-d", required=True, type=parse_date)
    parser.add_argument("--region-config", "-r", default=str(default_region_config()))
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="JSON path. The PDF is written beside it unless --pdf-output is set.",
    )
    parser.add_argument(
        "--pdf-output",
        help="PDF path. Defaults to --output with a .pdf suffix.",
    )
    parser.add_argument(
        "--aqi-config",
        help="Optional AQI style YAML; defaults to the bundled European AQI resource.",
    )
    parser.add_argument("--raw-forecast-dir")
    parser.add_argument("--aqi-forecast-dir")
    parser.add_argument("--uq-dir")
    parser.add_argument("--open-ifs-root")
    parser.add_argument("--force-confidence", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--policy-dir")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def pdf_output_path(json_path: str | Path, pdf_output: str | None = None) -> Path:
    if pdf_output:
        return Path(pdf_output)
    return Path(json_path).with_suffix(".pdf")


def write_bulletin_pdf(
    json_path: str | Path,
    pdf_path: str | Path,
    aqi_config: str | Path | None = None,
) -> Path:
    if aqi_config:
        return generate_pdf_from_json(json_path, pdf_path, aqi_config)
    with as_file(resource("aqi/europe.yaml")) as bundled:
        return generate_pdf_from_json(json_path, pdf_path, bundled)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_json_parser().parse_args(argv)
    configure_logging(args.log_level)
    json_output = generate_report_json(
        report_date=args.date,
        region_config_path=args.region_config,
        output_path=args.output,
        raw_forecast_dir=args.raw_forecast_dir,
        aqi_forecast_dir=args.aqi_forecast_dir,
        uq_dir=args.uq_dir,
        open_ifs_root=args.open_ifs_root,
        force_confidence=args.force_confidence,
        device=args.device,
        policy_dir=args.policy_dir,
    )
    pdf_output = write_bulletin_pdf(
        json_output,
        pdf_output_path(json_output, args.pdf_output),
        args.aqi_config,
    )
    print(f"Wrote JSON to {json_output}")
    print(f"Wrote PDF to {pdf_output}")
    _remind_if_previous_policy_run(json_output)
    return 0


def json_main(argv: Sequence[str] | None = None) -> int:
    return main(argv)


def build_pdf_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a daily bulletin PDF from JSON input."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument(
        "--config",
        "-c",
        help="Optional AQI style YAML; defaults to the bundled European AQI resource.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def pdf_main(argv: Sequence[str] | None = None) -> int:
    args = build_pdf_parser().parse_args(argv)
    configure_logging(args.log_level)
    output = write_bulletin_pdf(args.input, args.output, args.config)
    print(f"Wrote PDF to {output}")
    _remind_if_previous_policy_run(args.input)
    return 0


def _remind_if_previous_policy_run(json_path: str | Path) -> None:
    report = load_report(json_path)
    section = report.transboundary_pollution
    lead_hours = section.lead_hours if section is not None else None
    log_previous_run_reminder(report.metadata.report_date, lead_hours)


def build_transboundary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise country/source contributions to city PM10 and PM2.5 "
            "from cached CAMS Policy source-receptor forecasts."
        )
    )
    parser.add_argument("--date", "-d", required=True, type=parse_date)
    parser.add_argument("--output", "-o")
    parser.add_argument("--policy-dir")
    parser.add_argument("--country", default=DEFAULT_COUNTRY)
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def transboundary_main(argv: Sequence[str] | None = None) -> int:
    args = build_transboundary_parser().parse_args(argv)
    configure_logging(args.log_level)
    output = generate_tbi_report(
        args.date,
        output_path=args.output,
        policy_dir=args.policy_dir,
        country=args.country,
        inventory=args.inventory,
    )
    print(f"Wrote TBC JSON to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
