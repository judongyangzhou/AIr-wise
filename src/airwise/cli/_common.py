"""Shared command-line parsing helpers."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

SUPPORTED_COUNTRY = "Germany"


def parse_supported_country(value: str) -> str:
    """Accept only the country the daily bulletin is configured for."""
    if value != SUPPORTED_COUNTRY:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not supported; only {SUPPORTED_COUNTRY} is currently supported"
        )
    return value


def parse_date(value: str) -> date:
    for pattern in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            pass
    raise ValueError(f"Invalid date {value!r}; expected YYYYMMDD or YYYY-MM-DD.")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
