"""Shared command-line parsing helpers."""

from __future__ import annotations

import logging
from datetime import date, datetime


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
