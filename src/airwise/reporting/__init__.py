"""Automated report generation for AIr-wise."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DailyBulletinReport",
    "generate_overview",
    "generate_pdf",
    "generate_pdf_from_json",
    "load_report",
]


def __getattr__(name: str) -> Any:
    if name in {"DailyBulletinReport", "load_report"}:
        from airwise.domain.bulletin import DailyBulletinReport, load_report

        return DailyBulletinReport if name == "DailyBulletinReport" else load_report
    if name == "generate_overview":
        from airwise.reporting.overview import generate_overview

        return generate_overview
    if name in {"generate_pdf", "generate_pdf_from_json"}:
        from airwise.reporting.pdf_generator import generate_pdf, generate_pdf_from_json

        return generate_pdf if name == "generate_pdf" else generate_pdf_from_json
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
