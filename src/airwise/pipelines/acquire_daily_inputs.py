"""Explicitly acquire all remote inputs needed for one daily bulletin."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from airwise.config import AppSettings, load_settings
from airwise.data.download.cams_forecast import download_cams_forecast_day
from airwise.data.download.openifs import download_open_ifs_day
from airwise.data.download.policy import download_country_city_forecasts


@dataclass(frozen=True)
class DailyInputPaths:
    cams_forecast: Path
    openifs_control: tuple[Path, ...]
    policy_forecasts: tuple[Path, ...]


def _section(label: str, report_date: date) -> None:
    print(f"===== Download {label}: {report_date.isoformat()} =====")


def acquire_daily_inputs(
    report_date: date,
    *,
    settings: AppSettings | None = None,
    country: str = "Germany",
    skip_existing: bool = True,
    include_policy: bool = True,
    cams_downloader: Callable = download_cams_forecast_day,
    openifs_downloader: Callable = download_open_ifs_day,
    policy_downloader: Callable = download_country_city_forecasts,
) -> DailyInputPaths:
    """Download CAMS, OpenIFS control, and optional Policy inputs explicitly."""
    resolved = settings or load_settings()
    _section("CAMS forecast", report_date)
    cams_path = cams_downloader(
        report_date,
        out_dir=resolved.paths.cams_forecast_daily,
        skip_existing=skip_existing,
    )
    openifs_paths = openifs_downloader(
        report_date,
        output_dir=resolved.paths.open_ifs_data,
    )
    policy_paths = []
    if include_policy:
        _section("policy forecasts", report_date)
        policy_paths = policy_downloader(
            report_date,
            country=country,
            out_dir=resolved.paths.cams_policy_forecast,
            skip_existing=skip_existing,
        )
    return DailyInputPaths(
        cams_forecast=Path(cams_path),
        openifs_control=tuple(Path(path) for path in openifs_paths),
        policy_forecasts=tuple(Path(path) for path in policy_paths),
    )
