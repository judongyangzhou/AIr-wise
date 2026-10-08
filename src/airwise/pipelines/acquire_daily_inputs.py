"""Explicitly acquire all remote inputs needed for one daily bulletin."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from airwise.config import AppSettings, load_settings
from airwise.data.download.cams_forecast import download_cams_forecast_day
from airwise.data.download.openifs import download_open_ifs_day
from airwise.data.download.policy import (
    PolicyProductNotFoundError,
    download_country_city_forecasts,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyInputPaths:
    cams_forecast: Path
    openifs_control: tuple[Path, ...]
    policy_forecasts: tuple[Path, ...]
    policy_run_date: date | None = None


def _section(label: str, report_date: date) -> None:
    print(f"===== Download {label}: {report_date.isoformat()} =====")


def _download_policy_for_report(
    report_date: date,
    *,
    country: str,
    out_dir: Path,
    skip_existing: bool,
    policy_downloader: Callable,
) -> tuple[tuple[Path, ...], date | None]:
    """Download the report-date Policy run, or the previous run when it is unpublished.

    A complete HTTP 404 for every city means the daily product is not published yet.
    Partial downloads and any non-404 failure still abort. If the previous day's
    product is also unpublished, return an empty path list so the bulletin can omit
    the transboundary table.
    """
    try:
        paths = policy_downloader(
            report_date,
            country=country,
            out_dir=out_dir,
            skip_existing=skip_existing,
        )
    except PolicyProductNotFoundError:
        previous = report_date - timedelta(days=1)
        logger.warning(
            "CAMS Policy product for %s is not published; trying the %s run",
            report_date.isoformat(),
            previous.isoformat(),
        )
        print(
            f"Policy product for {report_date.isoformat()} was not published; "
            f"trying the {previous.isoformat()} run."
        )
        try:
            paths = policy_downloader(
                previous,
                country=country,
                out_dir=out_dir,
                skip_existing=skip_existing,
            )
        except PolicyProductNotFoundError:
            logger.warning(
                "CAMS Policy product for %s is also not published; "
                "the transboundary table will be omitted",
                previous.isoformat(),
            )
            return tuple(), None
        return tuple(Path(path) for path in paths), previous
    return tuple(Path(path) for path in paths), report_date


def acquire_daily_inputs(
    report_date: date,
    *,
    settings: AppSettings | None = None,
    country: str = "Germany",
    skip_existing: bool = True,
    cams_downloader: Callable = download_cams_forecast_day,
    openifs_downloader: Callable = download_open_ifs_day,
    policy_downloader: Callable = download_country_city_forecasts,
) -> DailyInputPaths:
    """Download CAMS, OpenIFS control, and Policy city forecasts.

    The OpenIFS channel comes from ``open_ifs.channel`` unless the downloader
    is called with an explicit channel.
    """
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
        overwrite=not skip_existing,
    )
    _section("policy forecasts", report_date)
    policy_paths, policy_run_date = _download_policy_for_report(
        report_date,
        country=country,
        out_dir=resolved.paths.cams_policy_forecast,
        skip_existing=skip_existing,
        policy_downloader=policy_downloader,
    )
    return DailyInputPaths(
        cams_forecast=Path(cams_path),
        openifs_control=tuple(Path(path) for path in openifs_paths),
        policy_forecasts=policy_paths,
        policy_run_date=policy_run_date,
    )
