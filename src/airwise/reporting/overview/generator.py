"""Generate overview bullet points from diagnostics and templates."""

from __future__ import annotations

from pathlib import Path

import yaml

from airwise.domain.bulletin import DailyBulletinReport
from airwise.resources import resource
from airwise.reporting.overview.diagnostics import OverviewDiagnostics, diagnose_overview


def _default_templates_path():
    return resource("reporting/overview_templates.yaml")


def load_overview_templates(path: str | Path | None = None) -> dict:
    template_path = Path(path) if path is not None else _default_templates_path()
    with template_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_oxford_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def bold(text: str) -> str:
    return f"<b>{text}</b>"


def format_oxford_list_bold(items: list[str]) -> str:
    return format_oxford_list([bold(item) for item in items])


MAX_LISTED_REGIONS = 5


def format_region_list(items: list[str], *, limit: int = MAX_LISTED_REGIONS) -> str:
    """Oxford-join regions; if more than ``limit``, list the first ones and add etc."""
    if len(items) <= limit:
        return format_oxford_list(items)
    return f"{', '.join(items[:limit])}, etc."


def format_region_list_bold(items: list[str], *, limit: int = MAX_LISTED_REGIONS) -> str:
    if len(items) <= limit:
        return format_oxford_list_bold(items)
    return f"{', '.join(bold(item) for item in items[:limit])}, etc."

def _fill_template(template: str, values: dict[str, str]) -> str:
    return template.format(**values)


def _render_overall_bullet(diagnostics: OverviewDiagnostics, templates: dict) -> str:
    template = templates["overall"][diagnostics.risk_level]
    return _fill_template(
        template,
        {
            "country": bold(diagnostics.country),
            "overall_aqi_label": bold(diagnostics.overall_aqi_label),
            "report_date": bold(diagnostics.report_date_display),
        },
    )


def _render_pollutant_summary_bullet(diagnostics: OverviewDiagnostics, templates: dict) -> str:
    summary = diagnostics.pollutant_summary
    section = templates["pollutant_summary"]
    if summary.template_key == "none":
        return _fill_template(
            section["none"],
            {"report_date": bold(diagnostics.report_date_display)},
        )
    if summary.template_key == "single":
        leader = summary.leaders[0]
        return _fill_template(
            section["single"],
            {
                "pollutant": bold(leader.label),
                "count": bold(str(leader.count)),
                "total": bold(str(summary.total_regions)),
            },
        )
    return _fill_template(
        section["combined"],
        {
            "pollutant_list": format_oxford_list_bold([entry.label for entry in summary.leaders]),
            "count": bold(str(summary.leaders[0].count)),
            "total": bold(str(summary.total_regions)),
        },
    )


def _format_peak_time(peak) -> str:
    return bold(f"{peak.max_time} UTC")


def _format_peak_value(peak) -> str:
    return bold(f"{peak.max_concentration:.2f}")


def _render_peak_detail(peak, section: dict) -> str:
    return _fill_template(
        section["peak_detail"],
        {
            "pollutant": bold(peak.pollutant_label),
            "value": _format_peak_value(peak),
            "unit": peak.unit,
            "time": _format_peak_time(peak),
            "region": bold(peak.region),
        },
    )


def _render_forecast_peak_bullet(diagnostics: OverviewDiagnostics, templates: dict) -> str | None:
    forecast = diagnostics.forecast_peak
    section = templates["forecast_peak"]
    key = forecast.template_key

    if key == "none":
        return None
    if key == "stable":
        return section["stable"]

    if key == "single_pollutant_single_region":
        peak = forecast.peaks[0]
        return _fill_template(
            section[key],
            {
                "region": bold(peak.region),
                "time": _format_peak_time(peak),
                "pollutant": bold(peak.pollutant_label),
                "value": _format_peak_value(peak),
                "unit": peak.unit,
                "aqi_label": bold(peak.aqi_label),
            },
        )

    if key == "single_pollutant_multiple_regions":
        peak = forecast.peaks[0]
        return _fill_template(
            section[key],
            {
                "aqi_label": bold(forecast.aqi_label),
                "region_list": format_region_list_bold(forecast.regions),
                "peak_region": bold(peak.region),
                "time": _format_peak_time(peak),
                "pollutant": bold(peak.pollutant_label),
                "value": _format_peak_value(peak),
                "unit": peak.unit,
            },
        )

    if key == "multiple_pollutants_single_region":
        intro = _fill_template(
            section["multiple_pollutants_single_region_intro"],
            {
                "aqi_label": bold(forecast.aqi_label),
                "region": bold(forecast.regions[0]),
                "pollutant_list": format_oxford_list_bold(forecast.pollutants),
            },
        )
        details = " ".join(_render_peak_detail(peak, section) for peak in forecast.peaks)
        return f"{intro} {details}"

    if key == "multiple_pollutants_multiple_regions":
        intro = _fill_template(
            section["multiple_pollutants_multiple_regions_intro"],
            {
                "n": bold(str(len(forecast.peaks))),
                "aqi_label": bold(forecast.aqi_label),
            },
        )
        details = " ".join(_render_peak_detail(peak, section) for peak in forecast.peaks)
        return f"{intro} {details}"

    return None


def render_overview_bullets(
    diagnostics: OverviewDiagnostics,
    templates: dict,
) -> list[str]:
    bullets = [
        _render_overall_bullet(diagnostics, templates),
        _render_pollutant_summary_bullet(diagnostics, templates),
    ]
    forecast_bullet = _render_forecast_peak_bullet(diagnostics, templates)
    if forecast_bullet:
        bullets.append(forecast_bullet)
    return bullets


def generate_overview(
    report: DailyBulletinReport,
    aqi_config_path: str | Path,
    templates_path: str | Path | None = None,
) -> list[str]:
    diagnostics = diagnose_overview(report, aqi_config_path)
    templates = load_overview_templates(templates_path)
    return render_overview_bullets(diagnostics, templates)
