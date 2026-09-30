"""Data models for daily air quality bulletin reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from airwise.domain.pollutants import POLLUTANT_IDS


@dataclass(frozen=True)
class PollutantMeta:
    label: str
    unit: str
    variable: str


@dataclass(frozen=True)
class PollutantCell:
    max_concentration: float
    max_time: str
    aqi_level: int
    confidence_score: float | None = None
    max_lat: float | None = None
    max_lon: float | None = None


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    cells: dict[str, PollutantCell]
    representative_city: str | None = None
    lat: float | None = None
    lon: float | None = None
    nuts_id: str | None = None


@dataclass(frozen=True)
class ReportMetadata:
    title: str
    country: str
    report_date: date
    project_ref: str | None = None
    project_title: str | None = None


@dataclass(frozen=True)
class PollutantSummaryByRegion:
    section_title: str
    interpretation: str
    pollutant_order: list[str]
    pollutants: dict[str, PollutantMeta]
    regions: list[Region]


@dataclass(frozen=True)
class PollutantSeries:
    values: list[float]
    lower: list[float] | None = None
    upper: list[float] | None = None
    distribution: list[float] | None = None


@dataclass(frozen=True)
class RegionTimeSeries:
    region_id: str
    series: dict[str, PollutantSeries]


@dataclass(frozen=True)
class ForecastTimeSeries:
    section_title: str
    times: list[str]
    regions: list[RegionTimeSeries]


@dataclass(frozen=True)
class RegionLeadtimeHeatmap:
    section_title: str
    interpretation: str


@dataclass(frozen=True)
class TransboundaryContributor:
    code: str
    label: str
    share: float


@dataclass(frozen=True)
class TransboundaryPollutant:
    label: str
    top_contributors: list[TransboundaryContributor]


@dataclass(frozen=True)
class TransboundaryCity:
    id: str
    name: str
    pollutants: dict[str, TransboundaryPollutant]


@dataclass(frozen=True)
class TransboundaryPollution:
    section_title: str
    interpretation: str
    lead_hours: list[int]
    top_n: int
    pollutant_order: list[str]
    cities: list[TransboundaryCity]


@dataclass(frozen=True)
class DailyBulletinReport:
    schema_version: str
    report_type: str
    metadata: ReportMetadata
    pollutant_summary_by_region: PollutantSummaryByRegion
    forecast_time_series: ForecastTimeSeries | None = None
    region_leadtime_heatmap: RegionLeadtimeHeatmap | None = None
    transboundary_pollution: TransboundaryPollution | None = None


class ReportValidationError(ValueError):
    """Raised when report JSON fails validation."""


def _require_mapping(data: Any, path: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ReportValidationError(f"{path} must be an object")
    return data


def _require_str(data: Any, path: str) -> str:
    if not isinstance(data, str):
        raise ReportValidationError(f"{path} must be a string")
    return data


def _require_number(data: Any, path: str) -> float:
    if not isinstance(data, (int, float)):
        raise ReportValidationError(f"{path} must be a number")
    return float(data)


def _parse_number_list(data: Any, path: str, expected_length: int) -> list[float]:
    if not isinstance(data, list):
        raise ReportValidationError(f"{path} must be an array")
    if len(data) != expected_length:
        raise ReportValidationError(f"{path} must contain {expected_length} values")
    return [_require_number(item, f"{path}[{idx}]") for idx, item in enumerate(data)]


def _parse_pollutant_cell(data: dict[str, Any], path: str) -> PollutantCell:
    max_time = _require_str(data.get("max_time"), f"{path}.max_time")
    if len(max_time) != 5 or max_time[2] != ":":
        raise ReportValidationError(f"{path}.max_time must use HH:MM format")

    aqi_level = data.get("aqi_level")
    if not isinstance(aqi_level, int) or not 1 <= aqi_level <= 6:
        raise ReportValidationError(f"{path}.aqi_level must be an integer between 1 and 6")

    confidence = data.get("confidence_score")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise ReportValidationError(f"{path}.confidence_score must be a number or null")
    if confidence is not None and not 0 <= float(confidence) <= 100:
        raise ReportValidationError(f"{path}.confidence_score must be between 0 and 100")

    max_lat = data.get("max_lat")
    max_lon = data.get("max_lon")
    if max_lat is not None and not isinstance(max_lat, (int, float)):
        raise ReportValidationError(f"{path}.max_lat must be a number")
    if max_lon is not None and not isinstance(max_lon, (int, float)):
        raise ReportValidationError(f"{path}.max_lon must be a number")

    return PollutantCell(
        max_concentration=_require_number(data.get("max_concentration"), f"{path}.max_concentration"),
        max_time=max_time,
        aqi_level=aqi_level,
        confidence_score=None if confidence is None else float(confidence),
        max_lat=None if max_lat is None else float(max_lat),
        max_lon=None if max_lon is None else float(max_lon),
    )


def _parse_region(data: dict[str, Any], path: str) -> Region:
    cells_data = _require_mapping(data.get("cells"), f"{path}.cells")
    cells: dict[str, PollutantCell] = {}
    for pollutant_id in POLLUTANT_IDS:
        if pollutant_id not in cells_data:
            raise ReportValidationError(f"{path}.cells.{pollutant_id} is required")
        cells[pollutant_id] = _parse_pollutant_cell(
            _require_mapping(cells_data[pollutant_id], f"{path}.cells.{pollutant_id}"),
            f"{path}.cells.{pollutant_id}",
        )

    lat = data.get("lat")
    lon = data.get("lon")
    if lat is not None and not isinstance(lat, (int, float)):
        raise ReportValidationError(f"{path}.lat must be a number")
    if lon is not None and not isinstance(lon, (int, float)):
        raise ReportValidationError(f"{path}.lon must be a number")

    representative_city = data.get("representative_city")
    if representative_city is not None and not isinstance(representative_city, str):
        raise ReportValidationError(f"{path}.representative_city must be a string")
    nuts_id = data.get("nuts_id")
    if nuts_id is not None and not isinstance(nuts_id, str):
        raise ReportValidationError(f"{path}.nuts_id must be a string")

    return Region(
        id=_require_str(data.get("id"), f"{path}.id"),
        name=_require_str(data.get("name"), f"{path}.name"),
        cells=cells,
        representative_city=representative_city,
        lat=None if lat is None else float(lat),
        lon=None if lon is None else float(lon),
        nuts_id=nuts_id,
    )


def _parse_number_list_free(data: Any, path: str) -> list[float]:
    if not isinstance(data, list) or not data:
        raise ReportValidationError(f"{path} must be a non-empty array")
    return [_require_number(item, f"{path}[{idx}]") for idx, item in enumerate(data)]


def _parse_pollutant_series(data: dict[str, Any], path: str, expected_length: int) -> PollutantSeries:
    values = _parse_number_list(data.get("values"), f"{path}.values", expected_length)
    lower_data = data.get("lower")
    upper_data = data.get("upper")

    if (lower_data is None) != (upper_data is None):
        raise ReportValidationError(f"{path}.lower and {path}.upper must be provided together")

    lower = None
    upper = None
    if lower_data is not None and upper_data is not None:
        lower = _parse_number_list(lower_data, f"{path}.lower", expected_length)
        upper = _parse_number_list(upper_data, f"{path}.upper", expected_length)
        for idx, (low, value, high) in enumerate(zip(lower, values, upper)):
            if low > high:
                raise ReportValidationError(f"{path}.lower[{idx}] must be less than or equal to upper[{idx}]")
            if not low <= value <= high:
                raise ReportValidationError(f"{path}.values[{idx}] must lie between lower[{idx}] and upper[{idx}]")

    distribution = None
    if data.get("distribution") is not None:
        distribution = _parse_number_list_free(data.get("distribution"), f"{path}.distribution")

    return PollutantSeries(values=values, lower=lower, upper=upper, distribution=distribution)


def _parse_forecast_time_series(
    data: Any,
    summary_regions: list[Region],
) -> ForecastTimeSeries | None:
    if data is None:
        return None

    time_series_data = _require_mapping(data, "forecast_time_series")
    times_data = time_series_data.get("times")
    if not isinstance(times_data, list) or not times_data:
        raise ReportValidationError("forecast_time_series.times must be a non-empty array")
    times = [
        _require_str(item, f"forecast_time_series.times[{idx}]")
        for idx, item in enumerate(times_data)
    ]

    regions_data = time_series_data.get("regions")
    if not isinstance(regions_data, list) or not regions_data:
        raise ReportValidationError("forecast_time_series.regions must be a non-empty array")

    expected_region_ids = [region.id for region in summary_regions]
    seen_region_ids: set[str] = set()
    regions: list[RegionTimeSeries] = []
    for idx, item in enumerate(regions_data):
        path = f"forecast_time_series.regions[{idx}]"
        region_data = _require_mapping(item, path)
        region_id = _require_str(region_data.get("region_id"), f"{path}.region_id")
        if region_id not in expected_region_ids:
            raise ReportValidationError(f"{path}.region_id must match a pollutant_summary_by_region region id")
        if region_id in seen_region_ids:
            raise ReportValidationError(f"{path}.region_id must be unique")
        seen_region_ids.add(region_id)

        series_data = _require_mapping(region_data.get("series"), f"{path}.series")
        series: dict[str, PollutantSeries] = {}
        for pollutant_id in POLLUTANT_IDS:
            if pollutant_id not in series_data:
                raise ReportValidationError(f"{path}.series.{pollutant_id} is required")
            series[pollutant_id] = _parse_pollutant_series(
                _require_mapping(series_data[pollutant_id], f"{path}.series.{pollutant_id}"),
                f"{path}.series.{pollutant_id}",
                len(times),
            )
        regions.append(RegionTimeSeries(region_id=region_id, series=series))

    if set(seen_region_ids) != set(expected_region_ids):
        raise ReportValidationError("forecast_time_series.regions must include every pollutant_summary_by_region region")

    return ForecastTimeSeries(
        section_title=_require_str(time_series_data.get("section_title"), "forecast_time_series.section_title"),
        times=times,
        regions=regions,
    )


def _parse_region_leadtime_heatmap(data: Any) -> RegionLeadtimeHeatmap | None:
    if data is None:
        return None

    heatmap_data = _require_mapping(data, "region_leadtime_heatmap")
    return RegionLeadtimeHeatmap(
        section_title=_require_str(
            heatmap_data.get("section_title"),
            "region_leadtime_heatmap.section_title",
        ),
        interpretation=_require_str(
            heatmap_data.get("interpretation"),
            "region_leadtime_heatmap.interpretation",
        ),
    )


TRANSBOUNDARY_POLLUTANT_IDS = ("pm10", "pm25")


def _parse_transboundary_pollution(data: Any) -> TransboundaryPollution | None:
    if data is None:
        return None

    section = _require_mapping(data, "transboundary_pollution")
    pollutant_order = section.get("pollutant_order")
    if pollutant_order != list(TRANSBOUNDARY_POLLUTANT_IDS):
        raise ReportValidationError(
            "transboundary_pollution.pollutant_order must be ['pm10', 'pm25']"
        )

    lead_hours_raw = section.get("lead_hours")
    if not isinstance(lead_hours_raw, list) or len(lead_hours_raw) != 2:
        raise ReportValidationError("transboundary_pollution.lead_hours must contain two integers")
    lead_hours = [int(_require_number(item, f"transboundary_pollution.lead_hours[{idx}]")) for idx, item in enumerate(lead_hours_raw)]

    top_n_raw = section.get("top_n")
    if not isinstance(top_n_raw, int) or top_n_raw < 1:
        raise ReportValidationError("transboundary_pollution.top_n must be a positive integer")

    cities_data = section.get("cities")
    if not isinstance(cities_data, list) or not cities_data:
        raise ReportValidationError("transboundary_pollution.cities must be a non-empty array")

    cities: list[TransboundaryCity] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(cities_data):
        path = f"transboundary_pollution.cities[{idx}]"
        city_data = _require_mapping(item, path)
        city_id = _require_str(city_data.get("id"), f"{path}.id")
        if city_id in seen_ids:
            raise ReportValidationError(f"{path}.id must be unique")
        seen_ids.add(city_id)
        pollutants_data = _require_mapping(city_data.get("pollutants"), f"{path}.pollutants")
        pollutants: dict[str, TransboundaryPollutant] = {}
        for pollutant_id in TRANSBOUNDARY_POLLUTANT_IDS:
            poll_path = f"{path}.pollutants.{pollutant_id}"
            poll_data = _require_mapping(pollutants_data.get(pollutant_id), poll_path)
            contributors_data = poll_data.get("top_contributors")
            if not isinstance(contributors_data, list) or not contributors_data:
                raise ReportValidationError(f"{poll_path}.top_contributors must be a non-empty array")
            contributors: list[TransboundaryContributor] = []
            for contrib_idx, contrib_item in enumerate(contributors_data):
                contrib_path = f"{poll_path}.top_contributors[{contrib_idx}]"
                contrib = _require_mapping(contrib_item, contrib_path)
                share = _require_number(contrib.get("share"), f"{contrib_path}.share")
                if not 0 <= share <= 1:
                    raise ReportValidationError(f"{contrib_path}.share must be between 0 and 1")
                contributors.append(
                    TransboundaryContributor(
                        code=_require_str(contrib.get("code"), f"{contrib_path}.code"),
                        label=_require_str(contrib.get("label"), f"{contrib_path}.label"),
                        share=share,
                    )
                )
            pollutants[pollutant_id] = TransboundaryPollutant(
                label=_require_str(poll_data.get("label"), f"{poll_path}.label"),
                top_contributors=contributors,
            )
        cities.append(
            TransboundaryCity(
                id=city_id,
                name=_require_str(city_data.get("name"), f"{path}.name"),
                pollutants=pollutants,
            )
        )

    return TransboundaryPollution(
        section_title=_require_str(section.get("section_title"), "transboundary_pollution.section_title"),
        interpretation=_require_str(section.get("interpretation"), "transboundary_pollution.interpretation"),
        lead_hours=lead_hours,
        top_n=top_n_raw,
        pollutant_order=list(TRANSBOUNDARY_POLLUTANT_IDS),
        cities=cities,
    )


def parse_report(data: dict[str, Any]) -> DailyBulletinReport:
    schema_version = _require_str(data.get("schema_version"), "schema_version")
    if schema_version != "1.0":
        raise ReportValidationError("schema_version must be '1.0'")

    report_type = _require_str(data.get("report_type"), "report_type")
    if report_type != "daily_air_quality_bulletin":
        raise ReportValidationError("report_type must be 'daily_air_quality_bulletin'")

    metadata_data = _require_mapping(data.get("metadata"), "metadata")
    report_date_raw = _require_str(metadata_data.get("report_date"), "metadata.report_date")
    try:
        report_date = date.fromisoformat(report_date_raw)
    except ValueError as exc:
        raise ReportValidationError("metadata.report_date must be an ISO date (YYYY-MM-DD)") from exc

    metadata = ReportMetadata(
        title=_require_str(metadata_data.get("title"), "metadata.title"),
        country=_require_str(metadata_data.get("country"), "metadata.country"),
        report_date=report_date,
        project_ref=metadata_data.get("project_ref"),
        project_title=metadata_data.get("project_title"),
    )

    summary_data = _require_mapping(
        data.get("pollutant_summary_by_region"),
        "pollutant_summary_by_region",
    )
    pollutant_order = summary_data.get("pollutant_order")
    if pollutant_order != list(POLLUTANT_IDS):
        raise ReportValidationError(
            "pollutant_summary_by_region.pollutant_order must be "
            "['no2', 'o3', 'pm25', 'pm10']"
        )

    pollutants_data = _require_mapping(summary_data.get("pollutants"), "pollutant_summary_by_region.pollutants")
    pollutants: dict[str, PollutantMeta] = {}
    for pollutant_id in POLLUTANT_IDS:
        meta = _require_mapping(
            pollutants_data.get(pollutant_id),
            f"pollutant_summary_by_region.pollutants.{pollutant_id}",
        )
        pollutants[pollutant_id] = PollutantMeta(
            label=_require_str(meta.get("label"), f"pollutants.{pollutant_id}.label"),
            unit=_require_str(meta.get("unit"), f"pollutants.{pollutant_id}.unit"),
            variable=_require_str(meta.get("variable"), f"pollutants.{pollutant_id}.variable"),
        )

    regions_data = summary_data.get("regions")
    if not isinstance(regions_data, list) or not regions_data:
        raise ReportValidationError("pollutant_summary_by_region.regions must be a non-empty array")

    regions = [
        _parse_region(_require_mapping(item, f"pollutant_summary_by_region.regions[{idx}]"), f"pollutant_summary_by_region.regions[{idx}]")
        for idx, item in enumerate(regions_data)
    ]

    summary = PollutantSummaryByRegion(
        section_title=_require_str(summary_data.get("section_title"), "pollutant_summary_by_region.section_title"),
        interpretation=_require_str(summary_data.get("interpretation"), "pollutant_summary_by_region.interpretation"),
        pollutant_order=list(POLLUTANT_IDS),
        pollutants=pollutants,
        regions=regions,
    )
    forecast_time_series = _parse_forecast_time_series(data.get("forecast_time_series"), regions)
    region_leadtime_heatmap = _parse_region_leadtime_heatmap(data.get("region_leadtime_heatmap"))
    transboundary_pollution = _parse_transboundary_pollution(data.get("transboundary_pollution"))

    return DailyBulletinReport(
        schema_version=schema_version,
        report_type=report_type,
        metadata=metadata,
        pollutant_summary_by_region=summary,
        forecast_time_series=forecast_time_series,
        region_leadtime_heatmap=region_leadtime_heatmap,
        transboundary_pollution=transboundary_pollution,
    )


def load_report(path: str | Path) -> DailyBulletinReport:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_report(data)


def format_time_12h(time_24h: str) -> str:
    """Convert HH:MM (24h) to 12-hour display, e.g. '08:00' -> '08:00 am'."""
    parsed = datetime.strptime(time_24h, "%H:%M")
    return parsed.strftime("%I:%M %p").lower()


def format_confidence(score: float | None) -> str:
    if score is None:
        return "—"
    if float(score).is_integer():
        return f"{int(score)}%"
    return f"{score:.1f}%"


def format_cell_text(cell: PollutantCell) -> str:
    return (
        f"{cell.max_concentration:.2f} | "
        f"{format_time_12h(cell.max_time)} | "
        f"{format_confidence(cell.confidence_score)}"
    )


def format_contributor_text(contributors: list[TransboundaryContributor]) -> str:
    return ", ".join(f"{item.label} {item.share * 100:.1f}%" for item in contributors)
