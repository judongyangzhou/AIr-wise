"""Extract overview diagnostics from daily bulletin report data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from airwise.domain.bulletin import DailyBulletinReport, format_time_12h
from airwise.domain.pollutants import POLLUTANT_IDS
from airwise.reporting.styles import format_concentration_unit, load_aqi_levels


@dataclass(frozen=True)
class RegionSummary:
    name: str
    max_aqi_level: int
    dominant_pollutant_ids: tuple[str, ...]
    dominant_pollutant_labels: tuple[str, ...]


@dataclass(frozen=True)
class PeakEvent:
    region: str
    pollutant_id: str
    pollutant_label: str
    max_time: str
    max_concentration: float
    unit: str
    aqi_level: int
    aqi_label: str


@dataclass(frozen=True)
class PollutantDominanceEntry:
    label: str
    count: int


@dataclass(frozen=True)
class PollutantSummary:
    total_regions: int
    leaders: tuple[PollutantDominanceEntry, ...]
    template_key: str


@dataclass(frozen=True)
class MainConcern:
    pollutants: list[str]
    regions: list[str]
    pollutant_scope: str
    region_scope: str
    template_key: str


@dataclass(frozen=True)
class ForecastPeak:
    template_key: str
    aqi_label: str
    regions: list[str]
    pollutants: list[str]
    peaks: tuple[PeakEvent, ...]


@dataclass(frozen=True)
class OverviewDiagnostics:
    country: str
    report_date_display: str
    overall_aqi_level: int
    overall_aqi_label: str
    risk_level: str
    pollutant_summary: PollutantSummary
    hotspot_regions: list[str]
    hotspot_count: int
    hotspot_is_widespread: bool
    near_threshold_regions: list[str]
    main_concern: MainConcern
    peaks: tuple[PeakEvent, ...]
    forecast_peak: ForecastPeak
    include_near_threshold: bool


def _risk_level_from_aqi(aqi_level: int) -> str:
    if aqi_level <= 2:
        return "low"
    if aqi_level == 3:
        return "moderate"
    if aqi_level == 4:
        return "elevated"
    return "high"


def _dominant_pollutants_for_region(region, pollutants: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    max_aqi = max(cell.aqi_level for cell in region.cells.values())
    dominant_ids = tuple(
        pollutant_id
        for pollutant_id in POLLUTANT_IDS
        if region.cells[pollutant_id].aqi_level == max_aqi
    )
    dominant_labels = tuple(pollutants[pollutant_id].label for pollutant_id in dominant_ids)
    return dominant_ids, dominant_labels


def _summarize_regions(report: DailyBulletinReport) -> list[RegionSummary]:
    pollutants = report.pollutant_summary_by_region.pollutants
    summaries: list[RegionSummary] = []
    for region in report.pollutant_summary_by_region.regions:
        dominant_ids, dominant_labels = _dominant_pollutants_for_region(region, pollutants)
        max_aqi = max(cell.aqi_level for cell in region.cells.values())
        summaries.append(
            RegionSummary(
                name=region.name,
                max_aqi_level=max_aqi,
                dominant_pollutant_ids=dominant_ids,
                dominant_pollutant_labels=dominant_labels,
            )
        )
    return summaries


def _pollutant_summary(
    report: DailyBulletinReport,
    region_summaries: list[RegionSummary],
) -> PollutantSummary:
    counts: Counter[str] = Counter()
    for summary in region_summaries:
        counts.update(summary.dominant_pollutant_ids)

    total_regions = len(region_summaries)
    if not counts:
        return PollutantSummary(total_regions=total_regions, leaders=(), template_key="none")

    pollutant_order = report.pollutant_summary_by_region.pollutant_order
    if (
        len(counts) == len(pollutant_order)
        and len(set(counts.values())) == 1
    ):
        return PollutantSummary(total_regions=total_regions, leaders=(), template_key="none")

    max_count = max(counts.values())
    pollutants = report.pollutant_summary_by_region.pollutants
    leader_ids = [
        pollutant_id
        for pollutant_id in report.pollutant_summary_by_region.pollutant_order
        if counts.get(pollutant_id, 0) == max_count
    ]
    leaders = tuple(
        PollutantDominanceEntry(label=pollutants[pollutant_id].label, count=counts[pollutant_id])
        for pollutant_id in leader_ids
    )
    template_key = "single" if len(leaders) == 1 else "combined"
    return PollutantSummary(
        total_regions=total_regions,
        leaders=leaders,
        template_key=template_key,
    )


def _hotspot_regions(
    region_summaries: list[RegionSummary],
    overall_aqi_level: int,
) -> tuple[list[str], bool]:
    all_hotspots = sorted(
        summary.name for summary in region_summaries if summary.max_aqi_level == overall_aqi_level
    )
    is_widespread = len(all_hotspots) > 3
    return all_hotspots[:3], is_widespread


def _near_threshold_regions(
    region_summaries: list[RegionSummary],
    overall_aqi_level: int,
) -> list[str]:
    if overall_aqi_level <= 1:
        return []
    threshold = overall_aqi_level - 1
    return sorted(
        summary.name for summary in region_summaries if summary.max_aqi_level == threshold
    )


def _scope_for_count(count: int) -> str:
    if count == 0:
        return "none"
    if count == 1:
        return "single"
    return "multiple"


def _main_concern(
    report: DailyBulletinReport,
    region_summaries: list[RegionSummary],
    overall_aqi_level: int,
) -> MainConcern:
    if overall_aqi_level <= 2:
        return MainConcern(
            pollutants=[],
            regions=[],
            pollutant_scope="none",
            region_scope="none",
            template_key="none",
        )

    hotspot_summaries = [
        summary for summary in region_summaries if summary.max_aqi_level == overall_aqi_level
    ]
    regions = sorted(summary.name for summary in hotspot_summaries)

    pollutant_labels_by_id: dict[str, str] = {}
    for summary in hotspot_summaries:
        for pollutant_id, label in zip(
            summary.dominant_pollutant_ids,
            summary.dominant_pollutant_labels,
        ):
            pollutant_labels_by_id[pollutant_id] = label

    pollutants = [
        pollutant_labels_by_id[pollutant_id]
        for pollutant_id in report.pollutant_summary_by_region.pollutant_order
        if pollutant_id in pollutant_labels_by_id
    ]

    pollutant_scope = _scope_for_count(len(pollutants))
    region_scope = _scope_for_count(len(regions))
    if pollutant_scope == "none" or region_scope == "none":
        template_key = "none"
    else:
        pollutant_part = "pollutant" if pollutant_scope == "single" else "pollutants"
        region_part = "region" if region_scope == "single" else "regions"
        template_key = f"{pollutant_scope}_{pollutant_part}_{region_scope}_{region_part}"

    return MainConcern(
        pollutants=pollutants,
        regions=regions,
        pollutant_scope=pollutant_scope,
        region_scope=region_scope,
        template_key=template_key,
    )


def _global_max_aqi_level(report: DailyBulletinReport) -> int:
    return max(
        cell.aqi_level
        for region in report.pollutant_summary_by_region.regions
        for cell in region.cells.values()
    )


def _pollutants_at_aqi_level(report: DailyBulletinReport, aqi_level: int) -> tuple[str, ...]:
    return tuple(
        pollutant_id
        for pollutant_id in report.pollutant_summary_by_region.pollutant_order
        if any(
            region.cells[pollutant_id].aqi_level == aqi_level
            for region in report.pollutant_summary_by_region.regions
        )
    )


def _peak_for_pollutant_at_level(
    report: DailyBulletinReport,
    pollutant_id: str,
    aqi_level: int,
    aqi_labels: dict[int, str],
) -> PeakEvent:
    pollutants = report.pollutant_summary_by_region.pollutants
    best = None

    for region in report.pollutant_summary_by_region.regions:
        cell = region.cells[pollutant_id]
        if cell.aqi_level != aqi_level:
            continue
        candidate = (cell.max_concentration, region.name, cell)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        elif candidate[:2] == best[:2] and region.name < best[1]:
            best = candidate

    assert best is not None
    _, region_name, cell = best
    meta = pollutants[pollutant_id]
    return PeakEvent(
        region=region_name,
        pollutant_id=pollutant_id,
        pollutant_label=meta.label,
        max_time=format_time_12h(cell.max_time),
        max_concentration=cell.max_concentration,
        unit=format_concentration_unit(meta.unit),
        aqi_level=cell.aqi_level,
        aqi_label=aqi_labels[cell.aqi_level],
    )


def _find_peak_events(report: DailyBulletinReport, aqi_labels: dict[int, str]) -> tuple[PeakEvent, ...]:
    global_max = _global_max_aqi_level(report)
    if global_max <= 2:
        return ()

    pollutant_ids = _pollutants_at_aqi_level(report, global_max)
    return tuple(
        _peak_for_pollutant_at_level(report, pollutant_id, global_max, aqi_labels)
        for pollutant_id in pollutant_ids
    )


def _peak_in_region(
    report: DailyBulletinReport,
    region_name: str,
    pollutant_id: str,
    aqi_level: int,
    aqi_labels: dict[int, str],
) -> PeakEvent | None:
    pollutants = report.pollutant_summary_by_region.pollutants
    for region in report.pollutant_summary_by_region.regions:
        if region.name != region_name:
            continue
        cell = region.cells[pollutant_id]
        if cell.aqi_level != aqi_level:
            return None
        meta = pollutants[pollutant_id]
        return PeakEvent(
            region=region_name,
            pollutant_id=pollutant_id,
            pollutant_label=meta.label,
            max_time=format_time_12h(cell.max_time),
            max_concentration=cell.max_concentration,
            unit=format_concentration_unit(meta.unit),
            aqi_level=cell.aqi_level,
            aqi_label=aqi_labels[cell.aqi_level],
        )
    return None


def _pollutant_ids_for_labels(report: DailyBulletinReport, labels: list[str]) -> tuple[str, ...]:
    pollutants = report.pollutant_summary_by_region.pollutants
    label_set = set(labels)
    return tuple(
        pollutant_id
        for pollutant_id in report.pollutant_summary_by_region.pollutant_order
        if pollutants[pollutant_id].label in label_set
    )


def _peaks_for_concern(
    report: DailyBulletinReport,
    concern: MainConcern,
    overall_aqi_level: int,
    aqi_labels: dict[int, str],
) -> tuple[PeakEvent, ...]:
    pollutant_ids = _pollutant_ids_for_labels(report, concern.pollutants)
    if concern.template_key == "multiple_pollutants_single_region":
        region_name = concern.regions[0]
        peaks = [
            peak
            for pollutant_id in pollutant_ids
            if (peak := _peak_in_region(
                report, region_name, pollutant_id, overall_aqi_level, aqi_labels
            ))
        ]
        return tuple(peaks)

    all_peaks = _find_peak_events(report, aqi_labels)
    if len(pollutant_ids) == 1:
        pollutant_id = pollutant_ids[0]
        return tuple(peak for peak in all_peaks if peak.pollutant_id == pollutant_id)
    return tuple(
        peak
        for pollutant_id in pollutant_ids
        for peak in all_peaks
        if peak.pollutant_id == pollutant_id
    )


def _forecast_peak(
    report: DailyBulletinReport,
    region_summaries: list[RegionSummary],
    overall_aqi_level: int,
    risk_level: str,
    aqi_labels: dict[int, str],
) -> ForecastPeak:
    if risk_level == "low":
        return ForecastPeak(
            template_key="none",
            aqi_label=aqi_labels[overall_aqi_level],
            regions=[],
            pollutants=[],
            peaks=(),
        )
    if risk_level == "moderate":
        return ForecastPeak(
            template_key="stable",
            aqi_label=aqi_labels[overall_aqi_level],
            regions=[],
            pollutants=[],
            peaks=(),
        )

    concern = _main_concern(report, region_summaries, overall_aqi_level)
    peaks = _peaks_for_concern(report, concern, overall_aqi_level, aqi_labels)
    return ForecastPeak(
        template_key=concern.template_key,
        aqi_label=aqi_labels[overall_aqi_level],
        regions=concern.regions,
        pollutants=concern.pollutants,
        peaks=peaks,
    )


def _format_report_date_display(report: DailyBulletinReport) -> str:
    return report.metadata.report_date.strftime("%d %B %Y")


def diagnose_overview(
    report: DailyBulletinReport,
    aqi_config_path: str | Path,
) -> OverviewDiagnostics:
    aqi_levels = load_aqi_levels(aqi_config_path)
    aqi_labels = {level: item.label for level, item in aqi_levels.items()}

    region_summaries = _summarize_regions(report)
    overall_aqi_level = max(summary.max_aqi_level for summary in region_summaries)
    hotspot_regions, hotspot_is_widespread = _hotspot_regions(region_summaries, overall_aqi_level)
    hotspot_count = sum(
        1 for summary in region_summaries if summary.max_aqi_level == overall_aqi_level
    )
    near_threshold_regions = _near_threshold_regions(region_summaries, overall_aqi_level)
    include_near_threshold = (
        overall_aqi_level >= 3
        and hotspot_count <= 1
        and bool(near_threshold_regions)
    )
    risk_level = _risk_level_from_aqi(overall_aqi_level)

    return OverviewDiagnostics(
        country=report.metadata.country.title(),
        report_date_display=_format_report_date_display(report),
        overall_aqi_level=overall_aqi_level,
        overall_aqi_label=aqi_labels[overall_aqi_level],
        risk_level=risk_level,
        pollutant_summary=_pollutant_summary(report, region_summaries),
        hotspot_regions=hotspot_regions,
        hotspot_count=hotspot_count,
        hotspot_is_widespread=hotspot_is_widespread,
        near_threshold_regions=near_threshold_regions,
        main_concern=_main_concern(report, region_summaries, overall_aqi_level),
        peaks=_find_peak_events(report, aqi_labels),
        forecast_peak=_forecast_peak(
            report, region_summaries, overall_aqi_level, risk_level, aqi_labels
        ),
        include_near_threshold=include_near_threshold,
    )
