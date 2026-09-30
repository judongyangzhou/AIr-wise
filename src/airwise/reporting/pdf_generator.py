"""Assemble daily bulletin PDF documents."""

from __future__ import annotations

import logging
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
)

from airwise.reporting.aqi_legend import build_aqi_legend
from airwise.reporting.meteogram import figure_to_reportlab_image, render_region_meteogram_figure
from airwise.domain.bulletin import DailyBulletinReport, load_report
from airwise.reporting.overview import generate_overview
from airwise.reporting.region_heatmap import (
    HEATMAP_PAGE_POLLUTANTS,
    render_region_leadtime_heatmap_flowable,
)
from airwise.reporting.region_table import build_region_pollutant_table, region_meteogram_anchor
from airwise.reporting.source_receptor import COUNTRY_CONTRIBUTION_URL
from airwise.reporting.styles import bulletin_font_names, build_paragraph_styles, load_aqi_levels
from airwise.reporting.transboundary_table import build_transboundary_table

logger = logging.getLogger(__name__)

DEFAULT_HEATMAP_SECTION_TITLE = "3 Regional concentration by lead time"
DEFAULT_HEATMAP_INTERPRETATION = (
    "Each pollutant panel shows hourly concentration by region, "
    "coloured with European AQI thresholds. Box plots show the daily "
    "distribution for each region."
)


def _format_report_date(report_date) -> str:
    return report_date.strftime("%d %B %Y")


class BulletinDocTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename: str | Path,
        report: DailyBulletinReport,
        styles: dict,
    ) -> None:
        self.report = report
        self.styles = styles
        super().__init__(str(filename), pagesize=A4)
        self._setup_page_templates()

    def _setup_page_templates(self) -> None:
        width, height = A4
        margin = 0.5 * inch
        footer_height = 0.45 * inch
        header_height = 0.55 * inch

        frame = Frame(
            margin,
            margin + footer_height,
            width - 2 * margin,
            height - 2 * margin - footer_height - header_height,
            id="main",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        template = PageTemplate(id="main", frames=[frame], onPage=self._draw_header_footer)
        self.addPageTemplates([template])

    def _draw_header_footer(self, canvas, doc) -> None:
        width, _ = A4
        margin = 0.5 * inch
        metadata = self.report.metadata

        canvas.saveState()
        regular, bold = bulletin_font_names()
        canvas.setFont(bold, 14)
        canvas.setFillColor(colors.HexColor("#1F4E79"))
        canvas.drawString(margin, A4[1] - margin - 6, metadata.title)
        canvas.drawRightString(width - margin, A4[1] - margin - 6, _format_report_date(metadata.report_date))

        footer_y = margin * 0.6
        canvas.setFont(regular, 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(margin, footer_y, f"Page {doc.page}")

        footer_parts = []
        if metadata.project_ref:
            footer_parts.append(metadata.project_ref)
        if metadata.project_title:
            footer_parts.append(metadata.project_title)
        if footer_parts:
            canvas.drawRightString(width - margin, footer_y, " | ".join(footer_parts))

        canvas.restoreState()


def build_pdf_story(
    report: DailyBulletinReport,
    aqi_colors: dict[int, colors.Color],
    styles: dict,
    aqi_levels: dict | None = None,
    overview_bullets: list[str] | None = None,
    aqi_config_path: str | Path | None = None,
) -> list:
    summary = report.pollutant_summary_by_region
    story = [
        Paragraph("1 Overview", styles["section_title"]),
    ]
    if overview_bullets:
        logger.info("Adding overview section (%d bullets)", len(overview_bullets))
        story.extend(
            Paragraph(f"- {bullet}", styles["overview_bullet"]) for bullet in overview_bullets
        )
    story.append(Spacer(1, 12))
    story.extend(
        [
            Paragraph("2 Pollutant Summary per region", styles["section_title"]),
            Paragraph(summary.interpretation, styles["interpretation"]),
            Paragraph("Box colour based on AQI ranges:", styles["legend_caption"]),
        ]
    )
    if aqi_levels is not None:
        story.append(build_aqi_legend(aqi_levels))
    link_regions = report.forecast_time_series is not None
    logger.info(
        "Adding pollutant summary table (%d regions, meteogram links=%s)",
        len(summary.regions),
        link_regions,
    )
    story.extend(
        [
            Spacer(1, 8),
            build_region_pollutant_table(
                report,
                aqi_colors,
                link_regions_to_meteograms=link_regions,
                region_link_style=styles["region_link"],
            ),
        ]
    )
    if report.forecast_time_series is not None:
        if aqi_config_path is None:
            raise ValueError("aqi_config_path is required when forecast_time_series is present")
        logger.info("Adding region-leadtime heatmaps")
        story.extend(_build_region_leadtime_heatmap_pages(report, styles, aqi_config_path))
    if report.transboundary_pollution is not None:
        logger.info(
            "Adding transboundary pollution table (%d cities)",
            len(report.transboundary_pollution.cities),
        )
        story.extend(_build_transboundary_pages(report, styles))
    if report.forecast_time_series is not None:
        if aqi_config_path is None:
            raise ValueError("aqi_config_path is required when forecast_time_series is present")
        logger.info("Adding forecast meteograms")
        story.extend(_build_forecast_time_series_pages(report, styles, aqi_config_path))
    return story


def _heatmap_section_title(report: DailyBulletinReport) -> str:
    if report.region_leadtime_heatmap is not None:
        return report.region_leadtime_heatmap.section_title
    return DEFAULT_HEATMAP_SECTION_TITLE


def _heatmap_interpretation(report: DailyBulletinReport) -> str:
    if report.region_leadtime_heatmap is not None:
        return report.region_leadtime_heatmap.interpretation
    return DEFAULT_HEATMAP_INTERPRETATION


def _meteogram_section_title(report: DailyBulletinReport) -> str:
    forecast = report.forecast_time_series
    if forecast is None:
        raise ValueError("forecast_time_series is required")
    title = forecast.section_title
    if report.transboundary_pollution is not None and title.startswith("4 "):
        return "5 " + title[2:]
    if report.region_leadtime_heatmap is not None:
        return title
    if title.startswith("3 "):
        return "4 " + title[2:]
    return title


def _build_transboundary_pages(report: DailyBulletinReport, styles: dict) -> list:
    section = report.transboundary_pollution
    if section is None:
        return []
    interpretation = section.interpretation.replace(
        COUNTRY_CONTRIBUTION_URL,
        f'<a href="{COUNTRY_CONTRIBUTION_URL}" color="#1F4E79">{COUNTRY_CONTRIBUTION_URL}</a>',
    )
    return [
        PageBreak(),
        Paragraph(section.section_title, styles["section_title"]),
        Paragraph(interpretation, styles["interpretation"]),
        Spacer(1, 8),
        build_transboundary_table(report, styles),
    ]


def _build_region_leadtime_heatmap_pages(
    report: DailyBulletinReport,
    styles: dict,
    aqi_config_path: str | Path,
) -> list:
    pages: list = [
        PageBreak(),
        Paragraph(_heatmap_section_title(report), styles["section_title"]),
        Paragraph(_heatmap_interpretation(report), styles["interpretation"]),
        Spacer(1, 8),
    ]
    for page_index, pollutant_ids in enumerate(HEATMAP_PAGE_POLLUTANTS):
        if page_index > 0:
            pages.append(PageBreak())
        labels = ", ".join(pollutant_ids)
        logger.info(
            "Rendering heatmap page %d/%d (%s)",
            page_index + 1,
            len(HEATMAP_PAGE_POLLUTANTS),
            labels,
        )
        flowable = render_region_leadtime_heatmap_flowable(
            report,
            pollutant_ids,
            aqi_config_path,
            width=7.1 * inch,
            height=8.7 * inch,
        )
        pages.append(flowable)
    return pages


def _build_forecast_time_series_pages(
    report: DailyBulletinReport,
    styles: dict,
    aqi_config_path: str | Path,
) -> list:
    forecast = report.forecast_time_series
    if forecast is None:
        return []

    summary = report.pollutant_summary_by_region
    region_by_id = {region.id: region for region in summary.regions}
    series_by_region_id = {region_series.region_id: region_series for region_series in forecast.regions}

    pages: list = [PageBreak(), Paragraph(_meteogram_section_title(report), styles["section_title"])]
    n_regions = len(summary.regions)
    for region_index, region in enumerate(summary.regions):
        if region_index > 0:
            pages.append(PageBreak())
        logger.info(
            "Rendering meteogram %d/%d (%s)",
            region_index + 1,
            n_regions,
            region.name,
        )
        pages.append(Paragraph(f'<a name="{region_meteogram_anchor(region.id)}"/>', styles["region_link"]))
        pages.append(Paragraph(region.name, styles["subsection_title"]))
        region_series = series_by_region_id[region.id]
        fig = render_region_meteogram_figure(
            region_by_id[region.id],
            forecast.times,
            region_series.series,
            summary.pollutants,
            aqi_config_path,
        )
        pages.append(figure_to_reportlab_image(fig, width=7.0 * inch, height=8.8 * inch))
    return pages


def generate_pdf(
    report: DailyBulletinReport,
    output_path: str | Path,
    aqi_config_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building bulletin PDF for %s (%s) -> %s",
        report.metadata.country,
        report.metadata.report_date.isoformat(),
        output_path,
    )

    aqi_levels = load_aqi_levels(aqi_config_path)
    aqi_colors = {level: item.color for level, item in aqi_levels.items()}
    styles = build_paragraph_styles()
    logger.info("Generating overview bullets")
    overview_bullets = generate_overview(report, aqi_config_path)
    logger.info("Generated %d overview bullets", len(overview_bullets))

    doc = BulletinDocTemplate(output_path, report, styles)
    story = build_pdf_story(
        report,
        aqi_colors,
        styles,
        aqi_levels,
        overview_bullets,
        aqi_config_path,
    )
    logger.info("Writing PDF (%d story items)", len(story))
    doc.build(story)
    logger.info("Wrote bulletin PDF to %s", output_path)
    return output_path


def generate_pdf_from_json(
    input_path: str | Path,
    output_path: str | Path,
    aqi_config_path: str | Path,
) -> Path:
    input_path = Path(input_path)
    logger.info("Loading bulletin JSON from %s", input_path)
    report = load_report(input_path)
    logger.info(
        "Loaded %s report with %d regions",
        report.metadata.report_date.isoformat(),
        len(report.pollutant_summary_by_region.regions),
    )
    return generate_pdf(report, output_path, aqi_config_path)
