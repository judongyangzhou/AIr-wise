"""Build the Section 2 pollutant summary table."""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table

from airwise.domain.bulletin import DailyBulletinReport, format_cell_text
from airwise.reporting.styles import build_region_table_style, format_concentration_unit


def region_meteogram_anchor(region_id: str) -> str:
    return f"region_{region_id}"


def region_meteogram_link_markup(region_id: str, region_name: str) -> str:
    anchor = region_meteogram_anchor(region_id)
    return f'<a href="#{anchor}" color="#1F4E79">{region_name}</a>'


def build_region_pollutant_table(
    report: DailyBulletinReport,
    aqi_colors: dict[int, colors.Color],
    *,
    link_regions_to_meteograms: bool = False,
    region_link_style: ParagraphStyle | None = None,
) -> Table:
    summary = report.pollutant_summary_by_region
    pollutant_order = summary.pollutant_order

    header = ["Region\\Pollutant"]
    for pollutant_id in pollutant_order:
        meta = summary.pollutants[pollutant_id]
        header.append(f"{meta.label} ({format_concentration_unit(meta.unit)})")

    rows = [header]
    cell_aqi_levels: list[list[int | None]] = []

    for region in summary.regions:
        if link_regions_to_meteograms:
            if region_link_style is None:
                raise ValueError("region_link_style is required when link_regions_to_meteograms is True")
            row = [
                Paragraph(
                    region_meteogram_link_markup(region.id, region.name),
                    region_link_style,
                )
            ]
        else:
            row = [region.name]
        level_row: list[int | None] = []
        for pollutant_id in pollutant_order:
            cell = region.cells[pollutant_id]
            row.append(format_cell_text(cell))
            level_row.append(cell.aqi_level)
        rows.append(row)
        cell_aqi_levels.append(level_row)

    table = Table(
        rows,
        repeatRows=1,
        colWidths=[120, 95, 95, 95, 95],
    )
    table.setStyle(
        build_region_table_style(
            row_count=len(rows),
            column_count=len(header),
            aqi_colors=aqi_colors,
            cell_aqi_levels=cell_aqi_levels,
        )
    )
    return table
