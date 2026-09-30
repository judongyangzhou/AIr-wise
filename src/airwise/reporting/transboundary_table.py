"""Build the transboundary-pollution city table."""

from __future__ import annotations

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Table

from airwise.domain.bulletin import DailyBulletinReport, format_contributor_text
from airwise.reporting.styles import build_transboundary_table_style


def build_transboundary_table(report: DailyBulletinReport, styles: dict) -> Table:
    section = report.transboundary_pollution
    if section is None:
        raise ValueError("transboundary_pollution is required to build the city table")

    header = [
        "Cities",
        "Pollutant",
        "Main contributors (Transboundary pollution)",
    ]
    rows: list[list] = [header]
    city_style = styles["transboundary_city"]
    pollutant_style = styles["transboundary_pollutant"]
    contributors_style = styles["transboundary_contributors"]

    span_commands: list[tuple] = []
    for city in section.cities:
        first_row = len(rows)
        for pollutant_id in section.pollutant_order:
            block = city.pollutants[pollutant_id]
            rows.append(
                [
                    Paragraph(city.name, city_style),
                    Paragraph(block.label, pollutant_style),
                    Paragraph(format_contributor_text(block.top_contributors), contributors_style),
                ]
            )
        last_row = len(rows) - 1
        if last_row > first_row:
            span_commands.append(("SPAN", (0, first_row), (0, last_row)))

    usable_width = A4[0] - inch
    col_widths = [1.35 * inch, 0.85 * inch, usable_width - 2.2 * inch]
    table = Table(rows, repeatRows=1, colWidths=col_widths)
    table.setStyle(build_transboundary_table_style(len(rows), span_commands))
    return table
