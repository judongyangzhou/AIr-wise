"""Build the AQI colour legend for report tables."""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

from airwise.reporting.styles import AqiLevelStyle, GRID_GREY, bulletin_font_names


def build_aqi_legend(aqi_levels: dict[int, AqiLevelStyle]) -> Table:
    labels = [aqi_levels[level].label for level in range(1, 7)]
    _, bold = bulletin_font_names()
    table = Table([labels], colWidths=[80] * 6)
    style = TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), bold),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, GRID_GREY),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#333333")),
        ]
    )
    for column, level in enumerate(range(1, 7)):
        style.add("BACKGROUND", (column, 0), (column, 0), aqi_levels[level].color)

    table.setStyle(style)
    return table
