"""Report styling helpers."""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import TableStyle

from airwise.domain.aqi import load_aqi_definition

HEADER_BLUE = colors.HexColor("#1F4E79")
TEXT_GREY = colors.HexColor("#666666")
GRID_GREY = colors.grey

CONCENTRATION_UNIT = "ug/m3"
_UNIT_ALIASES = {
    "ug/m3",
    "ug m-3",
    "ug m⁻³",
    "µg m⁻³",
    "µg/m³",
    "ug/m³",
}

ARIAL_REGULAR_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
ARIAL_BOLD_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf")
ARIAL_ITALIC_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf")
ARIAL_BOLD_ITALIC_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf")
_FONTS_REGISTERED = False


def format_concentration_unit(unit: str | None = None) -> str:
    if unit is None or unit in _UNIT_ALIASES:
        return CONCENTRATION_UNIT
    return unit


def bulletin_font_names() -> tuple[str, str]:
    """Return (regular, bold) font names used by the bulletin PDF and plots."""
    global _FONTS_REGISTERED
    if ARIAL_REGULAR_PATH.is_file() and ARIAL_BOLD_PATH.is_file():
        if not _FONTS_REGISTERED:
            pdfmetrics.registerFont(TTFont("Arial", str(ARIAL_REGULAR_PATH)))
            pdfmetrics.registerFont(TTFont("Arial-Bold", str(ARIAL_BOLD_PATH)))
            family_kwargs = {"normal": "Arial", "bold": "Arial-Bold"}
            if ARIAL_ITALIC_PATH.is_file():
                pdfmetrics.registerFont(TTFont("Arial-Italic", str(ARIAL_ITALIC_PATH)))
                family_kwargs["italic"] = "Arial-Italic"
            if ARIAL_BOLD_ITALIC_PATH.is_file():
                pdfmetrics.registerFont(TTFont("Arial-BoldItalic", str(ARIAL_BOLD_ITALIC_PATH)))
                family_kwargs["boldItalic"] = "Arial-BoldItalic"
            # Needed so ReportLab <b>/<i> tags switch face; otherwise inline
            # overview bold stays on the regular Arial file.
            pdfmetrics.registerFontFamily("Arial", **family_kwargs)
            _FONTS_REGISTERED = True
        return "Arial", "Arial-Bold"
    return "Helvetica", "Helvetica-Bold"


@dataclass(frozen=True)
class AqiLevelStyle:
    level: int
    label: str
    color: colors.Color


def load_aqi_levels(config_path: str | Path) -> dict[int, AqiLevelStyle]:
    config = load_aqi_definition(config_path)
    levels = config.get("levels", {})
    level_map: dict[int, AqiLevelStyle] = {}
    for level_key, level_data in levels.items():
        level = int(level_key)
        hex_color = level_data.get("color")
        label = level_data.get("label")
        if not hex_color:
            raise ValueError(f"Missing color for AQI level {level} in {config_path}")
        if not label:
            raise ValueError(f"Missing label for AQI level {level} in {config_path}")
        level_map[level] = AqiLevelStyle(
            level=level,
            label=label,
            color=colors.HexColor(hex_color),
        )

    if set(level_map) != {1, 2, 3, 4, 5, 6}:
        raise ValueError(f"AQI config must define levels 1-6: {config_path}")

    return level_map


def load_aqi_colors(config_path: str | Path) -> dict[int, colors.Color]:
    return {level: item.color for level, item in load_aqi_levels(config_path).items()}


def build_paragraph_styles() -> dict[str, ParagraphStyle]:
    sheet = getSampleStyleSheet()
    regular, bold = bulletin_font_names()
    return {
        "header_title": ParagraphStyle(
            "header_title",
            parent=sheet["Normal"],
            fontName=bold,
            fontSize=14,
            textColor=HEADER_BLUE,
            alignment=TA_LEFT,
            leading=16,
        ),
        "header_date": ParagraphStyle(
            "header_date",
            parent=sheet["Normal"],
            fontName=bold,
            fontSize=14,
            textColor=HEADER_BLUE,
            alignment=TA_RIGHT,
            leading=16,
        ),
        "section_title": ParagraphStyle(
            "section_title",
            parent=sheet["Normal"],
            fontName=bold,
            fontSize=12,
            textColor=HEADER_BLUE,
            alignment=TA_LEFT,
            leading=14,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "subsection_title": ParagraphStyle(
            "subsection_title",
            parent=sheet["Normal"],
            fontName=bold,
            fontSize=10,
            textColor=HEADER_BLUE,
            alignment=TA_LEFT,
            leading=12,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "region_link": ParagraphStyle(
            "region_link",
            parent=sheet["Normal"],
            fontName=regular,
            fontSize=7,
            textColor=HEADER_BLUE,
            alignment=TA_LEFT,
            leading=9,
        ),
        "interpretation": ParagraphStyle(
            "interpretation",
            parent=sheet["Normal"],
            fontName=regular,
            fontSize=8,
            textColor=TEXT_GREY,
            alignment=TA_LEFT,
            leading=10,
            spaceAfter=4,
        ),
        "legend_caption": ParagraphStyle(
            "legend_caption",
            parent=sheet["Normal"],
            fontName=bold,
            fontSize=8,
            textColor=TEXT_GREY,
            alignment=TA_LEFT,
            leading=10,
            spaceBefore=2,
            spaceAfter=4,
        ),
        "overview_bullet": ParagraphStyle(
            "overview_bullet",
            parent=sheet["Normal"],
            fontName=regular,
            fontSize=9,
            textColor=colors.black,
            alignment=TA_LEFT,
            leading=12,
            leftIndent=12,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=sheet["Normal"],
            fontName=regular,
            fontSize=9,
            textColor=TEXT_GREY,
            alignment=TA_LEFT,
            leading=11,
        ),
        "footer_right": ParagraphStyle(
            "footer_right",
            parent=sheet["Normal"],
            fontName=regular,
            fontSize=9,
            textColor=TEXT_GREY,
            alignment=TA_RIGHT,
            leading=11,
        ),
        "transboundary_city": ParagraphStyle(
            "transboundary_city",
            parent=sheet["Normal"],
            fontName=regular,
            fontSize=8,
            textColor=colors.black,
            alignment=TA_LEFT,
            leading=10,
        ),
        "transboundary_pollutant": ParagraphStyle(
            "transboundary_pollutant",
            parent=sheet["Normal"],
            fontName=regular,
            fontSize=8,
            textColor=colors.black,
            alignment=TA_CENTER,
            leading=10,
        ),
        "transboundary_contributors": ParagraphStyle(
            "transboundary_contributors",
            parent=sheet["Normal"],
            fontName=regular,
            fontSize=8,
            textColor=colors.black,
            alignment=TA_LEFT,
            leading=10,
        ),
    }


def build_region_table_style(
    row_count: int,
    column_count: int,
    aqi_colors: dict[int, colors.Color],
    cell_aqi_levels: list[list[int | None]],
) -> TableStyle:
    regular, bold = bulletin_font_names()
    style = TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), regular),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 7),
            ("FONTNAME", (0, 0), (-1, 0), bold),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, GRID_GREY),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF4")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )

    for row_idx, row_levels in enumerate(cell_aqi_levels, start=1):
        for col_idx, aqi_level in enumerate(row_levels, start=1):
            if aqi_level is None:
                continue
            style.add(
                "BACKGROUND",
                (col_idx, row_idx),
                (col_idx, row_idx),
                aqi_colors[aqi_level],
            )

    return style


def build_transboundary_table_style(_row_count: int, span_commands: list[tuple] | None = None) -> TableStyle:
    regular, bold = bulletin_font_names()
    commands: list = [
        ("FONTNAME", (0, 0), (-1, 0), bold),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID_GREY),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF4")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if span_commands:
        commands.extend(span_commands)
    return TableStyle(commands)
