"""Meteogram plotting helpers for daily bulletin PDFs."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from reportlab.platypus import Image

from airwise.domain.aqi import load_aqi_definition
from airwise.domain.bulletin import PollutantMeta, PollutantSeries, Region
from airwise.domain.pollutants import POLLUTANT_IDS
from airwise.reporting.styles import ARIAL_BOLD_PATH, ARIAL_REGULAR_PATH, format_concentration_unit

FIGURE_PNG_DPI = 220


def apply_bulletin_plot_style() -> None:
    """Use the same Arial files as the bulletin PDF text."""
    if ARIAL_REGULAR_PATH.is_file():
        font_manager.fontManager.addfont(str(ARIAL_REGULAR_PATH))
        if ARIAL_BOLD_PATH.is_file():
            font_manager.fontManager.addfont(str(ARIAL_BOLD_PATH))
        family = "Arial"
        sans = ["Arial"]
    else:
        family = "sans-serif"
        sans = ["Liberation Sans", "Helvetica", "DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": sans,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 5.5,
            "legend.title_fontsize": 6,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
        }
    )


# Background tints tuned for chart readability; legend boxes match these colours.
METEOGRAM_BAND_COLORS: dict[int, str] = {
    1: "#D6F0FF",
    2: "#D9F2B0",
    3: "#FFF8B3",
    4: "#FFE0A3",
    5: "#FFB8B8",
    6: "#E0B8E0",
}


@dataclass(frozen=True)
class AqiThresholdBand:
    level: int
    label: str
    low: float
    high: float | None
    color: str


def load_aqi_threshold_bands(
    aqi_config_path: str | Path,
    pollutant_variable: str,
) -> list[AqiThresholdBand]:
    config = load_aqi_definition(aqi_config_path)
    threshold_config = config.get("thresholds", {}).get(pollutant_variable)
    if threshold_config is None:
        raise ValueError(f"Missing AQI thresholds for pollutant variable {pollutant_variable!r}")

    bins = threshold_config.get("bins")
    if not isinstance(bins, list) or len(bins) != 6:
        raise ValueError(f"AQI thresholds for {pollutant_variable!r} must define six bins")

    levels = config.get("levels", {})
    bands: list[AqiThresholdBand] = []
    for index, low in enumerate(bins):
        level = index + 1
        level_config = levels.get(level) or levels.get(str(level))
        if level_config is None:
            raise ValueError(f"Missing AQI level {level} for {pollutant_variable!r}")
        bands.append(
            AqiThresholdBand(
                level=level,
                label=level_config["label"],
                low=float(low),
                high=None if index == len(bins) - 1 else float(bins[index + 1]),
                color=level_config["color"],
            )
        )
    return bands


def _series_max(series: PollutantSeries) -> float:
    values = series.upper if series.upper is not None else series.values
    return max(values) if values else 0.0


def _axis_upper_limit(series: PollutantSeries) -> float:
    max_value = _series_max(series)
    if max_value <= 0:
        return 1.0
    padding = max(max_value * 0.08, 0.5)
    return max_value + padding


def _band_background_color(band: AqiThresholdBand) -> str:
    return METEOGRAM_BAND_COLORS.get(band.level, band.color)


def _add_aqi_level_legend(ax: Axes, bands: list[AqiThresholdBand]) -> None:
    handles = [
        Patch(
            facecolor=_band_background_color(band),
            edgecolor="#666666",
            linewidth=0.5,
            label=band.label,
        )
        for band in bands
    ]
    ax.legend(
        handles=handles,
        title="Air quality\nlevel",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=5.5,
        title_fontsize=6,
        frameon=False,
        handlelength=1.0,
        handleheight=0.9,
        labelspacing=0.35,
    )


def render_pollutant_meteogram(
    ax: Axes,
    times: list[str],
    series: PollutantSeries,
    pollutant_meta: PollutantMeta,
    aqi_bands: list[AqiThresholdBand],
) -> None:
    parsed_times = pd.to_datetime(times, utc=True).to_pydatetime()
    y_limit = _axis_upper_limit(series)

    for band in aqi_bands:
        if band.low >= y_limit:
            continue
        high = y_limit if band.high is None else min(band.high, y_limit)
        ax.axhspan(
            band.low,
            high,
            facecolor=_band_background_color(band),
            alpha=0.85,
            zorder=0,
        )

    if series.lower is not None and series.upper is not None:
        ax.fill_between(
            parsed_times,
            series.lower,
            series.upper,
            facecolor="#8C8C8C",
            alpha=0.25,
            linewidth=0,
            label="confidence interval",
            zorder=3,
        )

    ax.plot(parsed_times, series.values, color="black", linewidth=1.4, label="forecast", zorder=5)
    ax.set_title(pollutant_meta.label, loc="left", fontsize=9, pad=3)
    ax.set_ylabel(format_concentration_unit(pollutant_meta.unit), fontsize=8)
    ax.set_ylim(0, y_limit)
    ax.margins(x=0)
    ax.grid(visible=True, linestyle="--", linewidth=0.4, color="0.35", alpha=0.5, zorder=1)
    ax.tick_params(axis="both", labelsize=7)
    tzinfo = parsed_times[0].tzinfo
    ax.set_xlim(parsed_times[0], parsed_times[-1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3, tz=tzinfo))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H", tz=tzinfo))
    _add_aqi_level_legend(ax, aqi_bands)


def render_region_meteogram_figure(
    region: Region,
    times: list[str],
    series_by_pollutant: dict[str, PollutantSeries],
    pollutants_meta: dict[str, PollutantMeta],
    aqi_config_path: str | Path,
) -> Figure:
    apply_bulletin_plot_style()
    fig, axes = plt.subplots(
        len(POLLUTANT_IDS),
        1,
        figsize=(7.6, 9.2),
        sharex=True,
        facecolor="white",
    )
    fig.subplots_adjust(left=0.1, right=0.82, top=0.98, bottom=0.06, hspace=0.35)

    for ax, pollutant_id in zip(axes, POLLUTANT_IDS):
        pollutant_meta = pollutants_meta[pollutant_id]
        bands = load_aqi_threshold_bands(aqi_config_path, pollutant_meta.variable)
        render_pollutant_meteogram(
            ax,
            times,
            series_by_pollutant[pollutant_id],
            pollutant_meta,
            bands,
        )

    axes[-1].set_xlabel("time [UTC]", fontsize=8)
    return fig


def _png_pixel_size(png_bytes: bytes) -> tuple[int, int]:
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected PNG bytes")
    width, height = struct.unpack(">II", png_bytes[16:24])
    return int(width), int(height)


def png_bytes_to_reportlab_image(png_bytes: bytes, width: float, height: float) -> Image:
    """Fit a PNG into a ReportLab image without independently stretching axes."""
    pixel_width, pixel_height = _png_pixel_size(png_bytes)
    aspect = pixel_width / pixel_height
    draw_width, draw_height = float(width), float(width) / aspect
    if draw_height > height:
        draw_height = float(height)
        draw_width = float(height) * aspect
    return Image(io.BytesIO(png_bytes), width=draw_width, height=draw_height)


def figure_to_reportlab_image(fig: Figure, width: float, height: float) -> Image:
    """Embed a matplotlib figure without independently stretching width and height.

    Independent width/height would squash Arial and make plot text look narrower
    than the vector Arial used in the rest of the PDF.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=FIGURE_PNG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_bytes_to_reportlab_image(buffer.getvalue(), width, height)
