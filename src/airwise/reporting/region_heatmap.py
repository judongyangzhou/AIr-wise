"""Region-by-lead-time heatmaps with aligned daily box plots."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from reportlab.platypus import Flowable, Image

from airwise.reporting.meteogram import (
    FIGURE_PNG_DPI,
    AqiThresholdBand,
    apply_bulletin_plot_style,
    load_aqi_threshold_bands,
    png_bytes_to_reportlab_image,
)
from airwise.domain.bulletin import DailyBulletinReport, PollutantMeta, Region
from airwise.reporting.region_table import region_meteogram_anchor
from airwise.reporting.styles import format_concentration_unit

HEATMAP_PAGE_POLLUTANTS: tuple[tuple[str, ...], ...] = (
    ("no2", "o3"),
    ("pm25", "pm10"),
)
HEATMAP_AXES_GID = "region-leadtime-heatmap"
REGION_LINK_COLOR = "#1F4E79"
_LABEL_LINK_PAD_PX = 4.0

RegionLabelLink = tuple[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class RegionLeadtimeMatrix:
    pollutant_id: str
    values: np.ndarray
    region_ids: tuple[str, ...]
    region_names: tuple[str, ...]
    lead_hours: tuple[int, ...]
    distributions: tuple[np.ndarray, ...] | None = None


def _lead_hours(times: list[str]) -> tuple[int, ...]:
    parsed = pd.to_datetime(times, utc=True)
    origin = parsed[0]
    hours = ((parsed - origin) / pd.Timedelta(hours=1)).astype(int)
    return tuple(int(hour) for hour in hours.to_numpy())


def build_region_leadtime_matrix(
    report: DailyBulletinReport,
    pollutant_id: str,
) -> RegionLeadtimeMatrix:
    forecast = report.forecast_time_series
    if forecast is None:
        raise ValueError("forecast_time_series is required to build a region-leadtime heatmap")

    summary = report.pollutant_summary_by_region
    series_by_region = {item.region_id: item for item in forecast.regions}
    values = np.asarray(
        [series_by_region[region.id].series[pollutant_id].values for region in summary.regions],
        dtype=float,
    )
    if values.ndim != 2 or values.shape[1] != len(forecast.times):
        raise ValueError(
            f"Expected a (n_regions, n_times) matrix for {pollutant_id}, got {values.shape}"
        )
    distribution_rows = []
    has_distribution = False
    for region in summary.regions:
        series = series_by_region[region.id].series[pollutant_id]
        if series.distribution:
            distribution_rows.append(np.asarray(series.distribution, dtype=float))
            has_distribution = True
        else:
            distribution_rows.append(np.asarray(series.values, dtype=float))
    return RegionLeadtimeMatrix(
        pollutant_id=pollutant_id,
        values=values,
        region_ids=tuple(region.id for region in summary.regions),
        region_names=tuple(region.name for region in summary.regions),
        lead_hours=_lead_hours(forecast.times),
        distributions=tuple(distribution_rows) if has_distribution else None,
    )


def _colorbar_vmax(values: np.ndarray, bands: list[AqiThresholdBand]) -> float:
    if len(bands) != 6:
        raise ValueError(f"Expected 6 AQI bands, got {len(bands)}")
    last = bands[-1]
    data_max = float(np.nanmax(values)) if values.size else 0.0
    level5_width = max(last.low - bands[-2].low, 1.0)
    vmax = last.low + level5_width
    return max(vmax, data_max * 1.02)


def _aqi_colormap(
    values: np.ndarray,
    bands: list[AqiThresholdBand],
) -> tuple[ListedColormap, BoundaryNorm, list[float]]:
    if len(bands) != 6:
        raise ValueError(f"Expected 6 AQI bands, got {len(bands)}")
    vmax = _colorbar_vmax(values, bands)
    bounds = [band.low for band in bands] + [vmax]
    cmap = ListedColormap([band.color for band in bands])
    norm = BoundaryNorm(bounds, ncolors=6, clip=True)
    return cmap, norm, bounds


def _yaml_threshold_ticks(bands: list[AqiThresholdBand]) -> list[float]:
    return [band.low for band in bands]


def _floor_threshold_label(value: float) -> str:
    return str(int(np.floor(value)))


def _colorbar_tick_labels(bands: list[AqiThresholdBand]) -> list[str]:
    labels = [_floor_threshold_label(band.low) for band in bands]
    labels[-1] = f"{labels[-1]}+"
    return labels


def _colorbar_level_names(bands: list[AqiThresholdBand]) -> list[str]:
    names: list[str] = []
    for band in bands:
        if " " in band.label:
            names.append(band.label.replace(" ", "\n", 1))
        else:
            names.append(band.label)
    return names


def _draw_aqi_colorbar(ax_cbar: Axes, bands: list[AqiThresholdBand]) -> None:
    level_bounds = np.arange(7, dtype=float)
    cmap = ListedColormap([band.color for band in bands])
    norm = BoundaryNorm(level_bounds, ncolors=6)
    colorbar = ColorbarBase(
        ax_cbar,
        cmap=cmap,
        norm=norm,
        boundaries=level_bounds,
        ticks=np.arange(6),
        orientation="horizontal",
    )
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(labelsize=6.5, pad=1, length=2)
    colorbar.set_ticklabels(_colorbar_tick_labels(bands))

    transform = blended_transform_factory(ax_cbar.transData, ax_cbar.transAxes)
    for index, name in enumerate(_colorbar_level_names(bands)):
        ax_cbar.text(
            index + 0.5,
            -0.7,
            name,
            transform=transform,
            ha="center",
            va="top",
            fontsize=5.5,
            color="#222222",
            linespacing=1.05,
            clip_on=False,
        )


def _render_pollutant_panel(
    ax_title: Axes,
    ax_cbar: Axes,
    ax_heatmap: Axes,
    ax_box: Axes,
    ax_box_title: Axes,
    matrix: RegionLeadtimeMatrix,
    pollutant_meta: PollutantMeta,
    bands: list[AqiThresholdBand],
) -> None:
    cmap, norm, bounds = _aqi_colormap(matrix.values, bands)
    n_regions, n_times = matrix.values.shape
    unit = format_concentration_unit(pollutant_meta.unit)

    ax_heatmap.imshow(
        matrix.values,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(-0.5, n_times - 0.5, n_regions - 0.5, -0.5),
    )
    ax_heatmap.set_yticks(np.arange(n_regions))
    ax_heatmap.set_yticklabels(
        matrix.region_names,
        fontsize=6.5,
        color=REGION_LINK_COLOR,
    )
    ax_heatmap.set_gid(HEATMAP_AXES_GID)
    tick_step = 2 if n_times > 12 else 1
    tick_positions = [index for index, hour in enumerate(matrix.lead_hours) if hour % tick_step == 0]
    ax_heatmap.set_xticks(tick_positions)
    ax_heatmap.set_xticklabels([str(matrix.lead_hours[index]) for index in tick_positions], fontsize=7)
    ax_heatmap.set_xlabel("Lead time (h UTC)", fontsize=8)
    ax_heatmap.set_xlim(-0.5, n_times - 0.5)
    ax_heatmap.set_ylim(n_regions - 0.5, -0.5)
    ax_heatmap.set_xticks(np.arange(-0.5, n_times, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, n_regions, 1), minor=True)
    ax_heatmap.grid(which="minor", color="white", linestyle="-", linewidth=0.45, alpha=0.85)
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)
    ax_heatmap.tick_params(axis="y", length=0, labelcolor=REGION_LINK_COLOR)

    _draw_aqi_colorbar(ax_cbar, bands)

    ax_title.axis("off")
    ax_title.set_title(
        (
            f"{pollutant_meta.label} concentration, illustrative AQI-colour scale "
            f"({format_concentration_unit(pollutant_meta.unit)})"
        ),
        fontsize=9,
        pad=4,
    )

    ax_box_title.axis("off")
    ax_box_title.set_title("Daily distribution per region", fontsize=9, pad=4)

    ax_box.boxplot(
        [
            row[~np.isnan(row)]
            for row in (
                matrix.distributions
                if matrix.distributions is not None
                else matrix.values
            )
        ],
        vert=False,
        positions=np.arange(n_regions),
        widths=0.55,
        patch_artist=True,
        manage_ticks=False,
        flierprops={
            "marker": "D",
            "markersize": 3.2,
            "markerfacecolor": "#C00000",
            "markeredgecolor": "#C00000",
            "markeredgewidth": 0.0,
        },
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"facecolor": "#E6E6E6", "edgecolor": "black", "linewidth": 0.6},
        whiskerprops={"color": "black", "linewidth": 0.6},
        capprops={"color": "black", "linewidth": 0.6},
    )
    ax_box.set_ylim(n_regions - 0.5, -0.5)
    ax_box.set_yticks(np.arange(n_regions))
    ax_box.tick_params(axis="y", labelleft=False, length=0)
    ax_box.tick_params(axis="x", labelsize=7)
    ax_box.set_xlabel(f"{pollutant_meta.label} ({unit})", fontsize=8)
    ax_box.grid(visible=True, axis="x", linestyle="--", linewidth=0.4, color="0.6", alpha=0.7)
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)


def render_region_leadtime_heatmap_page(
    report: DailyBulletinReport,
    pollutant_ids: tuple[str, ...] | list[str],
    aqi_config_path: str | Path,
) -> Figure:
    if not pollutant_ids:
        raise ValueError("At least one pollutant is required")

    n_panels = len(pollutant_ids)
    n_regions = len(report.pollutant_summary_by_region.regions)
    apply_bulletin_plot_style()
    fig = plt.figure(figsize=(7.4, 4.7 * n_panels), facecolor="white")
    outer = fig.add_gridspec(
        n_panels,
        1,
        hspace=0.52 if n_panels > 1 else 0.12,
        left=0.24,
        right=0.98,
        top=0.93,
        bottom=0.07,
    )

    for index, pollutant_id in enumerate(pollutant_ids):
        block = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer[index],
            height_ratios=[2.0, float(n_regions)],
            hspace=0.16,
        )
        header = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=block[0],
            height_ratios=[1.15, 1.0],
            width_ratios=[3.15, 1.18],
            hspace=0.22,
            wspace=0.08,
        )
        body = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=block[1],
            width_ratios=[3.15, 1.18],
            wspace=0.08,
        )
        ax_title = fig.add_subplot(header[0, 0])
        ax_cbar = fig.add_subplot(header[1, 0])
        ax_box_title = fig.add_subplot(header[0, 1])
        ax_heatmap = fig.add_subplot(body[0, 0])
        ax_box = fig.add_subplot(body[0, 1])
        matrix = build_region_leadtime_matrix(report, pollutant_id)
        pollutant_meta = report.pollutant_summary_by_region.pollutants[pollutant_id]
        bands = load_aqi_threshold_bands(aqi_config_path, pollutant_meta.variable)
        _render_pollutant_panel(
            ax_title,
            ax_cbar,
            ax_heatmap,
            ax_box,
            ax_box_title,
            matrix,
            pollutant_meta,
            bands,
        )

    return fig


class ImageWithNamedDestLinks(Flowable):
    """A ReportLab image with invisible named-destination hotspots."""

    def __init__(self, image: Image, links: Sequence[RegionLabelLink]) -> None:
        super().__init__()
        self.image = image
        self.links = list(links)
        self.hAlign = getattr(image, "hAlign", "CENTER")
        self.width = float(image.drawWidth)
        self.height = float(image.drawHeight)

    def wrap(self, availWidth, availHeight):
        width, height = self.image.wrap(availWidth, availHeight)
        self.width = width
        self.height = height
        return width, height

    def draw(self) -> None:
        self.image.drawOn(self.canv, 0, 0)
        width = float(self.image.drawWidth)
        height = float(self.image.drawHeight)
        for destination, (x0, y0, x1, y1) in self.links:
            self.canv.linkRect(
                "",
                destination,
                (x0 * width, y0 * height, x1 * width, y1 * height),
                relative=1,
                thickness=0,
            )


def _savefig_pad_inches() -> float:
    pad_inches = matplotlib.rcParams["savefig.pad_inches"]
    if isinstance(pad_inches, (int, float)):
        return float(pad_inches)
    return 0.1


def _clip_unit_interval(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _display_bbox_to_image_fraction(
    display_bbox: Bbox,
    fig: Figure,
    image_bbox_inches: Bbox,
) -> tuple[float, float, float, float] | None:
    inches = TransformedBbox(display_bbox, fig.dpi_scale_trans.inverted())
    width = image_bbox_inches.width
    height = image_bbox_inches.height
    if width <= 0 or height <= 0:
        return None
    x0 = _clip_unit_interval((inches.x0 - image_bbox_inches.x0) / width)
    x1 = _clip_unit_interval((inches.x1 - image_bbox_inches.x0) / width)
    y0 = _clip_unit_interval((inches.y0 - image_bbox_inches.y0) / height)
    y1 = _clip_unit_interval((inches.y1 - image_bbox_inches.y0) / height)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _heatmap_axes(fig: Figure) -> list[Axes]:
    return [ax for ax in fig.axes if ax.get_gid() == HEATMAP_AXES_GID]


def _region_label_links(
    fig: Figure,
    renderer,
    crop_bbox: Bbox,
    region_ids: tuple[str, ...],
    region_names: tuple[str, ...],
) -> list[RegionLabelLink]:
    links: list[RegionLabelLink] = []
    heatmap_axes = _heatmap_axes(fig)
    if not heatmap_axes:
        raise RuntimeError("Heatmap figure has no labelled region axes")
    for ax in heatmap_axes:
        labels = list(ax.get_yticklabels())
        texts = tuple(label.get_text() for label in labels)
        if texts != region_names:
            raise RuntimeError("Heatmap y-tick labels do not match report regions")
        for index, (region_id, label) in enumerate(zip(region_ids, labels)):
            label_ext = label.get_window_extent(renderer)
            y_top = float(ax.transData.transform((0.0, index - 0.5))[1])
            y_bottom = float(ax.transData.transform((0.0, index + 0.5))[1])
            x_axes_left = float(ax.transAxes.transform((0.0, 0.0))[0])
            display_bbox = Bbox.from_extents(
                float(label_ext.x0) - _LABEL_LINK_PAD_PX,
                min(y_top, y_bottom),
                max(float(label_ext.x1), x_axes_left),
                max(y_top, y_bottom),
            )
            fraction = _display_bbox_to_image_fraction(display_bbox, fig, crop_bbox)
            if fraction is None:
                raise RuntimeError(f"Could not map heatmap label for region {region_id}")
            links.append((region_meteogram_anchor(region_id), fraction))
    return links


def heatmap_png_with_region_links(
    fig: Figure,
    regions: Sequence[Region],
) -> tuple[bytes, list[RegionLabelLink]]:
    """Save a heatmap PNG and map region-name hotspots onto the same crop."""
    region_ids = tuple(region.id for region in regions)
    region_names = tuple(region.name for region in regions)
    fig.set_dpi(FIGURE_PNG_DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    crop_bbox = fig.get_tightbbox(renderer).padded(_savefig_pad_inches())
    links = _region_label_links(fig, renderer, crop_bbox, region_ids, region_names)
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=FIGURE_PNG_DPI,
        bbox_inches=crop_bbox,
        facecolor="white",
    )
    return buffer.getvalue(), links


def render_region_leadtime_heatmap_flowable(
    report: DailyBulletinReport,
    pollutant_ids: tuple[str, ...] | list[str],
    aqi_config_path: str | Path,
    width: float,
    height: float,
) -> ImageWithNamedDestLinks:
    fig = render_region_leadtime_heatmap_page(report, pollutant_ids, aqi_config_path)
    try:
        png_bytes, links = heatmap_png_with_region_links(
            fig, report.pollutant_summary_by_region.regions
        )
    finally:
        plt.close(fig)
    image = png_bytes_to_reportlab_image(png_bytes, width, height)
    return ImageWithNamedDestLinks(image, links)
