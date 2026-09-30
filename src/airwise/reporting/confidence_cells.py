"""Sample a computed confidence grid into bulletin JSON cells."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def _nearest_index(coords: np.ndarray, value: float) -> int:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot sample an empty coordinate axis.")
    return int(np.abs(arr - float(value)).argmin())


def sample_cell_confidence(
    uq_ds: xr.Dataset,
    pollutant_id: str,
    lat: float,
    lon: float,
    hour: int,
) -> float:
    """Nearest-neighbour sample of hourly confidence at a concentration peak."""
    pollutants = np.asarray(uq_ds["pollutant"].values).astype(str)
    if pollutant_id not in set(pollutants):
        raise ValueError(
            f"Required neural confidence is missing pollutant {pollutant_id!r}."
        )
    lat_index = _nearest_index(uq_ds["latitude"].values, lat)
    lon_index = _nearest_index(uq_ds["longitude"].values, lon)
    point = uq_ds["confidence_score"].sel(pollutant=pollutant_id).isel(
        latitude=lat_index,
        longitude=lon_index,
        time=int(hour),
    )
    value = float(np.asarray(point.values).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(
            "Required neural confidence is not finite for "
            f"{pollutant_id} at ({lat}, {lon}) hour {hour:02d}:00."
        )
    return round(float(np.clip(value, 0.0, 100.0)), 1)


def attach_confidence_to_report(
    report_data: dict[str, Any],
    uq_ds: xr.Dataset,
) -> dict[str, Any]:
    """Fill pollutant-cell confidence_score from the gridded UQ product."""
    summary = report_data["pollutant_summary_by_region"]
    for region in summary["regions"]:
        for pollutant_id, cell in region["cells"].items():
            lat = cell.get("max_lat", region.get("lat"))
            lon = cell.get("max_lon", region.get("lon"))
            if lat is None or lon is None:
                raise ValueError(
                    f"Cannot sample required neural confidence for region {region.get('id')!r}: "
                    "peak coordinates are missing."
                )
            hour = int(str(cell["max_time"]).split(":", 1)[0])
            cell["confidence_score"] = sample_cell_confidence(
                uq_ds, pollutant_id, float(lat), float(lon), hour
            )
    base_interpretation = str(summary.get("interpretation", "")).rstrip()
    if base_interpretation and not base_interpretation.endswith("."):
        base_interpretation += "."
    summary["interpretation"] = (
        f"{base_interpretation} "
        "Confidence is P(CAMS forecast AQI is correct) from the 3-hourly spatial-std "
        "model; hours between leads interpolate mu/sigma, 22-23 UTC clamp to 21 UTC."
    ).strip()
    return report_data
