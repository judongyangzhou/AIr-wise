"""European AQI definitions and pure concentration classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr
import yaml

from airwise.resources import resource


DEFAULT_AQI_RESOURCE = "aqi/europe.yaml"
DEFAULT_AQI_CONFIG_PATH = resource(DEFAULT_AQI_RESOURCE)


@dataclass(frozen=True)
class AqiConcentrationBin:
    level: int
    label: str
    low: float
    high: float


def load_aqi_definition(path: str | Path | None = None) -> dict[str, Any]:
    """Load an AQI definition from an override path or the bundled default."""
    if path is None:
        target = resource(DEFAULT_AQI_RESOURCE)
        with target.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    else:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("AQI definition must contain a mapping.")
    return payload


def load_aqi_concentration_bins(
    pollutant: str,
    path: str | Path | None = None,
) -> list[AqiConcentrationBin]:
    """Return the configured concentration bins for one pollutant variable."""
    config = load_aqi_definition(path)
    threshold_config = config.get("thresholds", {}).get(pollutant)
    if threshold_config is None:
        available = list(config.get("thresholds", {}))
        raise KeyError(
            f"Missing AQI thresholds for pollutant {pollutant!r}. "
            f"Available: {available}"
        )

    edges_raw = threshold_config.get("bins")
    if not isinstance(edges_raw, list) or len(edges_raw) < 2:
        raise ValueError(
            f"AQI bins for {pollutant!r} must be a list with at least two edges."
        )

    levels = config.get("levels", {})
    edges = [float(value) for value in edges_raw]
    return [
        AqiConcentrationBin(
            level=index + 1,
            label=str(
                (levels.get(index + 1) or levels.get(str(index + 1)) or {}).get(
                    "label", f"level_{index + 1}"
                )
            ),
            low=low,
            high=float("inf") if index == len(edges) - 1 else edges[index + 1],
        )
        for index, low in enumerate(edges)
    ]


def load_pollutant_bins(path: str | Path | None = None) -> dict[str, list[float]]:
    """Return lower-bound edges suitable for ``numpy.digitize``."""
    config = load_aqi_definition(path)
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("AQI definition has no thresholds.")

    result: dict[str, list[float]] = {}
    for variable, spec in thresholds.items():
        edges = [float(value) for value in list(spec.get("bins") or [])]
        if len(edges) < 2:
            raise ValueError(
                f"AQI thresholds for {variable!r} must list at least two edges."
            )
        if not np.isinf(edges[-1]):
            edges.append(float("inf"))
        result[str(variable)] = edges
    return result


def classify_aqi(
    dataset: xr.Dataset,
    bins_by_variable: Mapping[str, list[float]] | None = None,
) -> xr.Dataset:
    """Classify concentration fields into European AQI levels 1–6."""
    bins_by_variable = bins_by_variable or load_pollutant_bins()
    classified = [
        xr.apply_ufunc(np.digitize, dataset[name], bins)
        for name in dataset.data_vars
        if (bins := bins_by_variable.get(name)) is not None
    ]
    if not classified:
        raise ValueError(
            "No recognised pollutant variables for AQI classification: "
            f"{list(dataset.data_vars)}"
        )
    result = xr.merge(classified)
    result.attrs = dict(dataset.attrs)
    result.attrs["aqi_standard"] = "European_AQI"
    return result
