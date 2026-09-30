"""Z-score statistics for error-modelling training.

On disk the meteorology and pollutant statistics are stored separately:

- ``models/error_modelling_era5_stats.json``: shared ERA5 z-score (reused by every pollutant)
- ``models/error_modelling_{pollutant}_stats.json``: CAMS forecast + residual for that species

The in-memory ``ErrorModellingStats`` object still holds both, so dataset / inference
code can keep a single lookup. Legacy combined JSON files (with an ``era5`` block)
remain loadable so existing PM10 inference paths keep working.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from airwise.modelling.config import get_error_modelling_settings

logger = logging.getLogger(__name__)

DEFAULT_STATS_DIR = Path("models")
DEFAULT_ERA5_STATS_NAME = "error_modelling_era5_stats.json"


@dataclass(frozen=True)
class ZScoreStats:
    mean: float
    std: float

    def normalise(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalise(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass(frozen=True)
class ErrorModellingStats:
    cams_forecast: ZScoreStats
    target_error: ZScoreStats
    era5: Mapping[str, ZScoreStats]


def default_era5_stats_path(directory: str | Path | None = None) -> Path:
    root = Path(directory) if directory is not None else DEFAULT_STATS_DIR
    return root / DEFAULT_ERA5_STATS_NAME


def default_pollutant_stats_path(
    pollutant: str,
    directory: str | Path | None = None,
) -> Path:
    root = Path(directory) if directory is not None else DEFAULT_STATS_DIR
    return root / f"error_modelling_{pollutant}_stats.json"


def pollutant_from_stats_filename(path: str | Path) -> str | None:
    """Return the pollutant encoded in ``error_modelling_{pollutant}_stats.json``."""
    name = Path(path).name
    prefix = "error_modelling_"
    suffix = "_stats.json"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    middle = name[len(prefix) : -len(suffix)]
    if not middle or middle == "era5":
        return None
    return middle


def stats_from_dict(payload: Mapping) -> ErrorModellingStats:
    if "cams_forecast" not in payload or "target_error" not in payload:
        raise ValueError("Combined or pollutant stats JSON must contain cams_forecast and target_error.")
    if "era5" not in payload:
        raise ValueError("Combined stats JSON must contain an era5 block. Use load_zscore_stats for split files.")
    return ErrorModellingStats(
        cams_forecast=ZScoreStats(**payload["cams_forecast"]),
        target_error=ZScoreStats(**payload["target_error"]),
        era5=_era5_from_mapping(payload["era5"]),
    )


def _era5_from_mapping(payload: Mapping) -> dict[str, ZScoreStats]:
    return {
        var_name: ZScoreStats(**stats_payload)
        for var_name, stats_payload in payload.items()
    }


def _is_mean_std_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and "mean" in value and "std" in value


def _payload_kind(payload: Mapping) -> str:
    kind = payload.get("kind")
    if kind in {"era5", "pollutant", "combined"}:
        return str(kind)
    has_forecast = "cams_forecast" in payload
    has_target = "target_error" in payload
    has_era5 = "era5" in payload
    if has_forecast and has_target and has_era5:
        return "combined"
    if has_forecast and has_target:
        return "pollutant"
    if has_era5 and not has_forecast:
        return "era5"
    if (
        not has_forecast
        and not has_target
        and payload
        and all(_is_mean_std_mapping(value) for value in payload.values())
    ):
        return "era5"
    raise ValueError(
        "Unrecognised z-score stats JSON. Expected a combined file, a pollutant file "
        "(cams_forecast + target_error), or an ERA5 file."
    )


def _read_json(path: str | Path) -> dict:
    stats_path = Path(path)
    with stats_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Stats JSON must be an object: {stats_path}")
    return payload


def load_era5_zscore_stats(path: str | Path) -> dict[str, ZScoreStats]:
    stats_path = Path(path)
    logger.info("Loading ERA5 z-score stats from %s", stats_path)
    payload = _read_json(stats_path)
    kind = _payload_kind(payload)
    if kind == "combined":
        return _era5_from_mapping(payload["era5"])
    if kind != "era5":
        raise ValueError(f"Expected an ERA5 stats JSON at {stats_path}, got kind={kind!r}.")
    era5_payload = payload["era5"] if "era5" in payload else payload
    return _era5_from_mapping(era5_payload)


def save_era5_zscore_stats(era5: Mapping[str, ZScoreStats], path: str | Path) -> None:
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "era5",
        "era5": {
            var_name: {"mean": stats.mean, "std": stats.std}
            for var_name, stats in era5.items()
        },
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved ERA5 z-score stats to %s", stats_path)


def save_pollutant_zscore_stats(
    stats: ErrorModellingStats,
    path: str | Path,
    pollutant: str | None = None,
) -> None:
    stats_path = Path(path)
    _guard_pollutant_stats_write(stats_path, pollutant)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "pollutant",
        "pollutant": pollutant,
        "cams_forecast": asdict(stats.cams_forecast),
        "target_error": asdict(stats.target_error),
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved pollutant z-score stats to %s", stats_path)


def save_zscore_stats(stats: ErrorModellingStats, path: str | Path) -> None:
    """Write a legacy combined JSON (pollutant + ERA5). Prefer the split savers."""
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    logger.info("Saved combined z-score stats to %s", stats_path)


def _guard_pollutant_stats_write(path: Path, pollutant: str | None) -> None:
    if path.name == DEFAULT_ERA5_STATS_NAME:
        raise ValueError(
            f"Refusing to write pollutant stats to the shared ERA5 file {path}."
        )
    named = pollutant_from_stats_filename(path)
    if named is not None and pollutant is not None and named != pollutant:
        raise ValueError(
            f"Refusing to write {pollutant!r} stats to {path} "
            f"(filename encodes {named!r}). This protects existing pollutant JSON files."
        )
    if not path.exists():
        return
    payload = _read_json(path)
    try:
        kind = _payload_kind(payload)
    except ValueError:
        return
    if kind == "combined":
        raise ValueError(
            f"Refusing to overwrite legacy combined stats {path}. "
            "Keep this file for existing inference and pass a pollutant-only "
            "--stats-output path (ERA5 lives in the shared ERA5 JSON)."
        )
    if pollutant is None:
        return
    existing = payload.get("pollutant")
    if existing and existing != pollutant:
        raise ValueError(
            f"Refusing to overwrite {path}: file belongs to pollutant {existing!r}, "
            f"not {pollutant!r}."
        )


def load_zscore_stats(
    path: str | Path,
    era5_stats_path: str | Path | None = None,
) -> ErrorModellingStats:
    """Load split or legacy-combined z-score statistics."""
    stats_path = Path(path)
    logger.info("Loading z-score stats from %s", stats_path)
    payload = _read_json(stats_path)
    kind = _payload_kind(payload)
    if kind == "era5":
        raise ValueError(
            f"{stats_path} is an ERA5-only stats file. Pass the pollutant JSON as --stats-path "
            "and the ERA5 file as --era5-stats-path."
        )
    if kind == "combined":
        return stats_from_dict(payload)

    era5_path = (
        Path(era5_stats_path)
        if era5_stats_path is not None
        else default_era5_stats_path(stats_path.parent)
    )
    if not era5_path.exists():
        raise FileNotFoundError(
            f"Pollutant stats {stats_path} have no embedded ERA5 block, and "
            f"ERA5 stats file was not found at {era5_path}."
        )
    era5 = load_era5_zscore_stats(era5_path)
    pollutant = payload.get("pollutant")
    logger.info(
        "Merged pollutant stats from %s with ERA5 stats from %s (pollutant=%s)",
        stats_path,
        era5_path,
        pollutant,
    )
    return ErrorModellingStats(
        cams_forecast=ZScoreStats(**payload["cams_forecast"]),
        target_error=ZScoreStats(**payload["target_error"]),
        era5=era5,
    )


def _safe_std(da: xr.DataArray) -> float:
    std = float(da.std(skipna=True).compute())
    if not np.isfinite(std) or std == 0:
        raise ValueError("Cannot z-score data with non-finite or zero standard deviation.")
    return std


def _zscore_stats(da: xr.DataArray) -> ZScoreStats:
    logger.info("Computing z-score stats for %s dims=%s", da.name or "<unnamed>", dict(da.sizes))
    mean = float(da.mean(skipna=True).compute())
    std = _safe_std(da)
    logger.info("Computed z-score stats for %s: mean=%.6g std=%.6g", da.name or "<unnamed>", mean, std)
    return ZScoreStats(mean=mean, std=std)


def compute_era5_zscore_stats(data) -> dict[str, ZScoreStats]:
    skip_zscore = get_error_modelling_settings().era5_skip_zscore
    return {
        var_name: _zscore_stats(data.era5[var_name])
        for var_name in data.era5.data_vars
        if var_name not in skip_zscore
    }


def compute_pollutant_zscore_stats(data) -> tuple[ZScoreStats, ZScoreStats]:
    pollutant = data.pollutant
    target_error = data.cams_analysis[pollutant] - data.cams_forecast[pollutant]
    return (
        _zscore_stats(data.cams_forecast[pollutant]),
        _zscore_stats(target_error),
    )


def compute_zscore_stats(data) -> ErrorModellingStats:
    """
    Compute z-score statistics from the training/validation period only.

    Variables listed under ``error_modelling.era5.skip_zscore`` (e.g. ``rain_mask``)
    are excluded.
    """
    logger.info("Computing z-score statistics from train/val data")
    cams_forecast, target_error = compute_pollutant_zscore_stats(data)
    return ErrorModellingStats(
        cams_forecast=cams_forecast,
        target_error=target_error,
        era5=compute_era5_zscore_stats(data),
    )


def _resolve_era5_stats(
    data,
    *,
    stats_path: Path | None,
    era5_stats_path: str | Path | None,
    era5_stats_mode: str,
) -> dict[str, ZScoreStats]:
    if era5_stats_mode not in {"auto", "load", "recompute"}:
        raise ValueError("era5_stats_mode must be one of: 'auto', 'load', 'recompute'.")

    resolved_era5_path = (
        Path(era5_stats_path)
        if era5_stats_path is not None
        else (default_era5_stats_path(stats_path.parent) if stats_path is not None else None)
    )

    if resolved_era5_path is not None and era5_stats_mode in {"auto", "load"} and resolved_era5_path.exists():
        return load_era5_zscore_stats(resolved_era5_path)
    if era5_stats_mode == "load":
        missing = resolved_era5_path if resolved_era5_path is not None else "<unset>"
        raise FileNotFoundError(f"ERA5 z-score stats file does not exist: {missing}")

    logger.info("Computing ERA5 z-score statistics from train/val data")
    era5 = compute_era5_zscore_stats(data)
    if resolved_era5_path is not None:
        save_era5_zscore_stats(era5, resolved_era5_path)
    return era5


def get_zscore_stats(
    data,
    stats_path: str | Path | None = None,
    stats_mode: str = "auto",
    era5_stats_path: str | Path | None = None,
    era5_stats_mode: str = "auto",
) -> ErrorModellingStats:
    """
    Load or compute z-score statistics.

    Pollutant and ERA5 files are handled independently:

    - ``stats_mode`` applies only to the pollutant JSON (CAMS forecast + residual).
    - ``era5_stats_mode`` applies only to the shared ERA5 JSON.
    - ``auto``: load if the file exists, otherwise compute and save.
    - ``load``: require the file to exist.
    - ``recompute``: compute from data and overwrite that file.

    Recomputing a new pollutant never overwrites another pollutant's JSON, and
    never overwrites the shared ERA5 file unless ``era5_stats_mode='recompute'``.
    """
    if stats_mode not in {"auto", "load", "recompute"}:
        raise ValueError("stats_mode must be one of: 'auto', 'load', 'recompute'.")

    resolved_stats_path = Path(stats_path) if stats_path is not None else None
    pollutant = getattr(data, "pollutant", None)

    if resolved_stats_path is not None and stats_mode in {"auto", "load"} and resolved_stats_path.exists():
        return load_zscore_stats(resolved_stats_path, era5_stats_path=era5_stats_path)
    if stats_mode == "load":
        raise FileNotFoundError(f"Z-score stats file does not exist: {resolved_stats_path}")
    if resolved_stats_path is not None:
        _guard_pollutant_stats_write(resolved_stats_path, pollutant)

    logger.info("Computing pollutant z-score statistics from train/val data (pollutant=%s)", pollutant)
    cams_forecast, target_error = compute_pollutant_zscore_stats(data)
    era5 = _resolve_era5_stats(
        data,
        stats_path=resolved_stats_path,
        era5_stats_path=era5_stats_path,
        era5_stats_mode=era5_stats_mode,
    )
    stats = ErrorModellingStats(
        cams_forecast=cams_forecast,
        target_error=target_error,
        era5=era5,
    )
    if resolved_stats_path is not None:
        save_pollutant_zscore_stats(stats, resolved_stats_path, pollutant=pollutant)
    return stats
