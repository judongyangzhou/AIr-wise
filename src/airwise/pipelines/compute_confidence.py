"""Pipeline for the daily control-forecast AQI-confidence product."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from airwise.config import load_config, pollutant_variables, require_config, resolve_repo_path
from airwise.data.io.openifs import (
    CONTROL_MEMBER,
    OPEN_IFS_MODEL_VARIABLES,
    load_open_ifs_day,
)
from airwise.data.io.uncertainty_product import (
    build_uq_dataset,
    bulletin_uq_path,
    load_uq_netcdf,
    write_uq_netcdf,
)
from airwise.data.preprocessing.masks import load_cams_land_mask
from airwise.domain.aqi import load_aqi_concentration_bins
from airwise.domain.pollutants import POLLUTANT_IDS
from airwise.modelling.checkpoints import inference_fingerprint
from airwise.modelling.config import (
    default_lead_hours,
    default_pollutant_log_transform,
)
from airwise.modelling.features import prepare_open_ifs_features
from airwise.modelling.inference import (
    load_spatial_std_model,
    model_space_name,
    predict_mu_sigma,
)
from airwise.modelling.stats import load_zscore_stats
from airwise.uncertainty import hourly_confidence_from_leads, is_native_lead_hour


logger = logging.getLogger(__name__)


def configured_lead_hours(
    lead_hours: Sequence[int] | None = None,
) -> tuple[int, ...]:
    return tuple(int(hour) for hour in (lead_hours or default_lead_hours()))


def resolve_bulletin_uq_dir(
    config: Mapping[str, Any] | None = None,
) -> Path:
    cfg = config or load_config()
    return resolve_repo_path(require_config(cfg, "reporting", "confidence", "output_dir"))


def _resolve_existing_path(path: str | Path) -> Path:
    return resolve_repo_path(path)


def _select_lead_hours(
    day_dataset: xr.Dataset,
    lead_hours: Sequence[int],
) -> xr.Dataset:
    hours = pd.to_datetime(day_dataset["time"].values).hour
    expected = set(int(hour) for hour in lead_hours)
    indices = [index for index, hour in enumerate(hours) if int(hour) in expected]
    if len(indices) != len(expected):
        raise ValueError(
            f"Expected leads {list(lead_hours)}, "
            f"found hours {sorted(set(int(hour) for hour in hours))}"
        )
    return day_dataset.isel(time=indices)


def _confidence_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return dict(require_config(config or load_config(), "reporting", "confidence"))


def _mapping_paths(
    section: Mapping[str, Any],
    key: str,
) -> dict[str, Path]:
    raw = require_config(section, key)
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"reporting.confidence.{key} must map pollutant ids to paths."
        )
    missing = [name for name in POLLUTANT_IDS if name not in raw]
    if missing:
        raise KeyError(f"reporting.confidence.{key} is missing {missing}.")
    return {name: _resolve_existing_path(raw[name]) for name in POLLUTANT_IDS}


def _pollutant_paths(
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    confidence = _confidence_config(config)
    checkpoints = _mapping_paths(confidence, "checkpoints")
    stats = _mapping_paths(confidence, "stats")
    era5_stats = _resolve_existing_path(require_config(confidence, "era5_stats"))
    return checkpoints, stats, era5_stats


def confidence_cache_fingerprint(
    config: Mapping[str, Any] | None = None,
    *,
    lead_hours: Sequence[int] | None = None,
) -> str:
    checkpoints, stats, era5_stats = _pollutant_paths(config)
    all_stats = {**stats, "era5": era5_stats}
    return inference_fingerprint(
        checkpoints,
        all_stats,
        settings={"lead_hours": list(configured_lead_hours(lead_hours))},
    )


def _open_ifs_control_meteorology(
    *,
    open_ifs_root: str | Path,
    init_date: date,
    latitude: np.ndarray,
    longitude: np.ndarray,
    stats,
) -> np.ndarray:
    target_grid = xr.Dataset(coords={"latitude": latitude, "longitude": longitude})
    with load_open_ifs_day(open_ifs_root, init_date) as open_ifs:
        features = prepare_open_ifs_features(
            open_ifs,
            target_grid,
            stats,
            numbers=(CONTROL_MEMBER,),
        ).load()
    stacked = np.stack(
        [
            features.sel(number=CONTROL_MEMBER, variable=name).values
            for name in OPEN_IFS_MODEL_VARIABLES
        ],
        axis=0,
    )
    return np.nan_to_num(
        stacked,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype("float32")


def build_daily_uq_dataset(
    *,
    report_date: date,
    conc_day: xr.Dataset,
    aqi_day: xr.Dataset,
    open_ifs_root: str | Path,
    checkpoints: Mapping[str, Path] | None = None,
    stats_paths: Mapping[str, Path] | None = None,
    era5_stats_path: str | Path | None = None,
    land_sea_mask_path: str | Path | None = None,
    aqi_config: str | Path | None = None,
    device: str | None = None,
    lead_hours: Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Run spatial-std inference using only OpenIFS control member 0."""
    import torch

    cfg = config or load_config()
    resolved_leads = configured_lead_hours(lead_hours)
    land_path = land_sea_mask_path or require_config(cfg, "paths", "land_sea_mask")
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    latitude = np.asarray(conc_day["latitude"].values)
    longitude = np.asarray(conc_day["longitude"].values)
    times = np.asarray(conc_day["time"].values)
    n_pollutants = len(POLLUTANT_IDS)
    n_leads = len(resolved_leads)
    height, width = latitude.size, longitude.size

    confidence = np.full((n_pollutants, 24, height, width), np.nan, dtype=np.float32)
    mu = np.full((n_pollutants, n_leads, height, width), np.nan, dtype=np.float32)
    sigma = np.full_like(mu, np.nan)
    cams_aqi = np.full((n_pollutants, 24, height, width), -1, dtype=np.int8)
    transforms: list[str] = []
    used_checkpoints: dict[str, str] = {}

    land_mask = load_cams_land_mask(
        _resolve_existing_path(land_path),
        latitude=latitude,
        longitude=longitude,
    )
    conc_leads = _select_lead_hours(conc_day, resolved_leads)
    default_checkpoints, default_stats, default_era5 = _pollutant_paths(cfg)
    checkpoints = checkpoints or default_checkpoints
    stats_paths = stats_paths or default_stats
    era5_stats_path = Path(era5_stats_path or default_era5)

    first_stats_path = next(
        (
            Path(stats_paths[name])
            for name in POLLUTANT_IDS
            if Path(stats_paths[name]).is_file()
        ),
        None,
    )
    if first_stats_path is None:
        raise FileNotFoundError("No pollutant z-score stats files were found.")
    meteo_stats = load_zscore_stats(
        first_stats_path,
        era5_stats_path=era5_stats_path,
    )
    meteorology = _open_ifs_control_meteorology(
        open_ifs_root=open_ifs_root,
        init_date=report_date,
        latitude=latitude,
        longitude=longitude,
        stats=meteo_stats,
    )

    native_flag = np.asarray(
        [is_native_lead_hour(hour, resolved_leads) for hour in range(24)],
        dtype=np.int8,
    )
    variables = pollutant_variables(cfg)
    for index, pollutant_id in enumerate(POLLUTANT_IDS):
        variable = variables[pollutant_id]
        transform = default_pollutant_log_transform(variable)
        transforms.append(model_space_name(transform))
        hourly_aqi = np.asarray(aqi_day[variable].values, dtype=np.float32)
        finite = np.isfinite(hourly_aqi)
        rounded = np.full(hourly_aqi.shape, -1, dtype=np.int8)
        rounded[finite] = np.rint(hourly_aqi[finite]).astype(np.int8)
        cams_aqi[index] = rounded

        checkpoint_path = Path(checkpoints[pollutant_id])
        stats_path = Path(stats_paths[pollutant_id])
        if not checkpoint_path.is_file() or not stats_path.is_file():
            logger.warning(
                "Skipping %s confidence; checkpoint or stats are unavailable.",
                pollutant_id,
            )
            continue
        stats = load_zscore_stats(stats_path, era5_stats_path=era5_stats_path)
        model, _ = load_spatial_std_model(
            checkpoint_path,
            len(OPEN_IFS_MODEL_VARIABLES),
            resolved_device,
            n_leads,
        )
        used_checkpoints[pollutant_id] = str(checkpoint_path)
        mu_lead, sigma_lead = predict_mu_sigma(
            model,
            np.asarray(conc_leads[variable].values, dtype=np.float32),
            meteorology,
            stats,
            resolved_device,
            transform=transform,
        )
        del model
        if resolved_device.startswith("cuda"):
            torch.cuda.empty_cache()
        mu[index] = mu_lead
        sigma[index] = sigma_lead
        bins = load_aqi_concentration_bins(variable, path=aqi_config)
        confidence[index], native_flag = hourly_confidence_from_leads(
            mu_lead,
            sigma_lead,
            cams_aqi[index],
            bins,
            transform=transform,
            land_mask=land_mask,
            lead_hours=resolved_leads,
        )

    fingerprint = inference_fingerprint(
        {name: Path(path) for name, path in checkpoints.items()},
        {
            **{name: Path(path) for name, path in stats_paths.items()},
            "era5": era5_stats_path,
        },
        settings={"lead_hours": list(resolved_leads)},
    )
    return build_uq_dataset(
        report_date=report_date,
        times=times,
        latitude=latitude,
        longitude=longitude,
        confidence=confidence,
        mu=mu,
        sigma=sigma,
        cams_aqi=cams_aqi,
        land_mask=land_mask.astype(np.int8),
        transforms=transforms,
        is_native_lead=native_flag,
        lead_hours=resolved_leads,
        attrs={
            "open_ifs_root": str(open_ifs_root),
            "meteorology": "IFS control forecast (oper/fc, number=0)",
            "device": resolved_device,
            "checkpoints": json.dumps(used_checkpoints),
            "input_fingerprint": fingerprint,
        },
    )


def ensure_daily_uq_dataset(
    *,
    report_date: date,
    conc_day: xr.Dataset,
    aqi_day: xr.Dataset,
    uq_dir: str | Path | None = None,
    open_ifs_root: str | Path | None = None,
    force: bool = False,
    device: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Load or build a versioned daily uncertainty product."""
    cfg = config or load_config()
    output_dir = Path(uq_dir) if uq_dir is not None else resolve_bulletin_uq_dir(cfg)
    path = bulletin_uq_path(output_dir, report_date)
    expected_fingerprint = confidence_cache_fingerprint(cfg)
    if path.is_file() and not force:
        existing = load_uq_netcdf(path)
        if existing.attrs.get("input_fingerprint") == expected_fingerprint:
            return existing
        existing.close()
        logger.info("Rebuilding stale uncertainty product %s", path)

    configured_root = open_ifs_root or (cfg.get("paths") or {}).get("open_ifs_data")
    if not configured_root:
        raise RuntimeError(
            "Cannot generate the required neural-confidence product: "
            "the OpenIFS root is not configured."
        )
    try:
        dataset = build_daily_uq_dataset(
            report_date=report_date,
            conc_day=conc_day,
            aqi_day=aqi_day,
            open_ifs_root=configured_root,
            device=device,
            config=cfg,
        )
    except (FileNotFoundError, ImportError) as exc:
        raise RuntimeError(
            f"Cannot generate the required neural-confidence product: {exc}"
        ) from exc
    write_uq_netcdf(dataset, path)
    return dataset
