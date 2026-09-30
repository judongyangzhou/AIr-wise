"""Load error-modelling defaults from configs/default.yaml."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from airwise.config import ConfigError, load_config, require_config


@dataclass(frozen=True)
class ErrorModellingSettings:
    pollutant: str
    train_val_years: tuple[int, ...]
    test_years: tuple[int, ...]
    lead_hours: tuple[int, ...]
    data_source: str
    pollutant_log_transform: dict[str, str | None]
    cams_training_temporal_resolution: int
    era5_grid: str
    era5_temporal_resolution: int
    era5_variables: tuple[str, ...]
    era5_log_transform: dict[str, str]
    era5_skip_zscore: frozenset[str]

    @property
    def default_pollutant_log_transform(self) -> str | None:
        return self.pollutant_log_transform_for(self.pollutant)

    def pollutant_log_transform_for(self, pollutant: str) -> str | None:
        if pollutant not in self.pollutant_log_transform:
            raise ConfigError(
                f"Missing error_modelling.pollutant_log_transform entry for {pollutant!r}. "
                "Set it in configs/default.yaml or override it in configs/local.yaml."
            )
        value = self.pollutant_log_transform[pollutant]
        return None if value in (None, False) else str(value)


def _as_int_tuple(values: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _as_optional_transform_map(payload: Mapping[str, Any] | str | None) -> dict[str, str | None]:
    if payload is None:
        return {}
    if isinstance(payload, str):
        return {}
    return {
        str(key): (None if value in (None, False) else str(value))
        for key, value in payload.items()
    }


@lru_cache(maxsize=4)
def get_error_modelling_settings(config_path: str | None = None) -> ErrorModellingSettings:
    """
    Return typed error-modelling defaults from YAML.

    ``config_path`` is optional; when omitted, ``load_config()`` resolution order is used.
    """
    cfg = load_config(config_path)
    section = require_config(cfg, "error_modelling")
    cams = require_config(section, "cams")
    training_input = require_config(cams, "training_input")
    era5 = require_config(section, "era5")

    pollutant = str(require_config(section, "pollutant"))
    pollutant_log_transform = _as_optional_transform_map(
        require_config(section, "pollutant_log_transform")
    )
    legacy_transform = training_input.get("log_transform")
    if pollutant not in pollutant_log_transform and legacy_transform not in (None, False):
        pollutant_log_transform = {**pollutant_log_transform, pollutant: str(legacy_transform)}
    if pollutant not in pollutant_log_transform:
        raise ConfigError(
            f"Missing error_modelling.pollutant_log_transform entry for {pollutant!r}. "
            "Set it in configs/default.yaml or override it in configs/local.yaml."
        )

    return ErrorModellingSettings(
        pollutant=pollutant,
        train_val_years=_as_int_tuple(require_config(section, "train_val_years")),
        test_years=_as_int_tuple(require_config(section, "test_years")),
        lead_hours=_as_int_tuple(require_config(section, "lead_hours")),
        data_source=str(require_config(section, "data_source")),
        pollutant_log_transform=pollutant_log_transform,
        cams_training_temporal_resolution=int(require_config(training_input, "temporal_resolution")),
        era5_grid=str(require_config(era5, "grid")),
        era5_temporal_resolution=int(require_config(era5, "temporal_resolution")),
        era5_variables=tuple(str(value) for value in require_config(era5, "variables")),
        era5_log_transform={
            str(key): str(value)
            for key, value in dict(require_config(era5, "log_transform")).items()
            if value not in (None, False)
        },
        era5_skip_zscore=frozenset(str(value) for value in require_config(era5, "skip_zscore")),
    )


def clear_error_modelling_settings_cache() -> None:
    get_error_modelling_settings.cache_clear()


# Convenience accessors used across the package.
def default_pollutant(config_path: str | Path | None = None) -> str:
    return get_error_modelling_settings(str(config_path) if config_path else None).pollutant


def default_lead_hours(config_path: str | Path | None = None) -> tuple[int, ...]:
    return get_error_modelling_settings(str(config_path) if config_path else None).lead_hours


def default_train_val_years(config_path: str | Path | None = None) -> tuple[int, ...]:
    return get_error_modelling_settings(str(config_path) if config_path else None).train_val_years


def default_test_years(config_path: str | Path | None = None) -> tuple[int, ...]:
    return get_error_modelling_settings(str(config_path) if config_path else None).test_years


def default_era5_variables(config_path: str | Path | None = None) -> tuple[str, ...]:
    return get_error_modelling_settings(str(config_path) if config_path else None).era5_variables


def default_era5_log_transform(config_path: str | Path | None = None) -> dict[str, str]:
    return dict(
        get_error_modelling_settings(str(config_path) if config_path else None).era5_log_transform
    )


def default_era5_skip_zscore(config_path: str | Path | None = None) -> frozenset[str]:
    return get_error_modelling_settings(str(config_path) if config_path else None).era5_skip_zscore


def default_era5_grid(config_path: str | Path | None = None) -> str:
    return get_error_modelling_settings(str(config_path) if config_path else None).era5_grid


def default_data_source(config_path: str | Path | None = None) -> str:
    return get_error_modelling_settings(str(config_path) if config_path else None).data_source


def default_pollutant_log_transform(
    pollutant: str | None = None,
    config_path: str | Path | None = None,
) -> str | None:
    settings = get_error_modelling_settings(str(config_path) if config_path else None)
    return settings.pollutant_log_transform_for(pollutant or settings.pollutant)
