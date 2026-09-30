"""PyTorch dataset and sample-time helpers for error modelling."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from airwise.modelling.config import get_error_modelling_settings
from airwise.modelling.training.data import (
    ErrorModellingData,
    load_error_modelling_data,
    load_error_modelling_data_from_zarr,
    resolve_training_zarr_root,
    training_zarr_available,
)
from airwise.modelling.stats import ErrorModellingStats, get_zscore_stats


logger = logging.getLogger(__name__)

# Time-axis batch size used while materialising the in-memory preload cache.
# 240 timestamps ~= 30 complete daily samples at 3-hourly resolution.
DEFAULT_PRELOAD_TIME_BATCH = 240


def make_daily_sample_times(
    times: Sequence[np.datetime64],
    lead_hours: Sequence[int] | None = None,
) -> list[pd.DatetimeIndex]:
    """
    Build one sample per day containing all requested 3-hour lead timestamps.
    """
    lead_hours = tuple(lead_hours or get_error_modelling_settings().lead_hours)
    time_index = pd.DatetimeIndex(pd.to_datetime(times)).sort_values().unique()
    available = set(time_index)
    sample_times: list[pd.DatetimeIndex] = []

    for day in pd.DatetimeIndex(time_index.normalize().unique()):
        day_times = pd.DatetimeIndex([day + pd.Timedelta(hours=hour) for hour in lead_hours])
        if all(time in available for time in day_times):
            sample_times.append(day_times)

    if not sample_times:
        raise ValueError("No complete daily 0,3,...,21 UTC samples were found.")

    logger.info(
        "Built %d complete daily samples from %s to %s",
        len(sample_times),
        sample_times[0][0],
        sample_times[-1][-1],
    )
    return sample_times


def split_sample_times(
    sample_times: Sequence[pd.DatetimeIndex],
    val_fraction: float = 0.2,
) -> tuple[list[pd.DatetimeIndex], list[pd.DatetimeIndex]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    ordered = list(sample_times)
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    if len(ordered) <= n_val:
        raise ValueError("Not enough samples to create both train and validation splits.")

    logger.info(
        "Split %d samples into train=%d and val=%d",
        len(ordered),
        len(ordered) - n_val,
        n_val,
    )
    return ordered[:-n_val], ordered[-n_val:]


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for neural network training. Install `torch`.") from exc
    return torch


class ErrorModellingDataset:
    """
    PyTorch dataset returning dual-resolution inputs and pollutant error targets.

    Each item contains:
    - ``cams_forecast``: ``[1, 8, 420, 700]``
    - ``era5``: ``[C, 8, H, W]`` (``H,W`` is ``420,700`` for ``grid="cams"``)
    - ``target_error``: ``[1, 8, 420, 700]``
    """

    def __init__(
        self,
        data: ErrorModellingData,
        stats: ErrorModellingStats,
        sample_times: Sequence[pd.DatetimeIndex] | None = None,
        era5_variables: Sequence[str] | None = None,
        fill_nan: bool = True,
        preload_to_memory: bool = False,
    ) -> None:
        self.data = data
        self.stats = stats
        self.sample_times = list(
            sample_times
            or make_daily_sample_times(data.cams_forecast["time"].values)
        )
        self.era5_variables = list(era5_variables or data.era5.data_vars)
        self.fill_nan = fill_nan
        self._preloaded_arrays: dict[str, np.ndarray] | None = None
        self._sample_indices: list[np.ndarray] | None = None
        skip_zscore = get_error_modelling_settings().era5_skip_zscore

        missing_stats = [
            var_name
            for var_name in self.era5_variables
            if var_name not in skip_zscore and var_name not in stats.era5
        ]
        if missing_stats:
            raise KeyError(f"Missing ERA5 z-score stats for variables: {missing_stats}")

        if preload_to_memory:
            self._preload_to_memory()

    def __len__(self) -> int:
        return len(self.sample_times)

    def close(self) -> None:
        self.data.close()

    def __getitem__(self, index: int) -> dict:
        torch = _require_torch()

        times = self.sample_times[index]
        if self._preloaded_arrays is not None and self._sample_indices is not None:
            sample_index = self._sample_indices[index]
            cams = self._preloaded_arrays["cams_forecast"][sample_index]
            target = self._preloaded_arrays["target_error"][sample_index]
            era5 = self._preloaded_arrays["era5"][:, sample_index, :, :]
        else:
            cams, target, era5 = self._load_sample_arrays(times)

        return {
            "cams_forecast": torch.from_numpy(cams[None, ...]),
            "era5": torch.from_numpy(era5),
            "target_error": torch.from_numpy(target[None, ...]),
            "time": [timestamp.isoformat() for timestamp in times],
        }

    def _load_sample_arrays(self, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pollutant = self.data.pollutant
        cams = self.data.cams_forecast[pollutant].sel(time=times).values.astype("float32")
        target = (
            self.data.cams_analysis[pollutant].sel(time=times)
            - self.data.cams_forecast[pollutant].sel(time=times)
        ).values.astype("float32")
        era5 = (
            self.data.era5[self.era5_variables]
            .to_array("variable")
            .sel(time=times)
            .transpose("variable", "time", "latitude", "longitude")
            .values.astype("float32")
        )

        cams = self.stats.cams_forecast.normalise(cams)
        target = self.stats.target_error.normalise(target)
        skip_zscore = get_error_modelling_settings().era5_skip_zscore
        for var_index, var_name in enumerate(self.era5_variables):
            if var_name in skip_zscore:
                continue
            era5[var_index] = self.stats.era5[var_name].normalise(era5[var_index])

        if self.fill_nan:
            cams = np.nan_to_num(cams, nan=0.0, posinf=0.0, neginf=0.0)
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)

        cams = cams.astype("float32", copy=False)
        target = target.astype("float32", copy=False)
        era5 = era5.astype("float32", copy=False)

        return cams, target, era5

    @staticmethod
    def _unique_times_from_samples(sample_times: Sequence[pd.DatetimeIndex]) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            sorted({timestamp for sample in sample_times for timestamp in sample})
        )

    @staticmethod
    def _sample_indices_for_times(
        sample_times: Sequence[pd.DatetimeIndex],
        time_to_index: Mapping[pd.Timestamp, int],
    ) -> list[np.ndarray]:
        return [
            np.asarray([time_to_index[timestamp] for timestamp in sample], dtype=np.int64)
            for sample in sample_times
        ]

    def _materialise_preloaded_arrays(
        self,
        unique_times: pd.DatetimeIndex,
        time_batch_size: int = DEFAULT_PRELOAD_TIME_BATCH,
    ) -> dict[str, np.ndarray]:
        pollutant = self.data.pollutant
        n_times = len(unique_times)
        if n_times == 0:
            raise ValueError("Cannot preload an empty timestamp set.")

        batch_size = max(1, int(time_batch_size))
        n_batches = (n_times + batch_size - 1) // batch_size
        n_lat = int(self.data.cams_forecast.sizes["latitude"])
        n_lon = int(self.data.cams_forecast.sizes["longitude"])
        n_era5 = len(self.era5_variables)
        skip_zscore = get_error_modelling_settings().era5_skip_zscore

        estimated_gib = (
            (2 * n_times * n_lat * n_lon)  # cams + target
            + (n_era5 * n_times * n_lat * n_lon)
        ) * np.dtype("float32").itemsize / 1024**3
        logger.info(
            "Materialising preload cache: %d timestamps in %d batches "
            "(batch_size=%d, grid=%dx%d, era5_vars=%d, ~%.2f GiB)",
            n_times,
            n_batches,
            batch_size,
            n_lat,
            n_lon,
            n_era5,
            estimated_gib,
        )

        cams_out = np.empty((n_times, n_lat, n_lon), dtype="float32")
        target_out = np.empty((n_times, n_lat, n_lon), dtype="float32")
        era5_out = np.empty((n_era5, n_times, n_lat, n_lon), dtype="float32")

        overall_start = time.perf_counter()
        for batch_index, start in enumerate(range(0, n_times, batch_size), start=1):
            stop = min(start + batch_size, n_times)
            batch_times = unique_times[start:stop]
            batch_start = time.perf_counter()
            logger.info(
                "Preload batch %d/%d: loading timestamps %d-%d / %d (%s -> %s)",
                batch_index,
                n_batches,
                start + 1,
                stop,
                n_times,
                batch_times[0],
                batch_times[-1],
            )

            stage = time.perf_counter()
            cams = self.data.cams_forecast[pollutant].sel(time=batch_times).values.astype("float32")
            logger.info(
                "Preload batch %d/%d: cams_forecast loaded in %.1fs shape=%s",
                batch_index,
                n_batches,
                time.perf_counter() - stage,
                tuple(cams.shape),
            )

            stage = time.perf_counter()
            analysis = (
                self.data.cams_analysis[pollutant].sel(time=batch_times).values.astype("float32")
            )
            target = analysis - cams
            logger.info(
                "Preload batch %d/%d: cams_analysis loaded + target built in %.1fs shape=%s",
                batch_index,
                n_batches,
                time.perf_counter() - stage,
                tuple(target.shape),
            )

            stage = time.perf_counter()
            era5 = (
                self.data.era5[self.era5_variables]
                .to_array("variable")
                .sel(time=batch_times)
                .transpose("variable", "time", "latitude", "longitude")
                .values.astype("float32")
            )
            logger.info(
                "Preload batch %d/%d: era5 loaded in %.1fs shape=%s",
                batch_index,
                n_batches,
                time.perf_counter() - stage,
                tuple(era5.shape),
            )

            stage = time.perf_counter()
            cams = self.stats.cams_forecast.normalise(cams)
            target = self.stats.target_error.normalise(target)
            for var_index, var_name in enumerate(self.era5_variables):
                if var_name in skip_zscore:
                    continue
                era5[var_index] = self.stats.era5[var_name].normalise(era5[var_index])

            if self.fill_nan:
                cams = np.nan_to_num(cams, nan=0.0, posinf=0.0, neginf=0.0)
                target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
                era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)

            cams_out[start:stop] = cams
            target_out[start:stop] = target
            era5_out[:, start:stop, :, :] = era5
            logger.info(
                "Preload batch %d/%d: normalised+copied in %.1fs; batch total %.1fs; "
                "elapsed %.1fs (%.1f%%)",
                batch_index,
                n_batches,
                time.perf_counter() - stage,
                time.perf_counter() - batch_start,
                time.perf_counter() - overall_start,
                100.0 * stop / n_times,
            )

        total_gib = (cams_out.nbytes + target_out.nbytes + era5_out.nbytes) / 1024**3
        logger.info(
            "Finished materialising preload cache in %.1fs (%.2f GiB resident)",
            time.perf_counter() - overall_start,
            total_gib,
        )
        return {
            "cams_forecast": cams_out,
            "target_error": target_out,
            "era5": era5_out,
        }

    def _attach_preloaded_arrays(
        self,
        arrays: dict[str, np.ndarray],
        time_to_index: Mapping[pd.Timestamp, int],
    ) -> None:
        self._preloaded_arrays = arrays
        self._sample_indices = self._sample_indices_for_times(self.sample_times, time_to_index)

    def _preload_to_memory(self) -> None:
        unique_times = self._unique_times_from_samples(self.sample_times)
        logger.info(
            "Preloading %d samples (%d unique timestamps) to memory",
            len(self.sample_times),
            len(unique_times),
        )
        arrays = self._materialise_preloaded_arrays(unique_times)
        time_to_index = {timestamp: index for index, timestamp in enumerate(unique_times)}
        self._attach_preloaded_arrays(arrays, time_to_index)
        total_gib = sum(array.nbytes for array in arrays.values()) / 1024**3
        logger.info("Finished preloading dataset to memory: %.2f GiB", total_gib)

    @classmethod
    def share_preloaded_memory(cls, datasets: Sequence[ErrorModellingDataset]) -> None:
        """
        Materialise one shared in-memory cache for multiple dataset splits.

        Train/val typically share the same underlying xarray handles; loading once
        for the union of their timestamps avoids a second full disk scan.
        """
        dataset_list = list(datasets)
        if not dataset_list:
            return

        reference = dataset_list[0]
        for dataset in dataset_list[1:]:
            if dataset.data is not reference.data:
                raise ValueError("share_preloaded_memory requires datasets to share the same ErrorModellingData.")
            if dataset.stats is not reference.stats:
                raise ValueError("share_preloaded_memory requires datasets to share the same stats.")
            if dataset.era5_variables != reference.era5_variables:
                raise ValueError("share_preloaded_memory requires identical ERA5 variable lists.")
            if dataset.fill_nan != reference.fill_nan:
                raise ValueError("share_preloaded_memory requires identical fill_nan settings.")

        all_sample_times = [sample for dataset in dataset_list for sample in dataset.sample_times]
        unique_times = cls._unique_times_from_samples(all_sample_times)
        logger.info(
            "Preloading %d shared samples across %d datasets (%d unique timestamps) to memory",
            len(all_sample_times),
            len(dataset_list),
            len(unique_times),
        )
        arrays = reference._materialise_preloaded_arrays(unique_times)
        time_to_index = {timestamp: index for index, timestamp in enumerate(unique_times)}
        for dataset in dataset_list:
            dataset._attach_preloaded_arrays(arrays, time_to_index)

        total_gib = sum(array.nbytes for array in arrays.values()) / 1024**3
        logger.info("Finished shared preloading to memory: %.2f GiB", total_gib)


def _load_split_data(
    *,
    years: Sequence[int],
    data_source: str,
    pollutant: str,
    chunks: dict | None,
    era5_grid: str,
    zarr_root: str | Path | None,
    pollutant_log_transform: str | None | bool | Mapping[str, str | None] | object,
) -> ErrorModellingData:
    if data_source == "zarr":
        return load_error_modelling_data_from_zarr(
            pollutant=pollutant,
            years=years,
            zarr_root=zarr_root,
            chunks=chunks,
            pollutant_log_transform=pollutant_log_transform,
        )
    if data_source == "netcdf":
        return load_error_modelling_data(
            years=years,
            pollutant=pollutant,
            chunks=chunks,
            era5_grid=era5_grid,
            pollutant_log_transform=pollutant_log_transform,
        )
    raise ValueError("data_source must be one of: 'zarr', 'netcdf', 'auto'.")


def build_default_datasets(
    train_val_years: Sequence[int] | None = None,
    test_years: Sequence[int] | None = None,
    val_fraction: float = 0.2,
    chunks: dict | None = None,
    build_test: bool = False,
    stats_path: str | Path | None = None,
    stats_mode: str = "auto",
    era5_stats_path: str | Path | None = None,
    era5_stats_mode: str = "auto",
    preload_to_memory: bool = False,
    era5_grid: str | None = None,
    data_source: str | None = None,
    zarr_root: str | Path | None = None,
    pollutant: str | None = None,
    pollutant_log_transform: str | None | bool | Mapping[str, str | None] | object = ...,
) -> tuple[ErrorModellingDataset, ErrorModellingDataset, ErrorModellingDataset | None, ErrorModellingStats]:
    """
    Build train/val/(optional test) datasets.

    Defaults come from ``configs/default.yaml`` ``error_modelling`` section.

    ``data_source``:
    - ``auto``: use the training Zarr cache when ``meta.json`` exists, else NetCDF
    - ``zarr``: require the training Zarr cache
    - ``netcdf``: load processed yearly NetCDF files
    """
    settings = get_error_modelling_settings()
    train_val_years = tuple(train_val_years or settings.train_val_years)
    test_years = tuple(test_years or settings.test_years)
    era5_grid = era5_grid or settings.era5_grid
    data_source = data_source or settings.data_source
    pollutant = pollutant or settings.pollutant

    if data_source not in {"auto", "zarr", "netcdf"}:
        raise ValueError("data_source must be one of: 'auto', 'zarr', 'netcdf'.")

    resolved_source = data_source
    if data_source == "auto":
        resolved_source = "zarr" if training_zarr_available(zarr_root=zarr_root) else "netcdf"
        logger.info(
            "Resolved data_source=auto to %s (zarr_root=%s)",
            resolved_source,
            resolve_training_zarr_root(zarr_root=zarr_root),
        )
    elif data_source == "zarr" and not training_zarr_available(zarr_root=zarr_root):
        raise FileNotFoundError(
            f"Training Zarr cache not found under {resolve_training_zarr_root(zarr_root=zarr_root)}. "
            "Build it with airwise-build-training-zarr or pass --data-source netcdf."
        )

    logger.info(
        "Building train/val datasets for years=%s source=%s pollutant=%s era5_grid=%s",
        list(train_val_years),
        resolved_source,
        pollutant,
        era5_grid,
    )
    train_val_data = _load_split_data(
        years=train_val_years,
        data_source=resolved_source,
        pollutant=pollutant,
        chunks=chunks,
        era5_grid=era5_grid,
        zarr_root=zarr_root,
        pollutant_log_transform=pollutant_log_transform,
    )
    stats = get_zscore_stats(
        train_val_data,
        stats_path=stats_path,
        stats_mode=stats_mode,
        era5_stats_path=era5_stats_path,
        era5_stats_mode=era5_stats_mode,
    )

    sample_times = make_daily_sample_times(train_val_data.cams_forecast["time"].values)
    train_times, val_times = split_sample_times(sample_times, val_fraction=val_fraction)

    test_dataset = None
    if build_test:
        logger.info("Building test dataset for years=%s", list(test_years))
        test_data = _load_split_data(
            years=test_years,
            data_source=resolved_source,
            pollutant=pollutant,
            chunks=chunks,
            era5_grid=era5_grid,
            zarr_root=zarr_root,
            pollutant_log_transform=pollutant_log_transform,
        )
        # Test uses a separate data handle, so it cannot share the train/val cache.
        test_dataset = ErrorModellingDataset(test_data, stats, preload_to_memory=preload_to_memory)
    else:
        logger.info("Skipping test dataset build")

    train_dataset = ErrorModellingDataset(train_val_data, stats, sample_times=train_times)
    val_dataset = ErrorModellingDataset(train_val_data, stats, sample_times=val_times)
    if preload_to_memory:
        ErrorModellingDataset.share_preloaded_memory([train_dataset, val_dataset])

    datasets = (
        train_dataset,
        val_dataset,
        test_dataset,
        stats,
    )
    test_size = len(test_dataset) if test_dataset is not None else 0
    logger.info("Dataset sizes: train=%d val=%d test=%d", len(datasets[0]), len(datasets[1]), test_size)
    return datasets
