"""Loss functions and AQI-aware weighting for neural error models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from airwise.domain.aqi import load_aqi_concentration_bins
from airwise.modelling.models import unpack_model_output


LOG_VARIANCE_MIN = -10.0
LOG_VARIANCE_MAX = 6.0
# Per-level weights for CAMS forecast AQI 1..6. High-AQI land points are rare
# (~1% at levels 4-5) and otherwise vanish in an unweighted NLL mean.
DEFAULT_CAMS_AQI_NLL_WEIGHTS = (1.0, 1.0, 3.0, 15.0, 25.0, 10.0)
# PM10 mean-backbone recipe stored in dual_encoder_biconvgru_unet/epoch_0016.pt.
DEFAULT_CAMS_AQI_HUBER_WEIGHTS = (1.0, 1.0, 2.0, 5.0, 10.0, 10.0)
DEFAULT_HUBER_BETA = 1.0
WEIGHTED_LOSS_NAMES = frozenset({"gaussian_nll", "aqi-weighted-huber"})
AQI_WEIGHT_SOURCES = ("forecast", "analysis", "max")
PHYSICAL_TAIL_KINDS = ("mae", "huber")
DEFAULT_PHYSICAL_TAIL_LAMBDA = 0.05
DEFAULT_PHYSICAL_TAIL_MIN_AQI = 5
DEFAULT_PHYSICAL_TAIL_HUBER_BETA = 1.0


def gaussian_nll_loss(mean, log_variance, target, weight=None):
    """Gaussian NLL without the constant 0.5*log(2π).

    When ``weight`` is omitted this is a plain mean over all pixels. When
    provided, the reduction is ``sum(w * nll) / sum(w)`` so the loss stays on
    the same scale as the unweighted NLL (importance sampling, not a global
    learning-rate rescaling).
    """
    torch = _require_torch()
    bounded = torch.clamp(log_variance, LOG_VARIANCE_MIN, LOG_VARIANCE_MAX)
    nll = 0.5 * (
        torch.exp(-bounded) * torch.square(target - mean) + bounded
    )
    if weight is None:
        return nll.mean()
    weight = weight.expand_as(nll).to(dtype=nll.dtype)
    return (nll * weight).sum() / weight.sum().clamp_min(1e-8)


def aqi_weighted_huber_loss(mean, target, weight=None, beta: float = DEFAULT_HUBER_BETA):
    """Smooth-L1 / Huber loss; optional CAMS-AQI importance weights.

    ``beta`` is the SmoothL1 transition point. At the historical value
    ``beta=1.0`` this matches PyTorch Huber with ``delta=1``.
    Weighted reduction is ``sum(w * huber) / sum(w)``, same as NLL.
    """
    if beta <= 0:
        raise ValueError("huber beta must be > 0.")
    torch = _require_torch()
    element = torch.nn.functional.smooth_l1_loss(
        mean, target, reduction="none", beta=beta
    )
    if weight is None:
        return element.mean()
    weight = weight.expand_as(element).to(dtype=element.dtype)
    return (element * weight).sum() / weight.sum().clamp_min(1e-8)


def prediction_loss(output, target, loss_name: str, weight=None, huber_beta: float = DEFAULT_HUBER_BETA):
    mean, log_variance = unpack_model_output(output)
    if loss_name == "mse":
        return _require_torch().nn.functional.mse_loss(mean, target), mean, None
    if loss_name in {"huber", "aqi-weighted-huber"}:
        return (
            aqi_weighted_huber_loss(mean, target, weight=weight, beta=huber_beta),
            mean,
            None,
        )
    if loss_name == "gaussian_nll":
        if log_variance is None:
            raise ValueError("gaussian_nll requires a learned-std model output.")
        return (
            gaussian_nll_loss(mean, log_variance, target, weight=weight),
            mean,
            log_variance,
        )
    raise ValueError(f"Unsupported loss_name: {loss_name!r}.")


@dataclass(frozen=True)
class CamsAqiNllWeightConfig:
    """Map denormalised CAMS concentration to per-pixel AQI weights.

    ``source`` chooses which physical field is bucketed into European AQI:
    - ``forecast``: CAMS forecast (legacy)
    - ``analysis``: CAMS analysis reconstructed from forecast + residual
    - ``max``: per-pixel maximum of forecast and analysis
    """

    cams_mean: float
    cams_std: float
    invert_log1p: bool
    aqi_thresholds: tuple[float, ...]
    level_weights: tuple[float, ...]
    land_mask: np.ndarray | None = None
    source: str = "forecast"
    target_mean: float | None = None
    target_std: float | None = None

    def _physical_concentration(self, normalised, mean: float, std: float):
        torch = _require_torch()
        space = normalised * std + mean
        if self.invert_log1p:
            return torch.expm1(space).clamp_min(0)
        return space.clamp_min(0)

    def _require_target_stats(self) -> None:
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Reconstructing analysis concentration requires target_error z-score stats.")

    def physical_forecast_concentration(self, cams):
        return self._physical_concentration(cams, self.cams_mean, self.cams_std)

    def physical_analysis_concentration(self, cams, residual_norm):
        """Invert forecast + z-scored residual to physical analysis concentration.

        Gradients flow through ``residual_norm`` (and ``cams`` if it requires grad).
        """
        torch = _require_torch()
        self._require_target_stats()
        residual_space = residual_norm * self.target_std + self.target_mean
        forecast_space = cams * self.cams_std + self.cams_mean
        analysis_space = forecast_space + residual_space
        if self.invert_log1p:
            return torch.expm1(analysis_space).clamp_min(0)
        return analysis_space.clamp_min(0)

    def concentration_threshold_for_aqi_level(self, min_aqi_level: int) -> float | None:
        """Lower physical bound for European AQI ``min_aqi_level`` (1-6).

        Level 1 has no lower bound beyond 0; returns ``None``.
        """
        if min_aqi_level < 1:
            raise ValueError(f"min_aqi_level must be >= 1, got {min_aqi_level}.")
        if min_aqi_level == 1:
            return None
        index = min_aqi_level - 2
        if index < 0 or index >= len(self.aqi_thresholds):
            raise ValueError(
                f"min_aqi_level {min_aqi_level} is outside AQI thresholds {self.aqi_thresholds}."
            )
        return float(self.aqi_thresholds[index])

    def _apply_land_mask(self, weight):
        if self.land_mask is None:
            return weight
        torch = _require_torch()
        land = torch.as_tensor(
            self.land_mask,
            device=weight.device,
            dtype=weight.dtype,
        )
        while land.ndim < weight.ndim:
            land = land.unsqueeze(0)
        return weight * land

    def analysis_aqi_ge_mask(self, cams, target, min_aqi_level: int):
        """Detached 0/1 mask where reconstructed analysis AQI >= ``min_aqi_level``."""
        torch = _require_torch()
        with torch.no_grad():
            analysis = self.physical_analysis_concentration(cams, target)
            threshold = self.concentration_threshold_for_aqi_level(min_aqi_level)
            if threshold is None:
                mask = torch.ones_like(analysis)
            else:
                mask = (analysis >= threshold).to(dtype=analysis.dtype)
            return self._apply_land_mask(mask)

    def _levels_for(self, concentration, device, dtype):
        torch = _require_torch()
        thresholds = torch.as_tensor(
            self.aqi_thresholds, device=device, dtype=dtype
        )
        return torch.bucketize(concentration, thresholds, right=False)

    def weights_for(self, cams, target=None):
        """Return a detached weight tensor matching ``cams``."""
        torch = _require_torch()
        source = self.source
        if source not in AQI_WEIGHT_SOURCES:
            raise ValueError(f"Unsupported AQI weight source: {source!r}.")
        with torch.no_grad():
            forecast_conc = self.physical_forecast_concentration(cams)
            if source == "forecast":
                concentration = forecast_conc
            else:
                if target is None:
                    raise ValueError(
                        f"AQI weight source {source!r} requires the residual target tensor."
                    )
                analysis_conc = self.physical_analysis_concentration(cams, target)
                concentration = (
                    analysis_conc
                    if source == "analysis"
                    else torch.maximum(forecast_conc, analysis_conc)
                )
            levels = self._levels_for(
                concentration, concentration.device, concentration.dtype
            )
            weights = torch.as_tensor(
                self.level_weights,
                device=concentration.device,
                dtype=concentration.dtype,
            )
            n_levels = int(weights.numel())
            if n_levels < 1:
                raise ValueError("level_weights must contain at least one AQI level.")
            weight = weights[levels.clamp(0, n_levels - 1)]
            return self._apply_land_mask(weight)


def build_cams_aqi_nll_weight_config(
    dataset,
    *,
    level_weights: Sequence[float] | None = None,
    aqi_config: str | Path | None = None,
    land_mask: np.ndarray | None = None,
    source: str = "forecast",
) -> CamsAqiNllWeightConfig:
    """Build CAMS AQI NLL/Huber weights from a training dataset."""
    if source not in AQI_WEIGHT_SOURCES:
        raise ValueError(
            f"source must be one of {AQI_WEIGHT_SOURCES}, got {source!r}."
        )
    stats = getattr(dataset, "stats", None)
    cams_stats = getattr(stats, "cams_forecast", None)
    if cams_stats is None:
        raise ValueError("CAMS AQI weighting requires dataset.stats.cams_forecast.")
    target_stats = getattr(stats, "target_error", None)
    if source != "forecast" and target_stats is None:
        raise ValueError(
            f"AQI weight source {source!r} requires dataset.stats.target_error."
        )
    data = getattr(dataset, "data", None)
    pollutant = str(getattr(data, "pollutant", "pm10_conc"))
    invert_log1p = getattr(data, "pollutant_log_transform", "log1p") == "log1p"
    bins = load_aqi_concentration_bins(pollutant, path=aqi_config)
    if len(bins) < 2:
        raise ValueError(f"Need at least two AQI bins for {pollutant!r}.")
    thresholds = tuple(float(item.low) for item in bins[1:])
    weights = tuple(
        float(value)
        for value in (level_weights or DEFAULT_CAMS_AQI_NLL_WEIGHTS)
    )
    if len(weights) != len(bins):
        raise ValueError(
            f"Expected {len(bins)} AQI NLL weights for {pollutant!r}, got {len(weights)}."
        )
    mask = None if land_mask is None else np.asarray(land_mask, dtype=bool)
    if mask is not None and mask.ndim != 2:
        raise ValueError(f"land_mask must be 2D [H, W], got shape {mask.shape}.")
    return CamsAqiNllWeightConfig(
        cams_mean=float(cams_stats.mean),
        cams_std=float(cams_stats.std),
        invert_log1p=invert_log1p,
        aqi_thresholds=thresholds,
        level_weights=weights,
        land_mask=mask,
        source=source,
        target_mean=None if target_stats is None else float(target_stats.mean),
        target_std=None if target_stats is None else float(target_stats.std),
    )


@dataclass(frozen=True)
class PhysicalTailLossConfig:
    """Physical-concentration MAE/Huber on analysis AQI >= ``min_aqi_level``."""

    min_aqi_level: int = DEFAULT_PHYSICAL_TAIL_MIN_AQI
    lambda_weight: float = DEFAULT_PHYSICAL_TAIL_LAMBDA
    kind: str = "mae"
    huber_beta: float = DEFAULT_PHYSICAL_TAIL_HUBER_BETA

    def __post_init__(self) -> None:
        if self.min_aqi_level < 1:
            raise ValueError(f"min_aqi_level must be >= 1, got {self.min_aqi_level}.")
        if self.lambda_weight < 0:
            raise ValueError("physical tail lambda must be >= 0.")
        if self.kind not in PHYSICAL_TAIL_KINDS:
            raise ValueError(
                f"physical tail kind must be one of {PHYSICAL_TAIL_KINDS}, got {self.kind!r}."
            )
        if self.huber_beta <= 0:
            raise ValueError("physical tail Huber beta must be > 0.")


def physical_tail_concentration_loss(
    pred_residual,
    cams,
    target,
    weight_config: CamsAqiNllWeightConfig,
    tail_config: PhysicalTailLossConfig,
):
    """Mean physical MAE/Huber over analysis-AQI tail pixels; 0 if the mask is empty."""
    torch = _require_torch()
    pred_phys = weight_config.physical_analysis_concentration(cams, pred_residual)
    with torch.no_grad():
        true_phys = weight_config.physical_analysis_concentration(cams, target)
        mask = weight_config.analysis_aqi_ge_mask(cams, target, tail_config.min_aqi_level)
    if tail_config.kind == "mae":
        element = torch.abs(pred_phys - true_phys)
    else:
        element = torch.nn.functional.smooth_l1_loss(
            pred_phys, true_phys, reduction="none", beta=tail_config.huber_beta
        )
    denom = mask.sum().clamp_min(1e-8)
    return (element * mask).sum() / denom


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for neural network training. Install `torch`.") from exc
    return torch


