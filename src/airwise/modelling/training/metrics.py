"""Streaming concentration metrics used during model training and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConcentrationMetrics:
    """
    Metrics for evaluating corrected concentration fields against raw forecasts.

    ``skill_score_vs_forecast`` is positive when the neural-network residual
    correction improves over the zero-residual baseline.

    ``forecast_corr`` / ``corrected_corr`` are Pearson correlations (R) against
    analysis. ``forecast_r2`` / ``corrected_r2`` are coefficients of determination.
    """

    n: int
    forecast_mae: float
    corrected_mae: float
    forecast_rmse: float
    corrected_rmse: float
    forecast_bias: float
    corrected_bias: float
    forecast_corr: float
    corrected_corr: float
    forecast_r2: float
    corrected_r2: float
    skill_score_vs_forecast: float
    improvement_rate: float
    mean_delta_abs_error: float

    def as_dict(self, prefix: str | None = None) -> dict[str, float]:
        payload = {
            "n": float(self.n),
            "forecast_mae": self.forecast_mae,
            "corrected_mae": self.corrected_mae,
            "forecast_rmse": self.forecast_rmse,
            "corrected_rmse": self.corrected_rmse,
            "forecast_bias": self.forecast_bias,
            "corrected_bias": self.corrected_bias,
            "forecast_corr": self.forecast_corr,
            "corrected_corr": self.corrected_corr,
            "forecast_r2": self.forecast_r2,
            "corrected_r2": self.corrected_r2,
            "skill_score_vs_forecast": self.skill_score_vs_forecast,
            "improvement_rate": self.improvement_rate,
            "mean_delta_abs_error": self.mean_delta_abs_error,
        }
        if prefix is None:
            return payload
        return {f"{prefix}_{name}": value for name, value in payload.items()}


def _as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "values"):
        values = values.values
    return np.asarray(values, dtype=np.float64)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def _pearson_corr(
    n: int,
    sum_pred: float,
    sum_target: float,
    sum_pred_sq: float,
    sum_target_sq: float,
    sum_pred_target: float,
) -> float:
    if n <= 0:
        return float("nan")
    pred_mean = sum_pred / n
    target_mean = sum_target / n
    pred_var = sum_pred_sq / n - pred_mean * pred_mean
    target_var = sum_target_sq / n - target_mean * target_mean
    covariance = sum_pred_target / n - pred_mean * target_mean
    if pred_var <= 0 or target_var <= 0:
        return float("nan")
    return float(covariance / np.sqrt(pred_var * target_var))


def _coefficient_of_determination(n: int, sse: float, sum_target: float, sum_target_sq: float) -> float:
    if n <= 0:
        return float("nan")
    sst = sum_target_sq - sum_target * sum_target / n
    if sst <= 0:
        return float("nan")
    return float(1.0 - sse / sst)


def _empty_concentration_metrics() -> ConcentrationMetrics:
    return ConcentrationMetrics(
        n=0,
        forecast_mae=float("nan"),
        corrected_mae=float("nan"),
        forecast_rmse=float("nan"),
        corrected_rmse=float("nan"),
        forecast_bias=float("nan"),
        corrected_bias=float("nan"),
        forecast_corr=float("nan"),
        corrected_corr=float("nan"),
        forecast_r2=float("nan"),
        corrected_r2=float("nan"),
        skill_score_vs_forecast=float("nan"),
        improvement_rate=float("nan"),
        mean_delta_abs_error=float("nan"),
    )


class ConcentrationMetricsAccumulator:
    """
    Streaming accumulator for concentration metrics on large forecast grids.
    """

    def __init__(self) -> None:
        self.n = 0
        self.forecast_sae = 0.0
        self.corrected_sae = 0.0
        self.forecast_sse = 0.0
        self.corrected_sse = 0.0
        self.forecast_error_sum = 0.0
        self.corrected_error_sum = 0.0
        self.improved_count = 0
        self.delta_abs_error_sum = 0.0
        self.sum_forecast = 0.0
        self.sum_corrected = 0.0
        self.sum_analysis = 0.0
        self.sum_forecast_sq = 0.0
        self.sum_corrected_sq = 0.0
        self.sum_analysis_sq = 0.0
        self.sum_forecast_analysis = 0.0
        self.sum_corrected_analysis = 0.0

    def update(self, forecast, analysis, corrected, mask=None) -> None:
        forecast_array = _as_numpy(forecast)
        analysis_array = _as_numpy(analysis)
        corrected_array = _as_numpy(corrected)
        forecast_array, analysis_array, corrected_array = np.broadcast_arrays(
            forecast_array,
            analysis_array,
            corrected_array,
        )
        valid = (
            np.isfinite(forecast_array)
            & np.isfinite(analysis_array)
            & np.isfinite(corrected_array)
        )
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        if not np.any(valid):
            return

        forecast_valid = forecast_array[valid]
        analysis_valid = analysis_array[valid]
        corrected_valid = corrected_array[valid]
        forecast_error = forecast_valid - analysis_valid
        corrected_error = corrected_valid - analysis_valid
        forecast_abs_error = np.abs(forecast_error)
        corrected_abs_error = np.abs(corrected_error)
        delta_abs_error = corrected_abs_error - forecast_abs_error

        self.n += int(valid.sum())
        self.forecast_sae += float(forecast_abs_error.sum())
        self.corrected_sae += float(corrected_abs_error.sum())
        self.forecast_sse += float((forecast_error * forecast_error).sum())
        self.corrected_sse += float((corrected_error * corrected_error).sum())
        self.forecast_error_sum += float(forecast_error.sum())
        self.corrected_error_sum += float(corrected_error.sum())
        self.improved_count += int(np.sum(delta_abs_error < 0))
        self.delta_abs_error_sum += float(delta_abs_error.sum())
        self.sum_forecast += float(forecast_valid.sum())
        self.sum_corrected += float(corrected_valid.sum())
        self.sum_analysis += float(analysis_valid.sum())
        self.sum_forecast_sq += float((forecast_valid * forecast_valid).sum())
        self.sum_corrected_sq += float((corrected_valid * corrected_valid).sum())
        self.sum_analysis_sq += float((analysis_valid * analysis_valid).sum())
        self.sum_forecast_analysis += float((forecast_valid * analysis_valid).sum())
        self.sum_corrected_analysis += float((corrected_valid * analysis_valid).sum())

    def compute(self) -> ConcentrationMetrics:
        if self.n == 0:
            return _empty_concentration_metrics()

        forecast_mae = self.forecast_sae / self.n
        corrected_mae = self.corrected_sae / self.n
        return ConcentrationMetrics(
            n=self.n,
            forecast_mae=float(forecast_mae),
            corrected_mae=float(corrected_mae),
            forecast_rmse=float(np.sqrt(self.forecast_sse / self.n)),
            corrected_rmse=float(np.sqrt(self.corrected_sse / self.n)),
            forecast_bias=float(self.forecast_error_sum / self.n),
            corrected_bias=float(self.corrected_error_sum / self.n),
            forecast_corr=_pearson_corr(
                self.n,
                self.sum_forecast,
                self.sum_analysis,
                self.sum_forecast_sq,
                self.sum_analysis_sq,
                self.sum_forecast_analysis,
            ),
            corrected_corr=_pearson_corr(
                self.n,
                self.sum_corrected,
                self.sum_analysis,
                self.sum_corrected_sq,
                self.sum_analysis_sq,
                self.sum_corrected_analysis,
            ),
            forecast_r2=_coefficient_of_determination(
                self.n,
                self.forecast_sse,
                self.sum_analysis,
                self.sum_analysis_sq,
            ),
            corrected_r2=_coefficient_of_determination(
                self.n,
                self.corrected_sse,
                self.sum_analysis,
                self.sum_analysis_sq,
            ),
            skill_score_vs_forecast=1.0 - _safe_divide(corrected_mae, forecast_mae),
            improvement_rate=float(self.improved_count / self.n),
            mean_delta_abs_error=float(self.delta_abs_error_sum / self.n),
        )


