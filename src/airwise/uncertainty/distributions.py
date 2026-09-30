"""Equal-weight Gaussian-mixture utilities in log1p concentration space."""

from __future__ import annotations

import numpy as np
from scipy import special, stats


def gaussian_aqi_bin_probability(
    mu: np.ndarray,
    sigma: np.ndarray | float,
    bin_low: float,
    bin_high: float,
    *,
    transform: str | None,
) -> np.ndarray:
    """Probability assigned to one physical AQI bin by a Gaussian forecast."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    if transform in {None, "none", "null", "false", "", "physical"}:
        scale = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
        lower = (
            0.0
            if bin_low == 0.0
            else stats.norm.cdf((bin_low - mu_arr) / scale)
        )
        upper = (
            1.0
            if not np.isfinite(bin_high)
            else stats.norm.cdf((bin_high - mu_arr) / scale)
        )
        return np.clip(upper - lower, 0.0, 1.0)
    return mixture_aqi_bin_probability(mu_arr[None, ...], sigma, bin_low, bin_high)


def mixture_log_cdf(
    log_value: np.ndarray,
    component_means: np.ndarray,
    sigma: np.ndarray | float,
) -> np.ndarray:
    """CDF of an equal-weight mixture; component axis is leading."""
    means = np.asarray(component_means, dtype=np.float64)
    value = np.asarray(log_value, dtype=np.float64)
    scale = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    return np.mean(stats.norm.cdf((value[None, ...] - means) / scale), axis=0)


def mixture_physical_cdf(
    concentration: np.ndarray | float,
    component_means: np.ndarray,
    sigma: np.ndarray | float,
) -> np.ndarray:
    value = np.maximum(np.asarray(concentration, dtype=np.float64), 0.0)
    return mixture_log_cdf(np.log1p(value), component_means, sigma)


def mixture_physical_quantile(
    probability: float,
    component_means: np.ndarray,
    sigma: np.ndarray | float,
    *,
    iterations: int = 32,
) -> np.ndarray:
    """Invert the mixture CDF with vectorised bisection."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1.")
    means = np.asarray(component_means, dtype=np.float64)
    scale = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    low = np.min(means, axis=0) - 9.0 * scale
    high = np.max(means, axis=0) + 9.0 * scale
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        cdf = mixture_log_cdf(middle, means, scale)
        low = np.where(cdf < probability, middle, low)
        high = np.where(cdf >= probability, middle, high)
    return np.maximum(np.expm1(0.5 * (low + high)), 0.0)


def mixture_aqi_bin_probability(
    component_means: np.ndarray,
    sigma: np.ndarray | float,
    bin_low: float,
    bin_high: float,
) -> np.ndarray:
    """Probability assigned to one physical AQI concentration bin."""
    if bin_low < 0 or not np.isfinite(bin_low):
        raise ValueError("bin_low must be finite and non-negative.")
    lower = (
        0.0
        if bin_low == 0.0
        else mixture_physical_cdf(bin_low, component_means, sigma)
    )
    upper = (
        1.0
        if not np.isfinite(bin_high)
        else mixture_physical_cdf(bin_high, component_means, sigma)
    )
    return np.clip(np.asarray(upper) - np.asarray(lower), 0.0, 1.0)


def _normal_absolute_moment(delta: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    scale_arr = np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)
    z = np.asarray(delta, dtype=np.float64) / scale_arr
    return (
        2.0 * scale_arr / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * z * z)
        + np.asarray(delta, dtype=np.float64) * special.erf(z / np.sqrt(2.0))
    )


def gaussian_mixture_crps(
    component_means: np.ndarray,
    sigma: np.ndarray | float,
    observation_log: np.ndarray,
) -> np.ndarray:
    """Closed-form CRPS for an equal-weight, shared-sigma Gaussian mixture.

    Intended for sampled/reservoir points because the pair term is O(K²).
    """
    means = np.asarray(component_means, dtype=np.float64)
    obs = np.asarray(observation_log, dtype=np.float64)
    scale = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    term_observation = np.mean(
        _normal_absolute_moment(obs[None, ...] - means, scale), axis=0
    )
    pair_sum = np.zeros_like(obs, dtype=np.float64)
    for first in means:
        for second in means:
            pair_sum += _normal_absolute_moment(
                first - second, np.sqrt(2.0) * scale
            )
    return term_observation - 0.5 * pair_sum / (means.shape[0] ** 2)
