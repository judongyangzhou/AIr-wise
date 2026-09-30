"""Compute AQI confidence from predictive mean and standard deviation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from airwise.domain.aqi import AqiConcentrationBin
from airwise.uncertainty.distributions import gaussian_aqi_bin_probability
from airwise.uncertainty.interpolation import (
    HOURLY_HOURS,
    interpolate_mu_sigma,
    is_native_lead_hour,
    resolve_lead_hours,
)


def confidence_percent_for_aqi_field(
    mu: np.ndarray,
    sigma: np.ndarray,
    aqi: np.ndarray,
    aqi_bins: Sequence[AqiConcentrationBin],
    *,
    transform: str | None,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a 0–100 confidence field for the assigned CAMS AQI classes."""
    mu_arr = np.asarray(mu)
    sigma_arr = np.asarray(sigma)
    aqi_arr = np.asarray(aqi)
    if mu_arr.shape != sigma_arr.shape or mu_arr.shape != aqi_arr.shape:
        raise ValueError(
            "mu, sigma, and aqi must share a shape; "
            f"got {mu_arr.shape}, {sigma_arr.shape}, {aqi_arr.shape}"
        )
    confidence = np.full(mu_arr.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(mu_arr) & np.isfinite(sigma_arr) & np.isfinite(aqi_arr)
    if land_mask is not None:
        valid &= np.asarray(land_mask, dtype=bool)
    bins_by_level = {item.level: item for item in aqi_bins}
    for level in np.unique(aqi_arr[valid]):
        definition = bins_by_level.get(int(level))
        if definition is None:
            continue
        mask = valid & (aqi_arr == level)
        confidence[mask] = (
            100.0
            * gaussian_aqi_bin_probability(
                mu_arr[mask],
                sigma_arr[mask],
                definition.low,
                definition.high,
                transform=transform,
            )
        ).astype(np.float32)
    return confidence


def hourly_confidence_from_leads(
    mu_lead: np.ndarray,
    sigma_lead: np.ndarray,
    hourly_aqi: np.ndarray,
    aqi_bins: Sequence[AqiConcentrationBin],
    *,
    transform: str | None,
    land_mask: np.ndarray | None = None,
    lead_hours: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate predictive parameters and calculate hourly confidence."""
    resolved_leads = resolve_lead_hours(lead_hours)
    hourly_aqi = np.asarray(hourly_aqi)
    if hourly_aqi.shape[0] != 24:
        raise ValueError(
            f"hourly_aqi must have 24 time steps, got {hourly_aqi.shape}"
        )
    mu_hour, sigma_hour = interpolate_mu_sigma(
        mu_lead,
        sigma_lead,
        hours=HOURLY_HOURS,
        lead_hours=resolved_leads,
    )
    confidence = np.empty(hourly_aqi.shape, dtype=np.float32)
    for hour in HOURLY_HOURS:
        confidence[hour] = confidence_percent_for_aqi_field(
            mu_hour[hour],
            sigma_hour[hour],
            hourly_aqi[hour],
            aqi_bins,
            transform=transform,
            land_mask=land_mask,
        )
    native = np.asarray(
        [is_native_lead_hour(hour, resolved_leads) for hour in HOURLY_HOURS],
        dtype=np.int8,
    )
    return confidence, native
