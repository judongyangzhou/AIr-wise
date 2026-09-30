"""Interpolate native model uncertainty parameters to hourly lead times."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_LEAD_HOURS = tuple(range(0, 24, 3))
HOURLY_HOURS = tuple(range(24))


def resolve_lead_hours(lead_hours: Sequence[int] | None = None) -> tuple[int, ...]:
    return tuple(int(hour) for hour in (lead_hours or DEFAULT_LEAD_HOURS))


def is_native_lead_hour(
    hour: int,
    lead_hours: Sequence[int] | None = None,
) -> bool:
    return int(hour) in set(resolve_lead_hours(lead_hours))


def interpolate_mu_sigma(
    mu_lead: np.ndarray,
    sigma_lead: np.ndarray,
    hours: Sequence[int] | np.ndarray | None = None,
    lead_hours: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate mu and log-sigma, clamping after the final lead."""
    resolved_leads = resolve_lead_hours(lead_hours)
    mu_lead = np.asarray(mu_lead)
    sigma_lead = np.asarray(sigma_lead)
    if mu_lead.shape != sigma_lead.shape:
        raise ValueError(
            f"mu/sigma lead shapes differ: {mu_lead.shape} vs {sigma_lead.shape}"
        )
    leads = np.asarray(resolved_leads, dtype=int)
    if mu_lead.shape[0] != leads.size:
        raise ValueError(
            f"Expected {leads.size} lead steps, got mu shape {mu_lead.shape}"
        )

    hour_values = np.arange(24) if hours is None else np.asarray(hours, dtype=int)
    spatial = mu_lead.shape[1:]
    out_mu = np.empty((hour_values.size, *spatial), dtype=np.float32)
    out_sigma = np.empty_like(out_mu)
    log_sigma = np.log(np.maximum(sigma_lead.astype(np.float64), 1e-8))
    last_lead = int(leads[-1])
    lead_index = {int(hour): index for index, hour in enumerate(leads)}

    for index, hour in enumerate(hour_values.tolist()):
        if hour >= last_lead:
            out_mu[index] = mu_lead[-1]
            out_sigma[index] = sigma_lead[-1]
            continue
        lower = max(int(leads[leads <= hour].max()), int(leads[0]))
        upper_candidates = leads[leads > hour]
        upper = int(upper_candidates[0]) if upper_candidates.size else last_lead
        lower_index = lead_index[lower]
        upper_index = lead_index[upper]
        alpha = 0.0 if upper == lower else (hour - lower) / (upper - lower)
        out_mu[index] = (
            (1.0 - alpha) * mu_lead[lower_index] + alpha * mu_lead[upper_index]
        ).astype(np.float32)
        out_sigma[index] = np.exp(
            (1.0 - alpha) * log_sigma[lower_index]
            + alpha * log_sigma[upper_index]
        ).astype(np.float32)
    return out_mu, out_sigma
