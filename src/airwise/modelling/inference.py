"""Control-forecast neural error-model loading and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from airwise.modelling.stats import ErrorModellingStats


def model_space_name(transform: str | None) -> str:
    if transform in {None, "none", "null", "false", ""}:
        return "physical"
    return str(transform)


def apply_forecast_transform(
    physical: np.ndarray,
    transform: str | None,
) -> np.ndarray:
    values = np.maximum(np.asarray(physical, dtype=np.float32), 0.0)
    resolved = model_space_name(transform)
    if resolved == "log1p":
        return np.log1p(values)
    if resolved == "physical":
        return values
    raise ValueError(f"Unsupported pollutant transform {transform!r}")


def _checkpoint_input_channels(state_dict: dict) -> int | None:
    for name, values in state_dict.items():
        if name.endswith("meteo_encoder.enc0.block.0.weight"):
            return int(values.shape[1])
    return None


def load_spatial_std_model(
    checkpoint_path: str | Path,
    meteo_channels: int,
    device: str,
    n_leads: int,
):
    import torch

    from airwise.modelling.models import build_model

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    checkpoint_channels = _checkpoint_input_channels(state_dict)
    if checkpoint_channels is not None and checkpoint_channels != meteo_channels:
        raise ValueError(
            f"{path} expects {checkpoint_channels} meteorology channels, "
            f"got {meteo_channels}."
        )
    model = build_model(
        model_name=str(
            checkpoint.get("model_name", "dual_encoder_biconvgru_unet")
        ),
        meteo_channels=meteo_channels,
        base_channels=int(checkpoint.get("base_channels", 32)),
        uncertainty_mode=str(checkpoint.get("uncertainty_mode", "learned_std")),
        std_mode=str(checkpoint.get("std_mode", "spatial")),
        n_leads=n_leads,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_mu_sigma(
    model,
    cams_physical_lead: np.ndarray,
    meteorology: np.ndarray,
    stats: ErrorModellingStats,
    device: str,
    *,
    transform: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model-space predictive parameters shaped ``lead, lat, lon``."""
    import torch

    from airwise.modelling.models import unpack_model_output

    space = apply_forecast_transform(cams_physical_lead, transform)
    normalised = stats.cams_forecast.normalise(space).astype("float32")
    cams = torch.from_numpy(normalised[None, None, ...]).to(device)
    meteo = torch.from_numpy(
        np.nan_to_num(
            np.asarray(meteorology, dtype="float32")[None, ...],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    ).to(device)
    with torch.inference_mode():
        prediction, log_variance = unpack_model_output(model(cams, meteo))
        if log_variance is None:
            raise RuntimeError(
                "Expected spatial log_variance from the confidence checkpoint."
            )
        residual = stats.target_error.denormalise(prediction)
        forecast_space = stats.cams_forecast.denormalise(cams)
        mu = (forecast_space + residual)[0, 0]
        sigma = torch.exp(0.5 * torch.clamp(log_variance, -10.0, 6.0)) * float(
            stats.target_error.std
        )
        sigma = torch.broadcast_to(sigma, prediction.shape)[0, 0]
    return (
        mu.detach().cpu().numpy().astype("float32"),
        sigma.detach().cpu().numpy().astype("float32"),
    )
