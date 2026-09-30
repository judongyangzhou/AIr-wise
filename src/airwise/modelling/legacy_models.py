"""Earlier CAMS error-model backbones kept for retraining.

The bulletin loads ``dual_encoder_biconvgru_unet`` from
``airwise.modelling.models``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

def _interpolate_to_cams_grid(meteo_features, cams_features, mode: str):
    kwargs = {
        "input": meteo_features,
        "size": cams_features.shape[2:],
        "mode": mode,
    }
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(**kwargs)


class PointwiseTemporalErrorCNN(nn.Module):
    """
    Fast baseline for CAMS PM10 forecast error prediction.

    This model avoids expensive 3x3 spatial convolutions on the 420x700 CAMS
    grid. It only mixes information across channels and the 8 lead-time steps,
    then predicts a high-resolution residual field.
    """

    def __init__(
        self,
        meteo_channels: int = 8,
        base_channels: int = 8,
        interpolate_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.cams_encoder = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GELU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=1),
            nn.GELU(),
        )
        self.meteo_encoder = nn.Sequential(
            nn.Conv3d(meteo_channels, base_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GELU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(base_channels, 1, kernel_size=1),
        )

    def forward(self, cams_forecast, era5):
        cams_features = self.cams_encoder(cams_forecast)
        meteo_features = self.meteo_encoder(era5)
        meteo_features = _interpolate_to_cams_grid(
            meteo_features,
            cams_features,
            mode=self.interpolate_mode,
        )
        return self.decoder(torch.cat([cams_features, meteo_features], dim=1))


class MultiScaleErrorCNN(nn.Module):
    """
    Original heavier dual-resolution 3D CNN.

    This keeps 3x3x3 convolutions on the high-resolution CAMS grid. It is much
    slower than ``PointwiseTemporalErrorCNN`` but can model local spatial
    neighbourhoods directly.
    """

    def __init__(
        self,
        meteo_channels: int = 8,
        base_channels: int = 16,
        interpolate_mode: str = "trilinear",
    ) -> None:
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.cams_encoder = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.meteo_encoder = nn.Sequential(
            nn.Conv3d(meteo_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(base_channels, 1, kernel_size=1),
        )

    def forward(self, cams_forecast, era5):
        cams_features = self.cams_encoder(cams_forecast)
        meteo_features = self.meteo_encoder(era5)
        meteo_features = _interpolate_to_cams_grid(
            meteo_features,
            cams_features,
            mode=self.interpolate_mode,
        )
        return self.decoder(torch.cat([cams_features, meteo_features], dim=1))



def build_legacy_backbone(
    model_name: str = "pointwise_temporal",
    meteo_channels: int = 7,
    base_channels: int = 8,
    interpolate_mode: str | None = None,
) -> nn.Module:
    """Build an early CNN backbone retained for checkpoint compatibility."""
    if model_name in {"pointwise_temporal", "fast"}:
        backbone = PointwiseTemporalErrorCNN(
            meteo_channels=meteo_channels,
            base_channels=base_channels,
            interpolate_mode=interpolate_mode or "nearest",
        )
    elif model_name in {"multiscale", "legacy"}:
        backbone = MultiScaleErrorCNN(
            meteo_channels=meteo_channels,
            base_channels=base_channels,
            interpolate_mode=interpolate_mode or "trilinear",
        )
    else:
        raise ValueError(f"Unsupported legacy model {model_name!r}.")
    return backbone
