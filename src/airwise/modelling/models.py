from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# ============================================================
# Basic blocks
# ============================================================

class ConvBlock(nn.Module):
    """
    Basic 2D convolution block:
    Conv2d -> GroupNorm -> SiLU -> Conv2d -> GroupNorm -> SiLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super().__init__()

        padding = kernel_size // 2

        # Make sure GroupNorm groups are valid
        groups = min(num_groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block:
    MaxPool2d -> ConvBlock
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """
    Upsampling block:
    Bilinear upsampling -> concat skip -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# ============================================================
# ConvGRU
# ============================================================

class ConvGRUCell(nn.Module):
    """
    ConvGRU cell for one time step.

    Input:
        x_t: [B, C_in, H, W]
        h_prev: [B, C_hidden, H, W]

    Output:
        h_t: [B, C_hidden, H, W]
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv_zr = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.conv_h = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=1)

        zr = self.conv_zr(combined)
        z, r = torch.chunk(zr, chunks=2, dim=1)

        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        combined_candidate = torch.cat([x_t, r * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_h(combined_candidate))

        h_t = (1 - z) * h_prev + z * h_candidate

        return h_t


class ConvGRU(nn.Module):
    """
    Unidirectional ConvGRU over a lead-time sequence.

    For 3-hourly CAMS forecasts this is typically:
        00 -> 03 -> 06 -> ... -> 21

    Input:
        x: [B, T, C, H, W]

    Output:
        output_seq: [B, T, C_hidden, H, W]
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.cell = ConvGRUCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        h = torch.zeros(
            B,
            self.hidden_channels,
            H,
            W,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        for t in range(T):
            h = self.cell(x[:, t], h)
            outputs.append(h)

        output_seq = torch.stack(outputs, dim=1)

        return output_seq


class BidirectionalConvGRU(nn.Module):
    """
    Bidirectional ConvGRU over a lead-time sequence.

    Forward direction:
        00 -> 03 -> 06 -> ... -> 21

    Backward direction:
        21 -> 18 -> 15 -> ... -> 00

    Each direction uses an independent ``ConvGRU``. Their outputs are
    concatenated along the channel axis and projected back to
    ``hidden_channels`` so the surrounding U-Net decoder keeps the same
    channel layout as the unidirectional model.

    Input:
        x: [B, T, C, H, W]

    Output:
        output_seq: [B, T, C_hidden, H, W]
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.forward_gru = ConvGRU(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.backward_gru = ConvGRU(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.merge = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # Forward: t = 0, 1, ..., T-1
        forward_seq = self.forward_gru(x)

        # Backward: reverse time, run ConvGRU, then flip back to original order.
        backward_input = torch.flip(x, dims=[1])
        backward_seq = self.backward_gru(backward_input)
        backward_seq = torch.flip(backward_seq, dims=[1])

        # [B, T, 2C, H, W] -> merge per timestep with shared 1x1 conv
        B, T, _, H, W = forward_seq.shape
        merged = torch.cat([forward_seq, backward_seq], dim=2)
        merged = merged.reshape(B * T, 2 * self.hidden_channels, H, W)
        merged = self.merge(merged)
        return merged.reshape(B, T, self.hidden_channels, H, W)


# ============================================================
# Encoder
# ============================================================

class Encoder2D(nn.Module):
    """
    2D U-Net style encoder.

    Input:
        x: [B*T, C, H, W]

    Output:
        f0: full resolution
        f1: 1/2 resolution
        f2: 1/4 resolution
        f3: 1/8 resolution
        f4: 1/16 resolution
    """

    def __init__(self, in_channels, base_channels=32):
        super().__init__()

        self.enc0 = ConvBlock(in_channels, base_channels)
        self.enc1 = DownBlock(base_channels, base_channels * 2)
        self.enc2 = DownBlock(base_channels * 2, base_channels * 4)
        self.enc3 = DownBlock(base_channels * 4, base_channels * 8)
        self.enc4 = DownBlock(base_channels * 8, base_channels * 8)

    def forward(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)

        return f0, f1, f2, f3, f4


# ============================================================
# Feature fusion
# ============================================================

class FeatureFusion(nn.Module):
    """
    Concatenate concentration and meteorology encoder features, then reduce
    channels with a 1x1 convolution.

    Both inputs are expected on the same spatial grid (e.g. CAMS 0.1deg).
    """

    def __init__(self, conc_channels, meteo_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(conc_channels + meteo_channels, out_channels, kernel_size=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, conc_feat, meteo_feat):
        if conc_feat.shape[-2:] != meteo_feat.shape[-2:]:
            raise ValueError(
                "Concentration and meteorology features must share the same spatial "
                f"size, got {tuple(conc_feat.shape[-2:])} and {tuple(meteo_feat.shape[-2:])}."
            )

        fused = torch.cat([conc_feat, meteo_feat], dim=1)
        return self.proj(fused)


# ============================================================
# Main model
# ============================================================

@dataclass
class ProbabilisticPrediction:
    """Mean and log variance in normalised residual space."""

    mean: torch.Tensor
    log_variance: torch.Tensor


class DualEncoderConvGRUUNet(nn.Module):
    """
    Dual-Encoder ConvGRU U-Net for same-grid concentration and meteorology.

    Inputs (same layout as ``ErrorModellingDataset``):
        forecast_conc:
            [B, 1, T, H, W]

        forecast_meteo:
            [B, C_meteo, T, H, W] on the same CAMS grid as ``forecast_conc``

    Output:
        predicted_error:
            [B, 1, T, H, W]

    The output represents:
        analysis - forecast

    Set ``bidirectional=True`` to use ``BidirectionalConvGRU`` at the
    bottleneck (00→21 and 21→00). Prefer selecting this via
    ``build_model("dual_encoder_biconvgru_unet")`` rather than constructing
    the flag by hand. Default ``False`` keeps unidirectional ``ConvGRU``.
    """

    def __init__(
        self,
        conc_channels=1,
        meteo_channels=7,
        out_channels=1,
        base_channels=32,
        convgru_kernel_size=3,
        pad_multiple=16,
        bidirectional=False,
        predict_log_variance=False,
    ):
        super().__init__()

        self.pad_multiple = pad_multiple
        self.bidirectional = bool(bidirectional)
        self.use_gradient_checkpointing = False
        C = base_channels

        self.conc_encoder = Encoder2D(
            in_channels=conc_channels,
            base_channels=C
        )

        self.meteo_encoder = Encoder2D(
            in_channels=meteo_channels,
            base_channels=C
        )

        # Feature-level fusion at each U-Net scale
        self.fuse0 = FeatureFusion(C, C, C)
        self.fuse1 = FeatureFusion(C * 2, C * 2, C * 2)
        self.fuse2 = FeatureFusion(C * 4, C * 4, C * 4)
        self.fuse3 = FeatureFusion(C * 8, C * 8, C * 8)
        self.fuse4 = FeatureFusion(C * 8, C * 8, C * 8)

        # Temporal module at bottleneck
        temporal_cls = BidirectionalConvGRU if self.bidirectional else ConvGRU
        self.temporal = temporal_cls(
            input_channels=C * 8,
            hidden_channels=C * 8,
            kernel_size=convgru_kernel_size
        )

        # Decoder
        self.up3 = UpBlock(
            in_channels=C * 8,
            skip_channels=C * 8,
            out_channels=C * 4
        )

        self.up2 = UpBlock(
            in_channels=C * 4,
            skip_channels=C * 4,
            out_channels=C * 2
        )

        self.up1 = UpBlock(
            in_channels=C * 2,
            skip_channels=C * 2,
            out_channels=C
        )

        self.up0 = UpBlock(
            in_channels=C,
            skip_channels=C,
            out_channels=C
        )

        # Output heads share the decoded features. The mean head is the
        # deterministic residual; the optional log-variance head predicts a
        # spatially varying Gaussian noise field in the same layout.
        self.out_head = nn.Conv2d(
            in_channels=C,
            out_channels=out_channels,
            kernel_size=1
        )
        self.logvar_head = None
        if predict_log_variance:
            self.logvar_head = nn.Conv2d(
                in_channels=C,
                out_channels=out_channels,
                kernel_size=1,
            )
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.zeros_(self.logvar_head.bias)

    def _run(self, module, *args):
        """Run ``module`` with optional activation checkpointing."""
        if not (self.use_gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return module(*args)

        # Non-reentrant checkpoint skips the region when no input requires grad,
        # which would zero-out encoder parameter gradients. Promote leaf inputs
        # (raw CAMS/ERA5 tensors) so checkpoint engages; cost is only input grads.
        prepared = []
        for argument in args:
            if torch.is_tensor(argument) and not argument.requires_grad:
                prepared.append(argument.detach().requires_grad_(True))
            else:
                prepared.append(argument)
        return torch_checkpoint(module, *prepared, use_reentrant=False)

    @staticmethod
    def _pad_to_multiple(x, multiple=16):
        """
        Pad H and W to be divisible by `multiple`.

        Input:
            x: [B, T, C, H, W]

        Output:
            x_padded
            original_size: (H, W)
        """

        B, T, C, H, W = x.shape

        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        # Padding format for 5D tensor:
        # F.pad pads the last dimensions.
        # Here: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h)) # TODO: set mode to "replicate"

        return x, (H, W)

    @staticmethod
    def _crop_to_original_size(x, original_size):
        """
        Crop output back to original H and W.

        Input:
            x: [B, T, C, H_pad, W_pad]
        """

        H, W = original_size
        return x[..., :H, :W]

    @staticmethod
    def _merge_time(x):
        """
        Convert [B, T, C, H, W] to [B*T, C, H, W].
        """

        B, T, C, H, W = x.shape
        return x.reshape(B * T, C, H, W)

    @staticmethod
    def _unmerge_time(x, B, T):
        """
        Convert [B*T, C, H, W] to [B, T, C, H, W].
        """

        BT, C, H, W = x.shape
        assert BT == B * T
        return x.reshape(B, T, C, H, W)

    def _decoder_features(self, forecast_conc, forecast_meteo):
        """
        Shared encoder/decoder features for the mean and log-variance heads.

        Returns:
            features: [B*T, C, H_pad, W_pad]
            original_size: (H, W)
            B, T
        """
        forecast_conc = forecast_conc.permute(0, 2, 1, 3, 4)
        forecast_meteo = forecast_meteo.permute(0, 2, 1, 3, 4)

        B, T, _, H, W = forecast_conc.shape
        if forecast_meteo.shape[-2:] != (H, W):
            raise ValueError(
                "forecast_meteo spatial size "
                f"{tuple(forecast_meteo.shape[-2:])} must match forecast_conc "
                f"{(H, W)}. Resample ERA5 onto the CAMS grid before training."
            )
        if forecast_meteo.shape[0] != B or forecast_meteo.shape[1] != T:
            raise ValueError(
                "forecast_meteo batch/time dimensions must match forecast_conc: "
                f"got meteo {(forecast_meteo.shape[0], forecast_meteo.shape[1])} "
                f"vs conc {(B, T)}."
            )

        forecast_conc, original_size = self._pad_to_multiple(
            forecast_conc,
            multiple=self.pad_multiple,
        )
        forecast_meteo, _ = self._pad_to_multiple(
            forecast_meteo,
            multiple=self.pad_multiple,
        )

        conc = self._merge_time(forecast_conc)
        meteo = self._merge_time(forecast_meteo)

        c0, c1, c2, c3, c4 = self._run(self.conc_encoder, conc)
        m0, m1, m2, m3, m4 = self._run(self.meteo_encoder, meteo)

        f0 = self._run(self.fuse0, c0, m0)
        f1 = self._run(self.fuse1, c1, m1)
        f2 = self._run(self.fuse2, c2, m2)
        f3 = self._run(self.fuse3, c3, m3)
        f4 = self._run(self.fuse4, c4, m4)

        f4_seq = self._unmerge_time(f4, B, T)
        f4_seq = self._run(self.temporal, f4_seq)
        f4 = self._merge_time(f4_seq)

        x = self._run(self.up3, f4, f3)
        x = self._run(self.up2, x, f2)
        x = self._run(self.up1, x, f1)
        x = self._run(self.up0, x, f0)
        return x, original_size, B, T

    def _field_from_head(self, head_out, original_size, B, T):
        head_out = self._unmerge_time(head_out, B, T)
        head_out = self._crop_to_original_size(head_out, original_size)
        return head_out.permute(0, 2, 1, 3, 4)

    def _shared_features_require_grad(self) -> bool:
        for name, parameter in self.named_parameters():
            if name.startswith("logvar_head"):
                continue
            if parameter.requires_grad:
                return True
        return False

    def forward(self, forecast_conc, forecast_meteo):
        """
        forecast_conc:
            [B, 1, T, H, W]

        forecast_meteo:
            [B, C_meteo, T, H, W] on the same grid as ``forecast_conc``

        return:
            predicted_error [B, 1, T, H, W], or ProbabilisticPrediction when
            ``logvar_head`` is present.
        """
        freeze_shared = (
            self.logvar_head is not None
            and not self._shared_features_require_grad()
        )
        with torch.set_grad_enabled(torch.is_grad_enabled() and not freeze_shared):
            features, original_size, B, T = self._decoder_features(
                forecast_conc,
                forecast_meteo,
            )
        if freeze_shared:
            features = features.detach()

        mean = self._field_from_head(self.out_head(features), original_size, B, T)
        if self.logvar_head is None:
            return mean
        log_variance = self._field_from_head(
            self.logvar_head(features),
            original_size,
            B,
            T,
        )
        return ProbabilisticPrediction(mean=mean, log_variance=log_variance)


class LearnedStdErrorModel(nn.Module):
    """Wrap a mean model with compact or spatially varying Gaussian variance."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        std_mode: str = "lead_specific",
        n_leads: int = 8,
        initial_log_variance: float = 0.0,
    ) -> None:
        super().__init__()
        if std_mode not in {"constant", "lead_specific", "spatial"}:
            raise ValueError(
                "std_mode must be 'constant', 'lead_specific', or 'spatial'."
            )
        if n_leads < 1:
            raise ValueError("n_leads must be >= 1.")
        if std_mode == "spatial" and getattr(backbone, "logvar_head", None) is None:
            raise ValueError(
                "spatial std_mode requires a DualEncoderConvGRUUNet backbone "
                "built with predict_log_variance=True."
            )
        self.backbone = backbone
        self.std_mode = std_mode
        self.n_leads = int(n_leads)
        if std_mode == "spatial":
            self.raw_log_variance = None
        else:
            size = 1 if std_mode == "constant" else self.n_leads
            self.raw_log_variance = nn.Parameter(
                torch.full((size,), float(initial_log_variance))
            )

    def forward(self, cams_forecast, forecast_meteo) -> ProbabilisticPrediction:
        if self.std_mode == "spatial":
            output = self.backbone(cams_forecast, forecast_meteo)
            if not isinstance(output, ProbabilisticPrediction):
                raise TypeError(
                    "spatial std_mode expects the backbone to return "
                    "ProbabilisticPrediction."
                )
            if output.mean.ndim != 5 or output.log_variance.ndim != 5:
                raise ValueError(
                    "Expected spatial mean/log_variance [B,C,T,H,W], got "
                    f"{output.mean.shape} and {output.log_variance.shape}."
                )
            return output

        # When the backbone is frozen (std warmup), skip autograd through it so
        # memory matches "std-only" training rather than accidentally retaining
        # a full U-Net graph that is never differentiated.
        backbone_trainable = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        if backbone_trainable:
            mean = self.backbone(cams_forecast, forecast_meteo)
        else:
            with torch.no_grad():
                mean = self.backbone(cams_forecast, forecast_meteo)
        if isinstance(mean, ProbabilisticPrediction):
            mean = mean.mean
        if mean.ndim != 5:
            raise ValueError(f"Expected mean [B,C,T,H,W], got {mean.shape}.")
        if self.std_mode == "lead_specific" and mean.shape[2] != self.n_leads:
            raise ValueError(
                f"Expected {self.n_leads} leads, got output shape {mean.shape}."
            )
        # Keep log-variance compact ([1,1,1,1,1] or [1,1,T,1,1]) and let
        # loss/broadcasting expand it. Materialising a full [B,C,T,H,W] copy
        # wastes several GiB on the CAMS grid for no numerical benefit.
        shape = (1, 1, 1, 1, 1)
        if self.std_mode == "lead_specific":
            shape = (1, 1, self.n_leads, 1, 1)
        log_variance = self.raw_log_variance.view(*shape)
        return ProbabilisticPrediction(mean=mean, log_variance=log_variance)

    def backbone_parameters(self):
        if self.std_mode == "spatial":
            return [
                parameter
                for name, parameter in self.backbone.named_parameters()
                if not name.startswith("logvar_head")
            ]
        return list(self.backbone.parameters())

    def std_parameters(self):
        if self.std_mode == "spatial":
            return list(self.backbone.logvar_head.parameters())
        return [self.raw_log_variance]

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the mean backbone; the std head stays trainable."""
        for parameter in self.backbone_parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.std_parameters():
            parameter.requires_grad_(True)

    def learned_sigma_normalised(self) -> torch.Tensor:
        if self.raw_log_variance is None:
            raise RuntimeError(
                "spatial std_mode has no compact sigma; run a forward pass."
            )
        return torch.exp(0.5 * self.raw_log_variance.detach())


def unpack_model_output(output) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (mean, log_variance), accepting deterministic outputs."""
    if isinstance(output, ProbabilisticPrediction):
        return output.mean, output.log_variance
    if torch.is_tensor(output):
        return output, None
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def build_model(
    model_name: str = "dual_encoder_biconvgru_unet",
    meteo_channels: int = 7,
    base_channels: int = 8,
    interpolate_mode: str | None = None,
    uncertainty_mode: str = "deterministic",
    std_mode: str = "lead_specific",
    n_leads: int = 8,
) -> nn.Module:
    if model_name in {"pointwise_temporal", "fast", "multiscale", "legacy"}:
        from airwise.modelling.legacy_models import build_legacy_backbone

        backbone = build_legacy_backbone(
            model_name=model_name,
            meteo_channels=meteo_channels,
            base_channels=base_channels,
            interpolate_mode=interpolate_mode,
        )
    elif model_name in {"dual_encoder_convgru_unet", "v1.0"}:
        backbone = DualEncoderConvGRUUNet(
            conc_channels=1,
            meteo_channels=meteo_channels,
            out_channels=1,
            base_channels=base_channels,
            convgru_kernel_size=3,
            bidirectional=False,
            predict_log_variance=(
                uncertainty_mode == "learned_std" and std_mode == "spatial"
            ),
        )
    elif model_name in {"dual_encoder_biconvgru_unet", "v1.5"}:
        backbone = DualEncoderConvGRUUNet(
            conc_channels=1,
            meteo_channels=meteo_channels,
            out_channels=1,
            base_channels=base_channels,
            convgru_kernel_size=3,
            bidirectional=True,
            predict_log_variance=(
                uncertainty_mode == "learned_std" and std_mode == "spatial"
            ),
        )
    else:
        raise ValueError(
            "model_name must be one of: "
            "'pointwise_temporal', 'fast', 'multiscale', 'legacy', "
            "'dual_encoder_convgru_unet', 'v1.0', "
            "'dual_encoder_biconvgru_unet', 'v1.5'."
        )
    if uncertainty_mode == "deterministic":
        return backbone
    if uncertainty_mode == "learned_std":
        if std_mode == "spatial" and not isinstance(backbone, DualEncoderConvGRUUNet):
            raise ValueError(
                "spatial std_mode is only supported for dual_encoder_* models."
            )
        return LearnedStdErrorModel(
            backbone,
            std_mode=std_mode,
            n_leads=n_leads,
        )
    raise ValueError("uncertainty_mode must be 'deterministic' or 'learned_std'.")
