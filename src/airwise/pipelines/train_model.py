from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from airwise.config import load_config, require_config, resolve_repo_path
from airwise.modelling.config import get_error_modelling_settings
from airwise.modelling.training.datasets import build_default_datasets
from airwise.modelling.stats import (
    default_era5_stats_path,
    default_pollutant_stats_path,
)
from airwise.data.preprocessing.masks import load_cams_land_mask
from airwise.modelling.models import LearnedStdErrorModel, build_model
from airwise.modelling.training.losses import (
    DEFAULT_CAMS_AQI_HUBER_WEIGHTS,
    DEFAULT_CAMS_AQI_NLL_WEIGHTS,
    DEFAULT_HUBER_BETA,
    DEFAULT_PHYSICAL_TAIL_HUBER_BETA,
    DEFAULT_PHYSICAL_TAIL_LAMBDA,
    DEFAULT_PHYSICAL_TAIL_MIN_AQI,
    PhysicalTailLossConfig,
    build_cams_aqi_nll_weight_config,
)
from airwise.modelling.training.trainer import (
    evaluate,
    fit_error_model,
)


logger = logging.getLogger(__name__)


def move_optimizer_state_to_device(optimizer, device: str) -> None:
    import torch

    target_device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target_device)


def parse_args() -> argparse.Namespace:
    settings = get_error_modelling_settings()
    config = load_config()
    default_transform = settings.default_pollutant_log_transform or "none"
    parser = argparse.ArgumentParser(description="Train the CAMS forecast error model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help=(
            "Number of micro-batches to accumulate before an optimizer step. "
            "Effective batch size is batch-size * accumulate-grad-batches."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--std-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument(
        "--model",
        default="pointwise_temporal",
        choices=[
            "pointwise_temporal",
            "fast",
            "multiscale",
            "legacy",
            "dual_encoder_convgru_unet",
            "v1.0",
            "dual_encoder_biconvgru_unet",
            "v1.5",
        ],
        help=(
            "Model architecture. pointwise_temporal is the fast default; "
            "multiscale is the original heavier 3D CNN; "
            "dual_encoder_convgru_unet (alias v1.0) uses unidirectional ConvGRU; "
            "dual_encoder_biconvgru_unet (alias v1.5) uses BidirectionalConvGRU."
        ),
    )
    parser.add_argument(
        "--interpolate-mode",
        default=None,
        help="Override ERA5-to-CAMS interpolation mode. Defaults: nearest for pointwise_temporal, trilinear for multiscale.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-preload-to-memory",
        action="store_true",
        help="Disable default train/val preloading and read samples lazily from xarray instead.",
    )
    parser.add_argument("--no-shuffle", action="store_true", help="Disable train dataloader shuffling.")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint-root",
        default=str(resolve_repo_path(require_config(config, "paths", "checkpoints"))),
        help="Root directory for checkpoints. A model-name subdirectory is created automatically.",
    )
    parser.add_argument(
        "--checkpoint-save-after",
        type=int,
        default=0,
        help="Do not save checkpoints for the first N epochs.",
    )
    parser.add_argument(
        "--checkpoint-save-every",
        type=int,
        default=1,
        help="Save one checkpoint every M epochs after --checkpoint-save-after.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a checkpoint .pt file to resume model and optimizer state from.",
    )
    parser.add_argument(
        "--warm-start-from",
        default=None,
        help="Load weights without optimizer state. Used to start learned-std from a mean backbone, or to finetune a deterministic run under a new loss.",
    )
    parser.add_argument(
        "--uncertainty-mode",
        choices=["deterministic", "learned_std"],
        default="deterministic",
    )
    parser.add_argument(
        "--std-mode",
        choices=["constant", "lead_specific", "spatial"],
        default="lead_specific",
        help=(
            "Learned-std layout. 'spatial' predicts a per-grid log-variance "
            "field from the shared DualEncoder decoder (dual_encoder_* only)."
        ),
    )
    parser.add_argument("--std-warmup-epochs", type=int, default=1)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Keep the mean backbone frozen for the whole run and train only "
            "the std / logvar head. Overrides --std-warmup-epochs."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "huber", "aqi-weighted-huber"],
        default="aqi-weighted-huber",
        help=(
            "Deterministic mean-training loss. "
            "aqi-weighted-huber is the PM10 mean-backbone recipe. "
            "learned_std training always uses gaussian_nll."
        ),
    )
    parser.add_argument(
        "--huber-beta",
        type=float,
        default=DEFAULT_HUBER_BETA,
        help="Smooth-L1 / Huber transition point (historical PM10 value: 1.0).",
    )
    parser.add_argument(
        "--huber-aqi-weights",
        type=float,
        nargs=6,
        default=None,
        metavar=("W1", "W2", "W3", "W4", "W5", "W6"),
        help=(
            "Per-CAMS-AQI Huber weights for levels 1-6. "
            f"Default: {' '.join(str(w) for w in DEFAULT_CAMS_AQI_HUBER_WEIGHTS)}."
        ),
    )
    parser.add_argument(
        "--aqi-weight-source",
        choices=["forecast", "analysis", "max"],
        default=None,
        help=(
            "Which physical concentration is mapped to AQI weights: "
            "forecast (legacy CAMS forecast), analysis, or the per-pixel max of both. "
            "Defaults to analysis for aqi-weighted-huber and forecast for NLL weighting."
        ),
    )
    parser.add_argument(
        "--physical-tail-lambda",
        type=float,
        default=0.0,
        help=(
            "Optional extra term: lambda * physical MAE/Huber on analysis AQI "
            ">= --physical-tail-min-aqi. Default 0 leaves the term unused "
            "(O3 mean and the original PM10/PM2.5 backbone). "
            f"Pass a positive value to enable it; suggested PM2.5: {DEFAULT_PHYSICAL_TAIL_LAMBDA}."
        ),
    )
    parser.add_argument(
        "--physical-tail-min-aqi",
        type=int,
        default=DEFAULT_PHYSICAL_TAIL_MIN_AQI,
        choices=[1, 2, 3, 4, 5, 6],
        help="Lowest analysis AQI level included in the physical tail term (default: 5).",
    )
    parser.add_argument(
        "--physical-tail-loss",
        choices=["mae", "huber"],
        default="mae",
        help="Physical-concentration tail loss. mae matches eval MAE; huber is Smooth-L1 in µg.",
    )
    parser.add_argument(
        "--physical-tail-huber-beta",
        type=float,
        default=DEFAULT_PHYSICAL_TAIL_HUBER_BETA,
        help="Smooth-L1 transition in µg when --physical-tail-loss huber (default: 1.0).",
    )
    parser.add_argument(
        "--nll-weight-by-cams-aqi",
        action="store_true",
        help=(
            "Weight the training Gaussian NLL by CAMS forecast AQI. "
            "Validation/test remain unweighted. Requires --uncertainty-mode learned_std."
        ),
    )
    parser.add_argument(
        "--nll-aqi-weights",
        type=float,
        nargs=6,
        default=None,
        metavar=("W1", "W2", "W3", "W4", "W5", "W6"),
        help=(
            "Per-CAMS-AQI NLL weights for levels 1-6. "
            f"Default: {' '.join(str(w) for w in DEFAULT_CAMS_AQI_NLL_WEIGHTS)}."
        ),
    )
    parser.add_argument(
        "--nll-weight-land-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When AQI weighting is on (NLL or Huber), zero out sea pixels. "
            "Disable with --no-nll-weight-land-only."
        ),
    )
    parser.add_argument(
        "--aqi-config",
        default=None,
        help=(
            "Optional AQI YAML used to map CAMS forecast concentration to levels. "
            "Defaults to the bundled European AQI definition."
        ),
    )
    parser.add_argument(
        "--land-sea-mask",
        default=str(resolve_repo_path(require_config(config, "paths", "land_sea_mask"))),
        help="Land-sea mask used when --nll-weight-land-only is enabled.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision (fp16) to reduce activation memory.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help=(
            "Enable activation checkpointing on DualEncoderConvGRUUNet blocks. "
            "Trades compute for lower peak GPU memory."
        ),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--tensorboard-root",
        default=str(resolve_repo_path(require_config(config, "paths", "tensorboard"))),
        help="Root directory for TensorBoard event files.",
    )
    parser.add_argument("--disable-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument(
        "--stats-output",
        default=None,
        help=(
            "Pollutant z-score JSON (CAMS forecast + residual). "
            "Defaults to models/error_modelling_{pollutant}_stats.json. "
            "Does not contain ERA5 statistics."
        ),
    )
    parser.add_argument(
        "--stats-mode",
        default="auto",
        choices=["auto", "load", "recompute"],
        help=(
            "Pollutant-stats file policy. "
            "auto: load if present else compute; load: require file; "
            "recompute: overwrite this pollutant's JSON only (never another species)."
        ),
    )
    parser.add_argument(
        "--era5-stats-path",
        default=str(default_era5_stats_path()),
        help="Shared ERA5 z-score JSON reused by every pollutant.",
    )
    parser.add_argument(
        "--era5-stats-mode",
        default="auto",
        choices=["auto", "load", "recompute"],
        help=(
            "ERA5-stats file policy, independent of --stats-mode. "
            "auto loads the shared file when it exists so training a new pollutant "
            "does not recompute or overwrite meteorology statistics."
        ),
    )
    parser.add_argument(
        "--era5-grid",
        default=settings.era5_grid,
        choices=["cams", "raw"],
        help=(
            "ERA5 spatial source when loading from NetCDF. "
            "Ignored when --data-source zarr is used."
        ),
    )
    parser.add_argument(
        "--data-source",
        default=settings.data_source,
        choices=["auto", "zarr", "netcdf"],
        help=(
            "Training data backend. "
            "'auto' uses the training Zarr cache when meta.json exists, otherwise NetCDF; "
            "'zarr' requires paths.training_zarr; "
            "'netcdf' loads processed yearly NetCDF files."
        ),
    )
    parser.add_argument(
        "--zarr-root",
        default=None,
        help="Training Zarr cache directory. Defaults to paths.training_zarr.",
    )
    parser.add_argument(
        "--pollutant",
        default=settings.pollutant,
        help="Pollutant variable to train on (must exist in the chosen data source).",
    )
    parser.add_argument(
        "--pollutant-log-transform",
        default=default_transform,
        choices=["log1p", "log", "none"],
        help="Pollutant-space transform applied at load time. Use 'none' for raw concentrations.",
    )
    parser.add_argument("--build-test", action="store_true", help="Also build/evaluate the test years from config.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log-every", type=int, default=10, help="Log every N batches; set 0 to disable batch logs.")
    args = parser.parse_args()
    if args.accumulate_grad_batches < 1:
        parser.error("--accumulate-grad-batches must be >= 1.")
    if args.std_mode == "spatial" and args.uncertainty_mode != "learned_std":
        parser.error("--std-mode spatial requires --uncertainty-mode learned_std.")
    if args.std_mode == "spatial" and args.model not in {
        "dual_encoder_convgru_unet",
        "v1.0",
        "dual_encoder_biconvgru_unet",
        "v1.5",
    }:
        parser.error("--std-mode spatial requires a dual_encoder_* model.")
    if args.nll_weight_by_cams_aqi and args.uncertainty_mode != "learned_std":
        parser.error("--nll-weight-by-cams-aqi requires --uncertainty-mode learned_std.")
    if args.nll_aqi_weights is not None and not args.nll_weight_by_cams_aqi:
        parser.error("--nll-aqi-weights requires --nll-weight-by-cams-aqi.")
    if args.huber_aqi_weights is not None and args.loss != "aqi-weighted-huber":
        parser.error("--huber-aqi-weights requires --loss aqi-weighted-huber.")
    if args.warm_start_from is not None and args.resume_from is not None:
        parser.error("Use either --warm-start-from (weights only) or --resume-from (full resume), not both.")
    if args.huber_beta <= 0:
        parser.error("--huber-beta must be > 0.")
    if args.freeze_backbone and args.uncertainty_mode != "learned_std":
        parser.error("--freeze-backbone requires --uncertainty-mode learned_std.")
    if any(weight < 0 for weight in (args.nll_aqi_weights or ())):
        parser.error("--nll-aqi-weights must be non-negative.")
    if any(weight < 0 for weight in (args.huber_aqi_weights or ())):
        parser.error("--huber-aqi-weights must be non-negative.")
    if args.physical_tail_lambda < 0:
        parser.error("--physical-tail-lambda must be >= 0.")
    if args.physical_tail_huber_beta <= 0:
        parser.error("--physical-tail-huber-beta must be > 0.")
    if args.physical_tail_lambda > 0 and args.uncertainty_mode != "deterministic":
        parser.error("--physical-tail-lambda requires --uncertainty-mode deterministic.")
    if args.stats_output is None:
        args.stats_output = str(default_pollutant_stats_path(args.pollutant))
    return args


def main() -> None:
    args = parse_args()
    settings = get_error_modelling_settings()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import torch
    from torch.utils.data import DataLoader

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)
    pollutant_log_transform = None if args.pollutant_log_transform == "none" else args.pollutant_log_transform
    logger.info(
        "Training config: epochs=%d batch_size=%d accumulate_grad_batches=%d "
        "effective_batch_size=%d lr=%g weight_decay=%g val_fraction=%g model=%s "
        "base_channels=%d stats_mode=%s stats_output=%s era5_stats_mode=%s "
        "era5_stats_path=%s data_source=%s zarr_root=%s pollutant=%s "
        "pollutant_log_transform=%s era5_grid=%s build_test=%s preload_to_memory=%s "
        "shuffle=%s amp=%s grad_checkpoint=%s std_mode=%s std_warmup_epochs=%d "
        "freeze_backbone=%s nll_weight_by_cams_aqi=%s loss=%s huber_beta=%s",
        args.epochs,
        args.batch_size,
        args.accumulate_grad_batches,
        args.batch_size * max(1, args.accumulate_grad_batches),
        args.lr,
        args.weight_decay,
        args.val_fraction,
        args.model,
        args.base_channels,
        args.stats_mode,
        args.stats_output,
        args.era5_stats_mode,
        args.era5_stats_path,
        args.data_source,
        args.zarr_root,
        args.pollutant,
        pollutant_log_transform,
        args.era5_grid,
        args.build_test,
        not args.no_preload_to_memory,
        not args.no_shuffle,
        args.amp,
        args.grad_checkpoint,
        args.std_mode,
        args.std_warmup_epochs,
        args.freeze_backbone,
        args.nll_weight_by_cams_aqi,
        args.loss if args.uncertainty_mode == "deterministic" else "gaussian_nll",
        args.huber_beta,
    )

    logger.info("Building datasets")
    train_dataset, val_dataset, test_dataset, _stats = build_default_datasets(
        val_fraction=args.val_fraction,
        chunks={"time": 24},
        build_test=args.build_test,
        stats_path=args.stats_output,
        stats_mode=args.stats_mode,
        era5_stats_path=args.era5_stats_path,
        era5_stats_mode=args.era5_stats_mode,
        preload_to_memory=not args.no_preload_to_memory,
        era5_grid=args.era5_grid,
        data_source=args.data_source,
        zarr_root=args.zarr_root,
        pollutant=args.pollutant,
        pollutant_log_transform=pollutant_log_transform,
    )

    logger.info("Creating dataloaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    logger.info("Building model")
    model = build_model(
        model_name=args.model,
        meteo_channels=len(train_dataset.era5_variables),
        base_channels=args.base_channels,
        interpolate_mode=args.interpolate_mode,
        uncertainty_mode=args.uncertainty_mode,
        std_mode=args.std_mode,
        n_leads=len(settings.lead_hours),
    )
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Built model %s with %d parameters", model.__class__.__name__, n_parameters)
    if args.grad_checkpoint:
        backbone = model.backbone if isinstance(model, LearnedStdErrorModel) else model
        if not hasattr(backbone, "use_gradient_checkpointing"):
            raise ValueError(
                "--grad-checkpoint is only supported for DualEncoderConvGRUUNet backbones."
            )
        backbone.use_gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing on %s", backbone.__class__.__name__)
    if args.uncertainty_mode == "learned_std":
        if not isinstance(model, LearnedStdErrorModel):
            raise TypeError("Expected LearnedStdErrorModel.")
        if args.warm_start_from is None and args.resume_from is None:
            raise ValueError(
                "--warm-start-from is required for a new learned-std run."
            )
        if args.warm_start_from is not None and args.resume_from is None:
            warm_path = Path(args.warm_start_from)
            logger.info("Warm-starting deterministic backbone from %s", warm_path)
            warm_checkpoint = torch.load(warm_path, map_location=device)
            incompatible = model.backbone.load_state_dict(
                warm_checkpoint["model_state_dict"],
                strict=False,
            )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "Warm-start checkpoint has unexpected backbone keys: "
                    f"{incompatible.unexpected_keys}"
                )
            missing = list(incompatible.missing_keys)
            if args.std_mode == "spatial":
                unexpected_missing = [
                    key for key in missing if not key.startswith("logvar_head")
                ]
                if unexpected_missing:
                    raise ValueError(
                        "Warm-start is missing backbone weights: "
                        f"{unexpected_missing}"
                    )
                logger.info(
                    "Initialized spatial logvar_head randomly; loaded %d backbone tensors",
                    len(warm_checkpoint["model_state_dict"]),
                )
            elif missing:
                raise ValueError(
                    "Warm-start is missing backbone weights: " f"{missing}"
                )
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone_parameters(), "lr": args.backbone_lr},
                {"params": model.std_parameters(), "lr": args.std_lr},
            ],
            weight_decay=args.weight_decay,
        )
        loss_name = "gaussian_nll"
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        loss_name = args.loss
        if args.warm_start_from is not None:
            warm_path = Path(args.warm_start_from)
            logger.info("Warm-starting deterministic model from %s", warm_path)
            warm_checkpoint = torch.load(warm_path, map_location=device)
            model.load_state_dict(warm_checkpoint["model_state_dict"])
            logger.info(
                "Loaded deterministic weights from epoch=%s; starting a fresh optimizer",
                warm_checkpoint.get("epoch"),
            )
    start_epoch = 1
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        logger.info("Resuming training from checkpoint %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_uncertainty = checkpoint.get(
            "uncertainty_mode", "deterministic"
        )
        if checkpoint_uncertainty != args.uncertainty_mode:
            raise ValueError(
                "Resume checkpoint uncertainty_mode does not match CLI: "
                f"{checkpoint_uncertainty!r} vs {args.uncertainty_mode!r}."
            )
        if (
            args.uncertainty_mode == "learned_std"
            and checkpoint.get("std_mode") != args.std_mode
        ):
            raise ValueError(
                "Resume checkpoint std_mode does not match CLI: "
                f"{checkpoint.get('std_mode')!r} vs {args.std_mode!r}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info(
            "Loaded checkpoint epoch=%d; next epoch will be %d",
            checkpoint["epoch"],
            start_epoch,
        )

    default_run_name = (
        args.model
        if args.uncertainty_mode == "deterministic"
        else f"{args.model}_{args.std_mode}_std"
    )
    checkpoint_dir = Path(args.checkpoint_root) / (args.run_name or default_run_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = None
    if not args.disable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard logging requires tensorboard. Install it or pass --disable-tensorboard."
            ) from exc

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = Path(args.tensorboard_root) / args.model / run_name
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
        tensorboard_writer.add_text(
            "config",
            json.dumps(vars(args), indent=2),
            global_step=0,
        )
        logger.info("Writing TensorBoard logs to %s", tensorboard_dir)
    logger.info("Saving checkpoints to %s", checkpoint_dir)

    nll_weight_config = None
    aqi_weight_source = args.aqi_weight_source
    if aqi_weight_source is None:
        aqi_weight_source = "analysis" if loss_name == "aqi-weighted-huber" else "forecast"
    physical_tail_config = None
    if args.physical_tail_lambda > 0:
        physical_tail_config = PhysicalTailLossConfig(
            min_aqi_level=args.physical_tail_min_aqi,
            lambda_weight=args.physical_tail_lambda,
            kind=args.physical_tail_loss,
            huber_beta=args.physical_tail_huber_beta,
        )
        logger.info(
            "Physical tail loss enabled: lambda=%g min_aqi=%d kind=%s huber_beta=%g",
            physical_tail_config.lambda_weight,
            physical_tail_config.min_aqi_level,
            physical_tail_config.kind,
            physical_tail_config.huber_beta,
        )
    else:
        logger.info("Physical tail loss disabled (--physical-tail-lambda=0)")
    if (
        args.nll_weight_by_cams_aqi
        or loss_name == "aqi-weighted-huber"
        or physical_tail_config is not None
    ):
        land_mask = None
        if args.nll_weight_land_only:
            land_mask = load_cams_land_mask(
                args.land_sea_mask,
                latitude=train_dataset.data.cams_forecast["latitude"].values,
                longitude=train_dataset.data.cams_forecast["longitude"].values,
            )
        level_weights = (
            args.nll_aqi_weights
            if loss_name == "gaussian_nll"
            else (args.huber_aqi_weights or DEFAULT_CAMS_AQI_HUBER_WEIGHTS)
        )
        nll_weight_config = build_cams_aqi_nll_weight_config(
            train_dataset,
            level_weights=level_weights,
            aqi_config=args.aqi_config,
            land_mask=land_mask,
            source=aqi_weight_source,
        )
        logger.info(
            "CAMS AQI weighting enabled for %s: source=%s weights=%s land_only=%s "
            "invert_log1p=%s thresholds=%s huber_beta=%s physical_tail=%s",
            loss_name,
            nll_weight_config.source,
            nll_weight_config.level_weights,
            land_mask is not None,
            nll_weight_config.invert_log1p,
            nll_weight_config.aqi_thresholds,
            args.huber_beta if loss_name in {"huber", "aqi-weighted-huber"} else None,
            None
            if physical_tail_config is None
            else {
                "lambda": physical_tail_config.lambda_weight,
                "min_aqi": physical_tail_config.min_aqi_level,
                "kind": physical_tail_config.kind,
            },
        )
        if physical_tail_config is not None and nll_weight_config.target_mean is None:
            raise ValueError(
                "Physical tail loss requires dataset.stats.target_error "
                "to reconstruct analysis concentration."
            )

    try:
        history = fit_error_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=checkpoint_dir,
            checkpoint_save_after=args.checkpoint_save_after,
            checkpoint_save_every=args.checkpoint_save_every,
            tensorboard_writer=tensorboard_writer,
            start_epoch=start_epoch,
            log_every=args.log_every,
            loss_name=loss_name,
            std_warmup_epochs=(
                args.std_warmup_epochs
                if args.uncertainty_mode == "learned_std"
                else 0
            ),
            freeze_backbone=bool(args.freeze_backbone),
            nll_weight_config=nll_weight_config,
            use_amp=args.amp,
            accumulate_grad_batches=args.accumulate_grad_batches,
            huber_beta=args.huber_beta,
            physical_tail_config=physical_tail_config,
            checkpoint_metadata={
                "uncertainty_mode": args.uncertainty_mode,
                "std_mode": (
                    args.std_mode
                    if args.uncertainty_mode == "learned_std"
                    else "none"
                ),
                "lead_hours": list(settings.lead_hours),
                "model_name": args.model,
                "base_channels": args.base_channels,
                "warm_start_from": args.warm_start_from or "",
                "amp": bool(args.amp),
                "grad_checkpoint": bool(args.grad_checkpoint),
                "accumulate_grad_batches": int(args.accumulate_grad_batches),
                "freeze_backbone": bool(args.freeze_backbone),
                "nll_weight_by_cams_aqi": bool(args.nll_weight_by_cams_aqi),
                "loss": loss_name,
                "huber_beta": args.huber_beta,
                "aqi_weight_source": aqi_weight_source,
                "physical_tail_lambda": args.physical_tail_lambda,
                "physical_tail_min_aqi": args.physical_tail_min_aqi,
                "physical_tail_loss": args.physical_tail_loss,
            },
        )
        test_metrics = None
        if test_loader is not None:
            test_metrics = evaluate(
                model,
                test_loader,
                device=device,
                stage="test",
                log_every=args.log_every,
                loss_name=loss_name,
                use_amp=args.amp,
                huber_beta=args.huber_beta,
            )
            if tensorboard_writer is not None:
                for name, value in test_metrics.items():
                    tensorboard_writer.add_scalar(f"test/{name}", value, args.epochs)
                tensorboard_writer.flush()
            logger.info("Finished training run with test metrics: %s", test_metrics)
        else:
            logger.info("Finished training run without test evaluation")

        print(json.dumps({"history": history, "test_metrics": test_metrics}, indent=2))
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        closed_data_ids = set()
        for dataset in (train_dataset, val_dataset, test_dataset):
            if dataset is None:
                continue
            data_id = id(dataset.data)
            if data_id in closed_data_ids:
                continue
            dataset.close()
            closed_data_ids.add(data_id)


if __name__ == "__main__":
    main()
