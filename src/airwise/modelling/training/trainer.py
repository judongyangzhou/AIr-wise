"""Training and evaluation loops for the error model."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from airwise.modelling.config import get_error_modelling_settings
from airwise.modelling.models import LearnedStdErrorModel
from airwise.modelling.training.losses import (
    DEFAULT_HUBER_BETA,
    WEIGHTED_LOSS_NAMES,
    CamsAqiNllWeightConfig,
    PhysicalTailLossConfig,
    physical_tail_concentration_loss,
    prediction_loss,
)
from airwise.modelling.training.metrics import ConcentrationMetricsAccumulator


logger = logging.getLogger(__name__)

def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for neural network training. Install `torch`.") from exc
    return torch


class RegressionMetrics:
    """
    Streaming regression metrics for normalised targets.

    RMSE and MAE are reported after rescaling by the target standard deviation,
    so they are in the same log-error space as ``analysis - forecast``.
    Correlation and R2 are invariant to this shared linear scaling.
    """

    def __init__(self, target_std: float = 1.0) -> None:
        self.target_std = target_std
        self.n = 0
        self.sse = 0.0
        self.sae = 0.0
        self.sum_pred = 0.0
        self.sum_target = 0.0
        self.sum_pred_sq = 0.0
        self.sum_target_sq = 0.0
        self.sum_pred_target = 0.0

    def update(self, prediction, target) -> None:
        torch = _require_torch()
        diff = prediction - target
        pred = prediction.detach().float()
        tgt = target.detach().float()
        diff = diff.detach().float()

        self.n += pred.numel()
        batch_stats = torch.stack(
            [
                (diff * diff).sum(),
                diff.abs().sum(),
                pred.sum(),
                tgt.sum(),
                (pred * pred).sum(),
                (tgt * tgt).sum(),
                (pred * tgt).sum(),
            ]
        ).cpu()
        (
            sse,
            sae,
            sum_pred,
            sum_target,
            sum_pred_sq,
            sum_target_sq,
            sum_pred_target,
        ) = batch_stats.tolist()
        self.sse += sse
        self.sae += sae
        self.sum_pred += sum_pred
        self.sum_target += sum_target
        self.sum_pred_sq += sum_pred_sq
        self.sum_target_sq += sum_target_sq
        self.sum_pred_target += sum_pred_target

    def compute(self) -> dict[str, float]:
        if self.n == 0:
            return {
                "loss": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "corr": float("nan"),
                "r2": float("nan"),
            }

        mse = self.sse / self.n
        rmse = float(np.sqrt(mse) * self.target_std)
        mae = self.sae / self.n * self.target_std

        pred_mean = self.sum_pred / self.n
        target_mean = self.sum_target / self.n
        pred_var = self.sum_pred_sq / self.n - pred_mean * pred_mean
        target_var = self.sum_target_sq / self.n - target_mean * target_mean
        covariance = self.sum_pred_target / self.n - pred_mean * target_mean

        if pred_var <= 0 or target_var <= 0:
            corr = float("nan")
        else:
            corr = covariance / float(np.sqrt(pred_var * target_var))

        sst = self.sum_target_sq - self.sum_target * self.sum_target / self.n
        r2 = float("nan") if sst <= 0 else 1.0 - self.sse / sst

        return {
            "loss": mse,
            "rmse": rmse,
            "mae": mae,
            "corr": corr,
            "r2": r2,
        }


def _target_std_from_dataloader(dataloader) -> float:
    dataset = getattr(dataloader, "dataset", None)
    stats = getattr(dataset, "stats", None)
    target_error = getattr(stats, "target_error", None)
    return float(getattr(target_error, "std", 1.0))


def _format_metrics(metrics: Mapping[str, float]) -> str:
    summary = (
        f"loss={metrics['loss']:.6g} "
        f"rmse={metrics['rmse']:.6g} "
        f"mae={metrics['mae']:.6g} "
        f"corr={metrics['corr']:.6g} "
        f"r2={metrics['r2']:.6g}"
    )
    if "concentration_skill_score_vs_forecast" in metrics:
        summary += (
            f" concentration_skill={metrics['concentration_skill_score_vs_forecast']:.6g}"
            f" concentration_improvement_rate={metrics['concentration_improvement_rate']:.6g}"
        )
    if "learned_sigma_mean" in metrics:
        summary += f" sigma_mean={metrics['learned_sigma_mean']:.6g}"
    if "nll_weight_mean" in metrics:
        summary += (
            f" nll_weight_mean={metrics['nll_weight_mean']:.4g}"
            f" nll_weight_pos={metrics['nll_weight_positive_frac']:.3g}"
        )
    return summary


def _pollutant_uses_log1p(dataloader) -> bool:
    dataset = getattr(dataloader, "dataset", None)
    data = getattr(dataset, "data", None)
    transform = getattr(data, "pollutant_log_transform", "log1p")
    return transform == "log1p"


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: str = "cuda",
    epoch: int | None = None,
    log_every: int = 10,
    loss_name: str = "mse",
    use_amp: bool = False,
    grad_scaler=None,
    accumulate_grad_batches: int = 1,
    nll_weight_config: CamsAqiNllWeightConfig | None = None,
    huber_beta: float = DEFAULT_HUBER_BETA,
    physical_tail_config: PhysicalTailLossConfig | None = None,
) -> dict[str, float]:
    torch = _require_torch()
    model.train()
    objective_sum = 0.0
    log_space_sum = 0.0
    physical_tail_sum = 0.0
    physical_tail_frac_sum = 0.0
    weight_sum = 0.0
    weight_positive_sum = 0.0
    n_batches = 0
    total_data_time = 0.0
    total_step_time = 0.0
    metrics_accumulator = RegressionMetrics(target_std=_target_std_from_dataloader(dataloader))
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    scaler = grad_scaler
    accumulate_grad_batches = max(1, int(accumulate_grad_batches))

    logger.info("Starting training epoch%s", f" {epoch}" if epoch is not None else "")
    optimizer.zero_grad(set_to_none=True)
    batch_wait_start = time.perf_counter()
    n_loader_batches = len(dataloader)
    for batch in dataloader:
        data_ready_time = time.perf_counter()
        total_data_time += data_ready_time - batch_wait_start
        step_start = time.perf_counter()

        cams = batch["cams_forecast"].to(device, non_blocking=True)
        era5 = batch["era5"].to(device, non_blocking=True)
        target = batch["target_error"].to(device, non_blocking=True)
        weight = (
            nll_weight_config.weights_for(cams, target=target)
            if nll_weight_config is not None and loss_name in WEIGHTED_LOSS_NAMES
            else None
        )

        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            output = model(cams, era5)
            loss, prediction, _ = prediction_loss(
                output, target, loss_name, weight=weight, huber_beta=huber_beta
            )
        log_space_loss = loss
        physical_loss = None
        tail_frac = None
        if (
            physical_tail_config is not None
            and physical_tail_config.lambda_weight > 0
        ):
            if nll_weight_config is None:
                raise ValueError(
                    "Physical tail loss requires CAMS AQI weight config with target_error stats."
                )
            physical_loss = physical_tail_concentration_loss(
                prediction.float(),
                cams.float(),
                target.float(),
                nll_weight_config,
                physical_tail_config,
            )
            with torch.no_grad():
                tail_mask = nll_weight_config.analysis_aqi_ge_mask(
                    cams.float(),
                    target.float(),
                    physical_tail_config.min_aqi_level,
                )
                tail_frac = tail_mask.mean()
            loss = log_space_loss.float() + physical_tail_config.lambda_weight * physical_loss
        scaled_loss = loss / accumulate_grad_batches
        if scaler is not None and use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        n_batches += 1
        should_step = (
            n_batches % accumulate_grad_batches == 0
            or n_batches == n_loader_batches
        )
        if should_step:
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metrics_accumulator.update(prediction.detach(), target)
        objective_sum += float(loss.detach())
        log_space_sum += float(log_space_loss.detach())
        if physical_loss is not None:
            physical_tail_sum += float(physical_loss.detach())
            physical_tail_frac_sum += float(tail_frac.detach())
        if weight is not None:
            weight_sum += float(weight.detach().float().mean())
            weight_positive_sum += float((weight.detach() > 0).float().mean())
        step_elapsed = time.perf_counter() - step_start
        total_step_time += step_elapsed
        if log_every > 0 and n_batches % log_every == 0:
            metrics = metrics_accumulator.compute()
            logger.info(
                "Epoch%s train batch %d/%d %s sec/batch=%.2f data_sec/batch=%.2f step_sec/batch=%.2f",
                f" {epoch}" if epoch is not None else "",
                n_batches,
                n_loader_batches,
                _format_metrics(metrics),
                (total_data_time + total_step_time) / n_batches,
                total_data_time / n_batches,
                total_step_time / n_batches,
            )
        batch_wait_start = time.perf_counter()

    metrics = metrics_accumulator.compute()
    metrics["objective_loss"] = objective_sum / max(n_batches, 1)
    metrics["log_space_loss"] = log_space_sum / max(n_batches, 1)
    if physical_tail_config is not None and physical_tail_config.lambda_weight > 0:
        metrics["physical_tail_loss"] = physical_tail_sum / max(n_batches, 1)
        metrics["physical_tail_frac"] = physical_tail_frac_sum / max(n_batches, 1)
    if nll_weight_config is not None:
        metrics["nll_weight_mean"] = weight_sum / max(n_batches, 1)
        metrics["nll_weight_positive_frac"] = weight_positive_sum / max(n_batches, 1)
    logger.info(
        "Finished training epoch%s %s sec/batch=%.2f data_sec/batch=%.2f step_sec/batch=%.2f",
        f" {epoch}" if epoch is not None else "",
        _format_metrics(metrics),
        (total_data_time + total_step_time) / max(n_batches, 1),
        total_data_time / max(n_batches, 1),
        total_step_time / max(n_batches, 1),
    )
    return metrics


def evaluate(
    model,
    dataloader,
    device: str = "cuda",
    stage: str = "eval",
    log_every: int = 10,
    include_concentration_metrics: bool = True,
    loss_name: str = "mse",
    use_amp: bool = False,
    huber_beta: float = DEFAULT_HUBER_BETA,
) -> dict[str, float]:
    torch = _require_torch()
    model.eval()
    n_batches = 0
    total_data_time = 0.0
    total_step_time = 0.0
    metrics_accumulator = RegressionMetrics(target_std=_target_std_from_dataloader(dataloader))
    lead_hours = get_error_modelling_settings().lead_hours
    concentration_accumulator = ConcentrationMetricsAccumulator() if include_concentration_metrics else None
    lead_concentration_accumulators = (
        [ConcentrationMetricsAccumulator() for _ in lead_hours] if include_concentration_metrics else []
    )
    dataset_stats = getattr(getattr(dataloader, "dataset", None), "stats", None)
    if include_concentration_metrics and dataset_stats is None:
        raise ValueError("Concentration metrics require dataloader.dataset.stats.")
    invert_log1p = _pollutant_uses_log1p(dataloader)
    objective_sum = 0.0
    coverage_counts = {level: 0 for level in (0.5, 0.6, 0.7, 0.8)}
    probabilistic_n = 0
    sigma_sum = 0.0
    sigma_count = 0
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    logger.info("Starting %s evaluation", stage)
    with torch.no_grad():
        batch_wait_start = time.perf_counter()
        for batch in dataloader:
            data_ready_time = time.perf_counter()
            total_data_time += data_ready_time - batch_wait_start
            step_start = time.perf_counter()

            cams = batch["cams_forecast"].to(device, non_blocking=True)
            era5 = batch["era5"].to(device, non_blocking=True)
            target = batch["target_error"].to(device, non_blocking=True)
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                output = model(cams, era5)
                objective, prediction, log_variance = prediction_loss(
                    output, target, loss_name, huber_beta=huber_beta
                )
            objective_sum += float(objective.detach())
            if log_variance is not None:
                sigma = torch.exp(
                    0.5 * torch.clamp(
                        log_variance, LOG_VARIANCE_MIN, LOG_VARIANCE_MAX
                    )
                )
                sigma_sum += float(sigma.detach().float().sum())
                sigma_count += int(sigma.numel())
                normal = torch.distributions.Normal(0.0, 1.0)
                for level in coverage_counts:
                    z = normal.icdf(
                        torch.tensor(
                            (1.0 + level) / 2.0,
                            device=sigma.device,
                            dtype=sigma.dtype,
                        )
                    )
                    coverage_counts[level] += int(
                        ((target >= prediction - z * sigma)
                         & (target <= prediction + z * sigma)).sum()
                    )
                probabilistic_n += target.numel()

            metrics_accumulator.update(prediction, target)
            if concentration_accumulator is not None:
                forecast_space = dataset_stats.cams_forecast.denormalise(cams)
                true_residual_space = dataset_stats.target_error.denormalise(target)
                pred_residual_space = dataset_stats.target_error.denormalise(prediction)
                analysis_space = forecast_space + true_residual_space
                corrected_space = forecast_space + pred_residual_space

                if invert_log1p:
                    forecast = torch.expm1(forecast_space)
                    analysis = torch.expm1(analysis_space)
                    corrected = torch.expm1(corrected_space)
                else:
                    forecast = forecast_space
                    analysis = analysis_space
                    corrected = corrected_space
                concentration_accumulator.update(forecast, analysis, corrected)
                if forecast.shape[2] == len(lead_concentration_accumulators):
                    for lead_index, lead_accumulator in enumerate(lead_concentration_accumulators):
                        lead_accumulator.update(
                            forecast[:, :, lead_index],
                            analysis[:, :, lead_index],
                            corrected[:, :, lead_index],
                        )
            n_batches += 1
            step_elapsed = time.perf_counter() - step_start
            total_step_time += step_elapsed
            if log_every > 0 and n_batches % log_every == 0:
                metrics = metrics_accumulator.compute()
                logger.info(
                    "%s batch %d/%d %s sec/batch=%.2f data_sec/batch=%.2f step_sec/batch=%.2f",
                    stage,
                    n_batches,
                    len(dataloader),
                    _format_metrics(metrics),
                    (total_data_time + total_step_time) / n_batches,
                    total_data_time / n_batches,
                    total_step_time / n_batches,
                )
            batch_wait_start = time.perf_counter()

    metrics = metrics_accumulator.compute()
    metrics["objective_loss"] = objective_sum / max(n_batches, 1)
    if probabilistic_n:
        for level, count in coverage_counts.items():
            metrics[f"gaussian_coverage_{int(level * 100)}"] = count / probabilistic_n
        if sigma_count:
            metrics["learned_sigma_mean"] = sigma_sum / sigma_count
        if isinstance(model, LearnedStdErrorModel) and model.std_mode != "spatial":
            for index, sigma in enumerate(
                model.learned_sigma_normalised().cpu().tolist()
            ):
                metrics[f"learned_sigma_normalised_{index}"] = float(sigma)
    if concentration_accumulator is not None:
        concentration_metrics = concentration_accumulator.compute().as_dict(prefix="concentration")
        metrics.update(concentration_metrics)
        for lead_hour, lead_accumulator in zip(lead_hours, lead_concentration_accumulators):
            metrics.update(
                lead_accumulator.compute().as_dict(prefix=f"concentration_t_plus_{lead_hour}h")
            )
    logger.info(
        "Finished %s evaluation %s sec/batch=%.2f data_sec/batch=%.2f step_sec/batch=%.2f",
        stage,
        _format_metrics(metrics),
        (total_data_time + total_step_time) / max(n_batches, 1),
        total_data_time / max(n_batches, 1),
        total_step_time / max(n_batches, 1),
    )
    return metrics


def fit_error_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    optimizer,
    device: str = "cuda",
    checkpoint_dir: str | Path | None = None,
    checkpoint_save_after: int = 0,
    checkpoint_save_every: int = 1,
    tensorboard_writer=None,
    start_epoch: int = 1,
    log_every: int = 10,
    loss_name: str = "mse",
    std_warmup_epochs: int = 0,
    freeze_backbone: bool = False,
    nll_weight_config: CamsAqiNllWeightConfig | None = None,
    checkpoint_metadata: Mapping | None = None,
    use_amp: bool = False,
    accumulate_grad_batches: int = 1,
    huber_beta: float = DEFAULT_HUBER_BETA,
    physical_tail_config: PhysicalTailLossConfig | None = None,
) -> list[dict[str, float]]:
    torch = _require_torch()
    model.to(device)
    history: list[dict[str, float]] = []
    if checkpoint_save_after < 0:
        raise ValueError("checkpoint_save_after must be >= 0.")
    if checkpoint_save_every < 1:
        raise ValueError("checkpoint_save_every must be >= 1.")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if start_epoch < 1:
        raise ValueError("start_epoch must be >= 1.")
    accumulate_grad_batches = max(1, int(accumulate_grad_batches))

    end_epoch = start_epoch + epochs - 1
    use_amp = bool(use_amp) and str(device).startswith("cuda")
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    logger.info(
        "Starting training for epochs %d-%d on device=%s amp=%s accumulate_grad_batches=%d",
        start_epoch,
        end_epoch,
        device,
        use_amp,
        accumulate_grad_batches,
    )
    for epoch in range(start_epoch, end_epoch + 1):
        if isinstance(model, LearnedStdErrorModel):
            if freeze_backbone:
                backbone_trainable = False
            else:
                warmup_last_epoch = start_epoch + std_warmup_epochs - 1
                backbone_trainable = epoch > warmup_last_epoch
            model.set_backbone_trainable(backbone_trainable)
            if freeze_backbone:
                logger.info("Epoch %d: backbone frozen (std head only)", epoch)
            elif std_warmup_epochs > 0:
                logger.info(
                    "Epoch %d: backbone trainable=%s (std warmup through epoch %d)",
                    epoch,
                    backbone_trainable,
                    warmup_last_epoch,
                )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            epoch=epoch,
            log_every=log_every,
            loss_name=loss_name,
            use_amp=use_amp,
            grad_scaler=grad_scaler,
            accumulate_grad_batches=accumulate_grad_batches,
            nll_weight_config=nll_weight_config,
            huber_beta=huber_beta,
            physical_tail_config=physical_tail_config,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            stage=f"val epoch {epoch}",
            log_every=log_every,
            loss_name=loss_name,
            use_amp=use_amp,
            huber_beta=huber_beta,
        )
        val_loss = val_metrics["objective_loss"]
        history.append({
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
        })
        if tensorboard_writer is not None:
            for name, value in train_metrics.items():
                tensorboard_writer.add_scalar(f"train/{name}", value, epoch)
            for name, value in val_metrics.items():
                tensorboard_writer.add_scalar(f"val/{name}", value, epoch)
            tensorboard_writer.flush()
        logger.info(
            "Epoch %d/%d complete: train_%s val_%s",
            epoch,
            end_epoch,
            _format_metrics(train_metrics),
            _format_metrics(val_metrics),
        )

        should_save_checkpoint = (
            checkpoint_dir is not None
            and epoch > checkpoint_save_after
            and (epoch - checkpoint_save_after) % checkpoint_save_every == 0
        )
        if should_save_checkpoint:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "loss_config": {
                    "name": loss_name,
                    "huber_beta": (
                        float(huber_beta)
                        if loss_name in {"huber", "aqi-weighted-huber"}
                        else None
                    ),
                    "aqi_thresholds": (
                        list(nll_weight_config.aqi_thresholds)
                        if nll_weight_config is not None
                        else None
                    ),
                    "aqi_weights": (
                        list(nll_weight_config.level_weights)
                        if nll_weight_config is not None
                        else None
                    ),
                    "aqi_weight_source": (
                        nll_weight_config.source
                        if nll_weight_config is not None
                        else None
                    ),
                    "nll_weight_by_cams_aqi": (
                        nll_weight_config is not None and loss_name == "gaussian_nll"
                    ),
                    "nll_aqi_weights": (
                        list(nll_weight_config.level_weights)
                        if nll_weight_config is not None and loss_name == "gaussian_nll"
                        else None
                    ),
                    "nll_weight_land_only": (
                        nll_weight_config is not None
                        and nll_weight_config.land_mask is not None
                    ),
                    "physical_tail": (
                        None
                        if physical_tail_config is None
                        else {
                            "min_aqi_level": int(physical_tail_config.min_aqi_level),
                            "lambda_weight": float(physical_tail_config.lambda_weight),
                            "kind": physical_tail_config.kind,
                            "huber_beta": float(physical_tail_config.huber_beta),
                        }
                    ),
                    "freeze_backbone": bool(freeze_backbone),
                },
            }
            if checkpoint_metadata:
                checkpoint.update(dict(checkpoint_metadata))
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved checkpoint to %s with val_loss=%.6g", checkpoint_path, val_loss)

    return history
