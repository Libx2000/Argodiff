"""Multi-step free-running fine-tuning for the deterministic GLORYS model.

The next model input is normally the previous forecast, not GLORYS truth.
Gradients are truncated between forecast days so a long rollout does not keep
all U-Net activations in memory.  Validation is nevertheless a fully free
30-day rollout and checkpoint selection uses several forecast lead times.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from contextlib import nullcontext
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from diffargo.config import load_config
from diffargo.data import GlorysRolloutDataset, load_stats
from diffargo.model import build_model
from diffargo.utils import (
    EMA, atomic_torch_save, cosine_lr, distributed_cleanup, distributed_setup,
    is_main_process, seed_everything, write_json,
)


def make_condition(
    states: list[torch.Tensor],
    valid_masks: list[torch.Tensor],
    anchor_ordinals: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    static_mask: torch.Tensor,
    spatial_features: torch.Tensor,
) -> torch.Tensor:
    normalized = [
        torch.where(valid, (state - state_mean) / state_std, 0.0)
        for state, valid in zip(states, valid_masks)
    ]
    angles = []
    for ordinal in anchor_ordinals.detach().cpu().tolist():
        current = date.fromordinal(int(ordinal))
        angles.append(2.0 * math.pi * (current.timetuple().tm_yday - 1) / 365.2425)
    angle = torch.tensor(angles, device=states[0].device, dtype=states[0].dtype)
    batch, _, height, width = states[0].shape
    seasonal = torch.stack((angle.sin(), angle.cos()), dim=1)
    seasonal = seasonal[:, :, None, None].expand(batch, 2, height, width)
    return torch.cat(
        normalized + [
            static_mask.expand(batch, -1, -1, -1),
            spatial_features.expand(batch, -1, -1, -1),
            seasonal,
        ],
        dim=1,
    )


def predict_correction(model, condition: torch.Tensor, channels: int) -> torch.Tensor:
    batch, _, height, width = condition.shape
    zeros = condition.new_zeros((batch, channels, height, width))
    timestep = torch.zeros(batch, device=condition.device, dtype=torch.long)
    return model(zeros, condition, timestep)


def channel_weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    latitude_weight: torch.Tensor,
    channel_weight: torch.Tensor,
) -> torch.Tensor:
    weight = valid.float() * latitude_weight
    numerator = (weight * (prediction - target).square()).sum(dim=(0, 2, 3))
    denominator = weight.sum(dim=(0, 2, 3)).clamp_min(1.0)
    per_channel = numerator / denominator
    return (per_channel * channel_weight).sum() / channel_weight.sum()


def physical_range_penalty(
    forecast: torch.Tensor,
    valid: torch.Tensor,
    levels: int,
    temperature_range: tuple[float, float],
    salinity_range: tuple[float, float],
    temperature_scale: float,
    salinity_scale: float,
    latitude_weight: torch.Tensor,
) -> torch.Tensor:
    penalties = []
    for channel_slice, bounds, scale in (
        (slice(0, levels), temperature_range, temperature_scale),
        (slice(levels, 2 * levels), salinity_range, salinity_scale),
    ):
        values = forecast[:, channel_slice]
        mask = valid[:, channel_slice].float() * latitude_weight
        violation = torch.relu(float(bounds[0]) - values) + torch.relu(values - float(bounds[1]))
        penalties.append(
            (mask * (violation / float(scale)).square()).sum() / mask.sum().clamp_min(1.0)
        )
    return 0.5 * (penalties[0] + penalties[1])


def teacher_forcing_probability(train_cfg: dict, epoch: int, epochs: int) -> float:
    start = float(train_cfg.get("teacher_forcing_start", 0.0))
    end = float(train_cfg.get("teacher_forcing_end", 0.0))
    fraction = float(train_cfg.get("teacher_forcing_decay_fraction", 1.0))
    decay_epochs = max(int(math.ceil(epochs * fraction)), 1)
    progress = min(epoch / max(decay_epochs - 1, 1), 1.0)
    return start + progress * (end - start)


def select_validation_indices(dataset: GlorysRolloutDataset, stride_days: int) -> list[int]:
    selected = []
    previous = None
    for index, anchor in enumerate(dataset.anchors):
        if previous is None or (anchor - previous).days >= stride_days:
            selected.append(index)
            previous = anchor
    if not selected:
        raise ValueError("No validation initializations selected")
    return selected


@torch.no_grad()
def validate_rollout(
    model: torch.nn.Module,
    ema: EMA,
    dataset: GlorysRolloutDataset,
    indices: list[int],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    tensors: dict[str, torch.Tensor],
    monitored_leads: list[int],
) -> dict[str, float]:
    # CPU backup avoids a second model-sized allocation on the GPU.
    backup = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    ema.copy_to(model)
    model.eval()
    horizon = dataset.rollout_days
    levels = dataset.levels
    # lead, variable(T/S), [forecast SSE, persistence SSE, weight]
    sums = torch.zeros((horizon, 2, 3), device=device, dtype=torch.float64)
    extremes = {
        "temperature_min": float("inf"), "temperature_max": float("-inf"),
        "salinity_min": float("inf"), "salinity_max": float("-inf"),
    }
    for index in indices:
        sample = dataset[index]
        states = [value.unsqueeze(0).to(device) for value in sample["context_states"]]
        masks = [value.unsqueeze(0).to(device) for value in sample["context_valid"]]
        truths = sample["truth_states"].to(device)
        truth_masks = sample["truth_valid"].to(device)
        initial_state = states[-1].clone()
        initial_valid = masks[-1].clone()
        initial_anchor = int(sample["anchor_ordinal"])
        for lead in range(1, horizon + 1):
            ordinals = torch.tensor([initial_anchor + lead - 1], device=device)
            condition = make_condition(
                states, masks, ordinals, tensors["state_mean"], tensors["state_std"],
                tensors["static_mask"], tensors["spatial_features"],
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                correction = predict_correction(model, condition, dataset.state_channels)
            forecast = states[-1] + correction.float() * tensors["increment_std"]
            forecast = torch.where(masks[-1], forecast, 0.0)
            truth = truths[lead - 1].unsqueeze(0)
            truth_valid = truth_masks[lead - 1].unsqueeze(0)
            common = masks[-1] & initial_valid & truth_valid
            weight = common.float() * tensors["latitude_weight"]
            for variable, channel_slice in enumerate((slice(0, levels), slice(levels, 2 * levels))):
                current_weight = weight[:, channel_slice]
                sums[lead - 1, variable, 0] += (
                    current_weight * (forecast[:, channel_slice] - truth[:, channel_slice]).square()
                ).sum().double()
                sums[lead - 1, variable, 1] += (
                    current_weight * (initial_state[:, channel_slice] - truth[:, channel_slice]).square()
                ).sum().double()
                sums[lead - 1, variable, 2] += current_weight.sum().double()
            valid_t = forecast[:, :levels][masks[-1][:, :levels]]
            valid_s = forecast[:, levels:][masks[-1][:, levels:]]
            extremes["temperature_min"] = min(extremes["temperature_min"], float(valid_t.min()))
            extremes["temperature_max"] = max(extremes["temperature_max"], float(valid_t.max()))
            extremes["salinity_min"] = min(extremes["salinity_min"], float(valid_s.min()))
            extremes["salinity_max"] = max(extremes["salinity_max"], float(valid_s.max()))
            states = states[1:] + [forecast.detach()]
            masks = masks[1:] + [masks[-1]]

    result: dict[str, float] = dict(extremes)
    ratios = []
    for lead in monitored_leads:
        for variable, name in enumerate(("temperature", "salinity")):
            forecast_rmse = torch.sqrt(sums[lead - 1, variable, 0] / sums[lead - 1, variable, 2])
            persistence_rmse = torch.sqrt(sums[lead - 1, variable, 1] / sums[lead - 1, variable, 2])
            ratio = forecast_rmse / persistence_rmse
            result[f"lead_{lead}_{name}_rmse"] = float(forecast_rmse)
            result[f"lead_{lead}_{name}_persistence_rmse"] = float(persistence_rmse)
            result[f"lead_{lead}_{name}_skill"] = float(1.0 - ratio)
            ratios.append(ratio)
    result["selection_score"] = float(torch.stack(ratios).mean())
    result["n_initializations"] = len(indices)
    for name, parameter in model.named_parameters():
        parameter.copy_(backup[name].to(device))
    model.train()
    return result


def build_constant_tensors(dataset, stats, device):
    lat_radians = np.deg2rad(dataset.lat)
    return {
        "state_mean": torch.as_tensor(stats["state_mean"], device=device).view(1, -1, 1, 1),
        "state_std": torch.as_tensor(stats["state_std"], device=device).view(1, -1, 1, 1),
        "increment_std": torch.as_tensor(stats["increment_std"], device=device).view(1, -1, 1, 1),
        "static_mask": torch.as_tensor(dataset.static_mask, device=device).unsqueeze(0),
        "spatial_features": torch.as_tensor(dataset.spatial_features, device=device).unsqueeze(0),
        "latitude_weight": torch.as_tensor(np.cos(lat_radians), device=device).view(1, 1, -1, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GLORYS model with free rollouts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, help="Override config epochs")
    parser.add_argument("--output-dir", help="Override config output directory")
    parser.add_argument("--max-train-batches", type=int, help="Smoke test only")
    parser.add_argument("--max-val-initializations", type=int, help="Smoke test only")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("epochs must be positive")
        train_cfg["epochs"] = args.epochs
    if args.output_dir:
        train_cfg["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    if train_cfg.get("objective") != "rollout_finetune":
        raise ValueError("Expected train.objective: rollout_finetune")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max-train-batches must be positive")
    if args.max_val_initializations is not None and args.max_val_initializations < 1:
        parser.error("max-val-initializations must be positive")

    distributed, rank, local_rank, world_size = distributed_setup()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    seed_everything(int(cfg["seed"]), rank)
    stats = load_stats(cfg["data"]["stats_path"])
    rollout_days = int(train_cfg["rollout_days"])
    validation_days = int(train_cfg["validation_horizon_days"])
    train_dataset = GlorysRolloutDataset(
        cfg["data"], "train", rollout_days, stats,
        stride_days=int(train_cfg.get("trajectory_stride_days", 1)),
    )
    val_dataset = GlorysRolloutDataset(cfg["data"], "val", validation_days, stats)
    val_indices = select_validation_indices(
        val_dataset, int(train_cfg["validation_stride_days"])
    )
    if args.max_val_initializations is not None:
        val_indices = val_indices[:args.max_val_initializations]

    batch_size = int(train_cfg["batch_size"])
    if batch_size != 1:
        raise ValueError("Use batch_size=1 for memory-safe global rollout fine-tuning")
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(train_cfg["num_workers"]) > 0,
        drop_last=True,
    )

    channels = train_dataset.state_channels
    model = build_model(cfg, channels, train_dataset.condition_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    accumulation = int(train_cfg["gradient_accumulation"])
    epochs = int(train_cfg["epochs"])
    total_updates = max(epochs * math.ceil(len(train_loader) / accumulation), 1)
    warmup_updates = min(int(train_cfg.get("warmup_updates", 200)), max(total_updates // 10, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lr(step, total_updates, warmup_updates)
    )
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if train_cfg["amp_dtype"] == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    start_epoch = 0
    global_step = 0
    best_score = float("inf")
    best_epoch = 0
    resume_path = train_cfg.get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("objective") != "rollout_finetune":
            raise ValueError("Resume checkpoint is not a rollout_finetune checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
    else:
        source = torch.load(train_cfg["init_checkpoint"], map_location="cpu", weights_only=False)
        if source.get("objective") not in {"direct_increment", "rollout_finetune"}:
            raise ValueError("Initialization checkpoint is not deterministic")
        if source.get("smoke_test", False):
            raise ValueError("Refusing to initialize from a smoke-test checkpoint")
        if int(source["target_channels"]) != channels:
            raise ValueError("Initialization checkpoint target channels differ")
        if int(source["condition_channels"]) != train_dataset.condition_channels:
            raise ValueError("Initialization checkpoint condition channels differ")
        model.load_state_dict(source["model"], strict=True)
        source_ema = EMA(model, float(train_cfg["ema_decay"]))
        source_ema.load_state_dict(source["ema"])
        source_ema.copy_to(model)
        del source, source_ema

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    raw_model = model.module if distributed else model
    ema = EMA(raw_model, float(train_cfg["ema_decay"]))
    if resume_path:
        ema.load_state_dict(checkpoint["ema"])
        del checkpoint
    tensors = build_constant_tensors(train_dataset, stats, device)

    depth = torch.as_tensor(np.abs(train_dataset.depth), device=device)
    level_weight = torch.where(
        depth >= float(train_cfg["deep_threshold_m"]),
        float(train_cfg["deep_loss_weight"]), 1.0,
    )
    channel_weight = torch.cat((
        level_weight,
        level_weight * float(train_cfg["salinity_loss_weight"]),
    ))
    temperature_range = tuple(float(v) for v in train_cfg["temperature_soft_range"])
    salinity_range = tuple(float(v) for v in train_cfg["salinity_soft_range"])
    monitored_leads = sorted(set(int(v) for v in train_cfg["validation_leads"]))
    if monitored_leads[-1] > validation_days:
        raise ValueError("validation_leads exceed validation_horizon_days")

    output_dir = Path(train_cfg["output_dir"])
    writer = None
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(cfg, output_dir / "resolved_config.json")
        writer = SummaryWriter(output_dir / "tensorboard") if SummaryWriter else None
        print(
            f"device={device}, world_size={world_size}, parameters="
            f"{sum(p.numel() for p in raw_model.parameters()):,}", flush=True,
        )
        print(
            f"train_trajectories={len(train_dataset):,}, rollout={rollout_days}d, "
            f"val_initializations={len(val_indices)}, validation={validation_days}d",
            flush=True,
        )

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        teacher_probability = teacher_forcing_probability(train_cfg, epoch, epochs)
        lead_weights = torch.linspace(
            1.0, float(train_cfg["lead_weight_end"]), rollout_days, device=device
        )
        lead_weight_sum = float(lead_weights.sum())
        running_loss = 0.0
        processed = 0
        epoch_start = time.time()
        progress = tqdm(
            train_loader, disable=not is_main_process(),
            desc=f"rollout epoch {epoch + 1}/{epochs}", dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            context_states = [value.to(device, non_blocking=True) for value in batch["context_states"].unbind(1)]
            context_valid = [value.to(device, non_blocking=True) for value in batch["context_valid"].unbind(1)]
            truths = batch["truth_states"].to(device, non_blocking=True)
            truth_valid = batch["truth_valid"].to(device, non_blocking=True)
            anchor_ordinals = batch["anchor_ordinal"].to(device)
            trajectory_loss = 0.0
            loader_length = (
                min(len(train_loader), args.max_train_batches)
                if args.max_train_batches is not None else len(train_loader)
            )
            should_update = ((batch_index + 1) % accumulation == 0 or batch_index + 1 == loader_length)
            for lead in range(1, rollout_days + 1):
                current_ordinals = anchor_ordinals + lead - 1
                condition = make_condition(
                    context_states, context_valid, current_ordinals,
                    tensors["state_mean"], tensors["state_std"],
                    tensors["static_mask"], tensors["spatial_features"],
                )
                truth = truths[:, lead - 1]
                target_valid = truth_valid[:, lead - 1]
                valid = context_valid[-1] & target_valid
                synchronize = should_update and lead == rollout_days
                sync_context = nullcontext() if synchronize or not distributed else model.no_sync()
                with sync_context:
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        correction = predict_correction(model, condition, channels)
                        target_correction = (truth - context_states[-1]) / tensors["increment_std"]
                        forecast_loss = channel_weighted_loss(
                            correction, target_correction, valid,
                            tensors["latitude_weight"], channel_weight,
                        )
                        forecast = context_states[-1] + correction * tensors["increment_std"]
                        range_loss = physical_range_penalty(
                            forecast, valid, train_dataset.levels,
                            temperature_range, salinity_range,
                            float(train_cfg["temperature_penalty_scale"]),
                            float(train_cfg["salinity_penalty_scale"]),
                            tensors["latitude_weight"],
                        )
                        loss = (
                            lead_weights[lead - 1] / lead_weight_sum
                            * (forecast_loss + float(train_cfg["physical_penalty_weight"]) * range_loss)
                        )
                    scaler.scale(loss / accumulation).backward()
                trajectory_loss += float(loss.detach())
                forecast = torch.where(context_valid[-1], forecast.float(), 0.0).detach()
                use_truth = lead < rollout_days and random.random() < teacher_probability
                next_state = truth.detach() if use_truth else forecast
                next_valid = target_valid if use_truth else context_valid[-1]
                context_states = context_states[1:] + [next_state]
                context_valid = context_valid[1:] + [next_valid]

            running_loss += trajectory_loss
            processed += 1
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), float(train_cfg["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(raw_model)
                global_step += 1
                if writer and global_step % int(train_cfg["log_every_steps"]) == 0:
                    writer.add_scalar("train/rollout_loss", trajectory_loss, global_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
            if is_main_process():
                progress.set_postfix(loss=f"{trajectory_loss:.4f}", tf=f"{teacher_probability:.2f}")

        if distributed:
            dist.barrier()
        result = None
        if is_main_process():
            result = validate_rollout(
                raw_model, ema, val_dataset, val_indices, device,
                amp_enabled, amp_dtype, tensors, monitored_leads,
            )
            score = result["selection_score"]
            if not math.isfinite(score):
                raise RuntimeError("Non-finite validation score")
            improved = score < best_score
            if improved:
                best_score = score
                best_epoch = epoch + 1
            print(
                f"epoch={epoch + 1} train_loss={running_loss / max(processed, 1):.6f} "
                f"score={score:.6f} best_epoch={best_epoch} "
                f"L7_T={result['lead_7_temperature_skill']:.2%} "
                f"L7_S={result['lead_7_salinity_skill']:.2%} "
                f"L30_T={result['lead_30_temperature_skill']:.2%} "
                f"L30_S={result['lead_30_salinity_skill']:.2%}",
                flush=True,
            )
            if writer:
                writer.add_scalar("validation/selection_score", score, epoch + 1)
                for key, value in result.items():
                    if key.endswith("_skill"):
                        writer.add_scalar(f"validation/{key}", value, epoch + 1)
                writer.add_scalar("train/epoch_seconds", time.time() - epoch_start, epoch + 1)
            payload = {
                "objective": "rollout_finetune",
                "source_checkpoint": str(train_cfg["init_checkpoint"]),
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "smoke_test": args.max_train_batches is not None or args.max_val_initializations is not None,
                "validation": result,
                "config": cfg,
                "target_channels": channels,
                "condition_channels": train_dataset.condition_channels,
            }
            atomic_torch_save(payload, output_dir / "checkpoint_last.pt")
            if improved:
                best_payload = {
                    key: value for key, value in payload.items()
                    if key not in {"optimizer", "scheduler", "scaler"}
                }
                atomic_torch_save(best_payload, output_dir / "checkpoint_best.pt")

        state = torch.zeros(3, device=device, dtype=torch.float64)
        if is_main_process():
            state[:] = torch.tensor([best_score, best_epoch, result["selection_score"]], device=device)
        if distributed:
            dist.broadcast(state, src=0)
        best_score, best_epoch = float(state[0]), int(state[1])
        patience = int(train_cfg.get("early_stop_patience", 0))
        stop = patience > 0 and epoch + 1 - best_epoch >= patience
        if distributed:
            dist.barrier()
        if stop:
            if is_main_process():
                print(f"early stop after {patience} epochs without improvement", flush=True)
            break

    if writer:
        writer.close()
    train_dataset.archive.close()
    val_dataset.archive.close()
    distributed_cleanup()


if __name__ == "__main__":
    main()
