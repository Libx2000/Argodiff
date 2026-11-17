"""Train a persistence-anchored, deterministic one-day GLORYS forecaster.

The U-Net predicts a correction to zero physical increment. Unlike diffusion
noise loss, the training loss and checkpoint selection measure forecast skill.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from diffargo.config import load_config
from diffargo.data import GlorysSequenceDataset, load_stats
from diffargo.diffusion import masked_weighted_mse
from diffargo.model import build_model
from diffargo.utils import (
    EMA, atomic_torch_save, cosine_lr, distributed_cleanup, distributed_setup,
    is_main_process, seed_everything, write_json,
)


def direct_prediction(model, condition: torch.Tensor, channels: int,
                      persistence_normalized: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = condition.shape
    zeros = condition.new_zeros((batch, channels, height, width))
    timestep = torch.zeros(batch, device=condition.device, dtype=torch.long)
    return persistence_normalized + model(zeros, condition, timestep)


def physical_sums(prediction, target, mask, latitude_weight, increment_mean, increment_std):
    """Area-weighted SSE and weight sums; the same cells are used by both methods."""
    truth_increment = target * increment_std + increment_mean
    predicted_increment = prediction * increment_std + increment_mean
    weight = mask * latitude_weight
    error = predicted_increment - truth_increment
    midpoint = target.shape[1] // 2
    result = []
    for channels in (slice(0, midpoint), slice(midpoint, None)):
        current_weight = weight[:, channels]
        result.extend((
            (current_weight * error[:, channels].square()).sum(),
            (current_weight * truth_increment[:, channels].square()).sum(),
            current_weight.sum(),
        ))
    return torch.stack(result).double()


def score_from_sums(sums: torch.Tensor) -> dict[str, float]:
    values = sums.cpu().tolist()
    t_model = math.sqrt(values[0] / max(values[2], 1.0))
    t_persist = math.sqrt(values[1] / max(values[2], 1.0))
    s_model = math.sqrt(values[3] / max(values[5], 1.0))
    s_persist = math.sqrt(values[4] / max(values[5], 1.0))
    if t_persist == 0 or s_persist == 0:
        raise RuntimeError("Persistence validation RMSE is zero")
    return {
        "temperature_rmse": t_model,
        "temperature_persistence_rmse": t_persist,
        "salinity_rmse": s_model,
        "salinity_persistence_rmse": s_persist,
        "temperature_skill": 1.0 - t_model / t_persist,
        "salinity_skill": 1.0 - s_model / s_persist,
        "selection_score": 0.5 * (t_model / t_persist + s_model / s_persist),
    }


@torch.no_grad()
def validate_ema(model, ema, loader, device, amp_enabled, amp_dtype,
                 persistence_normalized, increment_mean, increment_std, max_batches=None):
    # Evaluate the EMA weights that will actually be used for inference.
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    ema.copy_to(model)
    model.eval()
    sums = torch.zeros(6, device=device, dtype=torch.float64)
    channels = persistence_normalized.shape[1]
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        latitude_weight = batch["latitude_weight"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction = direct_prediction(model, condition, channels, persistence_normalized)
        sums += physical_sums(
            prediction.float(), target, mask, latitude_weight,
            increment_mean, increment_std,
        )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    for name, parameter in model.named_parameters():
        parameter.copy_(original[name])
    model.train()
    return score_from_sums(sums)


def main():
    parser = argparse.ArgumentParser(description="Train deterministic GLORYS one-day forecast")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-train-batches", type=int, help="Smoke test only; never use for research training")
    parser.add_argument("--max-val-batches", type=int, help="Smoke test only; never use for research training")
    args = parser.parse_args()
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max-train-batches must be positive")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        parser.error("max-val-batches must be positive")
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    if train_cfg.get("objective") != "direct_increment":
        raise ValueError("Expected train.objective: direct_increment")

    distributed, rank, local_rank, world_size = distributed_setup()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    seed_everything(int(cfg["seed"]), rank)
    stats_path = Path(cfg["data"]["stats_path"])
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing {stats_path}; run compute_stats.py first")
    stats = load_stats(stats_path)
    train_dataset = GlorysSequenceDataset(cfg["data"], "train", stats)
    val_dataset = GlorysSequenceDataset(cfg["data"], "val", stats)
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
        drop_last=True,
    )
    # No duplicate padded validation samples across DDP ranks.
    val_subset = Subset(val_dataset, range(rank, len(val_dataset), world_size))
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )

    channels = train_dataset.state_channels
    model = build_model(cfg, channels, train_dataset.condition_channels).to(device)
    # The residual head starts at exactly zero, so epoch zero is persistence.
    torch.nn.init.zeros_(model.output_conv.conv.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    accumulation = int(train_cfg["gradient_accumulation"])
    epochs = int(train_cfg["epochs"])
    total_updates = max(epochs * math.ceil(len(train_loader) / accumulation), 1)
    warmup_updates = min(1000, max(total_updates // 20, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lr(step, total_updates, warmup_updates)
    )
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if train_cfg["amp_dtype"] == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    increment_mean = torch.as_tensor(stats["increment_mean"], device=device).view(1, -1, 1, 1)
    increment_std = torch.as_tensor(stats["increment_std"], device=device).view(1, -1, 1, 1)
    persistence_normalized = -increment_mean / increment_std

    start_epoch = 0
    global_step = 0
    best_score = float("inf")
    best_epoch = 0
    checkpoint = None
    if train_cfg.get("resume"):
        checkpoint = torch.load(train_cfg["resume"], map_location=device, weights_only=False)
        if checkpoint.get("objective") != "direct_increment":
            raise ValueError("Resume checkpoint has a different objective")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    raw_model = model.module if distributed else model
    # DDP broadcasts rank-0 parameters at construction. Initialize EMA only
    # afterward so every rank validates the same inference weights.
    ema = EMA(raw_model, float(train_cfg["ema_decay"]))
    if checkpoint is not None:
        ema.load_state_dict(checkpoint["ema"])
    output_dir = Path(train_cfg["output_dir"])
    writer = None
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(cfg, output_dir / "resolved_config.json")
        writer = SummaryWriter(output_dir / "tensorboard") if SummaryWriter else None
        print(f"device={device}, world_size={world_size}, parameters="
              f"{sum(p.numel() for p in raw_model.parameters()):,}", flush=True)
        print(f"train samples={len(train_dataset):,}, val samples={len(val_dataset):,}", flush=True)

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        processed_batches = 0
        epoch_start = time.time()
        progress = tqdm(train_loader, disable=not is_main_process(),
                        desc=f"direct epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            condition = batch["condition"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            latitude_weight = batch["latitude_weight"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                prediction = direct_prediction(model, condition, channels, persistence_normalized)
                loss = masked_weighted_mse(prediction, target, mask, latitude_weight)
            scaler.scale(loss / accumulation).backward()
            running_loss += float(loss.detach())
            processed_batches += 1
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(),
                                               float(train_cfg["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(raw_model)
                global_step += 1
                if writer and global_step % int(train_cfg["log_every_steps"]) == 0:
                    writer.add_scalar("train/normalized_increment_mse", loss.detach().item(), global_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
                if is_main_process():
                    progress.set_postfix(loss=f"{loss.detach().item():.4f}")

        result = validate_ema(raw_model, ema, val_loader, device, amp_enabled, amp_dtype,
                              persistence_normalized, increment_mean, increment_std,
                              args.max_val_batches)
        score = result["selection_score"]
        if not math.isfinite(score):
            raise RuntimeError("Non-finite validation score")
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch + 1
        if is_main_process():
            print(f"epoch={epoch + 1} train_loss={running_loss / max(processed_batches, 1):.6f} "
                  f"val_T={result['temperature_rmse']:.6f}/{result['temperature_persistence_rmse']:.6f} "
                  f"val_S={result['salinity_rmse']:.6f}/{result['salinity_persistence_rmse']:.6f} "
                  f"score={score:.6f} best_epoch={best_epoch}", flush=True)
            if writer:
                writer.add_scalar("validation/selection_score", score, epoch + 1)
                writer.add_scalar("validation/temperature_skill", result["temperature_skill"], epoch + 1)
                writer.add_scalar("validation/salinity_skill", result["salinity_skill"], epoch + 1)
                writer.add_scalar("train/epoch_seconds", time.time() - epoch_start, epoch + 1)
            # Only two checkpoint names: a resumable last checkpoint and a
            # lightweight inference-best checkpoint. No per-epoch accumulation.
            payload = {
                "objective": "direct_increment", "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "ema": ema.state_dict(),
                "epoch": epoch, "global_step": global_step,
                "best_score": best_score, "best_epoch": best_epoch,
                "smoke_test": args.max_train_batches is not None or args.max_val_batches is not None,
                "validation": result, "config": cfg,
                "target_channels": channels,
                "condition_channels": train_dataset.condition_channels,
            }
            atomic_torch_save(payload, output_dir / "checkpoint_last.pt")
            if improved:
                best_payload = {key: value for key, value in payload.items()
                                if key not in {"optimizer", "scheduler", "scaler"}}
                atomic_torch_save(best_payload, output_dir / "checkpoint_best.pt")
        patience = int(train_cfg.get("early_stop_patience", 0))
        if patience > 0 and epoch + 1 - best_epoch >= patience:
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
