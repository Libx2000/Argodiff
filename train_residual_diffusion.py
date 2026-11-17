"""Train a diffusion model for residuals around an existing deterministic forecast.

The deterministic checkpoint is frozen.  The diffusion target is
GLORYS(t+1) - deterministic_forecast(t+1), normalized with dedicated
training-set residual statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffargo.config import load_config
from diffargo.data import GlorysSequenceDataset, load_stats
from diffargo.diffusion import build_diffusion, masked_weighted_mse
from diffargo.model import build_model
from diffargo.utils import EMA, atomic_torch_save, seed_everything


@torch.no_grad()
def deterministic_forecast(model, condition, channels, persistence):
    z = condition.new_zeros((condition.shape[0], channels, condition.shape[-2], condition.shape[-1]))
    t = torch.zeros(condition.shape[0], device=condition.device, dtype=torch.long)
    return persistence + model(z, condition, t)


@torch.no_grad()
def validation_loss(
    model, deterministic_model, diffusion, loader, state_channels, inc_mean,
    inc_std, residual_mean, residual_std, latitude_weights, device, max_batches,
):
    """Reproducible validation noise loss on a bounded chronological subset."""
    model.eval()
    total = 0.0
    used = 0
    generator = torch.Generator(device=device.type)
    generator.manual_seed(20260928)
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        persistence = (-inc_mean / inc_std).expand(
            condition.shape[0], -1, condition.shape[-2], condition.shape[-1]
        )
        deterministic_normalized = deterministic_forecast(
            deterministic_model, condition, state_channels, persistence
        )
        residual_physical = (target - deterministic_normalized) * inc_std
        residual = (residual_physical - residual_mean) / residual_std
        residual = torch.where(mask.bool(), residual, torch.zeros_like(residual))
        timestep_value = (batch_index * 9973 + 17) % diffusion.timesteps
        timesteps = torch.full(
            (condition.shape[0],), timestep_value, device=device, dtype=torch.long
        )
        noise = torch.randn(
            residual.shape, generator=generator, device=device, dtype=residual.dtype
        )
        noisy, _ = diffusion.q_sample(residual, timesteps, noise=noise)
        prediction = model(noisy, condition, timesteps)
        total += masked_weighted_mse(prediction, noise, mask, latitude_weights).item()
        used += 1
    return total / max(used, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--deterministic-checkpoint", required=True)
    p.add_argument("--residual-stats", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-max-batches", type=int, default=366)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_stats(cfg["data"]["stats_path"])
    with np.load(args.residual_stats, allow_pickle=False) as archive:
        if not {"residual_mean", "residual_std"}.issubset(archive.files):
            raise KeyError("Residual statistics must contain residual_mean and residual_std")
        residual_mean_np = archive["residual_mean"].astype(np.float32)
        residual_std_np = archive["residual_std"].astype(np.float32)
    dataset = GlorysSequenceDataset(cfg["data"], "train", stats)
    validation_dataset = GlorysSequenceDataset(cfg["data"], "val", stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    det_ckpt = torch.load(args.deterministic_checkpoint, map_location=device, weights_only=False)
    det_cfg = dict(cfg)
    det_model = build_model(det_cfg, dataset.state_channels, dataset.condition_channels).to(device)
    det_model.load_state_dict(det_ckpt["model"])
    if "ema" in det_ckpt:
        det_ema = EMA(det_model, float(cfg["train"].get("ema_decay", 0.999)))
        det_ema.load_state_dict(det_ckpt["ema"])
        det_ema.copy_to(det_model)
        print("deterministic backbone: loaded EMA weights", flush=True)
    else:
        print("WARNING: deterministic checkpoint has no EMA; using raw weights", flush=True)
    det_model.eval()
    for q in det_model.parameters():
        q.requires_grad_(False)

    model = build_model(cfg, dataset.state_channels, dataset.condition_channels).to(device)
    diffusion = build_diffusion(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("learning_rate", 2e-4)),
                            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    ema = EMA(model, float(cfg["train"].get("ema_decay", 0.999)))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    inc_mean = torch.as_tensor(stats["increment_mean"], device=device).view(1, -1, 1, 1)
    inc_std = torch.as_tensor(stats["increment_std"], device=device).view(1, -1, 1, 1)
    if residual_mean_np.shape != (dataset.state_channels,) or residual_std_np.shape != (dataset.state_channels,):
        raise ValueError(
            f"Residual statistics have shapes {residual_mean_np.shape}/{residual_std_np.shape}; "
            f"expected ({dataset.state_channels},)"
        )
    if not np.all(np.isfinite(residual_std_np)) or np.any(residual_std_np <= 0):
        raise ValueError("Residual standard deviations must be finite and positive")
    residual_mean = torch.as_tensor(residual_mean_np, device=device).view(1, -1, 1, 1)
    residual_std = torch.as_tensor(residual_std_np, device=device).view(1, -1, 1, 1)
    latw = torch.as_tensor(dataset.latitude_weights, device=device)
    best_validation = float("inf")
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for batch in loader:
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            persistence = (-inc_mean / inc_std).expand(condition.shape[0], -1, condition.shape[-2], condition.shape[-1])
            with torch.no_grad():
                det_norm = deterministic_forecast(det_model, condition, dataset.state_channels, persistence)
                # Convert deterministic forecast error to physical units first,
                # then use residual-specific training-only normalization.
                residual_physical = (target - det_norm) * inc_std
                residual = (residual_physical - residual_mean) / residual_std
                residual = torch.where(mask.bool(), residual, torch.zeros_like(residual))
            t = torch.randint(0, diffusion.timesteps, (condition.shape[0],), device=device)
            noisy, noise = diffusion.q_sample(residual, t)
            pred = model(noisy, condition, t)
            loss = masked_weighted_mse(pred, noise, mask, latw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("gradient_clip_norm", 1.0)))
            opt.step(); ema.update(model); total += loss.detach().item()
        raw_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        ema.copy_to(model)
        val = validation_loss(
            model, det_model, diffusion, validation_loader, dataset.state_channels,
            inc_mean, inc_std, residual_mean, residual_std, latw, device,
            args.val_max_batches,
        )
        model.load_state_dict(raw_state)
        payload = {"epoch": epoch, "model": model.state_dict(), "ema": ema.state_dict(),
                   "config": cfg, "target_channels": dataset.state_channels,
                   "condition_channels": dataset.condition_channels,
                   "objective": "residual_diffusion", "source_checkpoint": str(Path(args.deterministic_checkpoint).resolve()),
                   "residual_stats": str(Path(args.residual_stats).resolve()),
                   "validation_loss": val}
        atomic_torch_save(payload, out / "checkpoint_last.pt")
        if val < best_validation:
            best_validation = val
            atomic_torch_save(payload, out / "checkpoint_best.pt")
        print(
            f"epoch={epoch+1}/{args.epochs} train_loss={total/max(len(loader),1):.6f} "
            f"val_loss={val:.6f} best_val={best_validation:.6f}", flush=True,
        )


if __name__ == "__main__":
    main()
