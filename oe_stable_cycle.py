"""Versioned stability experiment; original frozen argo_guided_cycle.py untouched."""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from diffargo.config import load_config
from diffargo.data import GlorysSequenceDataset, load_stats, parse_date
from oe_stable_diffusion import build_diffusion
from diffargo.model import build_model
from diffargo.utils import EMA, seed_everything


def apply_checkpoint(model, checkpoint, ema_decay: float, label: str) -> None:
    model.load_state_dict(checkpoint["model"], strict=True)
    if "ema" in checkpoint:
        ema = EMA(model, ema_decay)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
        print(f"{label}: loaded EMA weights", flush=True)
    else:
        print(f"WARNING: {label} checkpoint has no EMA; using raw weights", flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_residual_stats(path: str | Path, channels: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not {"residual_mean", "residual_std"}.issubset(archive.files):
            raise KeyError("Residual statistics must contain residual_mean and residual_std")
        mean = archive["residual_mean"].astype(np.float32)
        std = archive["residual_std"].astype(np.float32)
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError(f"Residual statistics must both have shape ({channels},)")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Residual statistics contain invalid values")
    return mean, std


def load_spread_factors(path: str | Path | None, channels: int) -> np.ndarray:
    if path is None:
        return np.ones(channels, dtype=np.float32)
    with np.load(path, allow_pickle=False) as archive:
        if "inflation_factor" not in archive.files:
            raise KeyError("Spread calibration must contain inflation_factor")
        factors = archive["inflation_factor"].astype(np.float32)
    if factors.ndim == 1 and factors.shape != (channels,):
        raise ValueError(f"Spread factors must have shape ({channels},), got {factors.shape}")
    if factors.ndim == 2 and factors.shape[1] != channels:
        raise ValueError(
            f"Lead-dependent spread factors must have shape (lead, {channels}), "
            f"got {factors.shape}"
        )
    if factors.ndim not in (1, 2):
        raise ValueError("Spread factors must be one- or two-dimensional")
    if not np.all(np.isfinite(factors)) or np.any(factors < 1.0):
        raise ValueError("Spread factors must be finite and >= 1")
    return factors


def load_obs(
    path: Path, levels: int, height: int, width: int, lat, lon,
    temperature_bounds, salinity_bounds,
):
    values = np.zeros((2 * levels, height, width), np.float32)
    mask = np.zeros_like(values, dtype=bool)
    with np.load(path, allow_pickle=False) as archive:
        if "latitude" in archive and not np.allclose(archive["latitude"], lat, atol=1e-5):
            raise ValueError(f"Argo latitude grid differs: {path}")
        if "longitude" in archive and not np.allclose(archive["longitude"], lon, atol=1e-5):
            raise ValueError(f"Argo longitude grid differs: {path}")
        grid_i = archive["grid_i"].astype(np.int64)
        grid_j = archive["grid_j"].astype(np.int64)
        rejected = 0
        for profile, (i, j) in enumerate(zip(grid_i, grid_j)):
            if not (0 <= i < height and 0 <= j < width):
                rejected += 2 * levels
                continue
            temperature = archive["temperature"][profile]
            salinity = archive["salinity"][profile]
            temperature_valid = (
                np.isfinite(temperature)
                & (temperature >= temperature_bounds[0])
                & (temperature <= temperature_bounds[1])
            )
            salinity_valid = (
                np.isfinite(salinity)
                & (salinity >= salinity_bounds[0])
                & (salinity <= salinity_bounds[1])
            )
            rejected += int((~temperature_valid).sum() + (~salinity_valid).sum())
            values[:levels, i, j] = np.nan_to_num(temperature)
            values[levels:, i, j] = np.nan_to_num(salinity)
            mask[:levels, i, j] = temperature_valid
            mask[levels:, i, j] = salinity_valid
    return values, mask, rejected


def cycle_condition(dataset, states, anchor):
    normalized = [dataset._normalized_state(raw, valid) for raw, valid in states]
    angle = 2.0 * np.pi * (anchor.timetuple().tm_yday - 1) / 365.2425
    shape = (len(dataset.lat), len(dataset.lon))
    seasonal = np.stack((np.full(shape, np.sin(angle), np.float32),
                         np.full(shape, np.cos(angle), np.float32)))
    return np.concatenate(
        normalized + [dataset.static_mask, dataset.spatial_features, seasonal]
    ).astype(np.float32)


def physical_guard(array, levels, temperature_bounds, salinity_bounds):
    before = array.copy()
    array[..., :levels, :, :] = np.clip(array[..., :levels, :, :], *temperature_bounds)
    array[..., levels:, :, :] = np.clip(array[..., levels:, :, :], *salinity_bounds)
    finite = np.isfinite(before)
    changed = finite & (before != array)
    return array, int(changed.sum()), int(finite.sum())


def require_finite(name: str, tensor: torch.Tensor, valid_mask: torch.Tensor) -> None:
    selected = tensor[valid_mask.expand_as(tensor).bool()]
    bad = int((~torch.isfinite(selected)).sum().item())
    if bad:
        raise FloatingPointError(
            f"{name} contains {bad}/{selected.numel()} NaN/Inf values on ocean points"
        )


def masked_ensemble_moments(ensemble: np.ndarray, valid: np.ndarray):
    """Mean/std without applying nanmean to all-NaN land slices."""
    safe = np.where(valid[None], ensemble, 0.0)
    mean = safe.mean(axis=0, dtype=np.float64).astype(np.float32)
    spread = safe.std(axis=0, dtype=np.float64).astype(np.float32)
    return np.where(valid, mean, np.nan), np.where(valid, spread, np.nan)


def bounded_zero_mean_anomalies(
    anomalies: torch.Tensor,
    center: torch.Tensor,
    inflation: torch.Tensor,
    levels: int,
    temperature_bounds,
    salinity_bounds,
    ocean_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inflate anomalies without changing their mean or crossing physical bounds."""
    scaled = anomalies * inflation
    lower = torch.empty_like(center)
    upper = torch.empty_like(center)
    lower[:, :levels] = temperature_bounds[0]
    upper[:, :levels] = temperature_bounds[1]
    lower[:, levels:] = salinity_bounds[0]
    upper[:, levels:] = salinity_bounds[1]
    positive = scaled.clamp_min(0.0).amax(dim=0, keepdim=True)
    negative = (-scaled.clamp_max(0.0)).amax(dim=0, keepdim=True)
    ones = torch.ones_like(center)
    positive_ratio = torch.where(
        positive > 0, (upper - center).clamp_min(0.0) / positive.clamp_min(1e-12), ones
    )
    negative_ratio = torch.where(
        negative > 0, (center - lower).clamp_min(0.0) / negative.clamp_min(1e-12), ones
    )
    safe_ratio = torch.minimum(
        ones, torch.minimum(positive_ratio, negative_ratio)
    ).clamp(0.0, 1.0)
    safe_ratio = torch.where(ocean_mask.bool(), safe_ratio, torch.zeros_like(safe_ratio))
    bounded = scaled * safe_ratio
    bounded -= bounded.mean(dim=0, keepdim=True)
    return bounded, safe_ratio


def bounded_center(center, levels, temperature_bounds, salinity_bounds, ocean_mask):
    result = center.clone()
    before = result.clone()
    result[:, :levels].clamp_(*temperature_bounds)
    result[:, levels:].clamp_(*salinity_bounds)
    changed = ((result != before) & ocean_mask.bool()).sum().item()
    total = ocean_mask.sum().item()
    return result, int(changed), int(total)


def update_calibration_statistics(
    dataset, target, lead_index, analysis, spread, squared_error_sum,
    ensemble_variance_sum, weight_sum,
):
    truth, truth_valid = dataset.archive.read_state(target)
    valid = truth_valid & np.isfinite(analysis) & np.isfinite(spread)
    latitude_weight = dataset.latitude_weights.astype(np.float64)
    weight = valid * latitude_weight
    error = np.where(valid, analysis - truth, 0.0).astype(np.float64)
    variance = np.where(valid, spread, 0.0).astype(np.float64) ** 2
    squared_error_sum[lead_index] += (weight * error ** 2).sum(axis=(1, 2))
    ensemble_variance_sum[lead_index] += (weight * variance).sum(axis=(1, 2))
    weight_sum[lead_index] += weight.sum(axis=(1, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stability-options', help='Versioned physical-gradient JSON; omit for exact no-physics ablation')
    parser.add_argument("--config", required=True)
    parser.add_argument("--deterministic-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--residual-stats", required=True)
    parser.add_argument("--argo-root", help="Mapped daily NPZ root; omit for no-Argo control")
    parser.add_argument("--require-argo", action="store_true")
    parser.add_argument("--start", required=True, help="Last observed analysis day at cycle start")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--spread-calibration")
    parser.add_argument("--spread-inflation", type=float, default=1.0)
    parser.add_argument(
        "--individual-member-guidance", action="store_true",
        help="Run Argo-guided DDIM separately for every stochastic member",
    )
    parser.add_argument("--temperature-guidance", type=float, default=0.10)
    parser.add_argument("--salinity-guidance", type=float, default=0.02)
    parser.add_argument("--guidance-start", type=float, default=0.40)
    parser.add_argument("--max-temperature-correction", type=float, default=1.0)
    parser.add_argument("--max-salinity-correction", type=float, default=0.5)
    parser.add_argument("--temperature-bounds", nargs=2, type=float, default=(-3.0, 40.0))
    parser.add_argument("--salinity-bounds", nargs=2, type=float, default=(0.0, 42.0))
    parser.add_argument(
        "--save-daily", choices=("full", "analysis", "compact", "none"), default="full"
    )
    parser.add_argument(
        "--ensemble-storage-dtype", choices=("float16", "float32"), default="float32"
    )
    parser.add_argument("--calibration-statistics-output")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--nudging-gain", type=float, default=0.0)
    parser.add_argument("--member-mean-cycle", action="store_true",
                        help="Use independently guided member mean as the cycling analysis")
    parser.add_argument("--sampling-eta", type=float,
                        help="Explicit DDIM eta for reproducible ablations")
    parser.add_argument("--strict-calibration-leads", action="store_true")
    parser.add_argument("--paper-holdout-root")
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--paired-guidance-center", action="store_true",
                        help="Keep deterministic center plus paired Argo-only member-mean correction")
    parser.add_argument("--argo-stop-after", type=int,
                        help="Use Argo for this many days, then propagate ONLY the deterministic backbone; point-forecast experiment")
    args = parser.parse_args()
    if args.argo_stop_after is not None and not 0 <= args.argo_stop_after <= args.days:
        parser.error("--argo-stop-after must be within [0, days]")
    if args.members < 2:
        parser.error("--members must be at least 2")
    if not 0.0 <= args.residual_scale <= 2.0:
        parser.error("--residual-scale must be within [0, 2]")
    if args.spread_inflation < 1.0:
        parser.error("--spread-inflation must be >= 1")
    if not 0.0 <= args.guidance_start < 1.0:
        parser.error("--guidance-start must be within [0, 1)")
    if min(args.temperature_guidance, args.salinity_guidance) < 0:
        parser.error("guidance strengths must be non-negative")
    if args.require_argo and not args.argo_root:
        parser.error("--require-argo requires --argo-root")
    if not 0 <= args.nudging_gain <= 1:
        parser.error("--nudging-gain must lie in [0, 1]")
    if args.nudging_gain and not args.deterministic_only:
        parser.error("Nudging baseline requires --deterministic-only")
    if args.snapshot_every < 1:
        parser.error("--snapshot-every must be positive")
    if args.days < 1:
        parser.error("--days must be positive")
    if args.sampling_eta is not None and not 0 <= args.sampling_eta <= 1:
        parser.error("--sampling-eta must lie in [0, 1]")
    if args.member_mean_cycle and args.sampling_eta not in (None, 0.0):
        parser.error("Paired member-mean experiments require --sampling-eta 0")
    if args.paired_guidance_center and (not args.member_mean_cycle or not args.individual_member_guidance):
        parser.error("--paired-guidance-center requires --member-mean-cycle and --individual-member-guidance")

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]) if args.seed is None else args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_stats(cfg["data"]["stats_path"])
    dataset = GlorysSequenceDataset(cfg["data"], "test", stats)
    levels = dataset.levels
    height, width = len(dataset.lat), len(dataset.lon)
    ema_decay = float(cfg["train"].get("ema_decay", 0.999))
    residual_mean_np, residual_std_np = load_residual_stats(args.residual_stats, dataset.state_channels)
    spread_factors_np = load_spread_factors(args.spread_calibration, dataset.state_channels)
    if (args.strict_calibration_leads and spread_factors_np.ndim == 2
            and len(spread_factors_np) < args.days):
        raise ValueError("Calibration horizon is shorter than requested cycle; refit on 2018")
    sampling_eta = float(cfg["diffusion"].get("ddim_eta", 0.0)) if args.sampling_eta is None else args.sampling_eta
    if args.member_mean_cycle:
        sampling_eta = 0.0

    deterministic_model = build_model(cfg, dataset.state_channels, dataset.condition_channels).to(device)
    deterministic_checkpoint = torch.load(
        args.deterministic_checkpoint, map_location=device, weights_only=False
    )
    apply_checkpoint(deterministic_model, deterministic_checkpoint, ema_decay, "deterministic")
    residual_model = None
    if not args.deterministic_only:
        residual_model = build_model(cfg, dataset.state_channels, dataset.condition_channels).to(device)
        residual_checkpoint = torch.load(args.residual_checkpoint, map_location=device, weights_only=False)
        if residual_checkpoint.get("objective") != "residual_diffusion":
            raise ValueError("Residual checkpoint has an unexpected objective")
        apply_checkpoint(residual_model, residual_checkpoint, ema_decay, "residual diffusion")

    diffusion = build_diffusion(cfg).to(device)
    stability = None
    if args.stability_options:
        from oe_stability import StabilityOperator
        settings = json.loads(Path(args.stability_options).read_text(encoding='utf-8'))
        stability = StabilityOperator(dataset.archive.coordinates()['depth'], dataset.lat, dataset.lon, settings)
        diffusion.stability_operator = stability
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.paper_holdout_root and any((output_dir / name).exists() for name in
        ("run_summary.csv", "paper_grid_metrics.csv", "paper_holdout_metrics.csv")):
        raise FileExistsError("Paper run already exists; choose a new output directory")
    current = parse_date(args.start)
    history = [
        dataset.archive.read_state(current - timedelta(days=offset))
        for offset in reversed(range(dataset.context_days))
    ]
    increment_mean = torch.as_tensor(stats["increment_mean"], device=device).view(1, -1, 1, 1)
    increment_std = torch.as_tensor(stats["increment_std"], device=device).view(1, -1, 1, 1)
    residual_mean = torch.as_tensor(residual_mean_np, device=device).view(1, -1, 1, 1)
    residual_std = torch.as_tensor(residual_std_np, device=device).view(1, -1, 1, 1)
    persistence = (-increment_mean / increment_std).expand(1, -1, height, width)
    argo_root = Path(args.argo_root) if args.argo_root else None
    rows: list[dict] = []
    squared_error_by_lead = np.zeros((args.days, dataset.state_channels), dtype=np.float64)
    ensemble_variance_by_lead = np.zeros_like(squared_error_by_lead)
    calibration_weight_by_lead = np.zeros_like(squared_error_by_lead)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for lead in range(1, args.days + 1):
        day_started = time.perf_counter()
        free_point_forecast = args.argo_stop_after is not None and lead > args.argo_stop_after
        if spread_factors_np.ndim == 1:
            current_spread_factors = spread_factors_np
        else:
            current_spread_factors = spread_factors_np[
                min(lead - 1, spread_factors_np.shape[0] - 1)
            ]
        spread_inflation = torch.as_tensor(
            current_spread_factors * args.spread_inflation, device=device
        ).view(1, -1, 1, 1)
        target = current + timedelta(days=1)
        condition_np = cycle_condition(dataset, history, current)
        last_raw, last_valid = history[-1]
        condition = torch.from_numpy(condition_np).unsqueeze(0).to(device)
        ocean_mask = torch.from_numpy(last_valid.astype(np.float32)).unsqueeze(0).to(device)
        with torch.inference_mode():
            zeros = torch.zeros_like(persistence)
            time_zero = torch.zeros(1, device=device, dtype=torch.long)
            deterministic_normalized = persistence + deterministic_model(
                zeros, condition, time_zero
            )
            deterministic_increment = deterministic_normalized * increment_std + increment_mean
            deterministic = (
                torch.from_numpy(last_raw).unsqueeze(0).to(device) + deterministic_increment
            )
        require_finite("deterministic forecast", deterministic, ocean_mask)

        observation_file = (
            None if argo_root is None or free_point_forecast
            else argo_root / str(target.year) / f"{target.isoformat()}.npz"
        )
        obs_used = bool(observation_file and observation_file.is_file())
        if args.require_argo and not obs_used and not free_point_forecast:
            raise FileNotFoundError(f"Required Argo observation is missing: {observation_file}")
        observation_np = np.zeros((dataset.state_channels, height, width), np.float32)
        argo_mask_np = np.zeros_like(observation_np, dtype=bool)
        rejected_argo_values = 0
        argo_outside_model_ocean = 0
        if obs_used:
            observation_np, argo_mask_np, rejected_argo_values = load_obs(
                observation_file, levels, height, width, dataset.lat, dataset.lon,
                tuple(args.temperature_bounds), tuple(args.salinity_bounds),
            )
            # Argo may contain a valid value at a depth/grid cell represented
            # as land or below-bottom in GLORYS. Such a point has a NaN
            # background and must never enter the guidance likelihood.
            raw_argo_count = int(argo_mask_np.sum())
            argo_mask_np &= last_valid
            argo_outside_model_ocean = raw_argo_count - int(argo_mask_np.sum())

        stability_observation = torch.from_numpy(observation_np).unsqueeze(0).to(device)
        stability_observation_mask = torch.from_numpy(argo_mask_np).unsqueeze(0).to(device)
        guidance_correction = torch.zeros_like(deterministic)
        if obs_used and argo_mask_np.any() and not args.deterministic_only:
            observation = torch.from_numpy(observation_np).unsqueeze(0).to(device)
            observation_mask = torch.from_numpy(argo_mask_np.astype(np.float32)).unsqueeze(0).to(device)
            paired_noise = torch.zeros_like(deterministic)
            with torch.inference_mode():
                paired_unguided = diffusion.ddim_sample(
                    residual_model, condition, dataset.state_channels, mask=ocean_mask,
                    sampling_steps=args.sampling_steps, eta=0.0, initial_noise=paired_noise,
                )
            require_finite("paired unguided DDIM residual", paired_unguided, ocean_mask)
            paired_guided = diffusion.ddim_sample_observation_guided(
                residual_model, condition, dataset.state_channels,
                background_state=deterministic,
                residual_mean=residual_mean,
                residual_std=residual_std,
                observation=observation,
                observation_mask=observation_mask,
                levels=levels,
                temperature_guidance=args.temperature_guidance,
                salinity_guidance=args.salinity_guidance,
                guidance_start=args.guidance_start,
                residual_scale=args.residual_scale,
                mask=ocean_mask,
                sampling_steps=args.sampling_steps,
                eta=0.0,
                initial_noise=paired_noise,
            )
            require_finite("paired Argo-guided DDIM residual", paired_guided, ocean_mask)
            guidance_correction = args.residual_scale * (
                paired_guided - paired_unguided
            ) * residual_std
            guidance_correction[:, :levels].clamp_(
                -args.max_temperature_correction, args.max_temperature_correction
            )
            guidance_correction[:, levels:].clamp_(
                -args.max_salinity_correction, args.max_salinity_correction
            )
            require_finite("Argo diffusion correction", guidance_correction, ocean_mask)

        residual_members = []
        unguided_residual_members = []
        for _ in range(args.members):
            if args.deterministic_only or free_point_forecast:
                normalized_residual = torch.zeros_like(deterministic)
            elif args.individual_member_guidance and obs_used and argo_mask_np.any():
                member_noise = torch.randn_like(deterministic)
                if args.member_mean_cycle:
                    with torch.inference_mode():
                        paired_member = diffusion.ddim_sample(
                            residual_model, condition, dataset.state_channels, mask=ocean_mask,
                            sampling_steps=args.sampling_steps, eta=0.0,
                            initial_noise=member_noise,
                        )
                    unguided_residual_members.append(paired_member * residual_std + residual_mean)
                normalized_residual = diffusion.ddim_sample_observation_guided(
                    residual_model, condition, dataset.state_channels,
                    background_state=deterministic,
                    residual_mean=residual_mean,
                    residual_std=residual_std,
                    observation=observation,
                    observation_mask=observation_mask,
                    levels=levels,
                    temperature_guidance=args.temperature_guidance,
                    salinity_guidance=args.salinity_guidance,
                    guidance_start=args.guidance_start,
                    residual_scale=args.residual_scale,
                    mask=ocean_mask,
                    sampling_steps=args.sampling_steps,
                    eta=0.0,
                    initial_noise=member_noise,
                )
            else:
                with torch.inference_mode():
                    normalized_residual = diffusion.ddim_sample(
                        residual_model, condition, dataset.state_channels, mask=ocean_mask,
                        sampling_steps=args.sampling_steps,
                        eta=sampling_eta,
                    )
            require_finite("stochastic DDIM residual", normalized_residual, ocean_mask)
            residual_members.append(normalized_residual * residual_std + residual_mean)
        residual_members_tensor = torch.cat(residual_members, dim=0)
        if args.deterministic_only or free_point_forecast:
            residual_members_tensor.zero_()
        member_center = residual_members_tensor.mean(dim=0, keepdim=True)
        if args.member_mean_cycle and not args.deterministic_only and not free_point_forecast:
            if unguided_residual_members:
                unguided_member_center = torch.cat(unguided_residual_members).mean(dim=0, keepdim=True)
            else:
                unguided_member_center = member_center
            unguided_background = deterministic + args.residual_scale * unguided_member_center
            guided_background = deterministic + args.residual_scale * member_center
            if not args.individual_member_guidance:
                guided_background = guided_background + guidance_correction
            guidance_correction = guided_background - unguided_background
            if args.paired_guidance_center:
                unguided_background = deterministic
                guided_background = deterministic + guidance_correction
        else:
            unguided_background = deterministic
            guided_background = deterministic + guidance_correction
        if args.nudging_gain and obs_used:
            observation = torch.from_numpy(observation_np).unsqueeze(0).to(device)
            observation_mask = torch.from_numpy(argo_mask_np).unsqueeze(0).to(device)
            guidance_correction = torch.where(
                observation_mask, args.nudging_gain * (observation - deterministic),
                torch.zeros_like(deterministic),
            )
            guided_background = deterministic + guidance_correction
        residual_members_tensor -= residual_members_tensor.mean(dim=0, keepdim=True)
        raw_anomalies = args.residual_scale * residual_members_tensor
        unguided_center, unguided_clipped, unguided_finite = bounded_center(
            unguided_background, levels, tuple(args.temperature_bounds),
            tuple(args.salinity_bounds), ocean_mask,
        )
        analysis_center, analysis_clipped, analysis_finite = bounded_center(
            guided_background, levels, tuple(args.temperature_bounds),
            tuple(args.salinity_bounds), ocean_mask,
        )
        if stability is not None and stability.c['center']:
            # This additional center-state physical regularization is explicit.
            # It is NOT the Argo observation assimilation step, which stays above.
            analysis_center = stability.correct_tensor(
                analysis_center, ocean_mask, stability.c['center_iterations'],
                stability_observation, stability_observation_mask, 'center',
            )
            require_finite('physical center correction', analysis_center, ocean_mask)
        unguided_anomalies, unguided_safe_ratio = bounded_zero_mean_anomalies(
            raw_anomalies, unguided_center, spread_inflation, levels,
            tuple(args.temperature_bounds), tuple(args.salinity_bounds), ocean_mask,
        )
        analysis_anomalies, analysis_safe_ratio = bounded_zero_mean_anomalies(
            raw_anomalies, analysis_center, spread_inflation, levels,
            tuple(args.temperature_bounds), tuple(args.salinity_bounds), ocean_mask,
        )
        unguided_ensemble_tensor = unguided_center + unguided_anomalies
        analysis_ensemble_tensor = analysis_center + analysis_anomalies
        require_finite("inflated unguided ensemble", unguided_ensemble_tensor, ocean_mask)
        require_finite("inflated guided ensemble", analysis_ensemble_tensor, ocean_mask)

        deterministic_np = deterministic.cpu().numpy()[0]
        unguided_ensemble = unguided_ensemble_tensor.cpu().numpy()
        analysis_ensemble = analysis_ensemble_tensor.cpu().numpy()
        valid_members = last_valid[None]
        unguided_ensemble = np.where(valid_members, unguided_ensemble, np.nan).astype(np.float32)
        analysis_ensemble = np.where(valid_members, analysis_ensemble, np.nan).astype(np.float32)
        ensemble_unguided_mean, unguided_spread = masked_ensemble_moments(
            unguided_ensemble, last_valid
        )
        ensemble_analysis_mean, analysis_spread = masked_ensemble_moments(
            analysis_ensemble, last_valid
        )
        unguided_mean = np.where(
            last_valid, unguided_center.cpu().numpy()[0], np.nan
        ).astype(np.float32)
        analysis = np.where(
            last_valid, analysis_center.cpu().numpy()[0], np.nan
        ).astype(np.float32)
        mean_error = max(
            float(np.max(np.abs(ensemble_unguided_mean[last_valid] - unguided_mean[last_valid]))),
            float(np.max(np.abs(ensemble_analysis_mean[last_valid] - analysis[last_valid]))),
        )
        if mean_error > 2e-5:
            raise FloatingPointError(
                f"Spread inflation changed ensemble mean by {mean_error:.3e}"
            )
        correction_np = guidance_correction.cpu().numpy()[0].astype(np.float32)
        if not np.all(np.isfinite(analysis[last_valid])):
            raise FloatingPointError(f"Non-finite ocean analysis at {target}")

        if args.paper_holdout_root:
            from paper_metrics import evaluate_day
            evaluate_day(dataset, target, lead, analysis, analysis_ensemble,
                         args.paper_holdout_root, output_dir, args.deterministic_only or free_point_forecast)
        if stability is not None:
            stability.write_day(output_dir, lead, target)

        if args.save_daily != "none" and (lead == 1 or lead % args.snapshot_every == 0 or lead == args.days):
            daily = {
                "deterministic_forecast": deterministic_np.astype(np.float32),
                "unguided_mean": unguided_mean,
                "unguided_spread": unguided_spread,
                "analysis": analysis,
                "analysis_spread": analysis_spread,
                "diffusion_guidance_correction": correction_np,
                "argo_mask": argo_mask_np,
            }
            if args.save_daily == "full":
                daily["unguided_ensemble"] = unguided_ensemble
            if args.save_daily in {"full", "analysis"}:
                storage_type = np.float16 if args.ensemble_storage_dtype == "float16" else np.float32
                daily["analysis_ensemble"] = analysis_ensemble.astype(storage_type)
            np.savez_compressed(output_dir / f"{target.isoformat()}.npz", **daily)

        if args.calibration_statistics_output:
            update_calibration_statistics(
                dataset, target, lead - 1, analysis, analysis_spread,
                squared_error_by_lead, ensemble_variance_by_lead,
                calibration_weight_by_lead,
            )
            statistics_path = Path(args.calibration_statistics_output)
            statistics_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                statistics_path,
                squared_error_sum=squared_error_by_lead[:lead].sum(axis=0),
                ensemble_variance_sum=ensemble_variance_by_lead[:lead].sum(axis=0),
                weight_sum=calibration_weight_by_lead[:lead].sum(axis=0),
                squared_error_by_lead=squared_error_by_lead[:lead],
                ensemble_variance_by_lead=ensemble_variance_by_lead[:lead],
                weight_by_lead=calibration_weight_by_lead[:lead],
                completed_days=np.asarray(lead, dtype=np.int32),
                start_date=np.asarray(args.start),
                end_date=np.asarray(target.isoformat()),
                members=np.asarray(args.members, dtype=np.int32),
            )
        correction_ocean = correction_np[last_valid]
        row = {
            "date": target.isoformat(), "lead": lead, "argo_used": int(obs_used),
            "free_forecast_lead": lead - args.argo_stop_after if free_point_forecast else 0,
            "point_forecast_after_stop": int(free_point_forecast),
            "wall_seconds": time.perf_counter() - day_started,
            "argo_values": int(argo_mask_np.sum()), "residual_scale": args.residual_scale,
            "rejected_argo_values": rejected_argo_values,
            "argo_outside_model_ocean": argo_outside_model_ocean,
            "temperature_guidance": args.temperature_guidance,
            "salinity_guidance": args.salinity_guidance,
            "guidance_start": args.guidance_start,
            "spread_inflation_mean": float(spread_inflation.mean().item()),
            "spread_inflation_min": float(spread_inflation.min().item()),
            "spread_inflation_max": float(spread_inflation.max().item()),
            "individual_member_guidance": int(args.individual_member_guidance),
            "spread_bound_limited_fraction": float(
                ((analysis_safe_ratio < 0.999999) & ocean_mask.bool()).sum().item()
                / max(ocean_mask.sum().item(), 1)
            ),
            "ensemble_mean_error_max": mean_error,
            "correction_rms": float(np.sqrt(np.mean(correction_ocean ** 2))),
            "correction_abs_max": float(np.max(np.abs(correction_ocean))),
            "temperature_min": float(np.nanmin(analysis[:levels])),
            "temperature_max": float(np.nanmax(analysis[:levels])),
            "salinity_min": float(np.nanmin(analysis[levels:])),
            "salinity_max": float(np.nanmax(analysis[levels:])),
            "temperature_spread_mean": float(np.nanmean(analysis_spread[:levels])),
            "salinity_spread_mean": float(np.nanmean(analysis_spread[levels:])),
            "unguided_clipped_fraction": unguided_clipped / max(unguided_finite, 1),
            "analysis_clipped_fraction": analysis_clipped / max(analysis_finite, 1),
        }
        rows.append(row)
        with (output_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)

        next_valid = np.isfinite(analysis) & last_valid
        history = (history + [(analysis, next_valid)])[1:]
        current = target
        print(
            f"{target} lead={lead} obs={obs_used} values={row['argo_values']} "
            f"correction_rms={row['correction_rms']:.6f} "
            f"T=[{row['temperature_min']:.3f},{row['temperature_max']:.3f}] "
            f"S=[{row['salinity_min']:.3f},{row['salinity_max']:.3f}] "
            f"clip={row['analysis_clipped_fraction']:.3e}", flush=True,
        )
    dataset.archive.close()
    with (output_dir / "compute_cost.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(total_wall_seconds=time.perf_counter()-started, days=args.days,
                       members=args.members, sampling_steps=args.sampling_steps,
                       device=str(device), torch_version=torch.__version__,
                       gpu_name=torch.cuda.get_device_name() if device.type == "cuda" else None,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                       argo_stop_after=args.argo_stop_after), stream, indent=2)


if __name__ == "__main__":
    main()
