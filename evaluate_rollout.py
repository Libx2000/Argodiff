"""Evaluate a deterministic one-day model in a free-running daily rollout.

Only the initial context is read from GLORYS.  Every later context window uses
the model's own forecasts.  GLORYS states inside the rollout are read solely
for verification and are never fed back into the model.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from diffargo.config import load_config
from diffargo.data import GlorysSequenceDataset, load_stats, parse_date
from diffargo.model import build_model
from diffargo.utils import EMA, seed_everything


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric(weight: float, forecast_sse: float, persistence_sse: float):
    if weight <= 0:
        return None, None, None
    forecast_rmse = float(np.sqrt(forecast_sse / weight))
    persistence_rmse = float(np.sqrt(persistence_sse / weight))
    skill = (
        float(1.0 - forecast_rmse / persistence_rmse)
        if persistence_rmse > 0
        else None
    )
    return forecast_rmse, persistence_rmse, skill


def weighted_sums(
    forecast: np.ndarray,
    persistence: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return area-weighted sums per state channel on exactly common cells."""
    common = valid & np.isfinite(forecast) & np.isfinite(persistence) & np.isfinite(truth)
    latitude_weight = np.cos(np.deg2rad(latitude)).astype(np.float64)[None, :, None]
    weight = np.where(common, latitude_weight, 0.0)
    forecast_error = np.where(common, forecast - truth, 0.0).astype(np.float64)
    persistence_error = np.where(common, persistence - truth, 0.0).astype(np.float64)
    return (
        weight.sum(axis=(1, 2)),
        (weight * forecast_error**2).sum(axis=(1, 2)),
        (weight * persistence_error**2).sum(axis=(1, 2)),
    )


def select_initializations(
    dataset: GlorysSequenceDataset,
    start: date | None,
    end: date | None,
    horizon_days: int,
    stride_days: int,
    maximum: int | None,
) -> list[date]:
    """Select initial anchors whose complete verification window is available."""
    available = dataset.archive.date_index
    selected: list[date] = []
    previous_first_target: date | None = None
    for anchor in dataset.anchors:
        first_target = anchor + timedelta(days=1)
        last_target = anchor + timedelta(days=horizon_days)
        if start is not None and first_target < start:
            continue
        if end is not None and last_target > end:
            continue
        if any(anchor + timedelta(days=lead) not in available
               for lead in range(1, horizon_days + 1)):
            continue
        if (previous_first_target is not None
                and (first_target - previous_first_target).days < stride_days):
            continue
        selected.append(anchor)
        previous_first_target = first_target
        if maximum is not None and len(selected) >= maximum:
            break
    if not selected:
        raise ValueError("No complete rollout windows selected; check dates, horizon and split")
    return selected


def make_rollout_condition(
    dataset: GlorysSequenceDataset,
    states: list[np.ndarray],
    valid_masks: list[np.ndarray],
    anchor: date,
) -> np.ndarray:
    """Build a condition from in-memory states without reading future GLORYS."""
    if len(states) != dataset.context_days or len(valid_masks) != dataset.context_days:
        raise ValueError("Rollout context length does not match the trained model")
    normalized = [
        dataset._normalized_state(state, valid)  # Same transform used in training.
        for state, valid in zip(states, valid_masks)
    ]
    day_of_year = anchor.timetuple().tm_yday
    angle = 2.0 * np.pi * (day_of_year - 1) / 365.2425
    height, width = len(dataset.lat), len(dataset.lon)
    seasonal = np.stack((
        np.full((height, width), np.sin(angle), dtype=np.float32),
        np.full((height, width), np.cos(angle), dtype=np.float32),
    ))
    return np.concatenate(
        normalized + [dataset.static_mask, dataset.spatial_features, seasonal], axis=0
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Free-running multi-day evaluation of a deterministic GLORYS model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--initialization-stride-days", type=int, default=30)
    parser.add_argument("--max-initializations", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.horizon_days < 1 or args.initialization_stride_days < 1:
        parser.error("horizon-days and initialization-stride-days must be positive")
    if args.max_initializations is not None and args.max_initializations < 1:
        parser.error("max-initializations must be positive")
    if args.start is not None and args.end is not None and args.start > args.end:
        parser.error("start must not be after end")

    cfg = load_config(args.config)
    if int(cfg["data"]["lead_days"]) != 1:
        raise ValueError("Free daily rollout requires data.lead_days == 1")
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_stats(cfg["data"]["stats_path"])
    dataset = GlorysSequenceDataset(cfg["data"], args.split, stats)
    initializations = select_initializations(
        dataset, args.start, args.end, args.horizon_days,
        args.initialization_stride_days, args.max_initializations,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    objective = checkpoint.get("objective")
    if objective not in {"direct_increment", "rollout_finetune"}:
        raise ValueError("Rollout evaluator requires a deterministic forecast checkpoint")
    if checkpoint.get("smoke_test", False):
        raise ValueError("Refusing to evaluate a smoke-test checkpoint")
    trained_cfg = checkpoint["config"]
    for key in ("context_days", "lead_days", "train_years", "val_years", "test_years"):
        if trained_cfg["data"][key] != cfg["data"][key]:
            raise ValueError(f"Current config differs from checkpoint for data.{key}")
    expected_train_years = np.arange(
        trained_cfg["data"]["train_years"][0],
        trained_cfg["data"]["train_years"][1] + 1,
    )
    if not np.array_equal(stats["train_years"], expected_train_years):
        raise ValueError("Normalization statistics train years differ from checkpoint")
    if int(checkpoint["target_channels"]) != dataset.state_channels:
        raise ValueError("Checkpoint target channel count differs from dataset")
    if int(checkpoint["condition_channels"]) != dataset.condition_channels:
        raise ValueError("Checkpoint condition channel count differs from dataset")

    checkpoint_epoch = int(checkpoint["epoch"]) + 1
    model = build_model(
        trained_cfg, int(checkpoint["target_channels"]),
        int(checkpoint["condition_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema = EMA(model, float(trained_cfg["train"]["ema_decay"]))
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(model)
    del checkpoint
    model.eval()

    levels = dataset.levels
    horizon = args.horizon_days
    # Last axis: common weight, model SSE, fixed-initial persistence SSE.
    sums = np.zeros((horizon, dataset.state_channels, 3), dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)
    case_rows: list[dict] = []

    print(
        f"device={device}, split={args.split}, initializations={len(initializations)}, "
        f"horizon_days={horizon}, stride_days={args.initialization_stride_days}",
        flush=True,
    )
    for initialization_index, initial_anchor in enumerate(initializations, start=1):
        context_dates = [
            initial_anchor - timedelta(days=offset)
            for offset in reversed(range(dataset.context_days))
        ]
        context_states: list[np.ndarray] = []
        context_valid: list[np.ndarray] = []
        for current_date in context_dates:
            state, valid = dataset.archive.read_state(current_date)
            context_states.append(state)
            context_valid.append(valid)

        initial_state = context_states[-1].copy()
        initial_valid = context_valid[-1].copy()
        for lead in range(1, horizon + 1):
            current_anchor = initial_anchor + timedelta(days=lead - 1)
            valid_date = initial_anchor + timedelta(days=lead)
            condition_np = make_rollout_condition(
                dataset, context_states, context_valid, current_anchor
            )
            condition = torch.from_numpy(condition_np).unsqueeze(0).to(device)
            with torch.inference_mode():
                zeros = condition.new_zeros((
                    1, dataset.state_channels, len(dataset.lat), len(dataset.lon)
                ))
                timestep = torch.zeros(1, device=device, dtype=torch.long)
                normalized_correction = model(zeros, condition, timestep)[0].float().cpu().numpy()

            last_state = context_states[-1]
            last_valid = context_valid[-1]
            # Training is persistence-anchored: model output is the correction
            # to zero physical increment, in increment-standard-deviation units.
            forecast = last_state + normalized_correction * dataset.increment_std
            if not np.all(np.isfinite(forecast[last_valid])):
                raise FloatingPointError(
                    f"Non-finite ocean forecast: initialization={initial_anchor}, lead={lead}"
                )
            forecast = np.where(last_valid, forecast, np.nan).astype(np.float32)
            forecast_valid = last_valid.copy()

            truth, truth_valid = dataset.archive.read_state(valid_date)
            common_valid = forecast_valid & initial_valid & truth_valid
            weight, forecast_sse, persistence_sse = weighted_sums(
                forecast, initial_state, truth, common_valid, dataset.lat
            )
            if np.any(weight <= 0):
                raise RuntimeError(
                    f"No common valid ocean cells: initialization={initial_anchor}, lead={lead}"
                )
            sums[lead - 1, :, 0] += weight
            sums[lead - 1, :, 1] += forecast_sse
            sums[lead - 1, :, 2] += persistence_sse
            counts[lead - 1] += 1

            for variable, channel_slice in (
                ("temperature", slice(0, levels)),
                ("salinity", slice(levels, 2 * levels)),
            ):
                pooled = np.array((
                    weight[channel_slice].sum(),
                    forecast_sse[channel_slice].sum(),
                    persistence_sse[channel_slice].sum(),
                ))
                forecast_rmse, persistence_rmse, skill = metric(*pooled)
                values = forecast[channel_slice]
                values = values[np.isfinite(values)]
                case_rows.append({
                    "initial_anchor": initial_anchor.isoformat(),
                    "valid_date": valid_date.isoformat(),
                    "lead_day": lead,
                    "variable": variable,
                    "forecast_rmse": forecast_rmse,
                    "persistence_rmse": persistence_rmse,
                    "skill_vs_initial_persistence": skill,
                    "forecast_min": float(values.min()),
                    "forecast_max": float(values.max()),
                })

            # This is the defining free-run step: append the prediction, never truth.
            context_states = context_states[1:] + [forecast]
            context_valid = context_valid[1:] + [forecast_valid]

        final_rows = case_rows[-2:]
        print(
            f"[{initialization_index}/{len(initializations)}] init={initial_anchor} "
            f"lead={horizon}d T={final_rows[0]['forecast_rmse']:.5f}/"
            f"{final_rows[0]['persistence_rmse']:.5f} "
            f"S={final_rows[1]['forecast_rmse']:.5f}/"
            f"{final_rows[1]['persistence_rmse']:.5f}",
            flush=True,
        )

    by_depth_rows: list[dict] = []
    by_lead_rows: list[dict] = []
    for lead in range(1, horizon + 1):
        for channel in range(dataset.state_channels):
            variable = "temperature" if channel < levels else "salinity"
            level = channel if channel < levels else channel - levels
            forecast_rmse, persistence_rmse, skill = metric(*sums[lead - 1, channel])
            by_depth_rows.append({
                "lead_day": lead,
                "variable": variable,
                "depth_m": float(dataset.depth[level]),
                "level_index": level,
                "n_initializations": int(counts[lead - 1]),
                "weight_sum": float(sums[lead - 1, channel, 0]),
                "forecast_rmse": forecast_rmse,
                "persistence_rmse": persistence_rmse,
                "skill_vs_initial_persistence": skill,
            })
        for variable, channel_slice in (
            ("temperature", slice(0, levels)),
            ("salinity", slice(levels, 2 * levels)),
        ):
            pooled = sums[lead - 1, channel_slice].sum(axis=0)
            forecast_rmse, persistence_rmse, skill = metric(*pooled)
            by_lead_rows.append({
                "lead_day": lead,
                "variable": variable,
                "n_initializations": int(counts[lead - 1]),
                "weight_sum": float(pooled[0]),
                "forecast_rmse": forecast_rmse,
                "persistence_rmse": persistence_rmse,
                "skill_vs_initial_persistence": skill,
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary_by_lead_depth.csv", by_depth_rows, list(by_depth_rows[0]))
    write_csv(output_dir / "summary_by_lead.csv", by_lead_rows, list(by_lead_rows[0]))
    write_csv(output_dir / "cases_by_lead.csv", case_rows, list(case_rows[0]))
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "config": args.config,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch_one_based": checkpoint_epoch,
            "objective": objective,
            "evaluation_mode": "free_running_autoregressive",
            "truth_feedback": False,
            "baseline": "fixed_initial_state_persistence",
            "split": args.split,
            "start": args.start.isoformat() if args.start else None,
            "end": args.end.isoformat() if args.end else None,
            "horizon_days": horizon,
            "initialization_stride_days": args.initialization_stride_days,
            "initial_anchors": [value.isoformat() for value in initializations],
            "seed": int(cfg["seed"]),
            "device": str(device),
            "note": (
                "GLORYS after initialization is used only as verification truth. "
                "All-level pooled metrics use area weights but are not volume weighted. "
                "No forecast clipping is applied."
            ),
        }, handle, ensure_ascii=False, indent=2)
    dataset.archive.close()

    for lead in sorted({1, min(3, horizon), min(7, horizon), horizon}):
        rows = [row for row in by_lead_rows if row["lead_day"] == lead]
        message = " ".join(
            f"{row['variable'][0].upper()}={row['forecast_rmse']:.6f}/"
            f"{row['persistence_rmse']:.6f} skill={row['skill_vs_initial_persistence']:.2%}"
            for row in rows
        )
        print(f"lead={lead}d {message}", flush=True)
    print(f"saved: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
