"""Fit validation-only channel or lead-by-channel ensemble inflation factors."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def isotonic_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit for a non-decreasing sequence."""
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(index), float(index), max(float(weight), 1e-12), float(value)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[2] + right[2]
            mean = (left[2] * left[3] + right[2] * right[3]) / total_weight
            blocks.append([left[0], right[1], total_weight, mean])
    result = np.empty_like(values, dtype=np.float64)
    for start, end, _, value in blocks:
        result[int(start): int(end) + 1] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lead-dependent", action="store_true")
    parser.add_argument("--monotonic-lead", action="store_true")
    parser.add_argument("--min-factor", type=float, default=1.0)
    parser.add_argument("--max-factor", type=float, default=50.0)
    parser.add_argument("--log-shrinkage", type=float, default=0.10)
    args = parser.parse_args()
    if not 1.0 <= args.min_factor <= args.max_factor:
        parser.error("Require 1 <= min-factor <= max-factor")
    if not 0.0 <= args.log_shrinkage < 1.0:
        parser.error("--log-shrinkage must be in [0, 1)")
    if args.monotonic_lead and not args.lead_dependent:
        parser.error("--monotonic-lead requires --lead-dependent")

    error_sum = variance_sum = weight_sum = None
    completed_days = 0
    source_count = 0
    for item in args.statistics:
        path = Path(item)
        with np.load(path, allow_pickle=False) as archive:
            if args.lead_dependent:
                required = {
                    "squared_error_by_lead", "ensemble_variance_by_lead", "weight_by_lead"
                }
                missing = required.difference(archive.files)
                if missing:
                    raise KeyError(f"{path} is missing {sorted(missing)}")
                current_error = archive["squared_error_by_lead"].astype(np.float64)
                current_variance = archive["ensemble_variance_by_lead"].astype(np.float64)
                current_weight = archive["weight_by_lead"].astype(np.float64)
            else:
                required = {"squared_error_sum", "ensemble_variance_sum", "weight_sum"}
                missing = required.difference(archive.files)
                if missing:
                    raise KeyError(f"{path} is missing {sorted(missing)}")
                current_error = archive["squared_error_sum"].astype(np.float64)
                current_variance = archive["ensemble_variance_sum"].astype(np.float64)
                current_weight = archive["weight_sum"].astype(np.float64)
            completed_days += int(archive["completed_days"].item()) if "completed_days" in archive else 0
        if error_sum is None:
            error_sum = np.zeros_like(current_error)
            variance_sum = np.zeros_like(current_variance)
            weight_sum = np.zeros_like(current_weight)
        if current_error.shape != error_sum.shape:
            raise ValueError("All statistics files must have the same shape")
        error_sum += current_error
        variance_sum += current_variance
        weight_sum += current_weight
        source_count += 1

    assert error_sum is not None and variance_sum is not None and weight_sum is not None
    if np.any(weight_sum <= 0):
        raise ValueError("At least one lead/channel cell has no validation weight")
    rmse = np.sqrt(error_sum / weight_sum)
    spread = np.sqrt(variance_sum / weight_sum)
    raw_factor = rmse / np.maximum(spread, 1e-12)
    log_factor = np.log(np.maximum(raw_factor, 1.0))
    if args.lead_dependent and args.monotonic_lead:
        for channel in range(log_factor.shape[1]):
            log_factor[:, channel] = isotonic_increasing(
                log_factor[:, channel], weight_sum[:, channel]
            )
    shrunk_factor = np.exp((1.0 - args.log_shrinkage) * log_factor)
    factor = np.clip(shrunk_factor, args.min_factor, args.max_factor).astype(np.float32)
    if not np.all(np.isfinite(factor)):
        raise FloatingPointError("Non-finite spread inflation factor")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        inflation_factor=factor,
        raw_factor=raw_factor.astype(np.float32),
        validation_rmse=rmse.astype(np.float32),
        validation_spread=spread.astype(np.float32),
        squared_error_sum=error_sum,
        ensemble_variance_sum=variance_sum,
        weight_sum=weight_sum,
        completed_days=np.asarray(completed_days, dtype=np.int32),
        lead_dependent=np.asarray(args.lead_dependent),
        monotonic_lead=np.asarray(args.monotonic_lead),
        log_shrinkage=np.asarray(args.log_shrinkage, dtype=np.float32),
        min_factor=np.asarray(args.min_factor, dtype=np.float32),
        max_factor=np.asarray(args.max_factor, dtype=np.float32),
        source_count=np.asarray(source_count, dtype=np.int32),
    )

    channels = factor.shape[-1]
    levels = channels // 2
    leads = factor.shape[0] if factor.ndim == 2 else 1
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "lead_day", "channel", "variable", "level_index", "validation_rmse",
            "validation_spread", "raw_factor", "inflation_factor",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for lead in range(leads):
            for channel in range(channels):
                index = (lead, channel) if factor.ndim == 2 else channel
                writer.writerow({
                    "lead_day": lead + 1 if factor.ndim == 2 else "all",
                    "channel": channel,
                    "variable": "temperature" if channel < levels else "salinity",
                    "level_index": channel if channel < levels else channel - levels,
                    "validation_rmse": rmse[index],
                    "validation_spread": spread[index],
                    "raw_factor": raw_factor[index],
                    "inflation_factor": factor[index],
                })
    print(f"saved: {output}", flush=True)
    print(
        f"validation sample-days={completed_days} shape={factor.shape} "
        f"raw={raw_factor.min():.3f}..{raw_factor.max():.3f} "
        f"applied={factor.min():.3f}..{factor.max():.3f}", flush=True,
    )
    if factor.ndim == 2:
        print(
            f"lead-1 mean={factor[0].mean():.3f}, "
            f"lead-{factor.shape[0]} mean={factor[-1].mean():.3f}", flush=True,
        )


if __name__ == "__main__":
    main()
