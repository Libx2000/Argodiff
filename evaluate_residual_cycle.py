"""Evaluate deterministic accuracy and probabilistic ensemble calibration."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from diffargo.config import load_config
from diffargo.data import GlorysArchive


def weighted_mean(values, valid, weights):
    weight = np.broadcast_to(weights, values.shape) * valid
    return float(np.sum(weight * np.where(valid, values, 0.0)) / max(weight.sum(), 1.0))


def rmse(prediction, truth, valid, weights):
    return float(np.sqrt(weighted_mean((prediction - truth) ** 2, valid, weights)))


def empirical_crps(ensemble, truth, valid, weights):
    first = np.mean(np.abs(ensemble - truth[None]), axis=0)
    members = ensemble.shape[0]
    ordered = np.sort(ensemble, axis=0)
    coefficients = (2.0 * np.arange(members) - members + 1.0).astype(np.float64)
    second = np.tensordot(coefficients, ordered, axes=(0, 0)) / (members * members)
    return weighted_mean(first - second, valid, weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    archive = GlorysArchive(cfg["data"])
    archive.build_date_index()
    coordinates = archive.coordinates()
    levels = len(coordinates["depth"])
    weights = np.cos(np.deg2rad(coordinates["lat"]))[None, :, None]
    files = sorted(Path(args.run_dir).glob("????-??-??.npz"))
    if not files:
        raise FileNotFoundError(f"No daily NPZ files in {args.run_dir}")
    first = datetime.strptime(files[0].stem, "%Y-%m-%d").date()
    initial, _ = archive.read_state(first - timedelta(days=1))
    rows = []

    for path in files:
        day = datetime.strptime(path.stem, "%Y-%m-%d").date()
        truth, truth_valid = archive.read_state(day)
        with np.load(path, allow_pickle=False) as result:
            unguided = result["unguided_mean"]
            analysis = result["analysis"]
            spread = result["analysis_spread"]
            argo_values = int(result["argo_mask"].sum())
            ensemble = (
                result["analysis_ensemble"].astype(np.float32)
                if "analysis_ensemble" in result.files else None
            )
        row = {
            "date": path.stem,
            "argo_values": argo_values,
            "ensemble_members": 0 if ensemble is None else ensemble.shape[0],
        }
        for name, channel_slice in (
            ("temperature", slice(0, levels)),
            ("salinity", slice(levels, None)),
        ):
            valid = (
                truth_valid[channel_slice]
                & np.isfinite(analysis[channel_slice])
                & np.isfinite(unguided[channel_slice])
            )
            current_rmse = rmse(
                analysis[channel_slice], truth[channel_slice], valid, weights
            )
            mean_spread = weighted_mean(spread[channel_slice], valid, weights)
            row[f"{name}_unguided_rmse"] = rmse(
                unguided[channel_slice], truth[channel_slice], valid, weights
            )
            row[f"{name}_analysis_rmse"] = current_rmse
            row[f"{name}_persistence_rmse"] = rmse(
                initial[channel_slice], truth[channel_slice], valid, weights
            )
            row[f"{name}_spread"] = mean_spread
            row[f"{name}_spread_skill_ratio"] = mean_spread / max(current_rmse, 1e-12)
            row[f"{name}_skill_vs_unguided"] = (
                1.0 - current_rmse / max(row[f"{name}_unguided_rmse"], 1e-12)
            )
            if ensemble is not None:
                sliced_ensemble = ensemble[:, channel_slice]
                ensemble_min = np.min(
                    np.where(np.isfinite(sliced_ensemble), sliced_ensemble, np.inf), axis=0
                )
                ensemble_max = np.max(
                    np.where(np.isfinite(sliced_ensemble), sliced_ensemble, -np.inf), axis=0
                )
                row[f"{name}_range_coverage"] = weighted_mean(
                    (truth[channel_slice] >= ensemble_min)
                    & (truth[channel_slice] <= ensemble_max),
                    valid, weights,
                )
                row[f"{name}_crps"] = empirical_crps(
                    sliced_ensemble, truth[channel_slice], valid, weights
                )
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    last = rows[-1]
    summary = {
        "days": len(rows),
        "argo_days": sum(row["argo_values"] > 0 for row in rows),
        "final_day": last,
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0] if key != "date"
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
