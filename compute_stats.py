from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from diffargo.config import inclusive_years, load_config
from diffargo.data import GlorysArchive


class ChannelMoments:
    def __init__(self, channels: int) -> None:
        self.count = np.zeros(channels, dtype=np.int64)
        self.total = np.zeros(channels, dtype=np.float64)
        self.total_square = np.zeros(channels, dtype=np.float64)

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        flat_values = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
        flat_valid = valid.reshape(valid.shape[0], -1)
        safe = np.where(flat_valid, flat_values, 0.0)
        self.count += flat_valid.sum(axis=1, dtype=np.int64)
        self.total += safe.sum(axis=1, dtype=np.float64)
        self.total_square += np.square(safe).sum(axis=1, dtype=np.float64)

    def finalize(self, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if np.any(self.count == 0):
            raise ValueError(f"No valid samples for channels {np.flatnonzero(self.count == 0).tolist()}")
        mean = self.total / self.count
        variance = self.total_square / self.count - mean**2
        std = np.sqrt(np.maximum(variance, floor**2))
        return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute training-only GLORYS statistics")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_years = set(inclusive_years(data_cfg["train_years"]))
    lead_days = int(data_cfg["lead_days"])
    if lead_days != 1:
        raise ValueError("This MVP computes one-day increment statistics; set lead_days: 1")

    archive = GlorysArchive(data_cfg)
    available = archive.available_years()
    archive.build_date_index(available)
    dates = sorted(archive.date_index)
    coordinates = archive.coordinates()
    levels = len(coordinates["depth"])
    channels = 2 * levels
    state_moments = ChannelMoments(channels)
    increment_moments = ChannelMoments(channels)

    previous_date = None
    previous_state = None
    previous_valid = None
    static_mask = None

    relevant_dates = [
        current
        for current in dates
        if current.year in train_years
        or (current + timedelta(days=1)).year in train_years
    ]
    for current in tqdm(relevant_dates, desc="Streaming GLORYS", unit="day"):
        state, valid = archive.read_state(current)
        if current.year in train_years:
            state_moments.update(state, valid)
            if static_mask is None:
                static_mask = valid[:levels].copy()

            if previous_date is not None and current - previous_date == timedelta(days=1):
                joint = valid & previous_valid
                increment_moments.update(state - previous_state, joint)

        previous_date = current
        previous_state = state
        previous_valid = valid

    state_mean, state_std = state_moments.finalize()
    increment_mean, increment_std = increment_moments.finalize(floor=1e-5)
    if static_mask is None:
        raise RuntimeError("No training state was read")

    output = Path(data_cfg["stats_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        state_mean=state_mean,
        state_std=state_std,
        increment_mean=increment_mean,
        increment_std=increment_std,
        static_mask=static_mask.astype(np.uint8),
        latitude=coordinates["lat"],
        longitude=coordinates["lon"],
        depth=coordinates["depth"],
        train_years=np.asarray(sorted(train_years), dtype=np.int32),
        lead_days=np.asarray(lead_days, dtype=np.int32),
    )
    temporary.replace(output)
    print(f"saved: {output}")
    print(f"state mean range: {state_mean.min():.5g} .. {state_mean.max():.5g}")
    print(f"state std range: {state_std.min():.5g} .. {state_std.max():.5g}")
    print(f"increment std range: {increment_std.min():.5g} .. {increment_std.max():.5g}")
    archive.close()


if __name__ == "__main__":
    main()
