from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import inclusive_years


def _as_date(value: Any) -> date:
    return date(int(value.year), int(value.month), int(value.day))


class GlorysArchive:
    """Lazy, worker-safe reader for paired annual GLORYS NetCDF files."""

    def __init__(self, data_cfg: dict[str, Any]) -> None:
        self.cfg = data_cfg
        self.root = Path(data_cfg["root"])
        self.cache_size = int(data_cfg.get("file_cache_size", 8))
        self._cache: OrderedDict[tuple[str, int], netCDF4.Dataset] = OrderedDict()
        self._date_index: dict[date, tuple[int, int]] | None = None
        self._coordinates: dict[str, np.ndarray] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        for dataset in self._cache.values():
            dataset.close()
        self._cache.clear()

    def _path(self, kind: str, year: int) -> Path:
        if kind == "temperature":
            folder = self.cfg["temperature_subdir"]
            pattern = self.cfg["temperature_pattern"]
        elif kind == "salinity":
            folder = self.cfg["salinity_subdir"]
            pattern = self.cfg["salinity_pattern"]
        else:
            raise ValueError(f"Unknown field kind: {kind}")
        return self.root / folder / pattern.format(year=year)

    def available_years(self) -> list[int]:
        years: list[int] = []
        for year in range(1900, 2201):
            if self._path("temperature", year).is_file() and self._path("salinity", year).is_file():
                years.append(year)
        return years

    def _open(self, kind: str, year: int) -> netCDF4.Dataset:
        key = (kind, year)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self._path(kind, year)
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset = netCDF4.Dataset(path, "r")
        self._cache[key] = dataset
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            old.close()
        return dataset

    def build_date_index(self, years: Iterable[int] | None = None) -> dict[date, tuple[int, int]]:
        years = list(years) if years is not None else self.available_years()
        index: dict[date, tuple[int, int]] = {}
        time_name = self.cfg["time_name"]
        for year in years:
            dataset = self._open("temperature", year)
            time_var = dataset.variables[time_name]
            decoded = netCDF4.num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=True,
            )
            for offset, value in enumerate(decoded):
                current = _as_date(value)
                if current in index:
                    raise ValueError(f"Duplicate GLORYS date: {current}")
                index[current] = (year, offset)
        self._date_index = index
        return index

    @property
    def date_index(self) -> dict[date, tuple[int, int]]:
        if self._date_index is None:
            return self.build_date_index()
        return self._date_index

    def coordinates(self) -> dict[str, np.ndarray]:
        if self._coordinates is None:
            years = self.available_years()
            if not years:
                raise FileNotFoundError(f"No paired annual files under {self.root}")
            dataset = self._open("temperature", years[0])
            self._coordinates = {
                "lon": np.asarray(dataset.variables[self.cfg["longitude_name"]][:], dtype=np.float32),
                "lat": np.asarray(dataset.variables[self.cfg["latitude_name"]][:], dtype=np.float32),
                "depth": np.asarray(dataset.variables[self.cfg["depth_name"]][:], dtype=np.float32),
            }
        return self._coordinates

    def read_field(self, kind: str, current_date: date) -> np.ndarray:
        try:
            year, offset = self.date_index[current_date]
        except KeyError as exc:
            raise KeyError(f"Date {current_date.isoformat()} is unavailable") from exc
        dataset = self._open(kind, year)
        variable_name = (
            self.cfg["temperature_variable"] if kind == "temperature" else self.cfg["salinity_variable"]
        )
        array = dataset.variables[variable_name][offset]
        if np.ma.isMaskedArray(array):
            array = array.filled(np.nan)
        return np.asarray(array, dtype=np.float32)

    def read_state(self, current_date: date) -> tuple[np.ndarray, np.ndarray]:
        temperature = self.read_field("temperature", current_date)
        salinity = self.read_field("salinity", current_date)
        state = np.concatenate((temperature, salinity), axis=0)
        valid_depth = np.isfinite(temperature) & np.isfinite(salinity)
        valid_state = np.concatenate((valid_depth, valid_depth), axis=0)
        return state, valid_state


def load_stats(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    required = {"state_mean", "state_std", "increment_mean", "increment_std", "static_mask"}
    missing = required.difference(result)
    if missing:
        raise KeyError(f"Statistics file is missing: {sorted(missing)}")
    return result


class GlorysSequenceDataset(Dataset):
    """Chronological context-to-increment samples with masked ocean loss."""

    def __init__(
        self,
        data_cfg: dict[str, Any],
        split: str,
        stats: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.data_cfg = data_cfg
        self.split = split
        self.archive = GlorysArchive(data_cfg)
        all_years = self.archive.available_years()
        if not all_years:
            raise FileNotFoundError(f"No data found below {data_cfg['root']}")
        self.archive.build_date_index(all_years)
        self.context_days = int(data_cfg["context_days"])
        self.lead_days = int(data_cfg["lead_days"])
        bounds = data_cfg[f"{split}_years"]
        split_years = set(inclusive_years(bounds))
        all_dates = set(self.archive.date_index)

        anchors: list[date] = []
        for target in sorted(all_dates):
            if target.year not in split_years:
                continue
            anchor = target - timedelta(days=self.lead_days)
            context = [anchor - timedelta(days=i) for i in reversed(range(self.context_days))]
            if all(day in all_dates for day in context) and anchor in all_dates:
                anchors.append(anchor)
        if not anchors:
            raise ValueError(f"No valid samples for split={split}, years={bounds}")
        self.anchors = anchors

        self.stats = stats if stats is not None else load_stats(data_cfg["stats_path"])
        coords = self.archive.coordinates()
        self.lat = coords["lat"]
        self.lon = coords["lon"]
        self.depth = coords["depth"]
        self.levels = len(self.depth)
        self.state_channels = 2 * self.levels
        self.static_mask = self.stats["static_mask"].astype(np.float32)

        expected_mask_shape = (self.levels, len(self.lat), len(self.lon))
        if self.static_mask.shape != expected_mask_shape:
            raise ValueError(
                f"static_mask shape {self.static_mask.shape} != expected {expected_mask_shape}"
            )

        self.state_mean = self.stats["state_mean"].astype(np.float32)[:, None, None]
        self.state_std = self.stats["state_std"].astype(np.float32)[:, None, None]
        self.increment_mean = self.stats["increment_mean"].astype(np.float32)[:, None, None]
        self.increment_std = self.stats["increment_std"].astype(np.float32)[:, None, None]

        lat_radians = np.deg2rad(self.lat)
        lon_radians = np.deg2rad(self.lon)
        lat_grid = np.broadcast_to(np.sin(lat_radians)[:, None], (len(self.lat), len(self.lon)))
        sin_lon = np.broadcast_to(np.sin(lon_radians)[None, :], (len(self.lat), len(self.lon)))
        cos_lon = np.broadcast_to(np.cos(lon_radians)[None, :], (len(self.lat), len(self.lon)))
        self.spatial_features = np.stack((lat_grid, sin_lon, cos_lon)).astype(np.float32)
        self.latitude_weights = np.cos(lat_radians).astype(np.float32)[None, :, None]

    @property
    def condition_channels(self) -> int:
        return self.context_days * self.state_channels + self.levels + 5

    def __len__(self) -> int:
        return len(self.anchors)

    def _normalized_state(self, state: np.ndarray, valid: np.ndarray) -> np.ndarray:
        normalized = (state - self.state_mean) / self.state_std
        return np.where(valid, normalized, 0.0).astype(np.float32)

    def make_condition(self, anchor: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context_dates = [anchor - timedelta(days=i) for i in reversed(range(self.context_days))]
        context_states: list[np.ndarray] = []
        last_raw: np.ndarray | None = None
        last_valid: np.ndarray | None = None
        for current in context_dates:
            raw, valid = self.archive.read_state(current)
            context_states.append(self._normalized_state(raw, valid))
            last_raw, last_valid = raw, valid

        day_of_year = anchor.timetuple().tm_yday
        angle = 2.0 * np.pi * (day_of_year - 1) / 365.2425
        h, w = len(self.lat), len(self.lon)
        seasonal = np.stack(
            (
                np.full((h, w), np.sin(angle), dtype=np.float32),
                np.full((h, w), np.cos(angle), dtype=np.float32),
            )
        )
        condition = np.concatenate(
            context_states + [self.static_mask, self.spatial_features, seasonal], axis=0
        ).astype(np.float32)
        assert last_raw is not None and last_valid is not None
        return condition, last_raw, last_valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor = self.anchors[index]
        target_date = anchor + timedelta(days=self.lead_days)
        condition, last_raw, last_valid = self.make_condition(anchor)
        target_raw, target_valid = self.archive.read_state(target_date)
        valid = last_valid & target_valid
        increment = target_raw - last_raw
        target = (increment - self.increment_mean) / self.increment_std
        target = np.where(valid, target, 0.0).astype(np.float32)
        loss_mask = valid.astype(np.float32)

        return {
            "condition": torch.from_numpy(condition),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(loss_mask),
            "latitude_weight": torch.from_numpy(self.latitude_weights.copy()),
            "anchor_ordinal": anchor.toordinal(),
            "target_ordinal": target_date.toordinal(),
        }

    def decode_increment(self, normalized: np.ndarray) -> np.ndarray:
        return normalized * self.increment_std + self.increment_mean


class GlorysRolloutDataset(GlorysSequenceDataset):
    """Initial context plus consecutive truth states for rollout fine-tuning.

    Invalid cells are stored as zero and accompanied by explicit masks.  The
    training loop decides whether truth or the model forecast enters the next
    context; this dataset never performs teacher forcing implicitly.
    """

    def __init__(
        self,
        data_cfg: dict[str, Any],
        split: str,
        rollout_days: int,
        stats: dict[str, np.ndarray] | None = None,
        stride_days: int = 1,
    ) -> None:
        if rollout_days < 1 or stride_days < 1:
            raise ValueError("rollout_days and stride_days must be positive")
        super().__init__(data_cfg, split, stats)
        if self.lead_days != 1:
            raise ValueError("Daily rollout training requires lead_days == 1")
        self.rollout_days = int(rollout_days)
        available = self.archive.date_index
        split_years = set(inclusive_years(data_cfg[f"{split}_years"]))
        complete: list[date] = []
        previous: date | None = None
        for anchor in self.anchors:
            targets = [anchor + timedelta(days=lead) for lead in range(1, rollout_days + 1)]
            if any(target not in available or target.year not in split_years for target in targets):
                continue
            if previous is not None and (anchor - previous).days < stride_days:
                continue
            complete.append(anchor)
            previous = anchor
        if not complete:
            raise ValueError(
                f"No complete {rollout_days}-day trajectories for split={split}"
            )
        self.anchors = complete

    @staticmethod
    def _safe_state(state: np.ndarray, valid: np.ndarray) -> np.ndarray:
        return np.where(valid, state, 0.0).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor = self.anchors[index]
        context_states: list[np.ndarray] = []
        context_valid: list[np.ndarray] = []
        for current in [
            anchor - timedelta(days=offset)
            for offset in reversed(range(self.context_days))
        ]:
            state, valid = self.archive.read_state(current)
            context_states.append(self._safe_state(state, valid))
            context_valid.append(valid)

        truth_states: list[np.ndarray] = []
        truth_valid: list[np.ndarray] = []
        for lead in range(1, self.rollout_days + 1):
            state, valid = self.archive.read_state(anchor + timedelta(days=lead))
            truth_states.append(self._safe_state(state, valid))
            truth_valid.append(valid)

        return {
            "context_states": torch.from_numpy(np.stack(context_states)),
            "context_valid": torch.from_numpy(np.stack(context_valid)),
            "truth_states": torch.from_numpy(np.stack(truth_states)),
            "truth_valid": torch.from_numpy(np.stack(truth_valid)),
            "anchor_ordinal": anchor.toordinal(),
        }


def parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()
