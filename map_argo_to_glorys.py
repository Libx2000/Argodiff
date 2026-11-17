"""Map daily Argo profiles to the GLORYS 1.5-degree grid and 15 depths.

The output is sparse: each daily NPZ stores only grid cells containing at least
one profile. Temperature and salinity are independently vertically
interpolated without extrapolation, then averaged within each grid cell/depth.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import netCDF4
import numpy as np
from tqdm import tqdm

from argo_common import read_profile_arrays


def parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def resolve_coord(dataset, candidates):
    names = {name.lower(): name for name in dataset.variables}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    raise KeyError(f"Missing coordinate; tried {candidates}")


def find_glorys_file(root: Path) -> Path:
    candidates = sorted(root.rglob("thetao_glor_*_0_1.5deg.nc"))
    if not candidates:
        candidates = sorted(root.rglob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No GLORYS NetCDF file under {root}")
    return candidates[0]


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netCDF4.Dataset(path, "r") as dataset:
        lat_name = resolve_coord(dataset, ("lat", "latitude", "LATITUDE"))
        lon_name = resolve_coord(dataset, ("lon", "longitude", "LONGITUDE"))
        depth_name = resolve_coord(dataset, ("elevation", "depth", "DEPTH", "depth_m"))
        lat = np.asarray(dataset.variables[lat_name][:], dtype=np.float32).reshape(-1)
        lon = np.asarray(dataset.variables[lon_name][:], dtype=np.float32).reshape(-1)
        depth = np.abs(np.asarray(dataset.variables[depth_name][:], dtype=np.float32).reshape(-1))
    if len(lat) != 120 or len(lon) != 240:
        raise ValueError(f"Expected 1.5-degree 120x240 grid, got {len(lat)}x{len(lon)}")
    # Preserve the exact GLORYS depth-channel order.  The current files are
    # deep-to-shallow; changing it here would silently misalign Argo values
    # with the trained model channels.  np.interp accepts unsorted targets.
    return lat, lon, depth


def nearest_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # Do not assume latitude is ascending; a number of GLORYS exports are
    # north-to-south.  The grid is only 120 points wide, so this is cheap.
    return np.abs(values[:, None] - targets[None, :]).argmin(axis=0).astype(np.int32)


def nearest_longitude_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    difference = np.abs((values[:, None] - targets[None, :] + 180.0) % 360.0 - 180.0)
    return difference.argmin(axis=0).astype(np.int32)


def interpolate_profile(pressure: np.ndarray, values: np.ndarray, valid: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid = valid & np.isfinite(pressure) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.full(len(targets), np.nan, dtype=np.float32)
    p = np.abs(pressure[valid])
    v = values[valid]
    order = np.argsort(p)
    p, v = p[order], v[order]
    unique, inverse = np.unique(p, return_inverse=True)
    if len(unique) != len(p):
        sums = np.zeros(len(unique), dtype=np.float64)
        counts = np.zeros(len(unique), dtype=np.float64)
        np.add.at(sums, inverse, v)
        np.add.at(counts, inverse, 1.0)
        p, v = unique, sums / counts
    if len(p) < 2:
        return np.full(len(targets), np.nan, dtype=np.float32)
    output = np.interp(targets, p, v, left=np.nan, right=np.nan)
    output[(targets < p[0]) | (targets > p[-1])] = np.nan
    return output.astype(np.float32)


def map_one_file(path: Path, grid_lat, grid_lon, target_depth, accepted_qc):
    data = read_profile_arrays(path, accepted_qc)
    longitude = np.mod(data["longitude"].astype(np.float64), 360.0)
    latitude = data["latitude"].astype(np.float64)
    position_valid = np.isfinite(latitude) & np.isfinite(longitude)
    profile_ids = np.flatnonzero(position_valid)
    if not len(profile_ids):
        return empty_mapping(target_depth), 0
    lat_idx = nearest_indices(grid_lat.astype(np.float64), latitude[profile_ids])
    lon_idx = nearest_longitude_indices(grid_lon.astype(np.float64), longitude[profile_ids])
    cell_ids = lat_idx.astype(np.int64) * len(grid_lon) + lon_idx.astype(np.int64)
    temp_profiles = []
    sal_profiles = []
    for local_index, profile_id in enumerate(profile_ids):
        pressure = data["pressure"][profile_id]
        temp_profiles.append(interpolate_profile(
            pressure, data["temperature"][profile_id],
            data["valid_temperature"][profile_id], target_depth,
        ))
        sal_profiles.append(interpolate_profile(
            pressure, data["salinity"][profile_id],
            data["valid_salinity"][profile_id], target_depth,
        ))
    temp_profiles = np.asarray(temp_profiles, dtype=np.float32)
    sal_profiles = np.asarray(sal_profiles, dtype=np.float32)
    return aggregate_profiles(cell_ids, temp_profiles, sal_profiles, target_depth), len(profile_ids)


def empty_mapping(target_depth):
    return {
        "cell_index": np.empty(0, dtype=np.int32),
        "grid_i": np.empty(0, dtype=np.int16),
        "grid_j": np.empty(0, dtype=np.int16),
        "temperature": np.empty((0, len(target_depth)), dtype=np.float32),
        "salinity": np.empty((0, len(target_depth)), dtype=np.float32),
        "temperature_count": np.empty((0, len(target_depth)), dtype=np.int16),
        "salinity_count": np.empty((0, len(target_depth)), dtype=np.int16),
        "profile_count": np.empty(0, dtype=np.int16),
    }


def aggregate_profiles(cell_ids, temperatures, salinities, target_depth):
    unique_cells = np.unique(cell_ids)
    n_cells, n_depth = len(unique_cells), len(target_depth)
    temperature_sum = np.zeros((n_cells, n_depth), dtype=np.float64)
    salinity_sum = np.zeros((n_cells, n_depth), dtype=np.float64)
    temperature_count = np.zeros((n_cells, n_depth), dtype=np.int32)
    salinity_count = np.zeros((n_cells, n_depth), dtype=np.int32)
    profile_count = np.zeros(n_cells, dtype=np.int32)
    lookup = {int(cell): index for index, cell in enumerate(unique_cells)}
    for profile_index, cell in enumerate(cell_ids):
        output_index = lookup[int(cell)]
        profile_count[output_index] += 1
        valid_t = np.isfinite(temperatures[profile_index])
        valid_s = np.isfinite(salinities[profile_index])
        temperature_sum[output_index, valid_t] += temperatures[profile_index, valid_t]
        salinity_sum[output_index, valid_s] += salinities[profile_index, valid_s]
        temperature_count[output_index, valid_t] += 1
        salinity_count[output_index, valid_s] += 1
    temperature = np.divide(temperature_sum, temperature_count, out=np.full_like(temperature_sum, np.nan), where=temperature_count > 0)
    salinity = np.divide(salinity_sum, salinity_count, out=np.full_like(salinity_sum, np.nan), where=salinity_count > 0)
    return {
        "cell_index": unique_cells.astype(np.int32),
        "grid_i": (unique_cells // 240).astype(np.int16),
        "grid_j": (unique_cells % 240).astype(np.int16),
        "temperature": temperature.astype(np.float32),
        "salinity": salinity.astype(np.float32),
        "temperature_count": np.minimum(temperature_count, np.iinfo(np.int16).max).astype(np.int16),
        "salinity_count": np.minimum(salinity_count, np.iinfo(np.int16).max).astype(np.int16),
        "profile_count": np.minimum(profile_count, np.iinfo(np.int16).max).astype(np.int16),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Map raw Argo profiles to GLORYS grid/depths")
    parser.add_argument("--inventory", default=r"D:\argo_derived\inventory\argo_daily_inventory.csv")
    parser.add_argument("--glorys-root", default=r"D:\reanalysis_31year")
    parser.add_argument("--output-dir", default=r"D:\argo_derived\mapped_1p5deg")
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    parser.add_argument("--limit-files", type=int, help="Only map a prefix for a smoke test")
    parser.add_argument("--qc-values", default="1,2")
    args = parser.parse_args()
    inventory_path = Path(args.inventory).expanduser().resolve()
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    if args.limit_files is not None and args.limit_files < 1:
        parser.error("limit-files must be positive")
    accepted_qc = tuple(value.strip() for value in args.qc_values.split(",") if value.strip())
    with inventory_path.open("r", newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    records = [record for record in records if record.get("status") == "ok" and record.get("date")]
    if args.start:
        records = [record for record in records if parse_date(record["date"]) >= args.start]
    if args.end:
        records = [record for record in records if parse_date(record["date"]) <= args.end]
    records.sort(key=lambda record: (record["date"], record["file"]))
    if args.limit_files is not None:
        records = records[: args.limit_files]
    if not records:
        raise ValueError("No inventory records selected")
    grid_file = find_glorys_file(Path(args.glorys_root).expanduser().resolve())
    grid_lat, grid_lon, target_depth = load_grid(grid_file)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    grouped = defaultdict(list)
    for record in records:
        grouped[record["date"]].append(record)
    for day, day_records in tqdm(sorted(grouped.items()), desc="map Argo days"):
        combined = []
        total_profiles = 0
        for record in day_records:
            mapped, profile_count = map_one_file(Path(record["file"]), grid_lat, grid_lon, target_depth, accepted_qc)
            total_profiles += profile_count
            if len(mapped["cell_index"]):
                combined.append((mapped, profile_count))
        if combined:
            # Re-aggregate daily profiles by expanding each mapped cell using counts.
            daily = merge_daily_mappings([item[0] for item in combined], target_depth)
        else:
            daily = empty_mapping(target_depth)
        out_path = output_dir / day[:4] / f"{day}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **daily, latitude=grid_lat, longitude=grid_lon, depth_m=target_depth)
        summary_rows.append({
            "date": day,
            "source_files": len(day_records),
            "profiles_with_valid_position": total_profiles,
            "mapped_grid_cells": len(daily["cell_index"]),
            "temperature_observations": int(np.count_nonzero(daily["temperature_count"])),
            "salinity_observations": int(np.count_nonzero(daily["salinity_count"])),
            "output_file": str(out_path),
        })
    with (output_dir / "daily_mapping_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"GLORYS grid file: {grid_file}", flush=True)
    print(f"mapped days: {len(summary_rows)}, output: {output_dir}", flush=True)


def merge_daily_mappings(mappings, target_depth):
    n_depth = len(target_depth)
    unique_cells = np.unique(np.concatenate([item["cell_index"] for item in mappings]))
    lookup = {int(cell): index for index, cell in enumerate(unique_cells)}
    t_sum = np.zeros((len(unique_cells), n_depth), dtype=np.float64)
    s_sum = np.zeros((len(unique_cells), n_depth), dtype=np.float64)
    t_count = np.zeros((len(unique_cells), n_depth), dtype=np.int32)
    s_count = np.zeros((len(unique_cells), n_depth), dtype=np.int32)
    p_count = np.zeros(len(unique_cells), dtype=np.int32)
    for mapping in mappings:
        for row, cell in enumerate(mapping["cell_index"]):
            index = lookup[int(cell)]
            tc = mapping["temperature_count"][row].astype(np.int32)
            sc = mapping["salinity_count"][row].astype(np.int32)
            t_sum[index] += np.nan_to_num(mapping["temperature"][row], nan=0.0) * tc
            s_sum[index] += np.nan_to_num(mapping["salinity"][row], nan=0.0) * sc
            t_count[index] += tc
            s_count[index] += sc
            p_count[index] += int(mapping["profile_count"][row])
    return {
        "cell_index": unique_cells.astype(np.int32),
        "grid_i": (unique_cells // 240).astype(np.int16),
        "grid_j": (unique_cells % 240).astype(np.int16),
        "temperature": np.divide(t_sum, t_count, out=np.full_like(t_sum, np.nan), where=t_count > 0).astype(np.float32),
        "salinity": np.divide(s_sum, s_count, out=np.full_like(s_sum, np.nan), where=s_count > 0).astype(np.float32),
        "temperature_count": np.minimum(t_count, np.iinfo(np.int16).max).astype(np.int16),
        "salinity_count": np.minimum(s_count, np.iinfo(np.int16).max).astype(np.int16),
        "profile_count": np.minimum(p_count, np.iinfo(np.int16).max).astype(np.int16),
    }


if __name__ == "__main__":
    main()
