"""Split raw daily profiles before aggregation, using a fixed spatial holdout.

The assignment is stable across years. A one-grid-cell exclusion buffer prevents
guidance in a holdout cell or its eight immediate neighbours. Fractions are
spatial-block targets, not forced daily profile counts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from argo_common import read_profile_arrays
from map_argo_to_glorys import (load_grid, find_glorys_file, nearest_indices,
    nearest_longitude_indices, interpolate_profile, aggregate_profiles)


def spatial_partition(lat, lon, block_degrees=9.0, fraction=0.30, seed=314159, buffer=1):
    if not 0 < fraction < 1 or block_degrees <= 0 or not 0 <= buffer < len(lat):
        raise ValueError("Invalid spatial split settings")
    if not np.isclose(360 / block_degrees, round(360 / block_degrees)):
        raise ValueError("block-degrees must divide 360 for a periodic longitude split")
    bi = np.floor((np.asarray(lat) + 90) / block_degrees).astype(int)
    bj = np.floor(np.mod(lon, 360) / block_degrees).astype(int)
    blocks = sorted({(int(i), int(j)) for i in bi for j in bj})
    # Exactly ~30% of global spatial blocks, selected without using values/dates.
    ranked = sorted(blocks, key=lambda b: hashlib.sha256(f"{seed}:{b[0]}:{b[1]}".encode()).digest())
    selected = set(ranked[:max(1, round(len(blocks) * fraction))])
    held = np.array([[(int(i), int(j)) in selected for j in bj] for i in bi], dtype=bool)
    excluded = held.copy()
    for di in range(-buffer, buffer + 1):
        for dj in range(-buffer, buffer + 1):
            shifted = np.roll(held, dj, axis=1)
            if di >= 0:
                excluded[di:] |= shifted[:len(lat)-di]
            else:
                excluded[:di] |= shifted[-di:]
    guide = ~excluded
    return guide, held


def profile_day_mask(path, expected, n):
    """Do not silently assign multi-day JULD profiles to the filename day."""
    import netCDF4
    from argo_common import resolve_name, TIME_NAMES
    with netCDF4.Dataset(path) as ds:
        name = resolve_name(ds, TIME_NAMES, required=False)
        if name is None or not getattr(ds.variables[name], "units", None):
            return np.ones(n, dtype=bool)
        var = ds.variables[name]
        values = np.ma.asarray(var[:]).reshape(-1)
        if len(values) not in (1, n):
            raise ValueError(f"Cannot associate profile times in {path}")
        mask = ~np.ma.getmaskarray(values) & np.isfinite(values.filled(np.nan))
        keep = np.zeros(len(values), dtype=bool)
        if mask.any():
            decoded = netCDF4.num2date(values.data[mask], var.units,
                                      calendar=getattr(var,"calendar","standard"))
            keep[mask] = [f"{v.year:04d}-{v.month:02d}-{v.day:02d}" == expected for v in decoded]
        if len(keep) == 1:
            keep = np.full(n, keep[0], dtype=bool)
        if not np.all(keep):
            raise ValueError(f"Mixed/invalid profile dates in daily input {path}; build a per-profile daily inventory first")
        return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True)
    p.add_argument("--glorys-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2022-01-01")
    p.add_argument("--block-degrees", type=float, default=9.0)
    p.add_argument("--holdout-fraction", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=314159)
    p.add_argument("--buffer-cells", type=int, default=1)
    a = p.parse_args()
    out = Path(a.output_dir)
    if (out / "split_manifest.json").exists():
        raise FileExistsError("Split already exists. Choose a new output-dir to preserve provenance.")
    lat, lon, depth = load_grid(find_glorys_file(Path(a.glorys_root)))
    guide, held = spatial_partition(lat, lon, a.block_degrees, a.holdout_fraction, a.seed, a.buffer_cells)
    with Path(a.inventory).open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    grouped = {}
    for r in records:
        if r.get("status") == "ok" and a.start <= r.get("date", "") <= a.end:
            grouped.setdefault(r["date"], []).append(r)
    if not grouped:
        raise ValueError("No daily files in requested date interval")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "spatial_partition.npz", guide=guide, holdout=held,
                        latitude=lat, longitude=lon, depth_m=depth)
    rows = []
    day = date.fromisoformat(a.start)
    while day <= date.fromisoformat(a.end):
        text = day.isoformat()
        all_i, all_j, all_t, all_s, all_lat, all_lon, all_ids = [], [], [], [], [], [], []
        for r in sorted(grouped.get(text, []), key=lambda x: x["file"]):
            source = Path(r["file"])
            data = read_profile_arrays(source, ("1", "2"))
            if data["date"] is None or data["date"].isoformat() != text:
                raise ValueError(f"Inventory/source date mismatch: {source}")
            ok = (np.isfinite(data["latitude"]) & np.isfinite(data["longitude"])
                  & (np.abs(data["latitude"]) <= 90))
            ok &= profile_day_mask(source, text, data["n_profiles"])
            ids = np.flatnonzero(ok)
            ii = nearest_indices(lat, data["latitude"][ids])
            jj = nearest_longitude_indices(lon, data["longitude"][ids])
            for k, i, j in zip(ids, ii, jj):
                t = interpolate_profile(data["pressure"][k], data["temperature"][k], data["valid_temperature"][k], depth)
                s = interpolate_profile(data["pressure"][k], data["salinity"][k], data["valid_salinity"][k], depth)
                t[(t < -3) | (t > 40)] = np.nan
                s[(s < 0) | (s > 42)] = np.nan
                all_i.append(i); all_j.append(j); all_t.append(t); all_s.append(s)
                all_lat.append(data["latitude"][k]); all_lon.append(data["longitude"][k])
                all_ids.append(f"{source.as_posix()}#{k}")
        ii, jj = np.asarray(all_i, dtype=np.int16), np.asarray(all_j, dtype=np.int16)
        t = np.asarray(all_t, dtype=np.float32).reshape(-1, len(depth))
        s = np.asarray(all_s, dtype=np.float32).reshape(-1, len(depth))
        g, h = guide[ii, jj], held[ii, jj]
        mapped = aggregate_profiles(ii[g].astype(np.int64)*len(lon)+jj[g], t[g], s[g], depth)
        paths = [out / kind / str(day.year) / f"{text}.npz" for kind in ("guidance", "holdout")]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(paths[0], **mapped, latitude=lat, longitude=lon, depth_m=depth)
        np.savez_compressed(paths[1], grid_i=ii[h], grid_j=jj[h], temperature=t[h], salinity=s[h],
                            profile_latitude=np.asarray(all_lat, np.float32)[h],
                            profile_longitude=np.asarray(all_lon, np.float32)[h],
                            profile_id=np.asarray(all_ids, dtype=str)[h],
                            latitude=lat, longitude=lon, depth_m=depth)
        rows.append(dict(date=text, source_files=len(grouped.get(text, [])),
                         profiles=len(ii), guidance_profiles=int(g.sum()),
                         holdout_profiles=int(h.sum()), buffer_profiles=int((~g & ~h).sum())))
        print(f"{text}: profiles={len(ii)} guide={g.sum()} held={h.sum()}", flush=True)
        day += timedelta(days=1)
    with (out / "split_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest = vars(a) | {"days":len(rows), "guidance_grid_cells":int(guide.sum()),
        "holdout_grid_cells":int(held.sum()), "missing_source_days":[r["date"] for r in rows if not r["source_files"]],
        "partition_sha256":hashlib.sha256(guide.tobytes()+held.tobytes()).hexdigest(),
        "note":"Fixed geographic blocks across years; guidance exclusion buffer. Held profiles remain individual. Vertical interpolation uses existing pressure-as-depth convention; nearest-cell observation operator."}
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
