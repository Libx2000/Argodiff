"""Create a bounded daily inventory of raw Argo profile NetCDF files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from argo_common import inspect_metadata, read_profile_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory daily Argo profile NetCDF files")
    parser.add_argument("--input-root", default=r"D:\argo_nc_files")
    parser.add_argument("--output-dir", default=r"D:\argo_derived\inventory")
    parser.add_argument("--limit-files", type=int, help="Only scan a prefix for a smoke test")
    parser.add_argument("--qc-values", default="1,2", help="Accepted QC codes, comma-separated")
    args = parser.parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if args.limit_files is not None and args.limit_files < 1:
        parser.error("limit-files must be positive")
    accepted_qc = tuple(value.strip() for value in args.qc_values.split(",") if value.strip())
    files = sorted(input_root.rglob("*.nc"))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise FileNotFoundError(f"No .nc files below {input_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    first_metadata = None
    for path in tqdm(files, desc="scan Argo files"):
        try:
            if first_metadata is None:
                first_metadata = inspect_metadata(path)
            data = read_profile_arrays(path, accepted_qc)
            temperature = data["temperature"]
            salinity = data["salinity"]
            valid_temperature = data["valid_temperature"]
            valid_salinity = data["valid_salinity"]
            valid_position = np.isfinite(data["latitude"]) & np.isfinite(data["longitude"])
            joint = valid_temperature & valid_salinity
            rows.append({
                "date": data["date"].isoformat() if data["date"] else "",
                "file": str(path),
                "file_size_bytes": path.stat().st_size,
                "n_profiles": data["n_profiles"],
                "profiles_with_valid_position": int(np.count_nonzero(valid_position)),
                "profiles_with_temperature": int(np.count_nonzero(np.any(valid_temperature, axis=1))),
                "profiles_with_salinity": int(np.count_nonzero(np.any(valid_salinity, axis=1))),
                "temperature_points_after_qc": int(np.count_nonzero(valid_temperature)),
                "salinity_points_after_qc": int(np.count_nonzero(valid_salinity)),
                "joint_temperature_salinity_points": int(np.count_nonzero(joint)),
                "temperature_qc_present": int(data["temperature_qc_present"]),
                "salinity_qc_present": int(data["salinity_qc_present"]),
                "temperature_min": float(np.nanmin(np.where(valid_temperature, temperature, np.nan))) if np.any(valid_temperature) else None,
                "temperature_max": float(np.nanmax(np.where(valid_temperature, temperature, np.nan))) if np.any(valid_temperature) else None,
                "salinity_min": float(np.nanmin(np.where(valid_salinity, salinity, np.nan))) if np.any(valid_salinity) else None,
                "salinity_max": float(np.nanmax(np.where(valid_salinity, salinity, np.nan))) if np.any(valid_salinity) else None,
                "status": "ok",
                "error": "",
            })
        except Exception as exc:  # keep the inventory complete and auditable
            failures.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            rows.append({"date": "", "file": str(path), "file_size_bytes": path.stat().st_size, "status": "error", "error": failures[-1]["error"]})

    fieldnames = [
        "date", "file", "file_size_bytes", "n_profiles", "profiles_with_valid_position",
        "profiles_with_temperature", "profiles_with_salinity", "temperature_points_after_qc",
        "salinity_points_after_qc", "joint_temperature_salinity_points", "temperature_qc_present",
        "salinity_qc_present", "temperature_min", "temperature_max", "salinity_min", "salinity_max",
        "status", "error",
    ]
    with (output_dir / "argo_daily_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    by_date = Counter(row["date"] for row in valid_rows if row.get("date"))
    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "files_scanned": len(files),
        "files_ok": len(valid_rows),
        "files_failed": len(failures),
        "unique_dates": len(by_date),
        "duplicate_date_files": sum(1 for count in by_date.values() if count > 1),
        "accepted_qc_values": accepted_qc,
        "date_min": min(by_date) if by_date else None,
        "date_max": max(by_date) if by_date else None,
        "files_with_zero_profiles": sum(int(row.get("n_profiles", 0)) == 0 for row in valid_rows),
        "total_profiles": sum(int(row.get("n_profiles", 0)) for row in valid_rows),
        "total_temperature_points_after_qc": sum(int(row.get("temperature_points_after_qc", 0)) for row in valid_rows),
        "total_salinity_points_after_qc": sum(int(row.get("salinity_points_after_qc", 0)) for row in valid_rows),
        "metadata_from_first_file": first_metadata,
        "failures": failures,
        "note": "This is an inventory and QC audit, not a complete scientific validation of every profile.",
    }
    with (output_dir / "argo_inventory_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: summary[key] for key in ("files_scanned", "files_ok", "files_failed", "unique_dates", "date_min", "date_max", "total_profiles")}, ensure_ascii=False), flush=True)
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
