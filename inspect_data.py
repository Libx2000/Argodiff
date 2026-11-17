from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import netCDF4
import numpy as np

from diffargo.config import load_config
from diffargo.data import GlorysArchive


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect paired annual GLORYS files")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    archive = GlorysArchive(data_cfg)

    if not archive.root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {archive.root}")
    years = archive.available_years()
    if not years:
        raise FileNotFoundError(f"No paired temperature/salinity files below {archive.root}")
    index = archive.build_date_index(years)
    dates = sorted(index)
    coordinates = archive.coordinates()

    missing_dates = []
    for previous, current in zip(dates[:-1], dates[1:]):
        if current - previous != timedelta(days=1):
            missing_dates.append((previous, current))

    sample_state, sample_valid = archive.read_state(dates[0])
    levels = len(coordinates["depth"])
    print(f"root: {archive.root}")
    print(f"paired years: {years[0]}--{years[-1]} ({len(years)} years)")
    print(f"dates: {dates[0]}--{dates[-1]} ({len(dates)} days)")
    print(
        "grid: "
        f"depth={levels}, lat={len(coordinates['lat'])}, lon={len(coordinates['lon'])}"
    )
    print(f"state shape (T then S): {sample_state.shape}")
    print(f"depth/elevation values: {coordinates['depth'].tolist()}")
    print(f"valid fraction: {sample_valid[:levels].mean():.4f}")
    print(f"date discontinuities: {len(missing_dates)}")
    for previous, current in missing_dates[:10]:
        print(f"  gap: {previous} -> {current}")

    for kind, variable_key in (
        ("temperature", "temperature_variable"),
        ("salinity", "salinity_variable"),
    ):
        path = archive._path(kind, years[0])
        with netCDF4.Dataset(path) as dataset:
            variable = dataset.variables[data_cfg[variable_key]]
            print(f"{kind}: {Path(path).name}, dims={variable.dimensions}, dtype={variable.dtype}")

    archive.close()


if __name__ == "__main__":
    main()

