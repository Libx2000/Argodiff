"""Shared, conservative readers for daily Argo NetCDF profile files."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np


TEMP_NAMES = ("TEMP", "temperature", "TEMP_ADJUSTED", "temperature_adjusted")
PSAL_NAMES = ("PSAL", "salinity", "PSAL_ADJUSTED", "salinity_adjusted")
PRES_NAMES = ("PRES", "pressure", "DEPTH", "depth")
LAT_NAMES = ("LATITUDE", "latitude", "lat")
LON_NAMES = ("LONGITUDE", "longitude", "lon")
TIME_NAMES = ("JULD", "time", "TIME", "date")


def resolve_name(dataset: netCDF4.Dataset, candidates: Iterable[str], required: bool = True) -> str | None:
    names = {name.lower(): name for name in dataset.variables}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    if required:
        raise KeyError(f"None of {tuple(candidates)} found; variables={sorted(dataset.variables)}")
    return None


def parse_filename_date(path: str | Path) -> date | None:
    stem = Path(path).stem
    for length, fmt in ((8, "%Y%m%d"), (6, "%Y%m")):
        if len(stem) >= length:
            for offset in range(len(stem) - length + 1):
                token = stem[offset : offset + length]
                if token.isdigit():
                    try:
                        return datetime.strptime(token, fmt).date()
                    except ValueError:
                        pass
    return None


def masked_array(variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values)


def profile_count(dataset: netCDF4.Dataset, latitude_name: str) -> int:
    values = masked_array(dataset.variables[latitude_name]).reshape(-1)
    return int(values.size)


def orient_profile_level(values: np.ndarray, n_profiles: int) -> np.ndarray:
    """Return an array shaped (profile, level), tolerating level/profile order."""
    values = np.asarray(values)
    if values.ndim == 1:
        if values.size == n_profiles:
            return values.reshape(n_profiles, 1)
        return values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D profile variable, got shape={values.shape}")
    if values.shape[0] == n_profiles:
        return values
    if values.shape[1] == n_profiles:
        return values.T
    raise ValueError(f"Cannot identify profile axis: shape={values.shape}, n_profiles={n_profiles}")


def qc_mask(values: np.ndarray | None, shape: tuple[int, ...], accepted=("1", "2")) -> np.ndarray:
    if values is None:
        return np.ones(shape, dtype=bool)
    values = np.asarray(values)
    if values.shape != shape:
        values = orient_profile_level(values, shape[0])
    if values.dtype.kind in {"S", "U", "O"}:
        text = values.astype("U1")
        return np.isin(text, np.asarray(accepted, dtype="U1"))
    numeric = values.astype(np.float64, copy=False)
    return np.isin(numeric, np.asarray([int(value) for value in accepted], dtype=np.float64))


def find_qc(dataset: netCDF4.Dataset, variable_name: str) -> str | None:
    base = variable_name.lower()
    candidates = [f"{variable_name}_QC", f"{variable_name}_qc"]
    if "_adjusted" in base:
        candidates.extend([variable_name.replace("_ADJUSTED", "_QC"), variable_name.replace("_adjusted", "_qc")])
    else:
        candidates.extend([f"{variable_name}_ADJUSTED_QC", f"{variable_name}_adjusted_qc"])
    return resolve_name(dataset, candidates, required=False)


def read_profile_arrays(path: str | Path, accepted_qc=("1", "2")) -> dict:
    """Read one file and return raw profile arrays plus valid masks."""
    with netCDF4.Dataset(path, "r") as dataset:
        temp_name = resolve_name(dataset, TEMP_NAMES)
        psal_name = resolve_name(dataset, PSAL_NAMES)
        pres_name = resolve_name(dataset, PRES_NAMES)
        lat_name = resolve_name(dataset, LAT_NAMES)
        lon_name = resolve_name(dataset, LON_NAMES)
        n_profiles = profile_count(dataset, lat_name)
        latitude = masked_array(dataset.variables[lat_name]).reshape(-1).astype(np.float32)
        longitude = masked_array(dataset.variables[lon_name]).reshape(-1).astype(np.float32)
        pressure = orient_profile_level(masked_array(dataset.variables[pres_name]), n_profiles).astype(np.float32)
        temperature = orient_profile_level(masked_array(dataset.variables[temp_name]), n_profiles).astype(np.float32)
        salinity = orient_profile_level(masked_array(dataset.variables[psal_name]), n_profiles).astype(np.float32)
        temp_qc_name = find_qc(dataset, temp_name)
        psal_qc_name = find_qc(dataset, psal_name)
        temp_qc = None if temp_qc_name is None else orient_profile_level(masked_array(dataset.variables[temp_qc_name]), n_profiles)
        psal_qc = None if psal_qc_name is None else orient_profile_level(masked_array(dataset.variables[psal_qc_name]), n_profiles)

        # Argo pressure is normally positive; some exports store depth as a
        # negative coordinate, so retain either sign and normalize in the
        # interpolation step.
        # Several historical daily exports use -99.999 as an undocumented
        # fill value (their ``missing_value`` attribute says -999).  Treat
        # the conventional negative sentinel as missing before QC and
        # interpolation; otherwise salinity can contaminate every mapped cell.
        valid_pressure = np.isfinite(pressure) & (pressure > -90.0)
        valid_temperature = (
            np.isfinite(temperature) & (temperature > -90.0) & valid_pressure
            & qc_mask(temp_qc, temperature.shape, accepted_qc)
        )
        valid_salinity = (
            np.isfinite(salinity) & (salinity > -90.0) & valid_pressure
            & qc_mask(psal_qc, salinity.shape, accepted_qc)
        )
        file_date = parse_filename_date(path)
        time_name = resolve_name(dataset, TIME_NAMES, required=False)
        if time_name is not None:
            time_variable = dataset.variables[time_name]
            time_values = masked_array(time_variable).reshape(-1)
            time_units = getattr(time_variable, "units", None)
            if time_units:
                try:
                    decoded = netCDF4.num2date(
                        time_values, units=time_units,
                        calendar=getattr(time_variable, "calendar", "standard"),
                        only_use_cftime_datetimes=True,
                    )
                    if file_date is None and len(decoded):
                        value = decoded[0]
                        file_date = date(int(value.year), int(value.month), int(value.day))
                except (ValueError, TypeError, OverflowError):
                    pass
        return {
            "temperature": temperature,
            "salinity": salinity,
            "pressure": pressure,
            "latitude": latitude,
            "longitude": longitude,
            "valid_temperature": valid_temperature,
            "valid_salinity": valid_salinity,
            "temperature_qc_present": temp_qc_name is not None,
            "salinity_qc_present": psal_qc_name is not None,
            "date": file_date,
            "n_profiles": n_profiles,
            "temperature_name": temp_name,
            "salinity_name": psal_name,
            "pressure_name": pres_name,
            "latitude_name": lat_name,
            "longitude_name": lon_name,
        }


def inspect_metadata(path: str | Path) -> dict:
    with netCDF4.Dataset(path, "r") as dataset:
        dimensions = {
            name: (None if dimension.isunlimited() else len(dimension))
            for name, dimension in dataset.dimensions.items()
        }
        variables = {
            name: {
                "dimensions": tuple(variable.dimensions),
                "shape": tuple(variable.shape),
                "dtype": str(variable.dtype),
            }
            for name, variable in dataset.variables.items()
        }
        selected = {
            "temperature": resolve_name(dataset, TEMP_NAMES, required=False),
            "salinity": resolve_name(dataset, PSAL_NAMES, required=False),
            "pressure": resolve_name(dataset, PRES_NAMES, required=False),
            "latitude": resolve_name(dataset, LAT_NAMES, required=False),
            "longitude": resolve_name(dataset, LON_NAMES, required=False),
        }
        return {
            "dimensions": dimensions,
            "variables": variables,
            "selected_variables": selected,
            "file_date_from_name": parse_filename_date(path).isoformat() if parse_filename_date(path) else None,
        }
