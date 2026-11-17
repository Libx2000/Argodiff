"""Reproducible stress tests of mapped observation COLUMNS, not raw profiles.

No forecast/reference values enter selection. Holdout files are never modified.
Noise is a declared synthetic perturbation, not an Argo instrument-error estimate.
"""
from __future__ import annotations

import hashlib
from datetime import date
import numpy as np


def validate_options(options):
    o = dict(retention=1.0, temperature_noise=0.0, salinity_noise=0.0,
             outage_leads=[], observation_seed=20261018, start_date=None, **{})
    o.update(options)
    if not 0 <= float(o['retention']) <= 1:
        raise ValueError('retention must be within [0,1]')
    if any(not np.isfinite(float(o[k])) or float(o[k]) < 0
           for k in ('temperature_noise', 'salinity_noise')):
        raise ValueError('Synthetic noise standard deviations must be finite and nonnegative')
    if o['start_date'] is None:
        raise ValueError('Explicit last-analysis start_date required')
    date.fromisoformat(o['start_date'])
    if any(not isinstance(x, int) or x < 1 for x in o['outage_leads']):
        raise ValueError('outage_leads must contain positive integers')
    return o


def column_key(seed, day, i, j):
    return hashlib.sha256(f'{int(seed)}|{day}|{int(i)}|{int(j)}'.encode('ascii')).digest()


def transform_observations(values, mask, target, options):
    """Keep nested spatial subsets and matched noise across all compared methods."""
    o = validate_options(options)
    values = np.asarray(values, np.float32).copy()
    mask = np.asarray(mask, bool).copy()
    if values.shape != mask.shape or values.ndim != 3 or values.shape[0] % 2:
        raise ValueError('Expected matching (2*depth,latitude,longitude) arrays')
    if not np.isfinite(values[mask]).all():
        raise FloatingPointError('Nonfinite valid input observation')
    lead = (date.fromisoformat(str(target)) - date.fromisoformat(o['start_date'])).days
    if lead < 1:
        raise ValueError('Observation day must follow the declared initial analysis')
    levels = len(values) // 2
    columns = np.argwhere(mask.any(axis=0))
    before = int(mask.sum())
    in_outage = lead in o['outage_leads']
    selected = []
    for i, j in columns:
        key = column_key(o['observation_seed'], target, i, j)
        uniform = int.from_bytes(key[:8], 'little') / 2**64
        if in_outage or uniform >= o['retention']:
            mask[:, i, j] = False
            continue
        selected.append((int(i), int(j)))
        if o['temperature_noise'] or o['salinity_noise']:
            rng = np.random.default_rng(int.from_bytes(key[8:16], 'little'))
            noise = rng.normal(size=2*levels)
            noise[:levels] *= o['temperature_noise']
            noise[levels:] *= o['salinity_noise']
            values[:, i, j] += np.where(mask[:, i, j], noise, 0).astype(np.float32)
    # No clipping to disguise outliers. The unchanged engine's documented range
    # and finite-value filters are applied again after adding synthetic noise.
    bounds = ((-3., 40.), (0., 42.))
    range_valid = np.zeros_like(mask)
    for sl, (lower, upper) in zip((slice(0, levels), slice(levels, None)), bounds):
        range_valid[sl] = np.isfinite(values[sl]) & (values[sl] >= lower) & (values[sl] <= upper)
    rejected_noise = int((mask & ~range_valid).sum())
    mask &= range_valid
    return values, mask, dict(
        date=str(target), lead=lead, outage=in_outage,
        input_columns=len(columns), retained_columns=int(mask.any(axis=0).sum()),
        input_values=before, retained_values=int(mask.sum()),
        values_rejected_after_synthetic_noise=rejected_noise,
        retained_column_digest=hashlib.sha256(np.asarray(selected, dtype='<i8').reshape(-1,2).tobytes()).hexdigest(),
        selection_unit='mapped_grid_column; not native Argo profile',
        synthetic_noise_not_sensor_error=True)
