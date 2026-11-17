"""Streaming hydrographic, thermodynamic and multivariate verification.

Full-grid legacy-compatible scores are retained. Additional joint-profile scores
use a fixed random sample of statically complete columns and reference-defined
finite support. Any nonfinite prediction on that support is an ERROR, not an
opportunity to remove an inconvenient profile. No future truth enters sampling.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def profile_scores(ensemble, truth, weights=None):
    """Empirical energy and adjacent-component variogram scores, exponent 1/2.

    Inputs: member x component x column, and component x column. Components must
    already be standardized by TRAINING scales. This is a finite empirical
    ensemble score, not a claim of unbiased scoring of the latent distribution.
    """
    e, y = np.asarray(ensemble, float), np.asarray(truth, float)
    if e.ndim != 3 or y.shape != e.shape[1:] or e.shape[0] < 1:
        raise ValueError('Expected member/component/column arrays')
    if not np.isfinite(e).all() or not np.isfinite(y).all():
        raise FloatingPointError('Joint scores require finite declared support')
    w = np.ones(y.shape[1]) if weights is None else np.asarray(weights, float)
    if w.shape != (y.shape[1],) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Invalid profile weights')
    if not len(w):
        return dict(count=0, weight=0., energy_score=None, variogram_score=None)
    m = len(e)
    first = np.linalg.norm(e-y[None], axis=1).mean(axis=0)
    pair = np.zeros_like(first)
    for member in e:
        pair += np.linalg.norm(e-member[None], axis=1).sum(axis=0)
    energy = first - pair/(2*m*m)
    # Adjacent depth pairs within each of the two physical variables; never
    # connect the deepest temperature to the shallowest salinity.
    if y.shape[0] % 2:
        raise ValueError('Expected two variables with equal depth counts')
    levels = y.shape[0]//2
    if levels < 2:
        raise ValueError('At least two depths required')
    indices = np.r_[np.arange(levels-1), np.arange(levels, 2*levels-1)]
    observed = np.sqrt(np.abs(y[indices+1]-y[indices]))
    predicted = np.sqrt(np.abs(e[:,indices+1]-e[:,indices])).mean(axis=0)
    variogram = ((observed-predicted)**2).mean(axis=0)
    return dict(count=len(w), weight=float(w.sum()),
                energy_score=float(np.dot(w, energy)/w.sum()),
                variogram_score=float(np.dot(w, variogram)/w.sum()))


class ProfileObserver:
    def __init__(self, config, candidates, output):
        self.c, self.candidates, self.out = config, candidates, Path(output)
        self.indices = None

    def __call__(self, dataset, target, lead, analysis, ensemble, deterministic):
        from oe_joint_calibration import calibrated_members, lead_factors
        coords = dataset.archive.coordinates()
        depth = np.abs(coords['depth'])
        order = np.argsort(depth)
        levels = len(depth)
        if self.indices is None:
            eligible = np.flatnonzero(np.all(dataset.static_mask.astype(bool), axis=0))
            if not len(eligible):
                raise ValueError('No statically complete columns for joint-profile evaluation')
            rng = np.random.default_rng(self.c['profile_sample_seed'])
            self.indices = np.sort(rng.choice(eligible, size=min(len(eligible), self.c['profile_sample_points']), replace=False))
            np.savez_compressed(self.out/'ems_profile_sample.npz', indices=self.indices,
                latitude=dataset.lat, longitude=dataset.lon, depth_m=depth,
                eligible_columns=np.asarray(len(eligible)), seed=np.asarray(self.c['profile_sample_seed']))
        i,j = self.indices//len(dataset.lon), self.indices%len(dataset.lon)
        raw, valid = dataset.archive.read_state(target)
        truth = raw[:,i,j].astype(float)
        support = np.all(valid[:,i,j] & np.isfinite(truth), axis=0)
        ens = ensemble[:1] if deterministic else ensemble
        ens = ens[:,:,i,j]
        center = analysis[:,i,j]
        if not np.isfinite(ens[:,:,support]).all():
            raise FloatingPointError('A forecast is invalid on fixed reference-defined profile support')
        with np.load(self.c['state_stats'], allow_pickle=False) as stats:
            std = stats['state_std'].astype(float)
        if std.shape != (2*levels,) or (std <= 0).any() or not np.isfinite(std).all():
            raise ValueError('Invalid TRAINING normalization for joint-profile scores')
        channel_order = np.r_[order, order+levels]
        digest = hashlib.sha256(self.indices.astype('<i8').tobytes()+np.packbits(support).tobytes()).hexdigest()
        factors = lead_factors(self.c['reference_calibration'], lead, 2*levels)
        candidates = [dict(id='point',kind='raw')] if deterministic else self.candidates
        rows = []
        for candidate, members, _ in calibrated_members(ens, center, depth, dataset.lat[i], dataset.lon[j],
                factors, candidates, self.c['regions'], self.c['regional_feather_degrees']):
            en = members[:,channel_order][:,:,support]/std[channel_order][None,:,None]
            tr = truth[channel_order][:,support]/std[channel_order,None]
            scores = profile_scores(en, tr, np.cos(np.deg2rad(dataset.lat[i[support]])))
            rows.append(dict(date=str(target), lead=lead, candidate=candidate['id'],
                members=len(en), support_digest=digest, selected_columns=len(i), **scores))
        path = self.out/'ems_multivariate.csv'
        exists = path.exists()
        with path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            if not exists: writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ems-observer', required=True)
    args, rest = parser.parse_known_args()
    definition = json.loads(Path(args.ems_observer).read_text(encoding='utf-8'))
    q = argparse.ArgumentParser(add_help=False)
    q.add_argument('--output-dir', required=True)
    out, _ = q.parse_known_args(rest)
    import paper_metrics
    from oe_joint_diagnostics import JointObserver
    from ems_cycle import main as cycle
    observer = JointObserver(definition['config'], definition['candidates'], out.output_dir, 'validate')
    profiles = ProfileObserver(definition['config'], definition['candidates'], out.output_dir)
    original = paper_metrics.evaluate_day
    def evaluate(dataset, target, lead, analysis, ensemble, holdout_root, output_dir, deterministic=False):
        original(dataset, target, lead, analysis, ensemble, holdout_root, output_dir, deterministic)
        observer(dataset, target, lead, analysis, ensemble, holdout_root, output_dir, deterministic)
        profiles(dataset, target, lead, analysis, ensemble, deterministic)
    saved = sys.argv
    paper_metrics.evaluate_day = evaluate
    sys.argv = [sys.argv[0]] + rest
    try:
        cycle()
    finally:
        paper_metrics.evaluate_day = original
        sys.argv = saved


if __name__ == '__main__':
    main()
