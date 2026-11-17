"""Streaming grid and held-profile scores, evaluated only after sampling ends."""
from pathlib import Path
import csv
import numpy as np


def score_samples(pred, truth, ensemble, weights=None):
    original_shape = np.shape(pred)
    pred, truth = np.asarray(pred).ravel(), np.asarray(truth).ravel()
    members = ensemble.shape[0]
    ens = ensemble.reshape(members, -1)
    valid = np.isfinite(pred) & np.isfinite(truth) & np.all(np.isfinite(ens), axis=0)
    w = np.ones_like(pred, dtype=np.float64) if weights is None else np.broadcast_to(weights, original_shape).ravel().astype(np.float64)
    if len(w) != len(pred):
        raise ValueError("Weights and observations have different lengths")
    pred, truth, ens, w = pred[valid], truth[valid], ens[:, valid], w[valid]
    count, total = int(valid.sum()), float(w.sum())
    if not count or total <= 0:
        return {"count":0, "weight":0.0, **{k:float("nan") for k in
            ("rmse", "bias", "mae", "rms_spread", "spread_skill_ratio", "crps", "range_coverage", "coverage90")}}
    delta = pred-truth
    ordered = np.sort(ens, axis=0)
    coefficients = 2*np.arange(members)-members+1
    crps = np.mean(np.abs(ens-truth), axis=0) - coefficients @ ordered/(members**2)
    mean = lambda x: float(np.dot(w, x)/total)
    rmse = np.sqrt(mean(delta**2))
    # ddof=0 retained for comparison; finite-ensemble target ~sqrt(M/(M+1)).
    spread = np.sqrt(mean(np.var(ens, axis=0)))
    result = dict(count=count, weight=total, rmse=float(rmse), bias=mean(delta), mae=mean(np.abs(delta)),
                  rms_spread=float(spread), spread_skill_ratio=float(spread/max(rmse, 1e-12)), crps=mean(crps),
                  range_coverage=mean((truth >= ordered[0]) & (truth <= ordered[-1])),
                  coverage90=mean((truth >= np.quantile(ens, .05, axis=0)) & (truth <= np.quantile(ens, .95, axis=0))))
    # Integrate a randomized rank over its interval: uniform under exchangeability,
    # including small ensembles and ties. Never leave the last bin empty for M=8.
    low = (ens < truth).sum(axis=0)/(members+1)
    high = ((ens <= truth).sum(axis=0)+1)/(members+1)
    hist = np.array([mean(np.maximum(0, np.minimum(high, (i+1)/10)
                       - np.maximum(low, i/10))/(high-low)) for i in range(10)])
    result.update({f"rank_bin_{i}":float(v) for i,v in enumerate(hist)})
    return result


def append_rows(path, rows):
    path = Path(path)
    if not rows:
        return
    # Equal schema even on days with no valid held profiles.
    fields = ["date", "lead", "domain", "variable", "depth_index", "depth_m", "members",
              "count", "weight", "rmse", "bias", "mae", "rms_spread", "spread_skill_ratio",
              "crps", "range_coverage", "coverage90"] + [f"rank_bin_{i}" for i in range(10)]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def evaluate_day(dataset, target, lead, analysis, ensemble, holdout_root, output_dir, deterministic=False):
    levels = dataset.levels
    coords = dataset.archive.coordinates()
    truth, valid = dataset.archive.read_state(target)
    ensemble = ensemble[:1] if deterministic else ensemble
    grid_rows, held_rows = [], []
    path = Path(holdout_root)/str(target.year)/f"{target.isoformat()}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Holdout day missing: {path}")
    with np.load(path, allow_pickle=False) as f:
        if not np.allclose(f["latitude"], dataset.lat) or not np.allclose(f["longitude"], dataset.lon):
            raise ValueError("Held observation grid differs from model grid")
        if not np.allclose(f["depth_m"], np.abs(coords["depth"])):
            raise ValueError("Held depth order differs from model channels")
        ii, jj = f["grid_i"].astype(int), f["grid_j"].astype(int)
        held = {v:f[v].copy() for v in ("temperature", "salinity")}
        held_lat = f["profile_latitude"].copy()
    for v, offset in (("temperature",0),("salinity",levels)):
        sl = slice(offset, offset+levels)
        gt = np.where(valid[sl], truth[sl], np.nan)
        weights = np.broadcast_to(dataset.latitude_weights, gt.shape)
        for domain, prediction, actual, ens, ws in (
            ("glorys", analysis[sl], gt, ensemble[:,sl], weights),
            ("holdout", analysis[sl][:,ii,jj].T, held[v], ensemble[:,sl][:,:,ii,jj].transpose(0,2,1), None),
        ):
            rows = grid_rows if domain == "glorys" else held_rows
            for k in [-1]+list(range(levels)):
                if k == -1:
                    pr, tr, en, w = prediction, actual, ens, ws
                elif domain == "glorys":
                    pr, tr, en, w = prediction[k], actual[k], ens[:,k], ws[k]
                else:
                    pr, tr, en, w = prediction[:,k], actual[:,k], ens[:,:,k], None
                stats = score_samples(pr, tr, en, w)
                rows.append(dict(date=target.isoformat(), lead=lead, domain=domain, variable=v,
                                 depth_index=k, depth_m="all" if k < 0 else abs(float(coords["depth"][k])),
                                 members=len(ensemble), **stats))
    append_rows(Path(output_dir)/"paper_grid_metrics.csv", grid_rows)
    append_rows(Path(output_dir)/"paper_holdout_metrics.csv", held_rows)
    # Predeclared latitude bands, not approximations to ocean-basin masks.
    regional = []
    for region, lower, upper in (("tropical",0,20),("midlatitude",20,50),("highlatitude",50,91)):
        grid_mask = (np.abs(dataset.lat) >= lower) & (np.abs(dataset.lat) < upper)
        profile_mask = (np.abs(held_lat) >= lower) & (np.abs(held_lat) < upper)
        for v, offset in (("temperature",0),("salinity",levels)):
            sl = slice(offset, offset+levels)
            gt = np.where(valid[sl], truth[sl], np.nan)
            for domain, pr, tr, en, ws in (
                ("glorys", analysis[sl][:,grid_mask], gt[:,grid_mask], ensemble[:,sl][:,:,grid_mask],
                 np.broadcast_to(dataset.latitude_weights, gt.shape)[:,grid_mask]),
                ("holdout",analysis[sl][:,ii[profile_mask],jj[profile_mask]].T, held[v][profile_mask],
                 ensemble[:,sl][:,:,ii[profile_mask],jj[profile_mask]].transpose(0,2,1),None)):
                for k in [-1]+list(range(levels)):
                    if k < 0:
                        args = (pr,tr,en,ws)
                    elif domain == "glorys":
                        args = (pr[k],tr[k],en[:,k],ws[k])
                    else:
                        args = (pr[:,k],tr[:,k],en[:,:,k],None)
                    regional.append(dict(date=target.isoformat(),lead=lead,domain=domain+":"+region,
                        variable=v,depth_index=k,depth_m="all" if k < 0 else abs(float(coords["depth"][k])),
                        members=len(ensemble),**score_samples(*args)))
    append_rows(Path(output_dir)/"paper_region_metrics.csv",regional)
