"""Daily original T/S verification, scheduled original physical verification."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: refusing a silent CPU fallback for this time-limited experiment')
    # Tiny normal runtime check, not a benchmark or memory stress test.
    x = torch.ones(1, device='cuda'); (x + 1).sum().item()
    print('CUDA_RUNTIME_OK', json.dumps(dict(torch=torch.__version__, name=torch.cuda.get_device_name(),
                                            visible_devices=torch.cuda.device_count())), flush=True)


def support_record(dataset, target, lead, analysis, ensemble, deterministic):
    import numpy as np
    raw, valid = dataset.archive.read_state(target)
    levels = dataset.levels
    coords = dataset.archive.coordinates()
    geometry = hashlib.sha256(b''.join(np.asarray(coords[n], dtype='<f8').tobytes()
                                      for n in ('lat', 'lon', 'depth'))).digest()
    en = ensemble[:1] if deterministic else ensemble
    result = dict(date=str(target), lead=lead)
    for name, sl in (('temperature', slice(0, levels)), ('salinity', slice(levels, 2 * levels))):
        actual = np.where(valid[sl], raw[sl], np.nan)
        mask = np.isfinite(actual) & np.isfinite(analysis[sl]) & np.all(np.isfinite(en[:, sl]), axis=0)
        weight = np.broadcast_to(dataset.latitude_weights, actual.shape).astype(float)
        result[name] = dict(count=int(mask.sum()), weight=float(weight[mask].sum()),
                            support_digest=hashlib.sha256(geometry + np.packbits(mask).tobytes()).hexdigest())
    return result


def run(arguments):
    import ems_robust_quick as quick
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--quick-protocol', required=True)
    args, rest = parser.parse_known_args(arguments)
    record = quick.checked_protocol(Path(args.quick_protocol))
    require_cuda()
    from ems_lowmem import resource_info
    resource_info()
    q = argparse.ArgumentParser(add_help=False)
    q.add_argument('--ems-observer', required=True); q.add_argument('--output-dir', required=True)
    opts, _ = q.parse_known_args(rest)
    definition = json.loads(Path(opts.ems_observer).read_text(encoding='utf-8'))
    scheduled = set(record['quick']['physics_leads'])
    from ems_chunked_diagnostics import ChunkedJointObserver
    from ems_observer import ProfileObserver
    observer = ChunkedJointObserver(definition['config'], definition['candidates'], opts.output_dir, 'validate',
                                    chunk_columns=record['quick']['chunk_columns'])
    profiles = ProfileObserver(definition['config'], definition['candidates'], opts.output_dir)
    import paper_metrics
    from ems_cycle import main as cycle
    original, saved = paper_metrics.evaluate_day, sys.argv
    output = Path(opts.output_dir)
    def evaluate(dataset, target, lead, analysis, ensemble, holdout_root, output_dir, deterministic=False):
        started = time.perf_counter()
        original(dataset, target, lead, analysis, ensemble, holdout_root, output_dir, deterministic)
        ts_seconds = time.perf_counter() - started
        support = support_record(dataset, target, lead, analysis, ensemble, deterministic)
        with (output / 'quick_support.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(support, allow_nan=False) + '\n')
        physical_seconds = profile_seconds = 0.0
        if lead in scheduled:
            before = time.perf_counter()
            observer(dataset, target, lead, analysis, ensemble, holdout_root, output_dir, deterministic)
            physical_seconds = time.perf_counter() - before
            before = time.perf_counter()
            profiles(dataset, target, lead, analysis, ensemble, deterministic)
            profile_seconds = time.perf_counter() - before
        row = dict(date=str(target), lead=lead, ts_seconds=ts_seconds, physical_seconds=physical_seconds,
                   profile_seconds=profile_seconds, scheduled_physics=int(lead in scheduled))
        file = output / 'quick_day_timing.csv'; exists = file.exists()
        with file.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists: writer.writeheader()
            writer.writerow(row)
        print('QUICK_DAY', json.dumps(row), flush=True)
    paper_metrics.evaluate_day = evaluate; sys.argv = ['ems_cycle.py', *rest]
    # Remove only the observer option; the forecast arguments stay unchanged.
    r = argparse.ArgumentParser(add_help=False); r.add_argument('--ems-observer', required=True)
    _, forwarded = r.parse_known_args(rest); sys.argv = ['ems_cycle.py', *forwarded]
    try:
        cycle()
    finally:
        paper_metrics.evaluate_day = original; sys.argv = saved


if __name__ == '__main__':
    run(sys.argv[1:])
