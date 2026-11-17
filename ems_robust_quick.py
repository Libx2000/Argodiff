"""Small, separately versioned robustness screen; no old protocol mutation."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
FILES = ('ems_robust_quick.py', 'ems_quick_observer.py', 'configs/ems_robust_quick_v1.json',
         'submit_ems_robust_quick.slurm', 'ems_chunked_diagnostics.py', 'ems_lowmem.py',
         'ems_resource_probe/detect_resources.py', 'ems_resource_probe/_common.py')
BUILD = 'ems-robust-quick-v1'


def settings(filename='configs/ems_robust_quick_v1.json'):
    import ems_experiments as w
    q = w.read(filename); c = w.settings(q['base_config'])
    if (q['schema'] != BUILD or q['days'] != 60 or q['members'] != 50 or q['sampling_steps'] != 10 or
            q['methods'] != ['guided_raw', 'nudging_selected'] or
            q['scenarios'] != ['nominal', 'columns50', 'outage15'] or
            q['physics_leads'] != [1, 15, 30, 45, 46, 60] or
            q['starts'] != [dict(season='winter', month=1), dict(season='summer', month=7)] or
            q['seed'] != 20261006 or q['chunk_columns'] != 512 or q['output_root'] != 'runs/ems_robust_quick_v1'):
        raise ValueError('This is a fixed reduced screen; changes require a new version')
    tasks = [dict(phase='robust', year=c['development_year'], method=method, scenario=scenario,
                  season=start['season'], month=start['month'], seed=q['seed'], days=q['days'],
                  members=q['members'], steps=q['sampling_steps'])
             for scenario in q['scenarios'] for method in q['methods'] for start in q['starts']]
    return q, c, tasks


def hashes():
    import ems_experiments as w
    return {name: w.sha(name) for name in FILES}


def checked_protocol(path):
    import ems_experiments as w
    record = w.read(path)
    if record['build'] != BUILD or record['files'] != hashes():
        raise RuntimeError('Quick-screen sources changed; preserve protocol, do not overwrite it')
    for file, digest in record['base_dependencies'].items():
        if w.sha(file) != digest: raise RuntimeError('Original dependency changed: ' + file)
    return record


@contextmanager
def lock(filename, wait=False):
    import fcntl
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('a') as stream:
        deadline = time.monotonic() + 30
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if not wait or time.monotonic() >= deadline: raise
                time.sleep(.1)
        yield


def initialize(q, c, tasks):
    import ems_experiments as w
    w.verify(c)
    root = w.path(q['output_root']); record = dict(build=BUILD, quick=q, tasks=tasks, files=hashes(),
        base_dependencies={str(w.path(c['output_root']) / name): digest
                           for name, digest in w.dependencies(tasks[0], c).items()},
        original_protocol_complete=False, exploratory_screen=True, formal_final_authorized=False)
    # A shared init lock makes the two method jobs safe to launch independently.
    with lock(root / 'init.lock', wait=True):
        destination = root / 'protocol.json'
        if destination.exists():
            if checked_protocol(destination) != record: raise RuntimeError('Quick protocol differs; preserve it')
        else: w.write_new(destination, record)
    return root, record


def folder(root, task):
    return root / 'cases' / task['scenario'] / task['method'] / task['season'] / str(task['seed'])


def completed(root, task):
    import ems_experiments as w
    filename = folder(root, task) / 'completed.json'
    if not filename.exists(): return None
    r = w.read(filename)
    if r['task'] != task or r['protocol_sha256'] != w.sha(root / 'protocol.json'):
        raise RuntimeError('Quick case identity/protocol changed: ' + str(filename))
    for name, digest in r['outputs'].items():
        if w.sha(filename.parent / r['attempt'] / name) != digest:
            raise RuntimeError('Quick archived output changed: ' + name)
    return r


def tail(filename, limit=12000):
    if not filename.is_file(): return ''
    with filename.open('rb') as stream:
        stream.seek(max(0, filename.stat().st_size - limit)); return stream.read(limit).decode('utf-8', errors='replace')


def run_case(index, task, q, c, root):
    import ems_experiments as w
    base = folder(root, task)
    with lock(base / 'task.lock'):
        if completed(root, task): print('SKIP verified quick task', index, flush=True); return
        number = 1
        while (base / f'attempt_{number:03d}').exists(): number += 1
        out = base / f'attempt_{number:03d}'; out.mkdir()
        original = w.build_command(task, c, out)
        if original[:3] != [sys.executable, '-u', 'ems_observer.py']:
            raise RuntimeError('Unexpected original forecast command')
        command = [sys.executable, '-B', '-u', 'ems_quick_observer.py', '--quick-protocol', str(root / 'protocol.json'), *original[3:]]
        w.write_new(out / 'recipe.json', dict(task=task, command=command, original_command=original,
                                             quick_protocol_sha256=w.sha(root / 'protocol.json')))
        print('START quick task', index, task, 'child_log=', out / 'execution.log', flush=True)
        started = time.perf_counter()
        with (out / 'execution.log').open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=w.ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while True:
                try: code = process.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    lines = tail(out / 'execution.log').splitlines()
                    print('HEARTBEAT task', index, 'elapsed_min=', round((time.perf_counter() - started) / 60, 1),
                          'last=', lines[-1][-500:] if lines else 'loading', flush=True)
        if code:
            print(tail(out / 'execution.log'), file=sys.stderr, flush=True)
            raise RuntimeError('Quick child failed; preserved log: ' + str(out / 'execution.log'))
        for name in ('run_summary.csv', 'paper_grid_metrics.csv', 'paper_holdout_metrics.csv', 'quick_day_timing.csv'):
            with (out / name).open(encoding='utf-8', newline='') as stream:
                leads = {int(row['lead']) for row in csv.DictReader(stream)}
            if leads != set(range(1, task['days'] + 1)): raise RuntimeError('Incomplete daily quick output: ' + name)
        if w.read(out / 'joint_status.json')['completed_leads'] != q['physics_leads']:
            raise RuntimeError('Scheduled physics output incomplete')
        with (out / 'ems_multivariate.csv').open(encoding='utf-8', newline='') as stream:
            if {int(r['lead']) for r in csv.DictReader(stream)} != set(q['physics_leads']):
                raise RuntimeError('Scheduled profile output incomplete')
        with (out / 'quick_support.jsonl').open(encoding='utf-8') as stream:
            if [json.loads(line)['lead'] for line in stream] != list(range(1, task['days'] + 1)):
                raise RuntimeError('Daily support audit incomplete')
        w.verify(c); checked_protocol(root / 'protocol.json')
        w.write_new(out / 'allocation_cost.json', dict(subprocess_wall_seconds=time.perf_counter() - started,
                    scope='New quick protocol only; includes T/S and scheduled physics; not comparable to old full-diagnostic timing'))
        w.write_new(out / 'completed.pending.json', dict(task=task, attempt=out.name,
                    protocol_sha256=w.sha(root / 'protocol.json'),
                    outputs={p.name: w.sha(p) for p in out.iterdir() if p.is_file()}))
        os.replace(out / 'completed.pending.json', base / 'completed.json')
        print('DONE quick task', index, 'wall_minutes=', round((time.perf_counter() - started) / 60, 2), flush=True)


def inspect_original(c):
    """Read existing child logs/rows, not device state on the login node."""
    import ems_experiments as w
    marked = 0; records = []
    for index, task in enumerate(w.task_list('robust', c)):
        base = w.folder(task, c)
        if (base / 'completed.json').is_file(): marked += 1; continue
        attempts = sorted(base.glob('attempt_*'))
        if not attempts: continue
        out = attempts[-1]; row = None; recent = []
        file = out / 'run_summary.csv'
        if file.is_file():
            with file.open(encoding='utf-8', newline='') as stream:
                for item in csv.DictReader(stream):
                    try: elapsed = float(item['wall_seconds']); int(item['lead'])
                    except (TypeError, ValueError, KeyError): continue  # possible concurrently written last row
                    row = item; recent.append(elapsed)
        records.append(dict(task_id=index, task=task, last_lead=int(row['lead']) if row else None,
                            recent_day_seconds=round(sum(recent[-5:]) / len(recent[-5:]), 2) if recent else None,
                            recent_log=tail(out / 'execution.log', 3000).splitlines()[-3:],
                            note='Attempt may be active OR abandoned; device/host contention not inferred from logs'))
    print(json.dumps(dict(original_completed_markers_found=marked, markers_not_hash_verified=True,
                          unfinished_attempts=records), indent=2, ensure_ascii=False), flush=True)


def summarize(q, tasks, root):
    """Paired global daily T/S descriptives; physical files remain separately auditable."""
    import numpy as np
    import ems_experiments as w
    data = {}; costs = []
    for task in tasks:
        record = completed(root, task)
        if not record: raise RuntimeError('Quick summary requires all twelve cases: ' + str(task))
        out = folder(root, task) / record['attempt']
        with (out / 'quick_support.jsonl').open(encoding='utf-8') as stream:
            support = {int(row['lead']): row for row in map(json.loads, stream)}
        with (out / 'paper_grid_metrics.csv').open(encoding='utf-8', newline='') as stream:
            rows = [row for row in csv.DictReader(stream) if row['depth_index'] == '-1']
        data[task['scenario'], task['method'], task['season']] = (rows, support)
        costs.append(dict(task=task, **w.read(out / 'allocation_cost.json'), stage_timing=str(out / 'quick_day_timing.csv')))
    results = []
    for scenario in q['scenarios']:
        for variable in ('temperature', 'salinity'):
            left = []; right = []; mismatches = []
            for start in q['starts']:
                a, sa = data[scenario, 'guided_raw', start['season']]
                b, sb = data[scenario, 'nudging_selected', start['season']]
                aa = {int(r['lead']): r for r in a if r['variable'] == variable}
                bb = {int(r['lead']): r for r in b if r['variable'] == variable}
                if set(aa) != set(range(1, 61)) or set(bb) != set(aa): raise RuntimeError('Incomplete T/S series')
                for lead in sorted(aa):
                    x, y = aa[lead], bb[lead]
                    for z, declared in ((x, sa[lead][variable]), (y, sb[lead][variable])):
                        if int(z['count']) != declared['count'] or not np.isclose(float(z['weight']), declared['weight'], rtol=1e-12, atol=1e-12):
                            raise RuntimeError('Daily T/S support sidecar disagrees with original score')
                    if sa[lead][variable] != sb[lead][variable] or x['date'] != y['date']:
                        mismatches.append(dict(season=start['season'], lead=lead))
                    left.append(x); right.append(y)
            entry = dict(scenario=scenario, variable=variable, comparison='guided_raw versus nudging_selected',
                         status='matched' if not mismatches else 'unavailable_support_mismatch', mismatches=mismatches)
            if not mismatches:
                def pooled(rows):
                    weight = sum(float(r['weight']) for r in rows)
                    return dict(rmse=math.sqrt(sum(float(r['weight']) * float(r['rmse'])**2 for r in rows) / weight),
                                crps=sum(float(r['weight']) * float(r['crps']) for r in rows) / weight,
                                coverage90=sum(float(r['weight']) * float(r['coverage90']) for r in rows) / weight)
                entry.update(diffusion=pooled(left), nudging=pooled(right))
            results.append(entry)
    payload = dict(build=BUILD, exploratory_screen=True, original_robust_complete=False,
                   formal_final_authorized=False, settings=q, ts_comparisons=results, costs=costs,
                   scheduled_physics_note='Only declared leads, not daily physical stability; cross-method physical support must be audited before paired claims',
                   limitations=['Two fixed starts and one seed; no independent-year inference',
                                '10 steps is a different sampler budget; use its own nominal reference',
                                'Unconfirmed Argo source/QC limitations unchanged',
                                'No claim about the omitted selected physical model, Gaussian baseline, noise scenarios or four-season robustness'])
    file = root / 'summary_quick.json'
    if file.exists():
        if w.read(file) != payload: raise RuntimeError('Quick summary changed; preserve it')
    else: w.write_new(file, payload)
    print('Saved', file, flush=True)


def main():
    import ems_experiments as w
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('inspect', 'pilot', 'diffusion', 'nudging', 'all', 'status', 'summarize'))
    args = p.parse_args(); q, c, tasks = settings()
    if args.action == 'inspect': inspect_original(c); return
    if args.action in ('status', 'summarize'):
        root = w.path(q['output_root']); w.verify(c); checked_protocol(root / 'protocol.json')
        if args.action == 'summarize': summarize(q, tasks, root)
        else: print('QUICK_STATUS', json.dumps(dict(total=12, completed=[i for i, t in enumerate(tasks) if completed(root, t)])))
        return
    root, _ = initialize(q, c, tasks)
    for index, task in enumerate(tasks):
        if args.action == 'pilot' and index != 0: continue
        if args.action == 'diffusion' and task['method'] != 'guided_raw': continue
        if args.action == 'nudging' and task['method'] != 'nudging_selected': continue
        if completed(root, task): print('SKIP verified quick task', index, flush=True); continue
        if time.time() + 1800 > float(os.environ.get('EMS_QUICK_DEADLINE', 'inf')):
            print('Wall-time budget stop; repeat same command to skip completed cases. Failed partial cases restart from day 1.', flush=True); return
        run_case(index, task, q, c, root)


if __name__ == '__main__':
    main()
