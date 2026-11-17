"""Additive v2: retain exact support checks and explicitly report unavailable pairs.

Aggregate CSV scores cannot recover a gridpoint intersection. A support mismatch
therefore produces NO paired difference/interval, not an adjusted count or date
intersection. Other, genuinely matched metrics can still be evaluated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import ems_validate61 as parent

BUILD = 'ems-validate61-support-v2'
FILES = ('ems_validate61_support.py', 'submit_ems_validate61_support.slurm')
OUTPUT = 'validate61_skip37_39_support_v2'
EXTRA_LIMITATIONS = [
    'common means within-case candidate intersection, not automatically cross-method common support.',
    'Support-mismatched metric series are explicitly unavailable for paired inference; no dates are dropped or counts overwritten.',
    'Native-support descriptives use different populations and must not be used to claim improvements.',
    'Stored aggregate scores cannot reconstruct gridpoint common-support scores; replay is needed if those pairs are required.',
]
FIELDS = ('scope', 'method', 'reference', 'candidate', 'region', 'variable', 'metric',
          'season', 'seed', 'date', 'field', 'left', 'right')


def integer(value):
    try:
        x = Decimal(str(value))
        if not x.is_finite() or x != x.to_integral_value() or x < 0:
            raise ValueError('Non-negative integer required')
        return str(int(x))
    except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid count/lead: ' + str(value)) from exc


def index_rows(rows):
    indexed = {}
    normalizations = 0
    for row in rows:
        out = dict(row)
        for name in ('count', 'lead'):
            value = integer(row[name])
            normalizations += str(row[name]) != value
            out[name] = value
        stamp = (out['season'], str(out['seed']), out['date'])
        if stamp in indexed:
            raise ValueError('Duplicate daily row: ' + str(stamp))
        indexed[stamp] = out
    return indexed, normalizations


def inspect_pair(left, right, tasks):
    """Full-series eligibility check; never takes the intersection of dates."""
    import numpy as np
    a, na = index_rows(left); b, nb = index_rows(right)
    expected = {(t['season'], str(t['seed']), lead) for t in tasks for lead in range(1, t['days'] + 1)}
    errors = []
    def error(stamp, field, x, y):
        errors.append(dict(season=stamp[0], seed=stamp[1], date=stamp[2], field=field,
                           left=str(x), right=str(y)))
    for side, indexed in (('left', a), ('right', b)):
        actual = {(r['season'], str(r['seed']), int(r['lead'])) for r in indexed.values()}
        if actual != expected:
            error(('', '', ''), side + '_lead_coverage', sorted(actual), sorted(expected))
    for stamp in sorted(a.keys() | b.keys()):
        if stamp not in a or stamp not in b:
            error(stamp, 'daily_record', 'present' if stamp in a else 'missing',
                  'present' if stamp in b else 'missing')
            continue
        x, y = a[stamp], b[stamp]
        for field in ('lead', 'count', 'support_digest'):
            if str(x[field]) != str(y[field]):
                error(stamp, field, x[field], y[field])
        try:
            wx, wy = float(x['weight']), float(y['weight'])
            vx, vy = float(x['value']), float(y['value'])
            if not np.isfinite([wx, wy, vx, vy]).all() or wx <= 0 or wy <= 0:
                error(stamp, 'nonfinite_or_empty_score', [wx, vx], [wy, vy])
            elif not np.isclose(wx, wy, rtol=1e-12, atol=1e-12):
                error(stamp, 'weight', wx, wy)
        except (TypeError, ValueError, OverflowError):
            error(stamp, 'invalid_numeric_score', x.get('value'), y.get('value'))
    return dict(eligible=not errors, left=list(a.values()), right=list(b.values()), errors=errors,
                formatting_only_normalizations=na + nb)


def native_summary(rows, tasks, root=False):
    """Separate own-support summary; no cross-method difference is calculated."""
    counts = Counter(t['season'] for t in tasks)
    total = numerator = 0.0
    used = 0
    for row in rows:
        weight, value = float(row['weight']), float(row['value'])
        if weight <= 0 or not math.isfinite(weight) or not math.isfinite(value):
            continue
        scale = counts[row['season']]
        total += weight / scale
        numerator += weight * value / scale
        used += 1
    estimate = numerator / total if total else None
    if root and estimate is not None:
        estimate = math.sqrt(max(0, estimate))
    return dict(value=estimate, weighted_support=total, daily_rows=used,
                interpretation='Descriptive only on this method native support; not a paired effect')


def declared_scores(method, config):
    candidates = ('raw', 'marginal') if method.split('|')[1] == 'selected' else ('raw',)
    regions = ['global'] + [r['name'] for r in config['diagnostics']['regions']]
    result = set()
    for candidate in candidates:
        for region in regions:
            for variable in ('temperature', 'salinity', 'rho', 'n2', 'pycnocline_depth'):
                for metric in ('rmse', 'crps', 'coverage90', 'width90'):
                    result.add((candidate, region, variable, metric))
        for metric in ('energy_score', 'variogram_score'):
            result.add((candidate, 'global_complete_columns', 'standardized_TS_profile', metric))
    return result


def audit_pairs(report, config, reader=None):
    from ems_evaluate import read_daily
    reader = read_daily if reader is None else reader
    daily = {case['task_id']: reader(case['output'], case['task'])
             for case in report['all_completed_descriptive']['cases']}
    jobs, audit, mismatches = [], [], []
    for name, scope in report['scopes'].items():
        data = defaultdict(lambda: defaultdict(list)); tasks = defaultdict(list)
        for case in scope['cases']:
            method = parent.key(case['task']); tasks[method].append(case['task'])
            for score, rows in daily[case['task_id']].items():
                data[method][score].extend(rows)
        for method in sorted(tasks):
            reference = 'nominal|deterministic|' + method.split('|')[2]
            if method == reference:
                continue
            # Include absent planned metrics too; never report only the survivors.
            plan = declared_scores(method, config) | data[method].keys()
            for candidate, region, variable, metric in sorted(plan):
                context = dict(scope=name, method=method, reference=reference, candidate=candidate,
                               region=region, variable=variable, metric=metric)
                rows = data[method].get((candidate, region, variable, metric), [])
                other = data[reference].get(('raw', region, variable, metric), [])
                checked = inspect_pair(rows, other, tasks[method])
                item = dict(context, status='matched' if checked['eligible'] else 'unavailable_support_or_coverage',
                            mismatch_rows=len(checked['errors']),
                            formatting_only_normalizations=checked['formatting_only_normalizations'],
                            examples=checked['errors'][:3],
                            native_left=native_summary(rows, tasks[method], metric == 'rmse'),
                            native_right=native_summary(other, tasks[method], metric == 'rmse'))
                audit.append(item)
                mismatches.extend(dict(context, **row) for row in checked['errors'])
                jobs.append(dict(context=context, checked=checked, audit=item,
                                 blocks=scope['paired_blocks']))
        print('Support audit complete:', name, flush=True)
    summary = dict(planned_metric_series=len(audit), matched=sum(x['status'] == 'matched' for x in audit),
                   unavailable=sum(x['status'] != 'matched' for x in audit),
                   unavailable_by_variable=dict(Counter(x['variable'] for x in audit if x['status'] != 'matched')),
                   formatting_only_normalizations=sum(x['formatting_only_normalizations'] for x in audit))
    print('PAIRED_SUPPORT_SUMMARY ' + json.dumps(summary, ensure_ascii=False), flush=True)
    return jobs, dict(build=BUILD, summary=summary, metric_series=audit, limitations=EXTRA_LIMITATIONS), mismatches


def write_differences(filename, rows):
    """Deterministic compressed diagnostic output; existing bytes must agree."""
    content = io.StringIO(newline='')
    writer = csv.DictWriter(content, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    data = gzip.compress(content.getvalue().encode('utf-8'), compresslevel=5, mtime=0)
    if filename.exists():
        if filename.read_bytes() != data:
            raise RuntimeError('Existing support diagnostics differ; preserve: ' + str(filename))
    else:
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open('xb') as stream:
            stream.write(data)


def evaluate(jobs, report, root, repetitions):
    import ems_experiments as w
    from ems_evaluate import paired_blocks
    comparisons = []
    for job in jobs:
        checked, item = job['checked'], job['audit']
        for days in (7, 14, 30):
            row = dict(job['context'], block_days=days, repetitions=repetitions,
                       paired_seed_blocks=job['blocks'])
            if checked['eligible']:
                value = paired_blocks(checked['left'], checked['right'], days, repetitions,
                                      root=(job['context']['metric'] == 'rmse'))
                row.update(status='matched', **value)
            else:
                row.update(status=item['status'], difference_left_minus_right=None,
                           conditional_95_interval=None, native_left=item['native_left'], native_right=item['native_right'],
                           mismatch_rows=item['mismatch_rows'], reason='No paired estimate: spatial/temporal support or coverage differs')
            comparisons.append(row)
    costs = []
    for case in report['all_completed_descriptive']['cases']:
        out = Path(case['output'])
        costs.append(dict(task_id=case['task_id'], task=case['task'],
                          allocation=w.read(out / 'allocation_cost.json'), cycle_loop=w.read(out / 'compute_cost.json')))
    return dict(build=BUILD, comparisons=comparisons, completed_case_costs=costs,
                report_sha256=w.sha(root / 'report_validate61_support.json'),
                support_audit_sha256=w.sha(root / 'support_audit_validate61.json'),
                original_design_complete=False, formal_final_authorized=False,
                limitations=parent.LIMITATIONS + EXTRA_LIMITATIONS)


def gate_diagnostics(report, config, audit):
    critical = [row for row in audit['metric_series'] if row['status'] != 'matched' and
                row['scope'] in ('matched_primary', 'matched_balanced') and
                row['variable'] in ('temperature', 'salinity', 'rho', 'n2')]
    pooled = {name: scope['support_errors'] for name, scope in report['scopes'].items()
              if name in ('matched_primary', 'matched_balanced') and scope['support_errors']}
    if critical or pooled:
        result = dict(passed=False, formal_final_authorized=False, diagnostic_assessments=None,
                      reason='Core gate support is not matched; numerical gate interpretation unavailable.')
    else:
        result = parent.diagnose_gates(report, config)
    result.update(build=BUILD, core_support_verified=not critical and not pooled,
                  unavailable_core_metric_series=len(critical), pooled_support_errors=pooled,
                  all_planned_metrics_paired=(audit['summary']['unavailable'] == 0))
    return result


def main(argv=None):
    import ems_experiments as w
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'all'), nargs='?', default='all')
    parser.add_argument('--policy', default='configs/ems_validate61_v1.json')
    parser.add_argument('--repetitions', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.repetitions < 2:
        parser.error('At least two repetitions required')
    print(BUILD, 'entry=', Path(__file__).resolve(), flush=True)
    policy, config, tasks = parent.policy_settings(args.policy)
    original_root = w.path(config['output_root'])
    source = original_root / policy['analysis_id']; root = original_root / OUTPUT
    with parent.compatible_runtime(config):
        status = parent.inventory(policy, config, tasks)
        if not status['ready']:
            raise RuntimeError('The same 61 cases must remain complete and checksum-valid')
        expected = parent.manifest(args.policy, policy, config, status)
        if w.read(source / 'analysis_protocol.json') != expected:
            raise RuntimeError('Original v1 analysis dependencies changed; preserve them, do not reset')
        report = parent.make_report(policy, config, status)
        if w.read(source / 'report_validate61.json') != report:
            raise RuntimeError('Original v1 report differs from its checked source scores')
        binding = dict(build=BUILD, parent_protocol_sha256=w.sha(source / 'analysis_protocol.json'),
                       parent_report_sha256=w.sha(source / 'report_validate61.json'),
                       files={name: w.sha(name) for name in FILES}, original_design_complete=False,
                       formal_final_authorized=False, limitations=EXTRA_LIMITATIONS)
        parent.store(root / 'analysis_protocol.json', binding)
        jobs, audit, mismatches = audit_pairs(report, config)
        parent.store(root / 'support_audit_validate61.json', audit)
        write_differences(root / 'paired_support_differences.csv.gz', mismatches)
        if args.action == 'all':
            exported = dict(report, source_build=report['build'], build=BUILD, paired_support_summary=audit['summary'],
                            comparison_warning='Only evaluation rows with status=matched support paired effects; other native-support metrics are descriptive only',
                            limitations=parent.LIMITATIONS + EXTRA_LIMITATIONS)
            parent.store(root / 'report_validate61_support.json', exported)
            parent.store(root / 'evaluation_validate61_support.json', evaluate(jobs, report, root, args.repetitions))
            decision = gate_diagnostics(report, config, audit)
            decision['support_audit_sha256'] = w.sha(root / 'support_audit_validate61.json')
            parent.store(root / 'decision_validate61_support.json', decision)
        final = parent.inventory(policy, config, tasks)
        if not final['ready'] or parent.manifest(args.policy, policy, config, final) != expected:
            raise RuntimeError('Source records changed during analysis; preserve outputs and inspect')
    print('Saved', root, '; unmatched series explicitly unavailable, not treated as paired.', flush=True)


if __name__ == '__main__':
    main()
