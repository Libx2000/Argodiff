"""Post-execution, explicitly incomplete validation; never rewrite frozen results.

Only the three named task IDs may be excluded. Full completion/output checks are
retained for every included case. All comparisons use matched start/seed blocks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

BUILD = 'ems-validate61-20261025-v1'
FILES = ('ems_validate61.py', 'configs/ems_validate61_v1.json', 'submit_ems_validate61.slurm')
LIMITATIONS = [
    'The original 64-case validation is incomplete; this is a post-execution amendment.',
    'Exclusion is author-reported runtime-driven; non-random missingness and runtime censoring cannot be ruled out.',
    'No autumn selected-method validation; summer has one seed, not two.',
    'Seeds are technical replicates nested within fixed starts; days and grid points are not independent experiments.',
    'Conditional moving-block intervals are not independent-year confidence intervals or hypothesis tests.',
    'Argo source/QC and independence from GLORYS assimilation remain unresolved in the legacy branch.',
    'Completed-case costs omit the censored cases and cannot establish an unconditional speedup.',
]


def block(task):
    return (task['season'], int(task['seed']))


def key(task):
    return f"{task['scenario']}|{task['method']}|m{task['members']}_k{task['steps']}"


def policy_settings(filename):
    import ems_experiments as w
    policy = w.read(filename)
    if policy['schema'] != 'ems-partial-validation-v1' or policy['phase'] != 'validate':
        raise ValueError('Only the explicit validate61 policy is supported')
    if policy['analysis_id'] != 'validate61_skip37_39_v1' or policy['formal_final_authorized'] is not False:
        raise ValueError('Do not relabel incomplete validation as a final-test authorization')
    config = w.settings(policy['base_config'])
    tasks = w.task_list('validate', config)
    ids = [item['task_id'] for item in policy['excluded_tasks']]
    if (ids != [37, 38, 39] or policy['expected_planned_count'] != 64 or
            policy['expected_included_count'] != 61 or len(tasks) != 64 or len(tasks) - len(ids) != 61):
        raise ValueError('The policy is bound to the original 64 tasks and exclusions 37,38,39')
    if config['development_year'] != 2019 or config['days'] != 90:
        raise ValueError('Original year/horizon must not change')
    for item in policy['excluded_tasks']:
        if any(tasks[item['task_id']][field] != value for field, value in item.items() if field != 'task_id'):
            raise ValueError('Task numbering/identity changed: ' + str(item))
    return policy, config, tasks


@contextmanager
def compatible_runtime(config):
    """Old-only records work without an amendment; mixed records use lowmem checks."""
    import ems_experiments as w
    amendment = w.path(config['output_root']) / 'runtime_memory_v1.json'
    restore = None
    if amendment.is_file():
        import ems_lowmem
        restore = ems_lowmem.install_workflow(config)  # verifies amendment, no initialization or reset
    try:
        yield
    finally:
        if restore is not None:
            restore()


def inventory(policy, config, tasks):
    import ems_experiments as w
    w.verify(config)
    excluded = {item['task_id'] for item in policy['excluded_tasks']}
    included, missing, invalid, omitted = [], [], [], []
    for index, task in enumerate(tasks):
        marker = w.folder(task, config) / 'completed.json'
        if index in excluded:
            omitted.append(dict(task_id=index, task=task, reason=policy['reason'],
                                completed_marker_exists=marker.is_file(), status='excluded_not_used'))
            continue
        try:
            record = w.completed(task, config)
            if record is None:
                missing.append(index)
                continue
            included.append(dict(task_id=index, task=task, marker=str(marker),
                                 completed_sha256=w.sha(marker),
                                 output=str(marker.parent / record['attempt'])))
        except Exception as exc:
            invalid.append(dict(task_id=index, marker=str(marker), error=f'{type(exc).__name__}: {exc}'))
    result = dict(planned=len(tasks), required=policy['expected_included_count'], verified=len(included),
                  pending_task_ids=missing, invalid=invalid, excluded=omitted, cases=included,
                  original_design_complete=False, ready=(not missing and not invalid and len(included) == 61))
    print('VALIDATE61_STATUS ' + json.dumps({k: v for k, v in result.items() if k != 'cases'},
                                         ensure_ascii=False), flush=True)
    return result


def store(destination, payload):
    """Immutable/idempotent outputs; changed analyses require a new version."""
    import ems_experiments as w
    if destination.exists():
        if w.read(destination) != payload:
            raise RuntimeError('Existing partial-analysis artifact differs; preserve it: ' + str(destination))
    else:
        w.write_new(destination, payload)


def manifest(filename, policy, config, status):
    import ems_experiments as w
    original = w.path(config['output_root'])
    dependencies = ['protocol.json', 'selection_tune.json', 'calibration_record.json', 'selected_calibration_2018.npz']
    if (original / 'runtime_memory_v1.json').is_file():
        dependencies.append('runtime_memory_v1.json')
    return dict(build=BUILD, policy=policy, policy_sha256=w.sha(filename),
                original_dependencies={name: w.sha(original / name) for name in dependencies},
                code_sha256={name: w.sha(name) for name in FILES},
                included_completed={item['marker']: item['completed_sha256'] for item in status['cases']},
                excluded_tasks=policy['excluded_tasks'], planned_count=64, included_count=61,
                original_design_complete=False, post_execution_amendment=True,
                formal_final_authorized=False, limitations=LIMITATIONS)


def scope_layout(cases, config, policy):
    by_method = defaultdict(set)
    for case in cases:
        by_method[key(case['task'])].add(block(case['task']))
    expected = {(season, int(seed)) for season in ('winter', 'spring', 'summer', 'autumn') for seed in config['seeds']}
    common = set.intersection(*by_method.values())
    balanced = {b for b in common if all((b[0], int(seed)) in common for seed in config['seeds'])}
    if len(common) != policy['expected_primary_blocks'] or len(balanced) != policy['expected_balanced_blocks']:
        raise ValueError('Unexpected matched coverage; do not silently widen/narrow the analysis')
    complete_methods = {method for method, blocks in by_method.items() if blocks == expected}
    scopes = dict(
        matched_primary=[case for case in cases if block(case['task']) in common],
        matched_balanced=[case for case in cases if block(case['task']) in balanced],
        complete_baselines=[case for case in cases if key(case['task']) in complete_methods],
    )
    return scopes, {method: {season: sorted(seed for s, seed in blocks if s == season)
                            for season in ('winter', 'spring', 'summer', 'autumn')}
                    for method, blocks in by_method.items()}


def aggregate(cases, loaded, config):
    """Each seed gets 1/n_seed within its own start; pool MSE before square root."""
    import ems_experiments as w
    from oe_joint_selection import combine
    denominators = Counter((key(case['task']), case['task']['season']) for case in cases)
    groups, held, supports = defaultdict(list), defaultdict(list), {}
    for case in cases:
        task = case['task']; method = key(task); payload, signature, holdout = loaded[case['task_id']]
        scale = 1.0 / denominators[method, task['season']]
        data = copy.deepcopy(payload)
        for table in ('scores', 'events'):
            for row in data[table]:
                row['weight'] *= scale
        groups[method].append(data)
        supports[method, *block(task)] = signature
        for row in holdout:
            held[method, row['variable']].append((row, scale))
    errors = []
    for (method, season, seed), signature in supports.items():
        reference = 'nominal|deterministic|' + method.split('|')[2]
        other = supports.get((reference, season, seed))
        if other is None or signature != other:
            errors.append(dict(method=method, season=season, seed=seed, reason='Missing/unmatched reference support'))
    rows = []
    for (method, variable), values in sorted(held.items()):
        total = sum(float(row['weight']) * scale for row, scale in values)
        rows.append(dict(key=method, variable=variable, weight=total,
                         count=sum(int(row['count']) for row, scale in values),
                         rmse=math.sqrt(sum(float(row['weight']) * scale * float(row['rmse'])**2
                                            for row, scale in values) / total)))
    blocks = sorted({block(case['task']) for case in cases})
    return dict(methods={method: combine(values) for method, values in groups.items()}, holdout=rows,
                support_errors=errors, cases=cases,
                paired_blocks=[dict(season=season, seed=seed) for season, seed in blocks],
                starts=sorted({season for season, seed in blocks}),
                normalization='Mean weighted losses over available paired seeds within each start, then pool spatial/time weights across starts',
                count_note='Counts are score occurrences including repeated seeds, NOT independent sample size',
                complete=False, amended_subset_complete=True,
                branch=config['branch'], formal_independent_observation_claim=False)


def make_report(policy, config, status):
    import ems_experiments as w
    loaded = {}
    for case in status['cases']:
        payload, signature, held, output = w.load_scores(case['task'], config)
        loaded[case['task_id']] = (payload, signature, held)
    scopes, coverage = scope_layout(status['cases'], config, policy)
    result = dict(build=BUILD, phase='validate', original_design_complete=False,
                  included_cases=61, planned_cases=64, excluded=status['excluded'],
                  coverage=coverage, limitations=LIMITATIONS,
                  all_completed_descriptive=aggregate(status['cases'], loaded, config),
                  scopes={name: aggregate(cases, loaded, config) for name, cases in scopes.items()})
    print('Matched scope case counts:', {name: len(cases) for name, cases in scopes.items()}, flush=True)
    return result


def evaluate_report(report, root, repetitions):
    import ems_experiments as w
    from ems_evaluate import paired_blocks, read_daily
    daily = {case['task_id']: read_daily(case['output'], case['task'])
             for case in report['all_completed_descriptive']['cases']}
    costs = []
    for case in report['all_completed_descriptive']['cases']:
        out = Path(case['output'])
        costs.append(dict(task_id=case['task_id'], task=case['task'],
                          allocation=w.read(out / 'allocation_cost.json'),
                          cycle_loop=w.read(out / 'compute_cost.json')))
    comparisons = []
    for name, scope in report['scopes'].items():
        if scope['support_errors']:
            raise ValueError('Unmatched support in ' + name)
        data = defaultdict(lambda: defaultdict(list))
        for case in scope['cases']:
            for score, rows in daily[case['task_id']].items():
                data[key(case['task'])][score].extend(rows)
        for method, scores in sorted(data.items()):
            reference = 'nominal|deterministic|' + method.split('|')[2]
            if method == reference:
                continue
            for (candidate, region, variable, metric), rows in scores.items():
                other = data[reference].get(('raw', region, variable, metric))
                if other is None:
                    raise ValueError('Missing matched reference metric: ' + str((name, method, metric)))
                expected = {(case['task']['season'], str(case['task']['seed']), lead)
                            for case in scope['cases'] if key(case['task']) == method
                            for lead in range(1, case['task']['days'] + 1)}
                for values in (rows, other):
                    actual = {(r['season'], str(r['seed']), int(r['lead'])) for r in values}
                    if actual != expected:
                        raise ValueError('Incomplete 90-day metric in scope ' + name)
                for days in (7, 14, 30):
                    value = paired_blocks(rows, other, days, repetitions, root=(metric == 'rmse'))
                    comparisons.append(dict(scope=name, method=method, reference=reference,
                                            candidate=candidate, region=region, variable=variable, metric=metric,
                                            paired_seed_blocks=scope['paired_blocks'], **value))
        print('Paired evaluation complete:', name, flush=True)
    return dict(build=BUILD, original_design_complete=False, formal_final_authorized=False,
                report_sha256=w.sha(root / 'report_validate61.json'), comparisons=comparisons,
                completed_case_costs=costs, costs_censored_task_ids=[37, 38, 39], limitations=LIMITATIONS)


def diagnose_gates(report, config):
    import ems_experiments as w
    from oe_stable_experiments import assess
    results = {}
    for name in ('matched_primary', 'matched_balanced'):
        view = w.assessment_view(report['scopes'][name], config)
        results[name] = [assess(view, 'selected', variant, config, False) for variant in ('raw', 'marginal')]
    return dict(build=BUILD, passed=False, formal_final_authorized=False, original_design_complete=False,
                diagnostic_assessments=results, physical_tuning_passed=w.selection(config)['physical_passed'],
                reason='Incomplete original design: diagnostic gate values do not authorize a full-protocol decision or unseen-year final test.',
                limitations=LIMITATIONS)


def main(argv=None):
    import ems_experiments as w
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'report', 'evaluate', 'decide', 'all'), nargs='?', default='all')
    parser.add_argument('--policy', default='configs/ems_validate61_v1.json')
    parser.add_argument('--repetitions', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.repetitions < 2:
        parser.error('At least two bootstrap repetitions are required')
    print(BUILD, 'entry=', Path(__file__).resolve(), 'action=', args.action, flush=True)
    policy, config, tasks = policy_settings(args.policy)
    root = w.path(config['output_root']) / policy['analysis_id']
    with compatible_runtime(config):
        status = inventory(policy, config, tasks)
        if not status['ready']:
            raise RuntimeError('The other 61 cases must be complete and checksum-valid; see VALIDATE61_STATUS')
        if args.action == 'status':
            return
        record = manifest(args.policy, policy, config, status)
        store(root / 'analysis_protocol.json', record)
        report = make_report(policy, config, status)
        store(root / 'report_validate61.json', report)
        if args.action in ('evaluate', 'all'):
            evaluation = evaluate_report(report, root, args.repetitions)
            store(root / 'evaluation_validate61.json', evaluation)
        if args.action in ('decide', 'all'):
            diagnostic = diagnose_gates(report, config)
            diagnostic['report_sha256'] = w.sha(root / 'report_validate61.json')
            store(root / 'decision_validate61.json', diagnostic)
        final_status = inventory(policy, config, tasks)
        if not final_status['ready'] or manifest(args.policy, policy, config, final_status) != record:
            raise RuntimeError('Sources changed during analysis; preserve outputs and inspect')
    print('Saved partial validation under', root, '; no original report/decision/freeze overwritten.', flush=True)


if __name__ == '__main__':
    main()
