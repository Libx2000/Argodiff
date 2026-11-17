"""Additive, checksum-bound EM&S experiments. No arrays, retraining, or overwrites.

Existing OE protocols/outputs stay untouched. Legacy data support diagnostic
experiments only; an explicitly confirmed source is mandatory for final testing.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import date, timedelta, datetime, timezone
import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PHASES = ('tune', 'calibrate', 'validate', 'robust', 'budget', 'final')
SEASONS = ('winter', 'spring', 'summer', 'autumn')
CODE = ('ems_experiments.py', 'ems_cycle.py', 'ems_observations.py', 'ems_gaussian.py',
        'ems_observer.py', 'ems_evaluate.py', 'oe_stable_diffusion.py', 'oe_stability.py',
        'oe_joint_calibration.py', 'oe_joint_diagnostics.py', 'oe_joint_selection.py',
        'oe_physics.py', 'paper_metrics.py', 'oe_experiments.py', 'oe_stable_experiments.py', 'fit_spread_calibration.py',
        'prepare_oe_argo.py', 'argo_common.py', 'prepare_paper_argo.py', 'map_argo_to_glorys.py',
        'diffargo/config.py', 'diffargo/data.py', 'diffargo/model.py',
        'diffargo/utils.py')


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT/p


def read(value):
    return json.loads(path(value).read_text(encoding='utf-8-sig'))


def sha(value):
    from oe_experiments import sha as cached_sha
    return cached_sha(path(value))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_new(value, payload):
    p = path(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def settings(filename):
    def merged(name, stack=()):
        if str(path(name).resolve()) in stack:
            raise ValueError('Cyclic configuration inheritance')
        c = read(name)
        files = [str(name)]
        if 'extends' in c:
            b, dependencies = merged(c['extends'], stack+(str(path(name).resolve()),))
            b.update(c); c=b; files+=dependencies
        return c,files
    overlay, files = merged(filename)
    c = read(overlay['base_config'])
    c.update(overlay)
    c['config_files'] = sorted(set(files+[overlay['base_config']]))
    if c['branch'] not in ('legacy_diagnostic', 'confirmed_source'):
        raise ValueError('Unknown source branch')
    if c['days'] != 90 or c['members'] < 50 or c['calibration_year'] != 2018 or c['development_year'] != 2019:
        raise ValueError('Production protocol requires 90 days, 50+ members, 2018 calibration and 2019 development')
    if c['final_year'] in (2018,2019,2020,2021):
        raise ValueError('Previously inspected years cannot be relabeled as a new final test')
    if len({s['id'] for s in c['scenarios']}) != len(c['scenarios']) or c['scenarios'][0]['id']!='nominal':
        raise ValueError('Unique scenarios with nominal first are required')
    if not 1<=c['tune_days']<=90 or c['robust_days']<45 or c['profile_sample_points']<1:
        raise ValueError('Invalid phase horizon or diagnostic sample size')
    return c


def catalog(c):
    result = {n:dict(id=n,kind=n) for n in
        ('deterministic','unguided_centered','guided_common','guided_raw','source_guided')}
    trust = c['trust_regions'][1]
    for kind in ('center','joint'):
        n = 'guided_'+kind
        options = dict(c['stability'], center=True, denoise=(kind=='joint'),
            max_temperature_change=trust['temperature'], max_salinity_change=trust['salinity'],
            temperature_step=trust['temperature']/c['stability']['center_iterations'],
            salinity_step=trust['salinity']/c['stability']['center_iterations'])
        result[n] = dict(id=n,kind='guided_stable',options=options)
    for g in c['nudging_gains']:
        n=f'nudging_{g:g}'; result[n]=dict(id=n,kind='nudging',gain=float(g))
    for r in c['gaussian_error_scales']:
        n=f'gaussian_{r:g}'; result[n]=dict(id=n,kind='gaussian',temperature_error=.5*r,salinity_error=.1*r)
    return result


def selection(c):
    record = read(path(c['output_root'])/'selection_tune.json')
    if record['report_sha256'] != sha(path(c['output_root'])/'report_tune.json'):
        raise RuntimeError('Tuning report changed after selection')
    for name in ('physical','nudging','gaussian'):
        if catalog(c).get(record[name]['id']) != record[name]:
            raise RuntimeError('Selected method no longer matches frozen catalogue')
    return record


def task_list(phase, c):
    if phase=='tune':
        # Source-residual portability is evaluated after selection, not tuned.
        methods=[n for n in catalog(c) if n!='source_guided']; scenarios=[c['scenarios'][0]]
    elif phase=='calibrate':
        methods=['selected']; scenarios=[c['scenarios'][0]]
    elif phase in ('validate','final'):
        methods=['deterministic','unguided_centered','guided_common','guided_raw',
                 'selected','nudging_selected','gaussian_selected','source_guided']
        scenarios=[c['scenarios'][0]]
    elif phase=='robust':
        methods=['guided_raw','selected','nudging_selected','gaussian_selected']
        scenarios=c['scenarios'][1:]
    elif phase=='budget':
        return [dict(phase=phase,year=c['development_year'],method='selected',scenario='nominal',
            season=s,month=1+3*i,seed=c['seeds'][0],days=c['budget_days'],members=m,steps=k)
            for m in c['budget_members'] for k in c['budget_steps'] for i,s in enumerate(SEASONS)]
    else:
        raise ValueError('Unknown phase')
    year=c['calibration_year'] if phase=='calibrate' else (c['final_year'] if phase=='final' else c['development_year'])
    seeds=c['seeds'][:1] if phase=='tune' else c['seeds']
    days=c['tune_days'] if phase=='tune' else (c['robust_days'] if phase=='robust' else c['days'])
    return [dict(phase=phase,year=year,method=m,scenario=scenario['id'],season=s,month=1+3*i,
        seed=int(seed),days=days,members=c['members'],steps=c['sampling_steps'])
        for scenario in scenarios for m in methods for i,s in enumerate(SEASONS) for seed in seeds]


def definition(t,c):
    aliases={'selected':'physical','nudging_selected':'nudging','gaussian_selected':'gaussian'}
    return selection(c)[aliases[t['method']]] if t['method'] in aliases else catalog(c)[t['method']]


def source_check(c, formal=False):
    split=path(c['split_root'])
    m=read(split/'split_manifest.json')
    if c['branch']=='confirmed_source':
        from prepare_oe_argo import validate_declaration
        declaration=read(c['source_declaration']); validate_declaration(declaration)
        if m.get('source_declaration')!=declaration or m.get('temperature_kind')!='potential_0dbar' or m.get('vertical_coordinate')!='depth_m':
            raise RuntimeError('Mapping and source declaration disagree. Never relabel an old mapping.')
        if formal and declaration['quality_policy']!='native_adjusted':
            raise RuntimeError('Formal independent-observation branch requires native QC/adjusted provenance')
    elif formal:
        raise RuntimeError('Legacy Argo definitions/QC are unresolved; formal final testing refused')
    return {str(split/n):sha(split/n) for n in ('split_manifest.json','spatial_partition.npz')}


def fingerprint(c, years):
    from diffargo.config import load_config
    from oe_experiments import input_hashes
    cfg=load_config(path(c['model_config']))
    if Path(cfg['data']['stats_path']).resolve()!=path(c['state_stats']).resolve():
        raise RuntimeError('Model normalization path differs from the declared training statistics')
    files=list(CODE)+c['config_files']+[c[k] for k in ('model_config','deterministic_checkpoint',
        'residual_checkpoint','source_residual_checkpoint','residual_stats','state_stats','parent_protocol')]
    if c['branch']=='confirmed_source': files.append(c['source_declaration'])
    data={str(y):{str(month):input_hashes(dict(config=c['model_config'],argo_split_root=c['split_root'],days=max(c['days'],c['robust_days'])),
        dict(year=y,month=month)) for month in (1,4,7,10)} for y in years}
    import numpy,torch,gsw
    return dict(files={n:sha(n) for n in sorted(set(files))},mapping=source_check(c),data=data,
        runtime=dict(numpy=numpy.__version__,torch=torch.__version__,gsw=gsw.__version__))


def initialize(c):
    root=path(c['output_root']); root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists(): verify(c); print('Existing protocol verified'); return
    from oe_experiments import verify_parent
    verify_parent(dict(parent_frozen_protocol=c['parent_protocol']))
    frozen=fingerprint(c,(c['calibration_year'],c['development_year']))
    import numpy as np
    with np.load(path(c['state_stats']),allow_pickle=False) as f: channels=len(f['state_mean'])
    if (root/'unit_factors.npz').exists(): raise FileExistsError('Partial init: preserve it and choose a new output_root')
    np.savez_compressed(root/'unit_factors.npz',inflation_factor=np.ones((max(c['days'],c['robust_days']),channels),np.float32))
    write_new(root/'protocol.json',dict(schema='ems-v1',created_utc=stamp(),settings=c,fingerprint=frozen,
        unit_factor_sha256=sha(root/'unit_factors.npz'),final_test_frozen=False,
        intent='Prospective local protocol, not public preregistration or journal approval'))
    print('INIT complete. Data provenance branch:',c['branch'])


def verify(c):
    record=read(path(c['output_root'])/'protocol.json')
    if record['settings']!=c or record['fingerprint']!=fingerprint(c,(c['calibration_year'],c['development_year'])):
        raise RuntimeError('Frozen code/config/data changed: preserve this experiment and use a NEW output version')
    if record['unit_factor_sha256']!=sha(path(c['output_root'])/'unit_factors.npz'):
        raise RuntimeError('Unit factors changed')
    return record


def dependencies(t,c):
    root=path(c['output_root']); names=['protocol.json']
    if t['phase']!='tune': names+=['selection_tune.json']
    if t['phase'] in ('validate','robust','budget','final'):
        names+=['calibration_record.json','selected_calibration_2018.npz']
        r=read(root/'calibration_record.json')
        if r['calibration_sha256']!=sha(root/'selected_calibration_2018.npz') or r['selection_sha256']!=sha(root/'selection_tune.json'):
            raise RuntimeError('Calibration source changed')
        for p,digest in r['sources'].items():
            if sha(p)!=digest: raise RuntimeError('Calibration completion record changed')
    if t['phase']=='final': names+=['decision_validate.json','final_freeze.json']
    return {n:sha(root/n) for n in names}


def folder(t,c,smoke=False):
    return path(c['output_root'])/('smoke' if smoke else 'cases')/t['phase']/t['scenario']/t['method']/f"m{t['members']}_k{t['steps']}"/t['season']/str(t['seed'])


def completed(t,c,smoke=False):
    mark=folder(t,c,smoke)/'completed.json'
    if not mark.exists(): return None
    r=read(mark)
    if r['task']!=t or r['dependencies']!=dependencies(t,c): raise RuntimeError('Completed case identity/dependencies changed')
    for name,digest in r['outputs'].items():
        if sha(mark.parent/r['attempt']/name)!=digest: raise RuntimeError('Archived output changed: '+name)
    return r


def build_command(t,c,out,smoke=False):
    d=definition(t,c); kind=d['kind']; root=path(c['output_root'])
    calibrated=t['phase'] in ('validate','robust','final') and t['method']=='selected'
    reference=root/'selected_calibration_2018.npz' if calibrated else root/'unit_factors.npz'
    dc=dict(c['diagnostics'],expected_days=t['days'],reference_calibration=str(reference),
        state_stats=str(path(c['state_stats'])),profile_sample_seed=c['profile_sample_seed'],profile_sample_points=c['profile_sample_points'])
    candidates=[dict(id='raw',kind='raw')]+([dict(id='marginal',kind='marginal')] if calibrated else [])
    observer=out/'observer.json';write_new(observer,dict(config=dc,candidates=candidates))
    start=date(t['year'],t['month'],1)-timedelta(days=1)
    scenario=next(x for x in c['scenarios'] if x['id']==t['scenario'])
    observation={k:v for k,v in scenario.items() if k not in ('id','outage_start','outage_end')}
    observation.update(start_date=str(start),observation_seed=c['observation_seed'],
        outage_leads=list(range(scenario.get('outage_start',1),scenario.get('outage_end',0)+1)))
    observation_file=out/'observation_options.json'; write_new(observation_file,observation)
    residual=c['source_residual_checkpoint'] if kind=='source_guided' else c['residual_checkpoint']
    cmd=[sys.executable,'-u','ems_observer.py','--ems-observer',str(observer),
        '--config',str(path(c['model_config'])),'--deterministic-checkpoint',str(path(c['deterministic_checkpoint'])),
        '--residual-checkpoint',str(path(residual)),'--residual-stats',str(path(c['residual_stats'])),
        '--paper-holdout-root',str(path(c['split_root'])/'holdout'),'--start',str(start),
        '--days',str(t['days']),'--members',str(t['members']),'--sampling-steps',str(t['steps']),
        '--sampling-eta','0','--residual-scale',str(c['guidance']['residual_scale']),
        '--seed',str(t['seed']),'--save-daily','none','--output-dir',str(out),
        '--ems-observation-options',str(observation_file)]
    if kind in ('deterministic','nudging'): cmd+=['--deterministic-only']
    if kind not in ('deterministic','unguided_centered'):
        cmd+=['--argo-root',str(path(c['split_root'])/'guidance'),'--require-argo']
    if kind=='nudging': cmd+=['--nudging-gain',str(d['gain'])]
    elif kind!='deterministic':
        if kind!='guided_common': cmd+=['--individual-member-guidance','--member-mean-cycle','--paired-guidance-center']
        cmd+=['--temperature-guidance',str(c['guidance']['temperature']),
              '--salinity-guidance',str(c['guidance']['salinity']),'--guidance-start',str(c['guidance']['start'])]
    if kind=='gaussian':
        cmd+=['--ems-backend','diagonal_gaussian','--ems-gaussian-temperature-error',str(d['temperature_error']),
              '--ems-gaussian-salinity-error',str(d['salinity_error'])]
    if 'options' in d:
        opts=out/'stability_options.json';write_new(opts,d['options']);cmd+=['--stability-options',str(opts)]
    if t['phase']=='calibrate': cmd+=['--calibration-statistics-output',str(out/'calibration_statistics.npz')]
    return cmd


def run_case(t,c,smoke=False):
    if t['phase']=='final': verify_final(c)
    t=dict(t,days=2,members=4,steps=4) if smoke else t
    base=folder(t,c,smoke);base.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (base/'task.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if completed(t,c,smoke): print('SKIP verified completed case:',base,flush=True);return
        n=1
        while (base/f'attempt_{n:03d}').exists(): n+=1
        out=base/f'attempt_{n:03d}';out.mkdir()
        dep=dependencies(t,c);cmd=build_command(t,c,out,smoke)
        write_new(out/'recipe.json',dict(task=t,command=cmd,method=definition(t,c),dependencies=dep,smoke=smoke,created_utc=stamp()))
        started=time.perf_counter()
        with (out/'execution.log').open('w',encoding='utf-8') as stream:
            code=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
        if code: raise RuntimeError(f'Case failed; preserved log: {out}/execution.log')
        verify(c)
        if dependencies(t,c)!=dep: raise RuntimeError('Dependencies changed during case')
        status=read(out/'joint_status.json')
        if status['completed_leads']!=list(range(1,t['days']+1)): raise RuntimeError('Incomplete physical diagnostics')
        for name in ('paper_grid_metrics.csv','paper_holdout_metrics.csv','ems_multivariate.csv','run_summary.csv'):
            with (out/name).open(encoding='utf-8',newline='') as stream:
                leads={int(r['lead']) for r in csv.DictReader(stream)}
            if leads!=set(range(1,t['days']+1)): raise RuntimeError('Incomplete score file: '+name)
        write_new(out/'allocation_cost.json',dict(subprocess_wall_seconds=time.perf_counter()-started,
            output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),
            scope='Entire child process including loading, scoring and I/O; not a training cost or pure sampler benchmark'))
        write_new(out/'completed.pending.json',dict(task=t,attempt=out.name,dependencies=dep,
            outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},completed_utc=stamp()))
        os.replace(out/'completed.pending.json',base/'completed.json')
        print('DONE',t,flush=True)


def load_scores(t,c):
    m=completed(t,c)
    if not m: raise RuntimeError('Incomplete case: '+str(t))
    out=folder(t,c)/m['attempt']
    with gzip.open(out/'joint_pooled.json.gz','rt',encoding='utf-8') as stream: payload=json.load(stream)
    for table in ('scores','events'):
        for r in payload[table]:
            if r['candidate']=='point':r['candidate']='raw'
    support={}
    with gzip.open(out/'joint_metrics.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for r in csv.DictReader(stream):
            if r['domain']=='glorys' and r['support']=='common' and r['candidate'] in ('point','raw') and r['variable'] in ('temperature','salinity','rho','n2'):
                key=tuple(r[n] for n in ('date','region','variable','depth_index'))
                support[key]=tuple(r[n] for n in ('support_digest','count','weight'))
    held=[]
    with (out/'paper_holdout_metrics.csv').open(encoding='utf-8',newline='') as stream:
        held=[r for r in csv.DictReader(stream) if r['depth_index']=='-1' and float(r['weight'])>0]
    return payload,support,held,out


def report(phase,c):
    from oe_joint_selection import combine
    verify(c)
    if phase=='final': verify_final(c)
    groups=defaultdict(list); held=defaultdict(list); supports={}; records=[]; outputs={}
    for t in task_list(phase,c):
        p,s,h,out=load_scores(t,c)
        key=f"{t['scenario']}|{t['method']}|m{t['members']}_k{t['steps']}"
        groups[key].append(p);supports[key,t['season'],t['seed']]=s
        for row in h:held[key,row['variable']].append(row)
        mark=folder(t,c)/'completed.json';outputs[str(mark)]=sha(mark)
        records.append(dict(task=t,output=str(out),completed_sha256=sha(mark)))
    support_errors=[]
    by_setting=defaultdict(list)
    for key,season,seed in supports:
        scenario=key.split('|')[0]
        by_setting[scenario,season,seed].append(key)
    for (scenario,season,seed),keys in by_setting.items():
        if phase=='budget':continue # Cost curve has explicit different M/K settings.
        base=keys[0]
        for key in keys[1:]:
            if supports[key,season,seed]!=supports[base,season,seed]:
                support_errors.append(dict(scenario=scenario,season=season,seed=seed,left=base,right=key))
    rows=[]
    for (key,var),values in held.items():
        weight=sum(float(r['weight']) for r in values)
        rows.append(dict(key=key,variable=var,weight=weight,count=sum(int(r['count']) for r in values),
            rmse=math.sqrt(sum(float(r['weight'])*float(r['rmse'])**2 for r in values)/weight)))
    result=dict(phase=phase,complete=True,methods={k:combine(v) for k,v in groups.items()},holdout=rows,
        support_errors=support_errors,cases=records,source_completed=outputs,branch=c['branch'],
        protocol_sha256=sha(path(c['output_root'])/'protocol.json'),
        inference='Descriptive paired seasonal comparisons; seeds nested within starts; no independent gridpoint/day inference',
        new_global_results=True,traceable_observation_source=(c['branch']=='confirmed_source'),
        formal_independent_observation_claim=False,
        independence_note='A holdout from guidance is not necessarily independent of GLORYS assimilation; source exclusion must be established separately.')
    dest=path(c['output_root'])/f'report_{phase}.json'
    if dest.exists():
        if read(dest)!=result:raise RuntimeError('Existing report changed; preserve the old version')
    else:write_new(dest,result)
    print(f'{phase}: complete {len(records)} cases; cross-method support errors {len(support_errors)}')
    return result


def method_key(c,method,scenario='nominal'):
    return f"{scenario}|{method}|m{c['members']}_k{c['sampling_steps']}"


def assessment_view(r,c):
    methods={k.split('|')[1]:v for k,v in r['methods'].items() if k.startswith('nominal|')}
    methods['argo_raw']=methods['guided_raw']
    held=[dict(method=x['key'].split('|')[1],variable=x['variable'],rmse=x['rmse']) for x in r['holdout'] if x['key'].startswith('nominal|')]
    held+=[dict(x,method='argo_raw') for x in held if x['method']=='guided_raw']
    return dict(methods=methods,holdout=held,support_errors=r['support_errors'])


def select(c):
    from oe_stable_experiments import assess
    r=report('tune',c); view=assessment_view(r,c)
    assessments=[assess(view,m,'raw',c,True) for m in ('guided_center','guided_joint')]
    good=[x for x in assessments if x['passed']]
    chosen=min(good or assessments,key=lambda x:(x['center_event'],x['objective']))
    def objective(m):
        result=[]
        for var in ('temperature','salinity'):
            x=next((x for x in r['holdout'] if x['key']==method_key(c,m) and x['variable']==var),None)
            ref=next((x for x in r['holdout'] if x['key']==method_key(c,'deterministic') and x['variable']==var),None)
            if x is None or ref is None: raise RuntimeError('No held observations for fair baseline tuning')
            result.append(x['rmse']/max(ref['rmse'],1e-12))
        return sum(result)/2
    catalogue=catalog(c)
    nudging=min((m for m in catalogue if m.startswith('nudging_')),key=objective)
    gaussian=min((m for m in catalogue if m.startswith('gaussian_')),key=objective)
    write_new(path(c['output_root'])/'selection_tune.json',dict(
        physical_passed=bool(good),physical=catalogue[chosen['method']],nudging=catalogue[nudging],gaussian=catalogue[gaussian],
        all_physical_assessments=assessments,nudging_objective=objective(nudging),gaussian_objective=objective(gaussian),
        report_sha256=sha(path(c['output_root'])/'report_tune.json'),created_utc=stamp(),
        diagnostic_only=not bool(good),
        note='If no physical method passes, the named provisional proposal is diagnostic only; formal final testing remains disabled.'))
    print('Physical screen passed:',bool(good),'proposal:',chosen['method'])


def fit(c):
    verify(c); sources={}; statistics=[]
    for t in task_list('calibrate',c):
        m=completed(t,c)
        if not m: raise RuntimeError('Calibration incomplete: '+str(t))
        statistics.append(str(folder(t,c)/m['attempt']/'calibration_statistics.npz'))
        p=folder(t,c)/'completed.json';sources[str(p)]=sha(p)
    root=path(c['output_root']);dest=root/'selected_calibration_2018.npz'
    if dest.exists():raise FileExistsError('Preserve existing calibration; do not overwrite')
    subprocess.run([sys.executable,'fit_spread_calibration.py','--statistics',*statistics,'--output',str(dest),
        '--lead-dependent','--monotonic-lead','--max-factor',str(c['calibration']['max_factor']),
        '--log-shrinkage',str(c['calibration']['log_shrinkage'])],cwd=ROOT,check=True)
    write_new(root/'calibration_record.json',dict(sources=sources,calibration_sha256=sha(dest),
        selection_sha256=sha(root/'selection_tune.json'),created_utc=stamp()))


def decide(c):
    from oe_stable_experiments import assess
    r=report('validate',c);view=assessment_view(r,c)
    values=[assess(view,'selected',v,c,False) for v in ('raw','marginal')]
    good=[x for x in values if x['passed']]
    chosen=min(good,key=lambda x:x['objective']) if good else None
    passed=bool(good) and selection(c)['physical_passed']
    write_new(path(c['output_root'])/'decision_validate.json',dict(passed=passed,
        variant=chosen['variant'] if chosen else None,assessments=values,
        report_sha256=sha(path(c['output_root'])/'report_validate.json'),created_utc=stamp(),
        independent_profile_physics_validated=False,
        interpretation='Research acceptance rules only, not hypothesis tests or a claim of independent Argo truth.'))
    print('Development acceptance:',passed)


def freeze_final(c,confirm):
    verify(c);source_check(c,formal=True)
    root=path(c['output_root']); d=read(root/'decision_validate.json')
    if not d['passed'] or not selection(c)['physical_passed']:raise RuntimeError('Physical development acceptance failed; no final freeze')
    if not confirm:raise RuntimeError('The human author must explicitly attest --confirm-unseen for the configured final year')
    if d['report_sha256']!=sha(root/'report_validate.json'):raise RuntimeError('Validation report changed')
    write_new(root/'final_freeze.json',dict(year=c['final_year'],author_attested_unseen=True,created_utc=stamp(),
        decision_sha256=sha(root/'decision_validate.json'),selection_sha256=sha(root/'selection_tune.json'),
        calibration_sha256=sha(root/'selected_calibration_2018.npz'),fingerprint=fingerprint(c,[c['final_year']]),
        variant=d['variant']))


def verify_final(c):
    root=path(c['output_root']); f=read(root/'final_freeze.json');source_check(c,formal=True)
    for key,name in [('decision_sha256','decision_validate.json'),('selection_sha256','selection_tune.json'),('calibration_sha256','selected_calibration_2018.npz')]:
        if sha(root/name)!=f[key]:raise RuntimeError('Final dependency changed')
    if f['fingerprint']!=fingerprint(c,[c['final_year']]):raise RuntimeError('Final inputs changed')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('plan','init','preflight','status','run','batch','report','select','fit','decide','freeze-final'))
    p.add_argument('--config',default='configs/ems_v1_legacy.json')
    p.add_argument('--phase',choices=PHASES,default='tune')
    p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int)
    p.add_argument('--task',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--confirm-unseen',action='store_true')
    a=p.parse_args(); c=settings(a.config)
    actions={'init':initialize,'preflight':verify,'select':select,'fit':fit,'decide':decide}
    if a.action in actions:actions[a.action](c);return
    if a.action=='freeze-final':freeze_final(c,a.confirm_unseen);return
    if a.action=='report':report(a.phase,c);return
    ts=task_list(a.phase,c)
    if a.action=='plan':
        for i,t in enumerate(ts):print(i,t)
        print('TOTAL',len(ts));return
    if a.action=='status':
        missing=[i for i,t in enumerate(ts) if not completed(t,c)]
        print(json.dumps(dict(total=len(ts),complete=len(ts)-len(missing),pending_task_ids=missing)));return
    verify(c)
    first=a.task if a.action=='run' else a.first
    last=a.task if a.action=='run' else (len(ts)-1 if a.last is None else a.last)
    if not 0<=first<=last<len(ts):p.error(f'Use task numbers 0..{len(ts)-1}')
    ids=list(range(first,last+1))
    if a.action=='batch':
        import numpy as np
        ids=np.random.default_rng(c['schedule_seed']+PHASES.index(a.phase)).permutation(ids).tolist()
    estimate=7200.;deadline=float(os.environ.get('EMS_DEADLINE','inf'))
    print('Seeded allocation order:',ids,flush=True)
    for i in ids:
        t=dict(ts[i],days=2,members=4,steps=4) if a.smoke else ts[i]
        if completed(t,c,a.smoke):print('SKIP task',i,flush=True);continue
        if time.time()+estimate+900>deadline:
            print('Budget stop before task',i,'; repeat SAME range to continue verified remaining cases.',flush=True);return
        before=time.time();run_case(ts[i],c,a.smoke)
        estimate=max(estimate,(time.time()-before)*1.3)


if __name__=='__main__':main()
