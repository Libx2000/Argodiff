"""Versioned stability ablations, honest gates, ordinary sequential allocations.

No checkpoint is trained or overwritten. All outputs are new versioned attempts.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, os, subprocess, sys, time
from collections import defaultdict
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CODE=('oe_stability.py','oe_stable_diffusion.py','oe_stable_cycle.py','oe_stable_observer.py',
      'oe_stable_experiments.py','oe_buoyancy.py','oe_joint_diagnostics.py','oe_joint_calibration.py',
      'oe_joint_selection.py','oe_physics.py','paper_metrics.py','fit_spread_calibration.py')
CODE=CODE+('oe_experiments.py','prepare_oe_argo.py','diffargo/config.py','diffargo/data.py',
           'diffargo/model.py','diffargo/utils.py')
PHASES=('screen','calibrate','validate','final')
SEASONS=('winter','spring','summer','autumn')


def path(p):return Path(p) if Path(p).is_absolute() else ROOT/p
def read(p):return json.loads(path(p).read_text(encoding='utf-8'))
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):
    from oe_experiments import sha as digest
    return digest(path(p))
def new_json(p,value):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False,ensure_ascii=False)


def settings(filename):
    c=read(filename);files=[str(filename)]
    if 'extends' in c:
        base=read(c['extends']);files.append(c['extends']);base.update(c);c=base
    c['config_files']=files
    if c['members']<50 or c['days']!=90 or c['calibration_year']!=2018 or c['development_year']!=2019:
        raise ValueError('Production design is 50+ members,90days,2018 calibration,2019 development')
    if c['final_year'] in (2018,2019,2020,2021):raise ValueError('Final year must not reuse inspected development/test years')
    if c['branch'] not in ('legacy_diagnostic','confirmed_source'):raise ValueError('Unknown source branch')
    return c


def source_check(c):
    split=path(c['split_root']);manifest=read(split/'split_manifest.json')
    if c['branch']=='confirmed_source':
        from prepare_oe_argo import validate_declaration
        declaration=read(c['source_declaration']);validate_declaration(declaration)
        if manifest.get('source_declaration')!=declaration:
            raise RuntimeError('Mapping declaration differs; create a new mapping version, never relabel old files')
        if (manifest.get('temperature_kind'),manifest.get('vertical_coordinate'))!=('potential_0dbar','depth_m'):
            raise RuntimeError('Declared mapping must contain potential temperature and meter depths')
    return {str(split/n):sha(split/n) for n in ('split_manifest.json','spatial_partition.npz')}


def fingerprint(c):
    names=list(CODE)+c['config_files']+[c[k] for k in ('parent_protocol','model_config','deterministic_checkpoint',
         'residual_checkpoint','residual_stats','state_stats')]
    if c['branch']=='confirmed_source':names.append(c['source_declaration'])
    from diffargo.config import load_config
    from oe_experiments import input_hashes
    model=load_config(path(c['model_config']))
    if Path(model['data']['stats_path']).resolve()!=path(c['state_stats']).resolve():
        raise RuntimeError('Model state-stats path differs from the declared state stats')
    inputs={}
    for year in (c['calibration_year'],c['development_year']):
        inputs[str(year)]={str(month):input_hashes(dict(config=c['model_config'],argo_split_root=c['split_root'],days=c['days']),
                                                        dict(year=year,month=month)) for month in (1,4,7,10)}
    return dict(files={n:sha(n) for n in sorted(set(names))},mapping=source_check(c),development_inputs=inputs)


def initialize(c):
    root=path(c['output_root']);root.mkdir(parents=True,exist_ok=True)
    record=root/'protocol.json'
    if record.exists():verify(c);print('Existing immutable protocol verified');return
    from oe_experiments import verify_parent
    verify_parent(dict(parent_frozen_protocol=c['parent_protocol']))
    import numpy as np, torch, gsw
    with np.load(path(c['state_stats']),allow_pickle=False) as f:channels=len(f['state_mean'])
    unit=root/'unit_factors.npz'
    if unit.exists():raise FileExistsError('Partial initialization exists; use a new output_root rather than overwrite')
    fingers=fingerprint(c)
    np.savez_compressed(unit,inflation_factor=np.ones((c['days'],channels),np.float32))
    new_json(record,dict(schema=3,created_utc=stamp(),settings=c,fingerprint=fingers,unit_factor_sha256=sha(unit),
                          runtime=dict(numpy=np.__version__,torch=torch.__version__,gsw=gsw.__version__),
                          intent='Prospective code/protocol freeze; not public preregistration or human scientific approval',
                          final_test_frozen=False,engineering_validation=False))
    print('INIT complete. Legacy source is diagnostic only; no unseen-year qualification implied.')


def verify(c):
    record=read(path(c['output_root'])/'protocol.json')
    if record['settings']!=c or record['fingerprint']!=fingerprint(c):
        raise RuntimeError('Frozen inputs/code/config changed. Preserve the old protocol; use a NEW output version.')
    if sha(path(c['output_root'])/'unit_factors.npz')!=record['unit_factor_sha256']:
        raise RuntimeError('Unit factors changed')
    import numpy,torch,gsw
    if record['runtime']!=dict(numpy=numpy.__version__,torch=torch.__version__,gsw=gsw.__version__):
        raise RuntimeError('Frozen numerical-library versions changed; use a new experiment version')
    return record


def catalog(c):
    definitions={m:dict(id=m,kind=m) for m in ('deterministic','unguided_centered','argo_raw')}
    for r in c['trust_regions']:
        for kind in ('det_center','argo_center','argo_joint'):
            name=f"{kind}_{r['id']}"
            opts=dict(c['stability'],center=True,denoise=(kind=='argo_joint'),
                      max_temperature_change=r['temperature'],max_salinity_change=r['salinity'],
                      temperature_step=r['temperature']/c['stability']['center_iterations'],
                      salinity_step=r['salinity']/c['stability']['center_iterations'])
            definitions[name]=dict(id=name,kind=kind,trust=r['id'],options=opts)
    r=c['trust_regions'][1]
    definitions['argo_denoise']=dict(id='argo_denoise',kind='argo_denoise',
        options=dict(c['stability'],denoise=True,center=False,max_temperature_change=r['temperature'],
                     max_salinity_change=r['salinity'],temperature_step=r['temperature']/8,salinity_step=r['salinity']/8))
    for gain in c['nudging_gains']:
        name=f'nudging_{gain:g}';definitions[name]=dict(id=name,kind='nudging',gain=float(gain))
    return definitions


def selection(c):
    record=read(path(c['output_root'])/'selection_screen.json')
    if record['report_sha256']!=sha(path(c['output_root'])/'report_screen.json'):
        raise RuntimeError('Screen report changed after selection')
    if not record['passed']:raise RuntimeError('No physical candidate passed screening; inspect failures before designing a NEW version')
    if catalog(c).get(record['method']['id'])!=record['method']:
        raise RuntimeError('Selected physical method differs from the frozen catalogue')
    return record


def task_list(phase,c):
    if phase=='screen':methods=list(catalog(c))
    else:
        s=selection(c)
        methods=['selected'] if phase=='calibrate' else ['deterministic','unguided_centered','argo_raw','det_selected','selected','nudging_selected']
    year=c['calibration_year'] if phase=='calibrate' else (c['final_year'] if phase=='final' else c['development_year'])
    seeds=c['seeds'][:1] if phase=='screen' else c['seeds']
    days=c['screen_days'] if phase=='screen' else c['days']
    return [dict(phase=phase,year=year,method=m,season=s,month=1+3*i,seed=int(seed),days=days)
            for m in methods for i,s in enumerate(SEASONS) for seed in seeds]


def method_definition(t,c):
    if t['phase']=='screen' or t['method'] in ('deterministic','unguided_centered','argo_raw'):
        return catalog(c)[t['method']]
    s=selection(c)
    if t['method']=='nudging_selected':return s['nudging']
    if t['method']=='det_selected':
        return dict(s['method'],id='det_selected',kind='det_center',options=dict(s['method']['options'],denoise=False,center=True))
    return s['method']


def folder(t,c,smoke=False):
    return path(c['output_root'])/('smoke' if smoke else 'cases')/t['phase']/t['method']/t['season']/str(t['seed'])


def checked_completion(t,c,smoke=False):
    mark=folder(t,c,smoke)/'completed.json'
    if not mark.exists():return None
    record=read(mark)
    if record['task']!=t or record['protocol_sha256']!=sha(path(c['output_root'])/'protocol.json'):
        raise RuntimeError('Completed task/protocol identity changed')
    if record['phase_dependencies']!=phase_dependencies(t['phase'],c):
        raise RuntimeError('Selection/calibration dependency changed since this case completed')
    for name,digest in record['outputs'].items():
        if sha(mark.parent/record['attempt']/name)!=digest:raise RuntimeError('Archived output changed: '+name)
    return record


def phase_dependencies(phase,c):
    root=path(c['output_root']);names=['protocol.json']
    if phase!='screen':names.append('selection_screen.json')
    if phase in ('validate','final'):
        names+=['selected_calibration_2018.npz','calibration_record.json']
        record=read(root/'calibration_record.json')
        if (record['calibration_sha256']!=sha(root/'selected_calibration_2018.npz') or
                record['selection_sha256']!=sha(root/'selection_screen.json')):
            raise RuntimeError('Calibration source selection or factors changed')
        if any(sha(p)!=digest for p,digest in record['source_completed'].items()):
            raise RuntimeError('Calibration completion records changed')
    if phase=='final':names+=['decision_2019.json','final_freeze.json']
    return {name:sha(root/name) for name in names}


def build_command(t,c,out,smoke=False):
    definition=method_definition(t,c);kind=definition['kind'];root=path(c['output_root'])
    calibration=root/'selected_calibration_2018.npz'
    calibrated=t['phase'] in ('validate','final') and t['method']=='selected'
    if calibrated and not calibration.exists():raise FileNotFoundError('Run calibrate and fit before validate')
    diag=dict(c['diagnostics'],expected_days=t['days'],reference_calibration=str(calibration if calibrated else root/'unit_factors.npz'))
    candidates=[dict(id='raw',kind='raw')]+([dict(id='marginal',kind='marginal')] if calibrated else [])
    observer=out/'observer.json';new_json(observer,dict(config=diag,candidates=candidates))
    first=date(t['year'],t['month'],1)
    cmd=[sys.executable,'-u','oe_stable_observer.py','--stable-observer',str(observer),
         '--config',str(path(c['model_config'])),'--deterministic-checkpoint',str(path(c['deterministic_checkpoint'])),
         '--residual-checkpoint',str(path(c['residual_checkpoint'])),'--residual-stats',str(path(c['residual_stats'])),
         '--paper-holdout-root',str(path(c['split_root'])/'holdout'),'--start',str(first-timedelta(days=1)),
         '--days',str(t['days']),'--members',str(4 if smoke else c['members']),
         '--sampling-steps',str(4 if smoke else c['sampling_steps']),'--sampling-eta','0',
         '--residual-scale',str(c['guidance']['residual_scale']),'--seed',str(t['seed']),
         '--save-daily','none','--output-dir',str(out)]
    if kind in ('deterministic','det_center','nudging'):cmd+=['--deterministic-only']
    if kind not in ('deterministic','det_center','unguided_centered'):
        cmd+=['--argo-root',str(path(c['split_root'])/'guidance'),'--require-argo']
    if kind=='nudging':cmd+=['--nudging-gain',str(definition['gain'])]
    elif kind not in ('deterministic','det_center'):
        cmd+=['--individual-member-guidance','--member-mean-cycle','--paired-guidance-center',
              '--temperature-guidance',str(c['guidance']['temperature']),
              '--salinity-guidance',str(c['guidance']['salinity']),
              '--guidance-start',str(c['guidance']['start'])]
    if 'options' in definition:
        options=out/'stability_options.json';new_json(options,definition['options']);cmd+=['--stability-options',str(options)]
    if t['phase']=='calibrate':cmd+=['--calibration-statistics-output',str(out/'calibration_statistics.npz')]
    return cmd


def run_case(t,c,smoke=False):
    if t['phase']=='final':verify_final(c)
    task=dict(t,days=2) if smoke else t
    base=folder(task,c,smoke);base.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (base/'task.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=checked_completion(task,c,smoke)
        if done:print('SKIP checksum-verified completed case:',base,flush=True);return
        number=1
        while (base/f'attempt_{number:03d}').exists():number+=1
        out=base/f'attempt_{number:03d}';out.mkdir()
        dependencies=phase_dependencies(t['phase'],c)
        cmd=build_command(task,c,out,smoke)
        new_json(out/'recipe.json',dict(task=task,command=cmd,created_utc=stamp(),smoke=smoke,
              method_definition=method_definition(task,c),phase_dependencies=dependencies,
              protocol_sha256=sha(path(c['output_root'])/'protocol.json')))
        with (out/'execution.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Case failed; preserved log: {out}/execution.log')
        verify(c)
        if dependencies!=phase_dependencies(t['phase'],c):raise RuntimeError('Phase inputs changed during this case')
        status=read(out/'joint_status.json')
        if status['completed_leads']!=list(range(1,task['days']+1)):raise RuntimeError('Diagnostic leads incomplete')
        for name in ('paper_grid_metrics.csv','paper_holdout_metrics.csv','joint_metrics.csv.gz','joint_events.csv.gz'):
            opener=gzip.open if name.endswith('.gz') else open
            with opener(out/name,'rt',encoding='utf-8',newline='') as f:leads={int(r['lead']) for r in csv.DictReader(f)}
            if leads!=set(range(1,task['days']+1)):raise RuntimeError('Incomplete output leads: '+name)
        marker=out/'completed.pending.json'
        new_json(marker,dict(task=task,attempt=out.name,protocol_sha256=sha(path(c['output_root'])/'protocol.json'),
            phase_dependencies=phase_dependencies(t['phase'],c),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},completed_utc=stamp()))
        os.replace(marker,base/'completed.json')
        print('DONE',task,flush=True)


def fit(c):
    verify(c);files=[];markers={}
    for t in task_list('calibrate',c):
        m=checked_completion(t,c)
        if not m:raise RuntimeError('Calibration incomplete: '+str(t))
        p=folder(t,c)/m['attempt']/'calibration_statistics.npz';files.append(str(p));markers[str(folder(t,c)/'completed.json')]=sha(folder(t,c)/'completed.json')
    destination=path(c['output_root'])/'selected_calibration_2018.npz'
    if destination.exists():raise FileExistsError('Preserve existing selected calibration')
    subprocess.run([sys.executable,'fit_spread_calibration.py','--statistics',*files,'--output',str(destination),
                    '--lead-dependent','--monotonic-lead','--max-factor',str(c['calibration']['max_factor']),
                    '--log-shrinkage',str(c['calibration']['log_shrinkage'])],cwd=ROOT,check=True)
    new_json(path(c['output_root'])/'calibration_record.json',dict(source_completed=markers,calibration_sha256=sha(destination),
              selection_sha256=sha(path(c['output_root'])/'selection_screen.json'),created_utc=stamp()))


def load_case_scores(t,c):
    m=checked_completion(t,c)
    if not m:raise RuntimeError('Incomplete case: '+str(t))
    out=folder(t,c)/m['attempt']
    with gzip.open(out/'joint_pooled.json.gz','rt',encoding='utf-8') as f:payload=json.load(f)
    for table in ('scores','events'):
        for r in payload[table]:
            if r['candidate']=='point':r['candidate']='raw'
    signatures={}
    with gzip.open(out/'joint_metrics.csv.gz','rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            k=tuple(r[n] for n in ('date','domain','support','region','variable','depth_index'))
            value=tuple(r[n] for n in ('support_digest','count','weight'))
            if k in signatures and signatures[k]!=value:raise RuntimeError('Within-case calibration changed support')
            signatures[k]=value
    holdout={}
    with (out/'paper_holdout_metrics.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['depth_index']=='-1' and float(r['weight'])>0:
                holdout.setdefault(r['variable'],[]).append(r)
    return payload,signatures,holdout


def pooled_holdout(rows):
    w=sum(float(r['weight']) for r in rows)
    return dict(weight=w,count=sum(int(r['count']) for r in rows),
                rmse=math.sqrt(sum(float(r['weight'])*float(r['rmse'])**2 for r in rows)/w)) if w else None


def report(phase,c):
    from oe_joint_selection import combine
    verify(c);groups=defaultdict(list);supports={};holdouts=defaultdict(list);markers={};case_records=[]
    if phase=='final':verify_final(c)
    tasks=task_list(phase,c)
    for t in tasks:
        payload,signature,held=load_case_scores(t,c)
        groups[t['method']].append(payload);supports[t['method'],t['season'],t['seed']]=signature
        for variable,rows in held.items():holdouts[t['method'],variable].extend(rows)
        mark=folder(t,c)/'completed.json';markers[str(mark)]=sha(mark)
        case_records.append(dict(task=t,scores=[r for r in payload['scores'] if r['domain']=='glorys' and r['support']=='common' and str(r['depth_index'])=='-1' and r['window']=='all']))
    failures=[]
    for (method,season,seed),sig in supports.items():
        other=supports.get(('deterministic',season,seed))
        if other is not None and sig!=other:failures.append(dict(method=method,season=season,seed=seed,reason='Cross-method physical support differs'))
    result=dict(phase=phase,complete=True,methods={m:combine(p) for m,p in groups.items()},
                holdout=[dict(method=m,variable=v,**pooled_holdout(rows)) for (m,v),rows in holdouts.items()],
                support_errors=failures,case_records=case_records,source_completed=markers,
                protocol_sha256=sha(path(c['output_root'])/'protocol.json'),
                inference='Descriptive seasonal paired comparisons; seeds nested within starts. No independent-gridpoint/day p-values.',
                engineering_validation=False)
    destination=path(c['output_root'])/f'report_{phase}.json'
    if destination.exists():
        if read(destination)!=result:raise RuntimeError('Existing report differs; preserve it')
    else:new_json(destination,result)
    print(f'{phase}: complete={len(tasks)} cases; cross-method support errors={len(failures)}')
    return result


def score(payload,variant,variable,region='global',window='all'):
    return next(r for r in payload['scores'] if r['candidate']==variant and r['domain']=='glorys' and r['support']=='common'
                and r['region']==region and r['variable']==variable and str(r['depth_index'])=='-1' and r['window']==window)


def severe(payload,estimator='cycle_center',variant='common_center',region='global',window='all'):
    return next(r for r in payload['events'] if r['candidate']==variant and r['estimator']==estimator and r['domain']=='glorys'
                and r['support']=='common' and r['region']==region and r['threshold_s2']==-1e-5 and r['window']==window)


def assess(report,method,variant,c,screen=False):
    p=report['methods'][method];baseline=report['methods']['argo_raw'];det=report['methods']['deterministic']
    g=c['gates'];checks=[]
    def check(label,value,limit):
        checks.append(dict(criterion=label,value=float(value),limit=float(limit),passed=bool(math.isfinite(value) and value<=limit)))
    check('cross_method_support_errors',len(report['support_errors']),0)
    for variable in ('temperature','salinity'):
        a,b,d=score(p,variant,variable),score(baseline,'raw',variable),score(det,'raw',variable)
        check(variable+'_rmse_vs_argo',a['rmse']/max(b['rmse'],1e-12),g['max_ts_rmse_ratio_to_argo'])
        check(variable+'_rmse_vs_det',a['rmse']/max(d['rmse'],1e-12),g['max_ts_rmse_ratio_to_deterministic'])
        h=next((r for r in report['holdout'] if r['method']==method and r['variable']==variable),None)
        hb=next((r for r in report['holdout'] if r['method']=='argo_raw' and r['variable']==variable),None)
        if h is None or hb is None:check(variable+'_holdout_missing',1,0)
        else:check(variable+'_holdout_rmse_vs_argo',h['rmse']/max(hb['rmse'],1e-12),g['max_holdout_rmse_ratio_to_argo'])
    objective=0.
    regions=['global'] if screen else ['global']+[r['name'] for r in c['diagnostics']['regions']]
    for region in regions:
        e,b=severe(p,region=region),severe(baseline,region=region)
        limit=b['forecast_frequency']*(g['screen_max_center_event_ratio'] if screen else g['validation_max_center_event_ratio'])
        if not screen:
            limit=min(limit,g['validation_absolute_center_event_limit'],
                      max(g['validation_reference_event_floor'],g['validation_reference_event_multiplier']*e['observed_frequency']))
        check(region+'_center_severe_frequency',e['forecast_frequency'],limit)
        for variable in ('rho','n2'):
            r,br=score(p,variant,variable,region),score(baseline,'raw',variable,region)
            ratio=r['crps']/max(br['crps'],1e-12);objective+=ratio
            check(region+'_'+variable+'_crps_vs_argo',ratio,g['max_derived_crps_ratio_to_argo'])
            if not screen:
                tolerance=g['global_coverage_tolerance'] if region=='global' else g['regional_coverage_tolerance']
                check(region+'_'+variable+'_coverage_distance',abs(r['coverage90']-.9),tolerance)
                if variable=='n2':check(region+'_n2_spread_skill',r['spread_skill_ratio'],g['max_n2_spread_skill'])
        a=severe(p,'member_probability',variant,region);b=severe(baseline,'member_probability','raw',region)
        check(region+'_severe_brier_vs_argo',a['brier']/max(b['brier'],1e-12),g['max_severe_brier_ratio_to_argo'])
    if not screen:
        for window in ('early','middle','late'):
            for variable in ('temperature','salinity'):
                a,b,d=(score(q,v,variable,window=window) for q,v in ((p,variant),(baseline,'raw'),(det,'raw')))
                check(window+'_'+variable+'_rmse_vs_argo',a['rmse']/max(b['rmse'],1e-12),g['max_ts_rmse_ratio_to_argo'])
                check(window+'_'+variable+'_rmse_vs_det',a['rmse']/max(d['rmse'],1e-12),g['max_ts_rmse_ratio_to_deterministic'])
            for variable in ('rho','n2'):
                a,b=score(p,variant,variable,window=window),score(baseline,'raw',variable,window=window)
                check(window+'_'+variable+'_crps_vs_argo',a['crps']/max(b['crps'],1e-12),g['max_derived_crps_ratio_to_argo'])
            e,b=severe(p,window=window),severe(baseline,window=window)
            limit=min(b['forecast_frequency']*g['validation_max_center_event_ratio'],g['validation_absolute_center_event_limit'],
                      max(g['validation_reference_event_floor'],g['validation_reference_event_multiplier']*e['observed_frequency']))
            check(window+'_center_severe_frequency',e['forecast_frequency'],limit)
    return dict(method=method,variant=variant,passed=all(x['passed'] for x in checks),checks=checks,
                objective=objective,center_event=severe(p)['forecast_frequency'])


def select_screen(c):
    r=report('screen',c);catalogue=catalog(c)
    assessments=[assess(r,m,'raw',c,True) for m in catalogue if m.startswith(('argo_center_','argo_joint_')) or m=='argo_denoise']
    eligible=[x for x in assessments if x['passed']]
    chosen=min(eligible,key=lambda x:(x['center_event'],x['objective'])) if eligible else None
    # Nudging gain is selected ONLY on the same 2019 screen, using held profiles.
    def objective(method):
        values=[]
        for variable in ('temperature','salinity'):
            h=next(x for x in r['holdout'] if x['method']==method and x['variable']==variable)
            b=next(x for x in r['holdout'] if x['method']=='deterministic' and x['variable']==variable)
            values.append(h['rmse']/max(b['rmse'],1e-12))
        return sum(values)/2
    nudge=min((m for m in catalogue if m.startswith('nudging_')),key=objective)
    result=dict(passed=bool(chosen),method=catalogue[chosen['method']] if chosen else None,
                nudging=catalogue[nudge],nudging_selection_objective=objective(nudge),all_assessments=assessments,
                report_sha256=sha(path(c['output_root'])/'report_screen.json'),created_utc=stamp())
    new_json(path(c['output_root'])/'selection_screen.json',result)
    print('SCREEN PASSED:',result['passed'],'chosen:',None if not chosen else chosen['method'])
    if not chosen:print('No automatic fallback winner. Failed cases remain available for diagnosis.')


def decide(c):
    r=report('validate',c)
    candidates=[assess(r,'selected',v,c) for v in ('raw','marginal')]
    good=[x for x in candidates if x['passed']]
    chosen=min(good,key=lambda x:x['objective']) if good else None
    result=dict(passed=bool(good),variant=None if chosen is None else chosen['variant'],assessments=candidates,
                report_sha256=sha(path(c['output_root'])/'report_validate.json'),created_utc=stamp(),
                limitations='Research acceptance only; source provenance, independent physics observations and actual engineering validation are separate requirements.')
    new_json(path(c['output_root'])/'decision_2019.json',result)
    print('VALIDATION PASSED:',result['passed'],'variant:',result['variant'])


def freeze_final(c,confirm):
    verify(c);root=path(c['output_root']);d=read(root/'decision_2019.json')
    if not d['passed']:raise RuntimeError('Validation failed. Do not bypass the gate for final-year testing.')
    if c['branch']!='confirmed_source':raise RuntimeError('Legacy Argo provenance unresolved. Final test requires a NEW confirmed-source branch and its full development/calibration.')
    if not confirm:raise RuntimeError('Accountable user must explicitly attest --confirm-unseen for the configured final year')
    if d['report_sha256']!=sha(root/'report_validate.json'):raise RuntimeError('Validation report changed after acceptance')
    from oe_experiments import input_hashes
    data={str(m):input_hashes(dict(config=c['model_config'],argo_split_root=c['split_root'],days=c['days']),dict(year=c['final_year'],month=m)) for m in (1,4,7,10)}
    new_json(root/'final_freeze.json',dict(year=c['final_year'],user_attested_unseen=True,created_utc=stamp(),
        protocol_sha256=sha(root/'protocol.json'),selection_sha256=sha(root/'selection_screen.json'),
        decision_sha256=sha(root/'decision_2019.json'),calibration_sha256=sha(root/'selected_calibration_2018.npz'),
        final_inputs=data,variant=d['variant'],engineering_validation=False))


def verify_final(c):
    root=path(c['output_root']);f=read(root/'final_freeze.json')
    for key,name in [('protocol_sha256','protocol.json'),('selection_sha256','selection_screen.json'),
                     ('decision_sha256','decision_2019.json'),('calibration_sha256','selected_calibration_2018.npz')]:
        if sha(root/name)!=f[key]:raise RuntimeError('Frozen final-test input changed: '+name)
    from oe_experiments import input_hashes
    current={str(m):input_hashes(dict(config=c['model_config'],argo_split_root=c['split_root'],days=c['days']),dict(year=c['final_year'],month=m)) for m in (1,4,7,10)}
    if current!=f['final_inputs']:raise RuntimeError('Final-year source inputs changed')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','status','init','preflight','run','batch','report','select','fit','decide','freeze-final'))
    p.add_argument('--config',default='configs/oe_stability_v3.json');p.add_argument('--phase',choices=PHASES,default='screen')
    p.add_argument('--task',type=int,default=0);p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int)
    p.add_argument('--smoke',action='store_true');p.add_argument('--confirm-unseen',action='store_true')
    a=p.parse_args();c=settings(a.config)
    if a.action=='init':initialize(c);return
    if a.action=='preflight':verify(c);print('All frozen code, mappings, weights and 2018/2019 inputs verified');return
    if a.action=='select':select_screen(c);return
    if a.action=='fit':fit(c);return
    if a.action=='decide':decide(c);return
    if a.action=='freeze-final':freeze_final(c,a.confirm_unseen);return
    if a.action=='report':report(a.phase,c);return
    tasks=task_list(a.phase,c)
    if a.action=='plan':
        for i,t in enumerate(tasks):print(i,t['method'],t['season'],t['seed'],t['days'])
        print('TOTAL',len(tasks));return
    if a.action=='status':
        complete=[];pending=[]
        for i,t in enumerate(tasks):
            (complete if checked_completion(t,c) else pending).append(i)
        print(json.dumps(dict(complete=len(complete),total=len(tasks),pending_task_ids=pending)));return
    verify(c)
    first=a.task if a.action=='run' else a.first
    last=a.task if a.action=='run' else (len(tasks)-1 if a.last is None else a.last)
    if not 0<=first<=last<len(tasks):p.error(f'Task range must be within0..{len(tasks)-1}')
    deadline=float(os.environ.get('OE_STABLE_DEADLINE','inf'));estimate=7200.
    order=list(range(first,last+1))
    if a.action=='batch':
        import numpy as np
        order=np.random.default_rng(c['schedule_seed']+PHASES.index(a.phase)).permutation(order).tolist()
        schedule=path(c['output_root'])/f'allocation_order_{a.phase}_{first}_{last}.json'
        record=dict(protocol_sha256=sha(path(c['output_root'])/'protocol.json'),task_ids=order,
                    purpose='Seeded execution order only; case dates and paired sampling seeds remain fixed')
        if schedule.exists():
            if read(schedule)!=record:raise RuntimeError('Archived task schedule differs')
        else:new_json(schedule,record)
    for i in order:
        check_task=dict(tasks[i],days=2) if a.smoke else tasks[i]
        if checked_completion(check_task,c,a.smoke):
            print('SKIP completed task',i,flush=True);continue
        if time.time()+estimate+900>deadline:
            print(f'Time budget reached; NEXT TASK={i}. Resubmit SAME range; completed cases are verified and skipped.');return
        print(f'RUN task={i}/{last} {tasks[i]}',flush=True);before=time.time()
        run_case(tasks[i],c,a.smoke);elapsed=time.time()-before
        if elapsed>30:estimate=max(estimate,elapsed*1.3)


if __name__=='__main__':main()
