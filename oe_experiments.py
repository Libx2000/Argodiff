"""Versioned OE extension: ordinary sequential jobs, immutable parent artifacts."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,os,subprocess,sys,time
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
PHASES=('legacy2021','consistent2021','consistent2022','calibrate')

def load_settings(path='configs/oe_experiments.json'):
    c=json.loads((ROOT/path).read_text(encoding='utf-8'))
    c['_config_path']=str(path)
    from paper_supplement import protocol
    parent=protocol()
    return c,parent

def tasks(phase,c):
    if phase not in PHASES:raise ValueError(phase)
    year=2018 if phase=='calibrate' else int(phase[-4:])
    methods=['fine_paired'] if phase=='calibrate' else c['methods']
    return [dict(phase=phase,year=year,method=m,month=1+3*i,season=s,seed=int(seed))
        for m in methods for i,s in enumerate(('winter','spring','summer','autumn')) for seed in c['seeds']]

def root_output(c):return ROOT/c['output_root']
def destination(t,c):return root_output(c)/t['phase']/t['method']/t['season']/str(t['seed'])

@lru_cache(maxsize=4096)
def cached_hash(path,size,mtime):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def sha(path):
    path=Path(path);st=path.stat()
    return cached_hash(str(path.resolve()),st.st_size,st.st_mtime_ns)

def verify_parent(c):
    frozen=json.loads((ROOT/c['parent_frozen_protocol']).read_text(encoding='utf-8'))
    for name,expected in frozen['fingerprints'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        if not p.is_file() or sha(p)!=expected:
            raise RuntimeError(f'Parent frozen artifact changed or unavailable: {p}. Do NOT overwrite the parent freeze.')
    audit=ROOT/Path(c['parent_frozen_protocol']).parent/'argo_audit/summary.json'
    if 'audit_sha256' in frozen and sha(audit)!=frozen['audit_sha256']:
        raise RuntimeError('Parent Argo audit differs from original freeze')
    return frozen

def input_hashes(c,t):
    import yaml
    split=Path(c['argo_split_root']);first=date(t['year'],t['month'],1)
    paths=[split/section/str(d.year)/f'{d.isoformat()}.npz'
        for section in ('guidance','holdout') for d in
        (first+timedelta(days=k) for k in range(c['days']))]
    config=yaml.safe_load((ROOT/c['config']).read_text(encoding='utf-8'))['data']
    data=Path(config['root']);data=data if data.is_absolute() else ROOT/data
    initial=first-timedelta(days=config.get('context_days',3));last=first+timedelta(days=c['days']-1)
    for year in range(initial.year,last.year+1):
        for variable in ('temperature','salinity'):
            paths.append(data/config[variable+'_subdir']/config[variable+'_pattern'].format(year=year))
    return {str(path):sha(path) for path in paths}

def artifacts(c,parent,phase):
    split=c['legacy_split_root'] if phase.startswith('legacy') else c['consistent_split_root']
    p=dict(parent,argo_split_root=split,days=c['days'])
    names=[parent[k] for k in ('config','deterministic_checkpoint','residual_checkpoint','fine_checkpoint','residual_stats','state_stats')]
    names += ['oe_experiments.py','oe_cycle.py','oe_physics.py','prepare_oe_argo.py','audit_oe_argo.py',
              c.get('_config_path','configs/oe_experiments.json'),c['diagnostics_config'],
              c['parent_frozen_protocol'],str(Path(split)/'split_manifest.json'),str(Path(split)/'spatial_partition.npz')]
    cal=c['legacy_calibration'] if phase.startswith('legacy') else c['consistent_calibration']
    if phase!='calibrate':names.append(cal)
    if not phase.startswith('legacy'):names.append('configs/oe_source_declaration.json')
    fingerprints={str(name):sha(Path(name) if Path(name).is_absolute() else ROOT/name) for name in sorted(set(names))}
    # Runtime versions bind numeric implementations, not only code and weights.
    import numpy,gsw,torch
    versions=dict(numpy=numpy.__version__,gsw=gsw.__version__,torch=torch.__version__)
    data={str(t['month']):input_hashes(p,t) for t in tasks(phase,c)[::len(c['seeds'])]}
    return dict(settings=c,parent=parent,split=split,fingerprints=fingerprints,versions=versions,data=data)

def freeze(phase,c,parent,confirm_unseen=False):
    old=verify_parent(c)
    record=artifacts(c,parent,phase)
    if phase=='legacy2021':
        from paper_supplement import SEEDS
        if c['seeds']!=list(SEEDS):raise RuntimeError('Legacy replay must use the original paired seeds')
        if record['data']!=old['test_data']['test2021']:
            raise RuntimeError('Legacy replay inputs differ from original 2021 freeze')
        for k in ('members','days','sampling_steps','residual_scale','temperature_guidance','salinity_guidance','guidance_start'):
            actual=c.get(k,parent.get(k))
            if k=='residual_scale':expected=1.0
            else:expected=old['protocol'][k]
            if actual!=expected:raise RuntimeError(f'Legacy replay setting differs: {k}')
    else:
        manifest=json.loads((Path(record['split'])/'split_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('temperature_kind')!='potential_0dbar' or manifest.get('vertical_coordinate')!='depth_m':
            raise RuntimeError('Use a declared TEOS-consistent split; unknown variable definitions refused')
        declaration=json.loads((ROOT/'configs/oe_source_declaration.json').read_text(encoding='utf-8'))
        if manifest.get('source_declaration')!=declaration:raise RuntimeError('Current declaration differs from the one used to map the observations')
    record.update(phase=phase,utc=datetime.now(timezone.utc).isoformat(),
        analysis_intent='new frozen test, contingent on author confirmation of unseen year' if phase=='consistent2022' and confirm_unseen else 'posthoc physical extension / sensitivity, not a new blind test',
        author_confirmed_unseen=bool(confirm_unseen),parent_frozen_artifacts_verified=True)
    out=root_output(c);out.mkdir(parents=True,exist_ok=True)
    path=out/f'frozen_{phase}.json'
    with path.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(f'Frozen {phase}: {path}',flush=True)

def build_command(t,c,parent,out):
    legacy=t['phase'].startswith('legacy')
    split=Path(c['legacy_split_root'] if legacy else c['consistent_split_root'])
    checkpoint=parent['residual_checkpoint'] if t['method']=='old_paired' else parent['fine_checkpoint']
    script='argo_guided_cycle.py' if t['phase']=='calibrate' else 'oe_cycle.py'
    cmd=[sys.executable,'-u',str(ROOT/script)]
    if script=='oe_cycle.py':cmd+=['--oe-config',str(ROOT/c['diagnostics_config'])]
    cmd+=['--config',parent['config'],'--deterministic-checkpoint',parent['deterministic_checkpoint'],
        '--residual-checkpoint',checkpoint,'--residual-stats',parent['residual_stats'],
        '--start',(date(t['year'],t['month'],1)-timedelta(days=1)).isoformat(),
        '--days',str(c['days']),'--members',str(c['members']),'--sampling-steps',str(c['sampling_steps']),
        '--sampling-eta','0','--residual-scale',str(c['residual_scale']),'--seed',str(t['seed']),
        '--paper-holdout-root',str(split/'holdout'),'--save-daily','none','--output-dir',str(out)]
    if t['method'] in ('deterministic','nudging'):cmd+=['--deterministic-only']
    if t['method'] not in ('deterministic','unguided_centered'):
        cmd+=['--argo-root',str(split/'guidance'),'--require-argo']
    if t['method']=='nudging':cmd+=['--nudging-gain',str(c['nudging_gain'])]
    elif t['method']!='deterministic':
        cmd+=['--individual-member-guidance','--member-mean-cycle','--paired-guidance-center']
        cmd+=['--temperature-guidance',str(parent['temperature_guidance']),
              '--salinity-guidance',str(parent['salinity_guidance']),'--guidance-start',str(parent['guidance_start'])]
    if t['method']=='fine_calibrated':
        cmd+=['--spread-calibration',str(ROOT/(c['legacy_calibration'] if legacy else c['consistent_calibration'])),
              '--strict-calibration-leads']
    if t['phase']=='calibrate':cmd+=['--calibration-statistics-output',str(out/'calibration_statistics.npz')]
    return cmd

def completed(t,c):
    folder=destination(t,c)
    mark=folder/'completed.json'
    if not mark.exists():return None
    record=json.loads(mark.read_text(encoding='utf-8'))
    if record['task']!=t:raise RuntimeError('Completed task identity mismatch')
    attempt=folder/record['attempt']
    for name,expected in record['outputs'].items():
        if sha(attempt/name)!=expected:raise RuntimeError(f'Completed result changed: {attempt/name}')
    return record

def validate_days(path,days):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8',newline='') as f:
        leads={int(r['lead']) for r in csv.DictReader(f)}
    if leads!=set(range(1,days+1)):raise RuntimeError(f'Incomplete score leads: {path}')

def run(t,c,parent,smoke=False):
    verify_parent(c)
    signature=artifacts(c,parent,t['phase'])
    frozen=json.loads((root_output(c)/f"frozen_{t['phase']}.json").read_text(encoding='utf-8'))
    for key in signature:
        if frozen[key]!=signature[key]:raise RuntimeError(f'OE freeze mismatch: {key}. Choose a new output version, never replace an existing freeze.')
    if smoke:
        # Separate 2-day outputs; never mistake smoke completion for the 90-day experiment.
        c=dict(c,days=2,output_root=c['output_root']+'/smoke')
    folder=destination(t,c);folder.mkdir(parents=True,exist_ok=True)
    # Advisory Linux lock releases on process death; no stale scheduler-ledger reattachment.
    import fcntl
    with (folder/'task.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=completed(t,c)
        if done:
            if done['signature']!=signature:raise RuntimeError('Completed run used another protocol')
            print(f'SKIP complete: {folder}',flush=True);return
        number=1
        while (folder/f'attempt_{number:03d}').exists():number+=1
        out=folder/f'attempt_{number:03d}';out.mkdir()
        cmd=build_command(t,c,parent,out)
        recipe=dict(task=t,signature=signature,command=cmd,smoke=smoke,utc=datetime.now(timezone.utc).isoformat())
        (out/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
        with (out/'execution.log').open('w',encoding='utf-8') as log:
            log.write(' '.join(cmd)+'\n');log.flush()
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Cycle failed, original attempt preserved: {out}/execution.log')
        for name in ('paper_grid_metrics.csv','paper_holdout_metrics.csv'):
            validate_days(out/name,c['days'])
        if t['phase']!='calibrate':
            validate_days(out/'oe_physics_metrics.csv.gz',c['days'])
            status=json.loads((out/'oe_diagnostic_status.json').read_text(encoding='utf-8'))
            if status['completed_leads']!=list(range(1,c['days']+1)):raise RuntimeError('Physics observer did not complete')
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='execution.log'}
        marker=dict(task=t,signature=signature,attempt=out.name,outputs=outputs,utc=datetime.now(timezone.utc).isoformat())
        tmp=folder/'completed.pending.json';tmp.write_text(json.dumps(marker,indent=2),encoding='utf-8')
        os.replace(tmp,folder/'completed.json')
        print(f'DONE {t}: {out}',flush=True)

def fit(c,parent):
    signature=artifacts(c,parent,'calibrate')
    frozen=json.loads((root_output(c)/'frozen_calibrate.json').read_text(encoding='utf-8'))
    if any(frozen[k]!=v for k,v in signature.items()):raise RuntimeError('Calibration freeze changed')
    files=[]
    for t in tasks('calibrate',c):
        record=completed(t,c)
        if not record:raise RuntimeError(f'Calibration task missing: {t}')
        if record['signature']!=signature:raise RuntimeError('Calibration recipe changed')
        files.append(str(destination(t,c)/record['attempt']/'calibration_statistics.npz'))
    out=ROOT/c['consistent_calibration']
    if out.exists():raise FileExistsError('Preserve existing consistent calibration')
    subprocess.run([sys.executable,'fit_spread_calibration.py','--statistics',*files,
        '--output',str(out),'--lead-dependent','--monotonic-lead','--max-factor',str(c['max_factor']),
        '--log-shrinkage',str(c['log_shrinkage'])],cwd=ROOT,check=True)

def pool(rows):
    selected=[r for r in rows if float(r['weight'])>0 and math.isfinite(float(r['rmse']))]
    total=sum(float(r['weight']) for r in selected)
    if not total:return dict(weight=0,rmse=None)
    avg=lambda k:sum(float(r['weight'])*float(r[k]) for r in selected)/total
    result=dict(weight=total,count=sum(int(r.get('count',0)) for r in selected),rmse=math.sqrt(avg('rmse_squared')) if 'rmse_squared' in selected[0] else
        math.sqrt(sum(float(r['weight'])*float(r['rmse'])**2 for r in selected)/total),
        bias=avg('bias'),mae=avg('mae'),crps=avg('crps'),coverage90=avg('coverage90'))
    if 'rms_spread' in selected[0]:
        spread=math.sqrt(sum(float(r['weight'])*float(r['rms_spread'])**2 for r in selected)/total)
        result.update(rms_spread=spread,spread_skill_ratio=spread/result['rmse'] if result['rmse'] else None,
            range_coverage=avg('range_coverage'))
    if all(r.get('rank_bin_0','')!='' for r in selected):
        result['rank_histogram']=[avg(f'rank_bin_{i}') for i in range(10)]
    return result

def report(phase,c):
    if phase=='calibrate':raise ValueError('Use fit after the eight calibration tasks; physics reports apply to the evaluation phases')
    _,parent=load_settings(c.get('_config_path','configs/oe_experiments.json'))
    verify_parent(c)
    signature=artifacts(c,parent,phase)
    frozen=json.loads((root_output(c)/f'frozen_{phase}.json').read_text(encoding='utf-8'))
    if any(frozen[k]!=v for k,v in signature.items()):raise RuntimeError('Report inputs differ from frozen protocol')
    groups=defaultdict(list);missing=[];replay=[];events=defaultdict(list);joint=defaultdict(list)
    for t in tasks(phase,c):
        record=completed(t,c)
        if not record:missing.append(t);continue
        if record['signature']!=signature:raise RuntimeError(f'Mixed protocols in completed outputs: {t}')
        out=destination(t,c)/record['attempt']
        with gzip.open(out/'oe_physics_metrics.csv.gz','rt',encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                key=tuple(row[k] for k in ('domain','region','variable','estimator','depth_index'))
                groups[(t['method'],*key)].append(dict(row,season=t['season'],seed=t['seed']))
        with gzip.open(out/'oe_event_metrics.csv.gz','rt',encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                key=tuple(row[k] for k in ('domain','region','event','reference_density_kgm3','tolerance_N_m3'))
                events[(t['method'],*key)].append(row)
        with gzip.open(out/'oe_joint_structure.csv.gz','rt',encoding='utf-8',newline='') as f:
            joint[t['method']].extend(csv.DictReader(f))
        if phase=='legacy2021' and t['method'] in ('deterministic','old_paired','fine_paired','fine_calibrated'):
            old=ROOT/'runs/paper_supplement_v1/test2021'/f"{t['method']}_scale1_continuous"/t['season']/str(t['seed'])
            for name in ('paper_grid_metrics.csv','paper_holdout_metrics.csv'):
                with (old/name).open(encoding='utf-8',newline='') as f:before=list(csv.DictReader(f))
                with (out/name).open(encoding='utf-8',newline='') as f:after=list(csv.DictReader(f))
                if len(before)!=len(after):raise RuntimeError('Legacy replay score length mismatch')
                delta=[]
                for b,a in zip(before,after):
                    for key in ('date','lead','domain','variable','depth_index','count','weight'):
                        if b[key]!=a[key]:raise RuntimeError('Legacy replay samples changed')
                    for key in ('rmse','crps','coverage90'):
                        x,y=float(b[key]),float(a[key])
                        if math.isfinite(x) and math.isfinite(y):delta.append(abs(x-y))
                replay.append(dict(task=t,file=name,max_absolute_score_difference=max(delta,default=0.)))
    summary=[];paired=[];seasonal=[];windows=[]
    for key,rows in groups.items():
        method,domain,region,var,est,depth=key
        summary.append(dict(method=method,domain=domain,region=region,variable=var,estimator=est,depth_index=depth,**pool(rows)))
        if method=='deterministic':continue
        baseline=groups.get(('deterministic',*key[1:]))
        if baseline is None:continue
        lookup={(r['season'],r['seed'],r['date']):r for r in baseline}
        matched=len(rows)==len(baseline) and all((r['season'],r['seed'],r['date']) in lookup and
            r['count']==lookup[(r['season'],r['seed'],r['date'])]['count'] and
            r['weight']==lookup[(r['season'],r['seed'],r['date'])]['weight'] for r in rows)
        if not matched:
            paired.append(dict(method=method,domain=domain,region=region,variable=var,estimator=est,depth_index=depth,comparable=False,reason='valid sample support differs'));continue
        a,b=pool(rows),pool(baseline)
        if a['weight'] and b['weight'] and b['rmse']:
            paired.append(dict(method=method,domain=domain,region=region,variable=var,estimator=est,depth_index=depth,
                comparable=True,rmse_gain_percent=100*(1-a['rmse']/b['rmse']),crps_change=a['crps']-b['crps']))
        if depth=='-1':
            for season in ('winter','spring','summer','autumn'):
                a,b=pool([r for r in rows if r['season']==season]),pool([r for r in baseline if r['season']==season])
                if a['weight'] and b['weight'] and b['rmse']:
                    seasonal.append(dict(method=method,domain=domain,region=region,variable=var,estimator=est,season=season,
                        rmse_gain_percent=100*(1-a['rmse']/b['rmse'])))
            for lo,hi in [(1,30),(31,60),(61,c['days']),(c['days'],c['days'])]:
                select=lambda rs:[r for r in rs if lo<=int(r['lead'])<=hi]
                a,b=pool(select(rows)),pool(select(baseline))
                if a['weight'] and b['weight'] and b['rmse']:
                    windows.append(dict(method=method,domain=domain,region=region,variable=var,estimator=est,
                        first_lead=lo,last_lead=hi,rmse=a['rmse'],deterministic_rmse=b['rmse'],
                        rmse_gain_percent=100*(1-a['rmse']/b['rmse']),crps=a['crps'],coverage90=a['coverage90']))
    event_summary=[]
    for key,rows in events.items():
        valid=[r for r in rows if float(r['weight'])>0];w=sum(float(r['weight']) for r in valid)
        if not w:continue
        reliability=[]
        for i in range(10):
            bw=sum(float(r[f'bin_{i}_weight']) for r in valid)
            binavg=lambda k:sum(float(r[f'bin_{i}_weight'])*float(r[f'bin_{i}_{k}']) for r in valid if float(r[f'bin_{i}_weight'])>0)/bw if bw else None
            reliability.append(dict(bin=i,weight=bw,probability=binavg('probability'),observed=binavg('observed')))
        event_summary.append(dict(method=key[0],domain=key[1],region=key[2],event=key[3],reference_density_kgm3=key[4],tolerance_N_m3=key[5],reliability=reliability,
            weight=w,**{k:sum(float(r['weight'])*float(r[k]) for r in valid)/w for k in ('brier','forecast_frequency','observed_frequency')}))
    structure=[]
    for method,rows in joint.items():
        stats={}
        for column,wkey in [('temperature_salinity_correlation','weight'),('temperature_salinity_covariance','weight'),('adjacent_density_correlation','adjacent_density_weight')]:
            valid=[r for r in rows if float(r[wkey])>0 and math.isfinite(float(r[column]))]
            total=sum(float(r[wkey]) for r in valid)
            stats[column]=sum(float(r[wkey])*float(r[column]) for r in valid)/total if total else None
        structure.append(dict(method=method,**stats))
    result=dict(phase=phase,complete=not missing,missing=missing,scores=summary,paired_vs_deterministic=paired,
        seasonal_comparisons=seasonal,lead_windows=windows,event_scores=event_summary,joint_structure=structure,legacy_replay_checks=replay,
        inference='Descriptive paired seasonal effects. Seeds, days, layers and grid points are not independent ocean replications; no significance claims.',
        engineering='Hypothetical unit-volume buoyancy environment, not device safety or resolved internal-wave load skill.')
    (root_output(c)/f'summary_{phase}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(f"{phase}: missing={len(missing)}; summary saved",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','freeze','run','batch','fit','report','preflight'))
    p.add_argument('--phase',choices=PHASES,default='legacy2021');p.add_argument('--config',default='configs/oe_experiments.json')
    p.add_argument('--task',type=int,default=0);p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int)
    p.add_argument('--smoke',action='store_true');p.add_argument('--confirm-unseen',action='store_true')
    a=p.parse_args();c,parent=load_settings(a.config);ts=tasks(a.phase,c)
    if a.action=='plan':
        for i,t in enumerate(ts):print(i,t['method'],t['season'],t['seed'],destination(t,c))
    elif a.action=='preflight':
        verify_parent(c);r=artifacts(c,parent,a.phase)
        print(json.dumps(dict(parent_verified=True,versions=r['versions'],days=c['days'],members=c['members'],tasks=len(ts)),indent=2))
    elif a.action=='freeze':freeze(a.phase,c,parent,a.confirm_unseen)
    elif a.action=='fit':fit(c,parent)
    elif a.action=='report':report(a.phase,c)
    else:
        first=a.task if a.action=='run' else a.first
        last=a.task if a.action=='run' else (len(ts)-1 if a.last is None else a.last)
        if not 0<=first<=last<len(ts):p.error(f'Valid task range 0..{len(ts)-1}')
        deadline=float(os.environ.get('OE_ALLOCATION_DEADLINE','inf'))
        estimate=5400.
        for i in range(first,last+1):
            if time.time()+estimate+600>deadline:
                print(f'Allocation nearly exhausted; next task={i}. Resubmit SAME range: completed tasks are skipped.',flush=True);return
            print(f'RUN {i}/{last}: {ts[i]}',flush=True)
            before=time.time();run(ts[i],c,parent,a.smoke);elapsed=time.time()-before
            if elapsed>60:estimate=max(estimate,elapsed*1.5)

if __name__=='__main__':main()
