"""Paired, support-checked summaries and conditional moving-block intervals.

Seeds are nested technical replicates and are averaged before resampling days.
Seasonal starts stay fixed. Intervals describe sensitivity within THESE windows,
not uncertainty over independent years or a population of global grid points.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np


def paired_blocks(left, right, block=14, repetitions=2000, seed=314159, root=False):
    """Rows have season/seed/date/lead/weight/count/support_digest/value.

    For RMSE, pass daily MSE and root=True. Resample matched day blocks within
    each season after averaging seed-specific weighted losses. No seed bootstrap.
    """
    def indexed(rows):
        result={}
        for r in rows:
            key=(r['season'],str(r['seed']),r['date'])
            if key in result: raise ValueError('Duplicate paired record')
            result[key]=r
        return result
    a,b=indexed(left),indexed(right)
    if not a or a.keys()!=b.keys(): raise ValueError('Missing/unmatched dates or seeds')
    daily=defaultdict(list)
    for key,x in a.items():
        y=b[key]
        for field in ('lead','count','support_digest'):
            if str(x[field])!=str(y[field]):raise ValueError('Paired support mismatch: '+field)
        wx,wy=float(x['weight']),float(y['weight'])
        if not np.isclose(wx,wy,rtol=1e-12,atol=1e-12):raise ValueError('Paired weights differ')
        vx,vy=float(x['value']),float(y['value'])
        if wx<=0 or not np.isfinite([wx,vx,vy]).all():raise ValueError('Invalid nonempty paired score')
        daily[key[0],int(x['lead']),key[2]].append((key[1],wx,vx,vy))
    seasons=defaultdict(list)
    for (season,lead,day),values in sorted(daily.items()):
        # One contribution per fixed day. The effective weight is the mean,
        # not sum, over random seeds, so duplicated seeds do not add replication.
        z=np.asarray([v[1:] for v in values],float)
        seasons[season].append((lead,tuple(sorted(v[0] for v in values)),
            z[:,0].mean(),np.mean(z[:,0]*z[:,1]),np.mean(z[:,0]*z[:,2])))
    arrays=[]
    for season,rows in sorted(seasons.items()):
        rows.sort()
        if [r[0] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Incomplete consecutive leads')
        if any(r[1]!=rows[0][1] for r in rows):raise ValueError('Seed coverage changes within a start')
        arrays.append(np.asarray([r[2:] for r in rows],float))
    if not isinstance(block,int) or block<1 or repetitions<2:raise ValueError('Invalid bootstrap settings')
    def estimate(sums):
        x,y=sums[1:]/sums[0]
        if root:x,y=np.sqrt(np.maximum([x,y],0))
        return np.array([x,y,x-y])
    actual=estimate(sum((a.sum(axis=0) for a in arrays),np.zeros(3)))
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(repetitions):
        total=np.zeros(3)
        for arr in arrays:
            n=len(arr);length=min(n,block)
            starts=rng.integers(0,n-length+1,size=int(np.ceil(n/length)))
            indices=np.concatenate([np.arange(s,s+length) for s in starts])[:n]
            total+=arr[indices].sum(axis=0)
        draws.append(estimate(total)[2])
    low,high=np.quantile(draws,[.025,.975])
    return dict(left=float(actual[0]),right=float(actual[1]),difference_left_minus_right=float(actual[2]),
        conditional_95_interval=[float(low),float(high)],block_days=block,repetitions=repetitions,
        fixed_starts=len(arrays),days=sum(len(a) for a in arrays),
        inference='Conditional moving-block sensitivity within fixed starts; not an independent-year CI or a hypothesis test')


def read_daily(out,t):
    result=defaultdict(list)
    with gzip.open(Path(out)/'joint_metrics.csv.gz','rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['domain']!='glorys' or r['support']!='common' or r['depth_index']!='-1' or float(r['weight'])<=0:continue
            if r['variable'] not in ('temperature','salinity','rho','n2','pycnocline_depth'):continue
            candidate='raw' if r['candidate']=='point' else r['candidate']
            for metric in ('rmse','crps','coverage90','width90'):
                row={k:r[k] for k in ('date','lead','count','weight','support_digest')}
                value=float(r[metric]);value=value**2 if metric=='rmse' else value
                result[candidate,r['region'],r['variable'],metric].append(dict(row,season=t['season'],seed=t['seed'],value=value))
    with (Path(out)/'ems_multivariate.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if float(r['weight'])<=0:raise ValueError('No complete profiles on a declared evaluation day')
            candidate='raw' if r['candidate']=='point' else r['candidate']
            for metric in ('energy_score','variogram_score'):
                row={k:r[k] for k in ('date','lead','count','weight','support_digest')}
                result[candidate,'global_complete_columns','standardized_TS_profile',metric].append(dict(row,
                    season=t['season'],seed=t['seed'],value=float(r[metric])))
    return result


def evaluate(phase,c,repetitions=2000):
    import ems_experiments as workflow
    report=workflow.report(phase,c)
    data={};costs=[]
    for case in report['cases']:
        t=case['task'];key=f"{t['scenario']}|{t['method']}|m{t['members']}_k{t['steps']}"
        for score,rows in read_daily(case['output'],t).items():data.setdefault(key,defaultdict(list))[score].extend(rows)
        out=Path(case['output']);allocation=json.loads((out/'allocation_cost.json').read_text())
        hardware=json.loads((out/'compute_cost.json').read_text())
        costs.append(dict(task=t,**allocation,cycle_loop=hardware))
    comparisons=[]
    for key,scores in sorted(data.items()):
        scenario,method,budget=key.split('|')
        if phase=='budget':reference=f'nominal|selected|m{max(c["budget_members"])}_k{max(c["budget_steps"])}'
        else:reference=f'{scenario}|'+('guided_raw' if phase=='robust' else 'deterministic')+'|'+budget
        if reference not in data or key==reference:continue
        if scores.keys()!=data[reference].keys():
            # A calibrated method legitimately has an additional candidate;
            # each candidate is paired to the reference raw ensemble.
            pass
        for (candidate,region,var,metric),rows in scores.items():
            other=data[reference].get(('raw',region,var,metric))
            if other is None:raise ValueError('Missing reference metric')
            for block in (7,14,30):
                interval=paired_blocks(rows,other,block,repetitions,root=(metric=='rmse'))
                comparisons.append(dict(method=key,reference=reference,candidate=candidate,region=region,variable=var,metric=metric,**interval))
    root=workflow.path(c['output_root']);dest=root/f'evaluation_{phase}.json'
    payload=dict(phase=phase,comparisons=comparisons,costs=costs,
        report_sha256=workflow.sha(root/f'report_{phase}.json'),
        limitations=['No independent-year population inference','No independent Argo thermodynamics claim',
            'Full-process cost includes scoring/I/O; compare hardware and member/step counts',
            'Gaussian baseline has diagonal covariance, not learned spatial covariance'])
    if dest.exists():
        if workflow.read(dest)!=payload:raise RuntimeError('Existing evaluation differs; preserve it')
    else:workflow.write_new(dest,payload)
    print('Saved',dest,'paired conditional comparisons',len(comparisons))


def main():
    import ems_experiments as workflow
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ems_v1_legacy.json')
    p.add_argument('--phase',choices=workflow.PHASES,default='validate');p.add_argument('--repetitions',type=int,default=2000)
    a=p.parse_args();evaluate(a.phase,workflow.settings(a.config),a.repetitions)


if __name__=='__main__':main()
