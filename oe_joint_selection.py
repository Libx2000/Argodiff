"""Prespecified 2018 fitting and 2019 joint-physics acceptance criteria."""
from __future__ import annotations
import math
from collections import defaultdict

SQUARED={'rmse','cycle_center_rmse','rms_spread'}


def combine(case_payloads):
    result={}
    for table,keys in [('scores',('candidate','domain','support','region','variable','depth_index','window')),
                       ('events',('candidate','estimator','domain','support','region','threshold_s2','window'))]:
        groups=defaultdict(list)
        for payload in case_payloads:
            for row in payload[table]:groups[tuple(str(row[k]) for k in keys)].append(row)
        rows=[]
        for key,items in groups.items():
            total=sum(r['weight'] for r in items)
            if not total:continue
            out={k:items[0][k] for k in keys};out.update(weight=total,count=sum(r['count'] for r in items))
            for column in items[0]:
                if column in keys or column in ('count','weight','spread_skill_ratio'):continue
                if column in SQUARED:out[column]=math.sqrt(max(0,sum(r['weight']*r[column]**2 for r in items)/total))
                elif isinstance(items[0][column],(int,float)):out[column]=sum(r['weight']*r[column] for r in items)/total
            if 'rms_spread' in out:out['spread_skill_ratio']=out['rms_spread']/max(out['rmse'],1e-12)
            rows.append(out)
        result[table]=rows
    return result


def assess(payload,candidate,config):
    """Same rule for training and validation; never selects using 2021/2022."""
    settings=config['acceptance'];support=config['selection_support'];failures=[];details=[];objective=0.
    index={(r['candidate'],r['domain'],r['support'],r['region'],r['variable'],str(r['depth_index']),r['window']):r for r in payload['scores']}
    event={(r['candidate'],r['domain'],r['support'],r['region'],float(r['threshold_s2']),r['window']):r for r in payload['events'] if r['estimator']=='member_probability'}
    def row(name,region,var,window='all'):
        key=(name,'glorys',support,region,var,'-1',window)
        if key not in index:raise ValueError(f'Missing acceptance metric: {key}')
        return index[key]
    def check(label,value,limit,relation='max'):
        good=math.isfinite(value) and (value<=limit if relation=='max' else value>=limit)
        details.append(dict(criterion=label,value=value,limit=limit,relation=relation,passed=good))
        if not good:failures.append(label)
    for region,weight in config['objective_region_weights'].items():
        rho=row(candidate,region,'rho');n2=row(candidate,region,'n2')
        marginal=row('marginal',region,'rho');raw=row('raw',region,'n2')
        if rho['count']!=marginal['count'] or rho['weight']!=marginal['weight'] or n2['count']!=raw['count'] or n2['weight']!=raw['weight']:
            raise ValueError('Acceptance comparisons do not share identical support')
        check(region+':density_crps_ratio',rho['crps']/max(marginal['crps'],1e-12),1+settings['crps_relative_tolerance'])
        check(region+':n2_crps_ratio',n2['crps']/max(raw['crps'],1e-12),1+settings['crps_relative_tolerance'])
        check(region+':density_rmse_ratio',rho['rmse']/max(marginal['rmse'],1e-12),1+settings['rmse_relative_tolerance'])
        check(region+':n2_rmse_ratio',n2['rmse']/max(raw['rmse'],1e-12),1+settings['rmse_relative_tolerance'])
        tolerance=settings['global_coverage_tolerance'] if region=='global' else settings['regional_coverage_tolerance']
        for variable,r in [('rho',rho),('n2',n2)]:
            check(region+':'+variable+'_coverage_low',r['coverage90'],.90-tolerance,'min')
            check(region+':'+variable+'_coverage_high',r['coverage90'],min(1.,.90+tolerance))
        check(region+':n2_spread_skill_max',n2['spread_skill_ratio'],settings['n2_max_spread_skill'])
        key=('raw','glorys',support,region,settings['severity_guard_threshold_s2'],'all')
        other=(candidate,*key[1:])
        if key not in event or other not in event:raise ValueError('Missing severe-negative-N2 event scores')
        check(region+':severe_n2_brier_ratio',event[other]['brier']/max(event[key]['brier'],1e-12),1+settings['severity_brier_relative_tolerance'])
        objective+=weight*(.5*(rho['crps']/max(marginal['crps'],1e-12)+n2['crps']/max(raw['crps'],1e-12))+
            settings['coverage_objective_weight']*(abs(rho['coverage90']-.9)+abs(n2['coverage90']-.9)))
    for window in ('early','middle','late'):
        for variable,reference in [('rho','marginal'),('n2','raw')]:
            a,b=row(candidate,'global',variable,window),row(reference,'global',variable,window)
            check(window+':'+variable+'_crps_ratio',a['crps']/max(b['crps'],1e-12),1+settings['window_crps_relative_tolerance'])
    return dict(candidate=candidate,passed=not failures,objective=objective,failures=failures,criteria=details)


def choose_2018(payload,candidates,config):
    assessments=[assess(payload,c['id'],config) for c in candidates if c['kind']=='joint']
    if not assessments:raise ValueError('No joint candidates')
    feasible=[a for a in assessments if a['passed']]
    winner=min(feasible or assessments,key=lambda r:(r['objective'],r['candidate']))
    selected=next(c for c in candidates if c['id']==winner['candidate'])
    return dict(selected=selected,training_gate_passed=bool(feasible),assessment=winner,
        all_assessments=assessments,selection_year=2018,validation_year=2019,
        provisional=not bool(feasible),
        note='If no candidate passes, the best proposal is retained for diagnostic validation only; deployment/final test stays blocked.')
