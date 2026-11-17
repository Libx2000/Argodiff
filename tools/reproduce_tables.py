"""Recompute manuscript Tables 2-6 from the released, explicitly versioned scores."""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

METHODS = {'deterministic':'Deterministic','unguided_centered':'Unguided',
 'guided_common':'Common-Diff','guided_raw':'Argo-Diff','selected':'SA-Diff',
 'nudging_selected':'Nudging','gaussian_selected':'Diagonal Gaussian','source_guided':'Source-Diff'}
BLOCKS={('winter','20261006'),('winter','20261017'),('spring','20261006'),
        ('spring','20261017'),('summer','20261006')}


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def only(rows,**filters):
    found=[r for r in rows if all(str(r.get(k))==str(v) for k,v in filters.items())]
    if len(found)!=1:raise ValueError(f'Expected one row, found {len(found)}: {filters}')
    return found[0]


def rmse(rows):
    return math.sqrt(sum(float(r['weight'])*float(r['rmse'])**2 for r in rows)/
                     sum(float(r['weight']) for r in rows))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args(argv)
    paper=a.data_root.resolve()/'paper';out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    scores=[r for r in read(paper/'source_data'/'validation_scores.csv') if
      r['scope']=='matched_primary' and r['domain']=='glorys' and r['support']=='common'
      and r['region']=='global' and r['depth_index']=='-1' and r['window']=='all']
    source=read(paper/'Figure4_recomputed_20261115.csv')
    revised={(m,v):rmse([r for r in source if r['method']==m and r['variable']==v])
      for m in ('deterministic','guided_raw','selected') for v in ('temperature','salinity')}
    revised.update({('unguided_centered',v):revised['deterministic',v] for v in ('temperature','salinity')})
    table2=[]
    for method,name in METHODS.items():
        get=lambda v:only(scores,method=method,variable=v,candidate='raw')
        t=revised.get((method,'temperature'),float(get('temperature')['rmse']))
        s=revised.get((method,'salinity'),float(get('salinity')['rmse']))
        table2.append(dict(method=name,temperature_rmse_degC=f'{t:.5f}',salinity_rmse=f'{s:.5f}',
         density_crps_kg_m3=f"{float(get('rho')['crps']):.5f}",n2_crps_1e_minus5_s2=f"{float(get('n2')['crps'])*1e5:.4f}",
         ts_source='revised_depth_summary' if (method,'temperature') in revised else 'archived_common_support'))
    write(out/'Table2.csv',table2)
    events=[r for r in read(paper/'source_data'/'validation_events.csv') if r['scope']=='matched_primary'
       and r['domain']=='glorys' and r['support']=='common' and r['region']=='global'
       and r['window']=='all' and float(r['threshold_s2'])==-1e-5]
    table3=[]
    for method in ('deterministic','guided_raw','selected'):
        center=only(events,method=method,candidate='common_center',estimator='cycle_center')
        member=only(events,method=method,candidate='raw',estimator='member_probability')
        table3.append(dict(method=METHODS[method],center_frequency_percent=f"{100*float(center['forecast_frequency']):.5f}",
          member_frequency_percent=f"{100*float(member['forecast_frequency']):.4f}",brier_score=f"{float(member['brier']):.5f}"))
    write(out/'Table3.csv',table3)
    profiles=[r for r in read(paper/'source_data'/'validation_profile_scores.csv')
      if (r['season'],r['seed']) in BLOCKS and r['candidate'] in ('raw','point')]
    table4=[]
    for method,name in METHODS.items():
        rr=[r for r in profiles if r['method']==method]
        if len(rr)!=450:raise ValueError(f'Expected 450 fixed-profile daily records for {method}, found {len(rr)}')
        weights=[float(r['weight'])/(1 if r['season']=='summer' else 2) for r in rr]
        energy=sum(w*float(r['energy_score']) for w,r in zip(weights,rr))/sum(weights)
        table4.append(dict(method=name,energy_score=f'{energy:.5f}'))
    write(out/'Table4.csv',table4)
    table5=[]
    for v in ('temperature','salinity','rho'):
        raw=only(scores,method='selected',variable=v,candidate='raw')
        cal=only(scores,method='selected',variable=v,candidate='marginal')
        table5.append(dict(variable=v,crps_raw=f"{float(raw['crps']):.5f}",crps_cal=f"{float(cal['crps']):.5f}",
         coverage_raw_percent=f"{100*float(raw['coverage90']):.2f}",coverage_cal_percent=f"{100*float(cal['coverage90']):.2f}",
         ssr_raw=f"{float(raw['spread_skill_ratio']):.3f}",ssr_cal=f"{float(cal['spread_skill_ratio']):.3f}",
         source='archived_common_support_not_revised_depth_summary'))
    write(out/'Table5.csv',table5)
    availability=paper/'source_data'/'availability_20261116'
    quick=read(availability/'quick_daily.csv');control=read(availability/'deterministic_daily_60.csv')
    table6=[]
    for scenario in ('nominal','columns50','outage15'):
        for v in ('temperature','salinity'):
            rr=[r for r in quick if r['scenario']==scenario and r['method']=='guided_raw' and r['variable']==v]
            bb=[r for r in control if r['variable']==v]
            assert len(rr)==len(bb)==120
            lookup={(r['season'],r['lead']):r for r in bb}
            for r in rr:
                other=lookup[r['season'],r['lead']]
                assert all(r[k]==other[k] for k in ('date','count','weight','domain','depth_index'))
            x,y=rmse(rr),rmse(bb)
            table6.append(dict(scenario=scenario,variable=v,argo_rmse=f'{x:.5f}',deterministic_rmse=f'{y:.5f}',
                              reduction_percent=f'{100*(1-x/y):.2f}'))
    write(out/'Table6.csv',table6)
    checks=dict(revised_sa_temperature_rmse=revised['selected','temperature'],revised_sa_salinity_rmse=revised['selected','salinity'],
      revised_sa_temperature_gain_percent=100*(1-revised['selected','temperature']/revised['deterministic','temperature']),
      revised_sa_salinity_gain_percent=100*(1-revised['selected','salinity']/revised['deterministic','salinity']),
      daily_records_reconciled_to_revised_summary=False,table2_sources_mixed_and_explicit=True)
    (out/'reconstruction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({'tables':5,'output':str(out),**checks},indent=2))


if __name__=='__main__':main()
