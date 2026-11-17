"""Four-season evaluation of fine-tuned diffusion versus an independent baseline."""
from __future__ import annotations
import argparse,csv,json,math,subprocess,sys,hashlib
from pathlib import Path
from datetime import date,timedelta
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
MODES=('deterministic','unguided','guided','paired_guided')
SEASONS=('winter','spring','summer','autumn')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def pooled(rows):
    rows=[r for r in rows if float(r['weight'])>0]
    if not rows:raise ValueError('No valid verification observations')
    total=sum(float(r['weight']) for r in rows)
    rmse=math.sqrt(sum(float(r['weight'])*float(r['rmse'])**2 for r in rows)/total)
    crps=sum(float(r['weight'])*float(r['crps']) for r in rows)/total
    return dict(rmse=rmse,crps=crps,weight=total)


def read_scores(path):
    with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if {int(r['lead']) for r in rows}!=set(range(1,91)):
        raise ValueError(f'Incomplete 90-day evaluation: {path}')
    keys=[(r['date'],r['variable'],r['depth_index']) for r in rows]
    if len(keys)!=len(set(keys)) or len(keys)!=90*32:raise ValueError(f'Incomplete/duplicate depth scores: {path}')
    return [r for r in rows if r['depth_index']=='-1']


def evaluate_gate(base,min_improvement=.005):
    result={'minimum_improvement':min_improvement,'methods':{}}
    data={}
    fingerprints=None;scale=None
    for mode in MODES:
        for season in SEASONS:
            recipe=json.loads((base/mode/season/'recipe.json').read_text(encoding='utf-8'))
            if fingerprints is None:fingerprints=recipe['fingerprints'];scale=recipe['scale']
            elif recipe['fingerprints']!=fingerprints or recipe['scale']!=scale:
                raise ValueError('Models, statistics, code or scales changed between experiments; mixed results cannot pass')
            for domain,name in (('glorys','paper_grid_metrics.csv'),('holdout','paper_holdout_metrics.csv')):
                data[mode,season,domain]=read_scores(base/mode/season/name)
    for mode in MODES[1:]:
        gates=[];details={}
        for domain in ('glorys','holdout'):
            details[domain]={}
            for variable in ('temperature','salinity'):
                candidate=[r for s in SEASONS for r in data[mode,s,domain] if r['variable']==variable]
                reference=[r for s in SEASONS for r in data['deterministic',s,domain] if r['variable']==variable]
                keyed={(r['date'],r['lead']):r for r in reference}
                if any(int(r['count'])!=int(keyed[r['date'],r['lead']]['count']) for r in candidate):
                    raise ValueError('Candidate and baseline scored on different sample counts')
                c,b=pooled(candidate),pooled(reference);ratio=c['rmse']/b['rmse']
                cf=pooled([r for r in candidate if int(r['lead'])==90]);bf=pooled([r for r in reference if int(r['lead'])==90])
                seasonal=[]
                for s in SEASONS:
                    cs=pooled([r for r in data[mode,s,domain] if r['variable']==variable])
                    bs=pooled([r for r in data['deterministic',s,domain] if r['variable']==variable])
                    seasonal.append(cs['rmse']/bs['rmse'])
                if domain=='glorys':
                    passed=ratio<1-min_improvement and cf['rmse']<bf['rmse'] and max(seasonal)<=1.001
                else:
                    # Holdout salt has unresolved tails. Enforce non-degradation,
                    # report its values, do not tune by deleting large errors.
                    passed=ratio<=1.001 and max(seasonal)<=1.001
                gates.append(passed)
                details[domain][variable]=dict(candidate=c,baseline=b,rmse_gain_percent=100*(1-ratio),
                    final_day_candidate=cf,final_day_baseline=bf,seasonal_ratios=dict(zip(SEASONS,seasonal)),passed=passed)
        result['methods'][mode]=dict(accepted=all(gates),details=details)
    result['any_accepted']=any(v['accepted'] for v in result['methods'].values())
    result['fingerprints']=fingerprints;result['residual_scale']=scale
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('cycle','gate'),required=True)
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--checkpoint',default='runs/residual_rollout_v1/checkpoint_best.pt')
    p.add_argument('--output-root',default='runs/residual_rollout_v1_validation90')
    p.add_argument('--scale',type=float,default=1.0)
    p.add_argument('--min-improvement',type=float,default=.005)
    a=p.parse_args();base=ROOT/a.output_root
    if a.phase=='gate':
        result=evaluate_gate(base,a.min_improvement)
        (base/'gate.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)
        return
    if not 0<=a.task<16:p.error('Task must lie in 0..15')
    if not 0<a.scale<=2:p.error('Scale must lie in (0,2]')
    cfg=json.loads((ROOT/'configs/paper_protocol.json').read_text(encoding='utf-8'))
    fingerprints={key:sha(path) for key,path in {
        'residual_checkpoint':ROOT/a.checkpoint,'deterministic_checkpoint':ROOT/cfg['deterministic_checkpoint'],
        'residual_stats':ROOT/cfg['residual_stats'],'state_stats':ROOT/cfg['state_stats'],
        'config':ROOT/cfg['config'],'cycle_code':ROOT/'argo_guided_cycle.py',
        'metrics_code':ROOT/'paper_metrics.py'}.items()}
    mode=MODES[a.task//4];season=SEASONS[a.task%4]
    start=date(2019,(1,4,7,10)[a.task%4],1)
    output=base/mode/season
    if any((output/n).exists() for n in ('run_summary.csv','paper_grid_metrics.csv','paper_holdout_metrics.csv')):
        raise FileExistsError(f'Existing run will not be overwritten: {output}')
    split=Path(cfg['argo_split_root'])
    for lead in range(1,91):
        target=start+timedelta(days=lead)
        for kind in ('guidance','holdout'):
            path=split/kind/str(target.year)/f'{target.isoformat()}.npz'
            if not path.is_file():raise FileNotFoundError(path)
    cmd=[sys.executable,ROOT/'argo_guided_cycle.py','--config',cfg['config'],
        '--deterministic-checkpoint',cfg['deterministic_checkpoint'],'--residual-checkpoint',a.checkpoint,
        '--residual-stats',cfg['residual_stats'],'--start',start.isoformat(),'--days','90',
        '--members','2' if mode=='deterministic' else '50','--sampling-steps','50','--sampling-eta','0',
        '--residual-scale',str(a.scale),'--seed','20261006',
        '--paper-holdout-root',split/'holdout','--save-daily','analysis','--snapshot-every','30',
        '--ensemble-storage-dtype','float32','--output-dir',output]
    if mode=='deterministic':cmd+=['--deterministic-only']
    else:cmd+=['--member-mean-cycle']
    if mode in ('guided','paired_guided'):
        cmd+=['--argo-root',split/'guidance','--require-argo','--individual-member-guidance',
              '--temperature-guidance',str(cfg['temperature_guidance']),
              '--salinity-guidance',str(cfg['salinity_guidance']),'--guidance-start',str(cfg['guidance_start'])]
    if mode=='paired_guided':cmd+=['--paired-guidance-center']
    output.mkdir(parents=True,exist_ok=True)
    (output/'recipe.json').write_text(json.dumps({'mode':mode,'start':start.isoformat(),'scale':a.scale,
        'fingerprints':fingerprints,
        'checkpoint':a.checkpoint,'command':list(map(str,cmd)),
        'spread_calibration':'none: mean-accuracy experiment; old-model factors are not reusable'},indent=2))
    print(' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)


if __name__=='__main__':main()
