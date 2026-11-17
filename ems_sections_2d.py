"""Export real longitude/latitude-depth sections without altering the forecast engine.

Replay reads a completed case's recipe, retains its sampling settings and replaces
expensive verification with a read-only section writer. Outputs use a separate
directory. No training, case-ledger update or overwrite of old results occurs.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=('deterministic','selected','guided_raw')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def option(argv,key):
    if argv.count(key)!=1:raise ValueError('Expected one '+key)
    return argv[argv.index(key)+1]

def set_option(argv,key,value):
    argv[argv.index(key)+1]=str(value)

def remap_project(value):
    text=str(value).replace('\\','/')
    if '/diffArgo/' in text:return str(ROOT/text.split('/diffArgo/',1)[1])
    return value

def extract_sections(pred,truth,valid,latitude,longitude,depth,zonal_lat=17.25,
                     longitude_range=(110.,180.),meridional_lon=135.,latitude_range=(-10.,50.)):
    """Select native grid rows/columns. Do not interpolate across land or sites."""
    lat=np.asarray(latitude,float);lon=np.mod(np.asarray(longitude,float),360.)
    d=np.abs(np.asarray(depth,float));order=np.argsort(d);levels=len(d)
    if np.any(np.diff(d[order])<=0):raise ValueError('Depths must be unique')
    shape=(2*levels,len(lat),len(lon))
    if np.shape(pred)!=shape or np.shape(truth)!=shape or np.shape(valid)!=shape:
        raise ValueError('Expected T-then-S channels on the full lat/lon grid')
    valid=np.asarray(valid,bool)&np.isfinite(truth)
    if not np.isfinite(np.asarray(pred)[valid]).all():
        raise FloatingPointError('Nonfinite prediction on reference-defined support')
    i=int(np.argmin(abs(lat-zonal_lat)));j=int(np.argmin(abs(lon-meridional_lon%360)))
    # Nearest native line is reported explicitly; reject distant out-of-domain requests.
    if len(lat)<2 or len(lon)<2:raise ValueError('A section requires a spatial grid')
    if abs(lat[i]-zonal_lat)>np.median(np.diff(np.sort(lat)))/2+1e-5:
        raise ValueError('Requested zonal line lies outside the grid')
    if abs(lon[j]-meridional_lon%360)>np.median(np.diff(np.sort(lon)))/2+1e-5:
        raise ValueError('Requested meridional line lies outside the grid')
    xx=np.flatnonzero((lon>=longitude_range[0])&(lon<=longitude_range[1]))
    yy=np.flatnonzero((lat>=latitude_range[0])&(lat<=latitude_range[1]))
    xx=xx[np.argsort(lon[xx])];yy=yy[np.argsort(lat[yy])]
    if len(xx)<3 or len(yy)<3:raise ValueError('Each line must contain at least three native columns')
    clean_truth=np.where(valid,truth,np.nan).reshape(2,levels,len(lat),len(lon))[:,order]
    clean_pred=np.where(valid,pred,np.nan).reshape(2,levels,len(lat),len(lon))[:,order]
    support=valid.reshape(2,levels,len(lat),len(lon))[:,order]
    return dict(depth_m=d[order].astype(np.float32),
        zonal_longitude=lon[xx].astype(np.float32),zonal_latitude=np.float32(lat[i]),
        meridional_latitude=lat[yy].astype(np.float32),meridional_longitude=np.float32(lon[j]),
        zonal_truth=clean_truth[:,:,i,:][:,:,xx].astype(np.float32),
        zonal_prediction=clean_pred[:,:,i,:][:,:,xx].astype(np.float32),
        zonal_valid=support[:,:,i,:][:,:,xx],
        meridional_truth=clean_truth[:,:,:,j][:,:,yy].astype(np.float32),
        meridional_prediction=clean_pred[:,:,:,j][:,:,yy].astype(np.float32),
        meridional_valid=support[:,:,:,j][:,:,yy])

def check_original(case,dataset,target,lead,analysis,ensemble):
    """Check global centre scores and all available archived local anchors."""
    truth,valid=dataset.archive.read_state(target)
    support=valid&np.isfinite(truth)&np.isfinite(analysis)&np.isfinite(ensemble).all(axis=0)
    records=[]
    with (case/'paper_grid_metrics.csv').open(encoding='utf-8',newline='') as f:
        original=[r for r in csv.DictReader(f) if int(r['lead'])==lead and
                  r['domain']=='glorys' and r['depth_index']=='-1']
    for v,var in enumerate(('temperature','salinity')):
        rr=[r for r in original if r['variable']==var]
        if len(rr)!=1:raise RuntimeError('Missing archived global score for '+var)
        sl=slice(v*dataset.levels,(v+1)*dataset.levels);mask=support[sl]
        weight=np.broadcast_to(dataset.latitude_weights,mask.shape)[mask].astype(float)
        delta=(analysis[sl]-truth[sl])[mask]
        rmse=float(np.sqrt(np.dot(weight,delta**2)/weight.sum()))
        reference=float(rr[0]['rmse'])
        if int(mask.sum())!=int(rr[0]['count']) or not np.isclose(rmse,reference,rtol=2e-4,atol=2e-5):
            raise RuntimeError(f'Replay differs from archived {var}: RMSE {rmse} vs {reference}. No manuscript section exported.')
        records.append(dict(variable=var,rmse=rmse,archived_rmse=reference,count=int(mask.sum())))
    anchor=case/f'joint_profiles_day{lead:03d}.npz'
    max_error=None
    if anchor.exists():
        with np.load(anchor,allow_pickle=False) as saved:
            ii=[int(np.argmin(abs(dataset.lat-x))) for x in saved['latitude']]
            jj=[int(np.argmin(abs(np.mod(dataset.lon,360)-x%360))) for x in saved['longitude']]
            np.testing.assert_allclose(np.abs(dataset.archive.coordinates()['depth']),saved['depth_m'],atol=1e-5,rtol=0)
            np.testing.assert_allclose(dataset.lat[ii],saved['latitude'],atol=1e-5,rtol=0)
            np.testing.assert_allclose(np.mod(dataset.lon[jj],360),np.mod(saved['longitude'],360),atol=1e-5,rtol=0)
            max_error=float(np.max(abs(analysis[:,ii,jj]-saved['center'])))
            if not np.isfinite(max_error) or max_error>2e-4:
                raise RuntimeError(f'Replay fixed-site mismatch {max_error}; no section exported')
    return truth,valid,dict(global_scores=records,anchor_max_abs_error=max_error,
        archive_profile_sha256=sha(anchor) if anchor.exists() else None,passed=True)

def case_path(args):
    return ROOT/args.case_root/args.method/'m50_k50'/args.season/str(args.seed)/args.attempt

def replay(args):
    case=case_path(args);recipe_path=case/'recipe.json'
    recipe=json.loads(recipe_path.read_text(encoding='utf-8'))
    task=recipe['task']
    if task['method']!=args.method or task['season']!=args.season or task['seed']!=args.seed:
        raise ValueError('Recipe does not match requested case')
    if args.end_lead>task['days'] or min(args.leads)<1 or max(args.leads)>args.end_lead:
        raise ValueError('Export leads must lie within the replay and original cycle')
    original=recipe['command']
    if original[2]!='ems_observer.py' or '--calibration-statistics-output' in original:
        raise ValueError('Expected an archived validation recipe')
    argv=[remap_project(x) for x in original[3:]]
    index=argv.index('--ems-observer');del argv[index:index+2]
    out_base=ROOT/args.output_root/args.method
    out_base.mkdir(parents=True,exist_ok=True)
    number=1
    while (out_base/f'attempt_{number:03d}').exists():number+=1
    out=out_base/f'attempt_{number:03d}';out.mkdir()
    set_option(argv,'--output-dir',out)
    set_option(argv,'--days',args.end_lead)
    set_option(argv,'--save-daily','none')
    for key in ('--config','--deterministic-checkpoint','--residual-checkpoint','--residual-stats','--ems-observation-options'):
        if not Path(option(argv,key)).is_file():raise FileNotFoundError(option(argv,key))
    for key in ('--argo-root','--stability-options'):
        if key in argv and not Path(option(argv,key)).exists():raise FileNotFoundError(option(argv,key))
    manifest=dict(source_recipe=str(recipe_path),source_recipe_sha256=sha(recipe_path),task=task,
        replay_end_lead=args.end_lead,export_leads=args.leads,command=['ems_cycle.py']+argv,
        numerical_changes='None except optional shortening of the end horizon; no members/steps/guidance changes.',
        verification='Full diagnostics omitted; exported days checked against archived global RMSE and available 3-site centres.',
        selection='Fixed zonal and meridional lines, chosen for geographic coverage, not prediction error.',
        created_utc=datetime.now(timezone.utc).isoformat(),human_verified=False,
        inputs={key:dict(path=option(argv,key),sha256=sha(option(argv,key))) for key in
                ('--config','--deterministic-checkpoint','--residual-checkpoint','--residual-stats')})
    dump(out/'section_recipe.json',manifest)
    import paper_metrics
    from ems_cycle import main as cycle
    original_evaluate=paper_metrics.evaluate_day;old_argv=sys.argv
    written=[]
    def capture(dataset,target,lead,analysis,ensemble,*unused,**kwargs):
        if lead not in args.leads:return
        truth,valid,checks=check_original(case,dataset,target,lead,analysis,ensemble)
        arrays=extract_sections(analysis,truth,valid,dataset.lat,dataset.lon,
            dataset.archive.coordinates()['depth'],args.zonal_lat,tuple(args.lon_range),
            args.meridional_lon,tuple(args.lat_range))
        arrays.update(method=np.asarray(args.method),date=np.asarray(str(target)),lead=np.int32(lead),
            source_recipe_sha256=np.asarray(sha(recipe_path)),replay_verified=np.bool_(True))
        filename=out/f'lead{lead:03d}_sections.npz'
        with filename.open('xb') as f:np.savez_compressed(f,**arrays)
        record=dict(path=filename.name,sha256=sha(filename),bytes=filename.stat().st_size,checks=checks)
        dump(out/f'lead{lead:03d}_checks.json',record);written.append(record)
        print(f'EXPORTED real native-grid sections: {filename}',flush=True)
    paper_metrics.evaluate_day=capture
    sys.argv=['ems_cycle.py']+argv
    try:cycle()
    finally:paper_metrics.evaluate_day=original_evaluate;sys.argv=old_argv
    if len(written)!=len(set(args.leads)):raise RuntimeError('Not all requested leads exported')
    dump(out/'section_complete.json',dict(files=written,complete=True,human_verified=False))
    print('SECTIONS COMPLETE:',out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--case-root',default='runs/ems_v1_legacy/cases/validate/nominal')
    p.add_argument('--season',choices=('winter','spring','summer','autumn'),default='winter')
    p.add_argument('--seed',type=int,default=20261006);p.add_argument('--attempt',default='attempt_001')
    p.add_argument('--end-lead',type=int,default=90);p.add_argument('--leads',nargs='+',type=int,default=[30,90])
    p.add_argument('--output-root',default='runs/ems_sections_2d_v1')
    p.add_argument('--zonal-lat',type=float,default=17.25)
    p.add_argument('--lon-range',nargs=2,type=float,default=[110.,180.])
    p.add_argument('--meridional-lon',type=float,default=135.)
    p.add_argument('--lat-range',nargs=2,type=float,default=[-10.,50.])
    a=p.parse_args()
    if len(set(a.leads))!=len(a.leads):p.error('Duplicate leads')
    replay(a)

if __name__=='__main__':main()
