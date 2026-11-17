"""Local numerical and real-model verification of the desktop OE extension."""
import csv,gzip,json,sys,warnings
from datetime import date,timedelta
from pathlib import Path
from unittest.mock import patch
import numpy as np
import netCDF4
import gsw
import pytest
import torch
import yaml
from oe_physics import derive_fields,member_mean,event_scores,pycnocline_depth,region_masks
from prepare_oe_argo import read_source,interpolate,validate_declaration
from oe_experiments import tasks,build_command,pool,completed,sha

def declaration():
    return dict(human_confirmed=True,vertical_coordinate='pressure_dbar',temperature_kind='in_situ',salinity_kind='practical_salinity',quality_policy='export_disclosed',quality_provenance='Synthetic numerical test',allow_filename_date=False)

def make_profiles(path,native=False):
    with netCDF4.Dataset(path,'w') as ds:
        ds.createDimension('N_PROF',2);ds.createDimension('N_LEVELS',3)
        for name,value in [('LATITUDE',[20,30]),('LONGITUDE',[115,135])]:ds.createVariable(name,'f8',('N_PROF',))[:]=value
        tv=ds.createVariable('JULD','f8',('N_PROF',));tv.units='days since 2021-01-01';tv[:]=[0,1]
        for name,value,units in [('PRES',[10,100,500],'dbar'),('TEMP',[20,15,8],'degree_Celsius'),('PSAL',[35,35.1,35.2],'1')]:
            v=ds.createVariable(name,'f8',('N_PROF','N_LEVELS'));v.units=units;v[:]=[value,value]
            if native:
                q=ds.createVariable(name+'_QC','S1',('N_PROF','N_LEVELS'));q[:]=np.array([list('111'),list('111')],dtype='S1')
                ds.createVariable(name+'_ADJUSTED','f8',('N_PROF','N_LEVELS'))[:]=np.array([value,value])+(0 if name=='PRES' else .5)
                q=ds.createVariable(name+'_ADJUSTED_QC','S1',('N_PROF','N_LEVELS'));q[:]=np.array([list('111'),list('141')],dtype='S1')
        if native:
            ds.createVariable('DATA_MODE','S1',('N_PROF',))[:]=np.array(list('RD'),dtype='S1')
            for name in ['POSITION_QC','JULD_QC']:ds.createVariable(name,'S1',('N_PROF',))[:]=np.array(list('11'),dtype='S1')

def test_teos_order_units_and_missing_layers():
    depth=np.array([500,100,10.]);lat=np.array([20.,30.]);lon=np.array([115.,135.])
    t=np.broadcast_to(np.array([8,15,20.])[None,:,None],(2,3,2)).copy();s=np.broadcast_to(np.array([35.2,35.1,35.])[None,:,None],t.shape).copy()
    f,d,o,g,info=derive_fields(t,s,depth,lat,lon)
    p=gsw.p_from_z(-d[:,None],lat[None]);sa=gsw.SA_from_SP(s[0,o],p,lon,lat);ct=gsw.CT_from_pt(sa,t[0,o])
    np.testing.assert_allclose(f['rho'][0],gsw.rho(sa,ct,p),rtol=1e-13)
    np.testing.assert_allclose(f['n2'][0],gsw.Nsquared(sa,ct,p,lat=lat,axis=0)[0],rtol=1e-13)
    np.testing.assert_allclose(f['buoyancy_per_volume'][0],f['rho'][0]*g)
    assert (f['n2']>0).all() and info['valid_values']==12
    t[:,1,0]=np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning);bad,*_=derive_fields(t,s,depth,lat,lon)
        assert np.isnan(bad['n2'][:,:,0]).all() and np.isnan(member_mean(bad['rho'])[1,0])

def test_memberwise_density_events_pycnocline_and_dateline():
    t=np.array([[[5.],[4.]],[[25.],[24.]]]);s=np.full_like(t,35.)
    f,*_=derive_fields(t,s,[10,100],[20],[115]);c,*_=derive_fields(t.mean(axis=0)[None],s.mean(axis=0)[None],[10,100],[20],[115])
    assert abs(member_mean(f['rho'])[0,0]-c['rho'][0,0,0])>.1
    e=np.array([[False,True],[True,True]]);r=event_scores(e,[False,True],np.ones(2,bool),[2.,1.])
    assert r['brier']==pytest.approx(1/6) and r['bin_5_weight']==2 and r['bin_9_weight']==1
    assert np.isnan(event_scores(e,[False,True],np.zeros(2,bool))['brier'])
    p=pycnocline_depth(np.array([[[1.],[3.]],[[np.nan],[3.]]]),np.array([10.,100.,500.]),[10,500])
    assert p[0,0]==300 and np.isnan(p[1,0])
    masks=region_masks(np.array([0.,0.,60]),np.array([355.,5.,180.]),{'regions':[dict(name='cross',latitude=[-5,5],longitude=[350,10])]})
    assert masks['cross'].tolist()==[True,True,False]

def test_raw_mapping_units_and_profile_dates(tmp_path):
    path=tmp_path/'daily_19900101.nc';make_profiles(path);a=read_source(path,declaration())
    assert a['dates'].tolist()==['2021-01-01','2021-01-02']
    p=np.array([[10,100,500],[10,100,500]]);lat=np.array([[20],[30]]);lon=np.array([[115],[135]])
    np.testing.assert_allclose(a['depth'],-gsw.z_from_p(p,lat))
    sa=gsw.SA_from_SP(np.array([[35,35.1,35.2],[35,35.1,35.2]]),p,lon,lat)
    np.testing.assert_allclose(a['temperature'],gsw.pt0_from_t(sa,np.array([[20,15,8],[20,15,8]]),p))
    assert not a['qc_present']['temperature']
    c=declaration();c['human_confirmed']=False
    with pytest.raises(ValueError,match='unconfirmed'):validate_declaration(c)

def test_native_adjusted_matching_qc(tmp_path):
    path=tmp_path/'native.nc';make_profiles(path,True);c=declaration();c['quality_policy']='native_adjusted';c['temperature_kind']='potential_0dbar'
    a=read_source(path,c)
    assert a['temperature'][0,1]==15 and a['temperature'][1,1]==15.5
    assert a['valid_temperature'][0,1] and not a['valid_temperature'][1,1]
    with netCDF4.Dataset(path,'a') as f:f.variables['DATA_MODE'][0]=b'X'
    assert not read_source(path,c)['position'][0]

def test_interpolation_gaps_and_no_extrapolation():
    r=interpolate(np.array([10.,100.,500.]),np.array([20.,15.,8.]),np.ones(3,bool),np.array([0.,10.,50.,100.,200.,500.,600.]),200)
    assert np.isnan(r[[0,4,6]]).all();np.testing.assert_allclose(r[[1,2,3,5]],[20,17.777777,15,8],rtol=1e-6)

def test_source_metadata_conflicts_and_native_missing_qc(tmp_path):
    path=tmp_path/'raw.nc';make_profiles(path);c=declaration();c['quality_policy']='native_adjusted'
    with pytest.raises(ValueError,match='POSITION_QC'):read_source(path,c)
    c=declaration();c['vertical_coordinate']='depth_m'
    with pytest.raises(ValueError,match='pressure metadata'):read_source(path,c)

def test_full_daily_mapping_serialization_and_spatial_holdout(tmp_path):
    from prepare_oe_argo import main
    from prepare_paper_argo import spatial_partition
    raw=tmp_path/'raw';grid=tmp_path/'grid';raw.mkdir();grid.mkdir();source=raw/'19900101.nc';make_profiles(source)
    lat=np.linspace(-89.25,89.25,120).astype('f4');lon=(np.arange(240)*1.5).astype('f4');depth=np.array([100,50,10.],dtype='f4')
    guide,held=spatial_partition(lat,lon,9.,.3,314159,1)
    candidates=(lat[:,None]>10)&(lat[:,None]<30)&np.ones((120,240),bool)
    gi,gj=np.argwhere(guide&candidates)[0];hi,hj=np.argwhere(held&candidates)[0]
    with netCDF4.Dataset(source,'a') as f:f.variables['LATITUDE'][:]=lat[[gi,hi]];f.variables['LONGITUDE'][:]=lon[[gj,hj]]
    with netCDF4.Dataset(grid/'grid.nc','w') as f:
        for name,value in [('lat',lat),('lon',lon),('depth',depth)]:f.createDimension(name,len(value));f.createVariable(name,'f4',(name,))[:]=value
    config=tmp_path/'declaration.json';config.write_text(json.dumps(declaration()));out=tmp_path/'mapped'
    args=['mapper','--raw-root',str(raw),'--glorys-root',str(grid),'--output-dir',str(out),'--declaration',str(config),'--years','2021']
    with patch.object(sys,'argv',args):main()
    manifest=json.loads((out/'split_manifest.json').read_text())
    assert manifest['source_counts']['profiles']==2 and len(manifest['missing_source_days'])==363
    with np.load(out/'guidance/2021/2021-01-01.npz') as f:
        assert f['grid_i'].tolist()==[int(gi)] and f['grid_j'].tolist()==[int(gj)]
        assert str(f['temperature_kind'])=='potential_0dbar'
    with np.load(out/'holdout/2021/2021-01-02.npz') as f:
        assert f['grid_i'].tolist()==[int(hi)] and f['grid_j'].tolist()==[int(hj)]
        assert f['temperature'].shape==(1,3) and str(f['vertical_coordinate'])=='depth_m'
    with np.load(out/'holdout/2021/2021-01-03.npz') as f:assert f['temperature'].shape==(0,3)

def test_report_aggregates_empty_regions_reliability_and_matched_windows(tmp_path):
    import oe_experiments as workflow
    from oe_physics import append_gzip,METRIC_FIELDS,EVENT_FIELDS
    c,p=workflow.load_settings();c=dict(c,output_root=str(tmp_path),days=2)
    ts=[dict(phase='consistent2021',year=2021,month=1,season='winter',seed=1,method=m) for m in ['deterministic','fine_paired']]
    signature={'test_signature':1};(tmp_path/'frozen_consistent2021.json').write_text(json.dumps(signature))
    for t in ts:
        folder=workflow.destination(t,c);out=folder/'attempt_001';out.mkdir(parents=True)
        metric=[];event=[];joint=[]
        for lead in [1,2]:
            error=1 if t['method']=='deterministic' else .5
            for region in ['global','empty']:
                score=dict(count=2 if region=='global' else 0,weight=2 if region=='global' else 0,rmse=error,bias=0,mae=error,rms_spread=.4,spread_skill_ratio=.4/error,crps=.2,range_coverage=1,coverage90=.9)
                metric.append(dict(date=f'2021-01-0{lead}',lead=lead,domain='glorys',region=region,variable='rho',unit='kg m-3',estimator='ensemble_mean',depth_index=-1,depth_m='all',members=3,**score))
            score=event_scores(np.array([[False,True],[True,True]]),[False,True],np.ones(2,bool))
            event.append(dict(date=f'2021-01-0{lead}',lead=lead,domain='glorys',region='global',event='negative_n2',reference_density_kgm3='',tolerance_N_m3='',members=2,**score))
            joint.append(dict(date=f'2021-01-0{lead}',lead=lead,weight=0,temperature_salinity_correlation=np.nan,temperature_salinity_covariance=np.nan,adjacent_density_weight=0,adjacent_density_correlation=np.nan))
        append_gzip(out/'oe_physics_metrics.csv.gz',metric,METRIC_FIELDS);append_gzip(out/'oe_event_metrics.csv.gz',event,EVENT_FIELDS);append_gzip(out/'oe_joint_structure.csv.gz',joint,list(joint[0]))
        marker=dict(task=t,signature=signature,attempt=out.name,outputs={x.name:sha(x) for x in out.iterdir()})
        (folder/'completed.json').write_text(json.dumps(marker))
    with patch.object(workflow,'tasks',return_value=ts),patch.object(workflow,'verify_parent'),patch.object(workflow,'artifacts',return_value=signature):workflow.report('consistent2021',c)
    result=json.loads((tmp_path/'summary_consistent2021.json').read_text())
    assert result['complete'] and result['paired_vs_deterministic'][0]['rmse_gain_percent']==50
    assert result['lead_windows'][-1]['last_lead']==2
    assert result['joint_structure'][0]['temperature_salinity_correlation'] is None
    assert result['event_scores'][0]['reliability'][5]['probability']==.5

def test_plan_commands_and_hash_protection(tmp_path):
    from oe_experiments import load_settings,destination
    c,p=load_settings();ts=tasks('legacy2021',c);assert len(ts)==48 and len(tasks('calibrate',c))==8
    assert [ts[i]['method'] for i in range(0,48,8)]==c['methods']
    for i in [0,8,16,24,32,40]:
        cmd=build_command(ts[i],c,p,tmp_path)
        assert ('--argo-root' in cmd)==(i in [8,24,32,40]);assert ('--deterministic-only' in cmd)==(i in [0,8]);assert ('--spread-calibration' in cmd)==(i==40)
    c=dict(c,output_root=str(tmp_path));folder=destination(ts[0],c);out=folder/'attempt_001';out.mkdir(parents=True);(out/'result.csv').write_text('original')
    marker=dict(task=ts[0],attempt=out.name,outputs={'result.csv':sha(out/'result.csv')});(folder/'completed.json').write_text(json.dumps(marker))
    assert completed(ts[0],c)['attempt']=='attempt_001'
    (out/'result.csv').write_text('changed')
    with pytest.raises(RuntimeError,match='changed'):completed(ts[0],c)
    pooled=pool([dict(weight=1,rmse=2,bias=0,mae=1,crps=1,coverage90=.9),dict(weight=3,rmse=4,bias=0,mae=1,crps=1,coverage90=.9)])
    assert pooled['rmse']==pytest.approx(np.sqrt(13))

def test_real_unet_cycle_replay(tmp_path):
    from diffargo.model import build_model
    from argo_guided_cycle import main as cycle
    from oe_cycle import main as observed
    torch.set_num_threads(2);data=tmp_path/'data';(data/'T').mkdir(parents=True);(data/'S').mkdir()
    days=[date(2019,1,1)+timedelta(days=i) for i in range(5)];lat=np.linspace(-60,60,8).astype('f4');lon=(np.arange(8)*45).astype('f4');depth=np.array([100,50,10.],dtype='f4');arrays={}
    for kind,var in [('T','t'),('S','s')]:
        values=np.stack([np.broadcast_to(((20-.02*depth) if kind=='T' else (35+.001*depth))[:,None,None]+.005*k,(3,8,8)).copy() for k in range(5)]).astype('f4');values[:,:,0,0]=np.nan;arrays[kind]=values
        with netCDF4.Dataset(data/kind/f'{kind}_2019.nc','w') as ds:
            for name,n in [('time',5),('depth',3),('lat',8),('lon',8)]:ds.createDimension(name,n)
            v=ds.createVariable('time','f8',('time',));v.units='days since 2019-01-01';v[:]=np.arange(5)
            for name,value in [('depth',depth),('lat',lat),('lon',lon)]:ds.createVariable(name,'f4',(name,))[:]=value
            ds.createVariable(var,'f4',('time','depth','lat','lon'))[:]=values
    mask=np.ones((3,8,8),np.float32);mask[:,0,0]=0;stats=tmp_path/'stats.npz';resstats=tmp_path/'resstats.npz'
    np.savez_compressed(stats,state_mean=np.array([18,19,20,35,35,35],np.float32),state_std=np.array([5,5,5,1,1,1],np.float32),increment_mean=np.zeros(6,np.float32),increment_std=np.full(6,.01,np.float32),static_mask=mask)
    np.savez_compressed(resstats,residual_mean=np.zeros(6,np.float32),residual_std=np.full(6,.01,np.float32))
    cfg=dict(seed=7,data=dict(root=str(data),temperature_subdir='T',salinity_subdir='S',temperature_pattern='T_{year}.nc',salinity_pattern='S_{year}.nc',temperature_variable='t',salinity_variable='s',depth_name='depth',latitude_name='lat',longitude_name='lon',time_name='time',train_years=[2019,2019],val_years=[2019,2019],test_years=[2019,2019],context_days=3,lead_days=1,stats_path=str(stats),file_cache_size=4),model=dict(base_channels=4,channel_multipliers=[1,2],time_embedding_dim=16,dropout=0.,attention_at_bottleneck=False),diffusion=dict(timesteps=16,beta_schedule='cosine'),train=dict(output_dir='unused',ema_decay=.99))
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg));model=build_model(cfg,6,26)
    with torch.no_grad():model.output_conv.conv.bias.zero_()
    det=tmp_path/'det.pt';res=tmp_path/'res.pt';torch.save({'model':model.state_dict()},det);torch.save({'model':model.state_dict(),'objective':'residual_diffusion'},res)
    held=tmp_path/'holdout'/'2019';held.mkdir(parents=True)
    for k in [3,4]:
        meta={} if k==3 else dict(temperature_kind=np.asarray('potential_0dbar'),vertical_coordinate=np.asarray('depth_m'))
        np.savez_compressed(held/f'{days[k]}.npz',grid_i=np.array([3]),grid_j=np.array([3]),temperature=arrays['T'][k,:,3,3][None],salinity=arrays['S'][k,:,3,3][None],latitude=lat,longitude=lon,depth_m=depth,profile_latitude=np.array([lat[3]]),profile_longitude=np.array([lon[3]]),**meta)
    from oe_experiments import ROOT
    diagnostic=ROOT/'configs/oe_diagnostics.json'
    common=['cycle','--config',str(config),'--deterministic-checkpoint',str(det),'--residual-checkpoint',str(res),'--residual-stats',str(resstats),'--start','2019-01-03','--days','2','--members','3','--sampling-steps','4','--sampling-eta','0','--residual-scale','1','--seed','7','--save-daily','compact','--paper-holdout-root',str(held.parent),'--individual-member-guidance','--member-mean-cycle','--paired-guidance-center']
    for name,entry,extra in [('original',cycle,[]),('observer',observed,['--oe-config',str(diagnostic)])]:
        with patch.object(sys,'argv',common+extra+['--output-dir',str(tmp_path/name)]),patch('torch.cuda.is_available',return_value=False):entry()
    for name in ['paper_grid_metrics.csv','paper_holdout_metrics.csv','paper_region_metrics.csv']:assert (tmp_path/'original'/name).read_bytes()==(tmp_path/'observer'/name).read_bytes()
    for day in days[3:]:
        with np.load(tmp_path/'original'/f'{day}.npz') as a,np.load(tmp_path/'observer'/f'{day}.npz') as b:np.testing.assert_array_equal(a['analysis'],b['analysis'])
    status=json.loads((tmp_path/'observer'/'oe_diagnostic_status.json').read_text());assert status['completed_leads']==[1,2];assert status['holdout_physics_skipped_unconfirmed']==['2019-01-04']
    with gzip.open(tmp_path/'observer'/'oe_physics_metrics.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    assert {r['domain'] for r in rows}=={'glorys','holdout'};assert {'rho','n2','pycnocline_depth'}.issubset({r['variable'] for r in rows})
