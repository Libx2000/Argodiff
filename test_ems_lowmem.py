import copy,csv,gzip,json
from datetime import date
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from oe_joint_diagnostics import JointObserver
from ems_chunked_diagnostics import ChunkedJointObserver


def configuration(tmp_path,levels):
    p=tmp_path/'factors.npz';np.savez_compressed(p,inflation_factor=np.full(2*levels,2.))
    return dict(reference_calibration=str(p),regional_feather_degrees=2.,pycnocline_depth_m=[10.,500.],
        negative_n2_thresholds_s2=[0.,-1e-6,-1e-5,-1e-4],regions=[])


def fixture(members=5,levels=4,points=23):
    rng=np.random.default_rng(16);depth=np.linspace(500,1,levels)
    lat=np.linspace(-60,60,points);lon=np.linspace(5,355,points)
    center=np.concatenate([20-.02*depth,35+.0001*depth])[:,None]+np.zeros((2*levels,points))
    truth=center+rng.normal(0,.02,center.shape)
    en=(center[None]+rng.normal(0,.1,(members,2*levels,points))).astype('f4')
    en[:,:,1]=np.nan;center[:,1]=np.nan;truth[:,1]=np.nan
    truth[0,3]=np.nan;en[:,2,4]=np.nan
    # Channels deliberately reversed in depth, missing layers, and zero-width
    # channels exercise exact masks and no extrapolation/bridging.
    en[:,0,5]=center[0,5]
    memberships=dict(global_=np.ones(points,bool),empty=np.zeros(points,bool),box=lat>0)
    memberships['global']=memberships.pop('global_')
    return center,en,truth,depth,lat,lon,memberships,np.cos(np.deg2rad(lat))


def rows(path):
    with gzip.open(path,'rt',newline='') as f:return list(csv.DictReader(f))


def compare_rows(a,b,keys):
    def index(rs):return {tuple(r[k] for k in keys):r for r in rs}
    aa,bb=index(a),index(b)
    assert len(aa)==len(a) and len(bb)==len(b) and aa.keys()==bb.keys()
    maximum=0.
    for key,x in aa.items():
        y=bb[key];assert x.keys()==y.keys()
        for col in x:
            if col in keys or col=='support_digest':assert x[col]==y[col]
            elif x[col]=='':assert y[col]==''
            else:
                u,v=float(x[col]),float(y[col])
                if np.isnan(u):assert np.isnan(v)
                elif col=='count':assert u==v
                else:
                    assert v==pytest.approx(u,rel=3e-12,abs=5e-13),(key,col,u,v)
                    maximum=max(maximum,abs(u-v))
    return maximum


@pytest.mark.parametrize('members,levels,points,chunk',[(1,4,23,5),(5,4,23,1),(5,4,23,7),(50,15,31,8)])
def test_daily_and_pooled_match_whole_grid(tmp_path,members,levels,points,chunk):
    c=configuration(tmp_path,levels);inputs=fixture(members,levels,points)
    candidates=[dict(id='raw',kind='raw'),dict(id='marginal',kind='marginal'),
        dict(id='joint',kind='joint',length_m=75.,mix=.5,max_renormalization=3.,amplitude=1.2,regional_gain=1.4)]
    oldout=tmp_path/'old';newout=tmp_path/'new';oldout.mkdir();newout.mkdir()
    old=JointObserver(c,candidates,oldout,'validate');new=ChunkedJointObserver(c,candidates,newout,'validate',chunk_columns=chunk)
    before=inputs[1].copy();state=np.random.get_state()
    old.diagnose(date(2018,7,1),1,'glorys',*inputs)
    new.diagnose(date(2018,7,1),1,'glorys',*inputs)
    np.testing.assert_array_equal(before,inputs[1]);after=np.random.get_state()
    assert state[0]==after[0] and state[2:]==after[2:];np.testing.assert_array_equal(state[1],after[1])
    compare_rows(rows(oldout/'joint_metrics.csv.gz'),rows(newout/'joint_metrics.csv.gz'),
        ['date','lead','candidate','domain','support','region','variable','depth_index','depth_m','members'])
    compare_rows(rows(oldout/'joint_events.csv.gz'),rows(newout/'joint_events.csv.gz'),
        ['date','lead','candidate','domain','support','region','threshold_s2','estimator'])
    assert old.funnels==new.funnels
    for table in ['scores','events']:
        a,b=old.acc.export()[table],new.acc.export()[table]
        compare_rows([{k:str(v) for k,v in r.items()} for r in a],[{k:str(v) for k,v in r.items()} for r in b],
            ['candidate','domain','support','region','variable','depth_index','window'] if table=='scores' else
            ['candidate','estimator','domain','support','region','threshold_s2','window'])


def test_physics_only_receives_bounded_columns(tmp_path):
    import ems_chunked_diagnostics as memory
    c=configuration(tmp_path,15);inputs=fixture(50,15,37);seen=[];original=memory.physics_fields
    def watched(state,*args,**kwargs):seen.append(state.shape[-1]);return original(state,*args,**kwargs)
    out=tmp_path/'bounded';out.mkdir()
    observer=memory.ChunkedJointObserver(c,[dict(id='raw',kind='raw')],out,'validate',chunk_columns=8)
    with patch.object(memory,'physics_fields',watched):observer.diagnose(date(2018,7,1),1,'glorys',*inputs)
    assert max(seen)==8 and len(seen)>3


def runtime_fixture(tmp_path):
    import ems_experiments as w
    import ems_lowmem as low
    root=tmp_path/'runs';root.mkdir();(root/'protocol.json').write_text('{}')
    c=dict(output_root=str(root))
    with patch.object(w,'verify',return_value={}):low.initialize(c,8)
    return w,low,c,root


def test_runtime_amendment_old_new_completion_and_tampering(tmp_path):
    w,low,c,root=runtime_fixture(tmp_path)
    original=(root/'protocol.json').read_bytes()
    t=dict(phase='calibrate',scenario='nominal',method='selected',members=50,steps=50,season='summer',seed=20261006,days=90)
    folder=w.folder(t,c);out=folder/'attempt_001';out.mkdir(parents=True);f=out/'score.csv';f.write_text('valid old output')
    base_deps={'protocol.json':low.file_hash(root/'protocol.json')}
    marker=dict(task=t,attempt=out.name,dependencies=base_deps,outputs={'score.csv':w.sha(f)})
    (folder/'completed.json').write_text(json.dumps(marker))
    new=dict(t,season='autumn')
    with patch.object(w,'dependencies',return_value=base_deps):
        restore=low.install_workflow(c)
        try:
            assert w.completed(t,c)['dependencies']==base_deps
            assert low.AMENDMENT in w.dependencies(new,c)
            newfolder=w.folder(new,c);newout=newfolder/'attempt_002';newout.mkdir(parents=True);g=newout/'score.csv';g.write_text('new output')
            newmarker=dict(task=new,attempt=newout.name,dependencies=w.dependencies(new,c),outputs={'score.csv':w.sha(g)})
            (newfolder/'completed.json').write_text(json.dumps(newmarker))
            assert w.completed(new,c)['attempt']=='attempt_002'
            assert (root/'protocol.json').read_bytes()==original
            f.write_text('bad old output')
            with pytest.raises(RuntimeError,match='changed'):w.completed(t,c)
            p=root/low.AMENDMENT;p.write_text(p.read_text()+' ')
            with pytest.raises(RuntimeError,match='changed during'):w.dependencies(new,c)
        finally:restore()


def test_amendment_reinitialization_and_code_mismatch(tmp_path):
    w,low,c,root=runtime_fixture(tmp_path)
    with patch.object(w,'verify',return_value={}):
        before=(root/low.AMENDMENT).read_bytes();low.initialize(c,8)
        assert (root/low.AMENDMENT).read_bytes()==before
        with pytest.raises(RuntimeError,match='already frozen'):low.initialize(c,16)
    with patch.object(low,'code_hashes',return_value={}):
        with pytest.raises(RuntimeError,match='code changed'):low.checked_record(root/low.AMENDMENT)


def test_real_unet_with_chunked_child_has_identical_forecast(tmp_path):
    import sys
    import test_ems_v1
    from ems_lowmem import child
    w,low,c,root=runtime_fixture(tmp_path)
    fixture_dir=tmp_path/'real';fixture_dir.mkdir()
    test_ems_v1.test_real_cycle_compatibility_and_gaussian_without_residual_weights(fixture_dir)
    output=fixture_dir/'lowmem';output.mkdir()
    args=['--memory-amendment',str(root/low.AMENDMENT),'--ems-observer',str(fixture_dir/'observer.json'),
        '--config',str(fixture_dir/'config.yaml'),'--deterministic-checkpoint',str(fixture_dir/'det.pt'),
        '--residual-checkpoint',str(fixture_dir/'res.pt'),'--residual-stats',str(fixture_dir/'resstats.npz'),
        '--start','2019-01-03','--days','2','--members','3','--sampling-steps','4','--sampling-eta','0',
        '--residual-scale','1','--seed','7','--save-daily','compact','--paper-holdout-root',str(fixture_dir/'holdout'),
        '--individual-member-guidance','--member-mean-cycle','--paired-guidance-center',
        '--argo-root',str(fixture_dir/'holdout'),'--require-argo','--ems-observation-options',str(fixture_dir/'stress.json'),
        '--output-dir',str(output)]
    with patch('torch.cuda.is_available',return_value=False):child(args)
    for day in ['2019-01-04','2019-01-05']:
        with np.load(fixture_dir/'wrapped'/f'{day}.npz') as a,np.load(output/f'{day}.npz') as b:
            np.testing.assert_array_equal(a['analysis'],b['analysis'])
    for name in ['paper_grid_metrics.csv','paper_holdout_metrics.csv','ems_multivariate.csv']:
        assert (fixture_dir/'wrapped'/name).read_bytes()==(output/name).read_bytes()
    assert (output/'memory_runtime.json').exists()
