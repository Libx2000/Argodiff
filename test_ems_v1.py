import copy
from datetime import date
import json
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
import pytest
import torch

from ems_observations import transform_observations,validate_options
from ems_observer import profile_scores
from ems_gaussian import DiagonalGaussianResidual
from ems_evaluate import paired_blocks


def test_nested_thinning_matched_noise_and_rng():
    x=np.ones((6,12,12),np.float32)*10;mask=np.ones_like(x,bool)
    o=dict(start_date='2019-01-01',temperature_noise=.1,salinity_noise=.02)
    torch.manual_seed(5);saved=torch.get_rng_state().clone()
    a,ma,aa=transform_observations(x,mask,'2019-01-02',dict(o,retention=.25))
    b,mb,bb=transform_observations(x,mask,'2019-01-02',dict(o,retention=.5))
    assert not (ma&~mb).any();np.testing.assert_array_equal(a[ma],b[ma])
    assert torch.equal(saved,torch.get_rng_state());assert mask.all() and (x==10).all()
    c,mc,cc=transform_observations(x,mask,'2019-01-02',dict(o,retention=.25))
    np.testing.assert_array_equal(a,c);assert aa==cc


def test_outage_resumes_and_bounds():
    x=np.ones((4,2,2),np.float32);mask=np.ones_like(x,bool)
    o=dict(start_date='2019-01-01',outage_leads=[2])
    assert transform_observations(x,mask,'2019-01-02',o)[1].all()
    assert not transform_observations(x,mask,'2019-01-03',o)[1].any()
    assert transform_observations(x,mask,'2019-01-04',o)[1].all()
    x[0,0,0]=45
    _,m,a=transform_observations(x,mask,'2019-01-02',o)
    assert not m[0,0,0] and a['values_rejected_after_synthetic_noise']==1
    with pytest.raises(ValueError):validate_options(dict(o,retention=2))


def test_gaussian_conditional_formula_and_mask():
    sampler=DiagonalGaussianResidual(.5,.1)
    shape=(1,4,2,2);bg=torch.ones(shape)*10;std=torch.ones((1,4,1,1))*.2
    mean=torch.ones_like(std)*.1;noise=torch.ones(shape)*.3;obs=bg+.5;mask=torch.ones(shape)
    om=torch.ones(shape);om[:,:,0,0]=0
    z=sampler.ddim_sample_observation_guided(None,torch.zeros((1,2,2,2)),4,
        background_state=bg,residual_mean=mean,residual_std=std,observation=obs,
        observation_mask=om,levels=2,mask=mask,initial_noise=noise)
    r=torch.tensor([.25,.25,.01,.01]).view(1,4,1,1);k=std.square()/(std.square()+r)
    expected=k*(obs-bg-mean)/std+torch.sqrt(1-k)*noise
    np.testing.assert_allclose(z[:,:,1,1],expected[:,:,1,1],rtol=1e-5)
    assert torch.equal(z[:,:,0,0],noise[:,:,0,0])


def test_energy_variogram_manual():
    e=np.array([[[0],[0],[0],[0]],[[2],[2],[2],[2]]]);y=np.zeros((4,1))
    # Mean distance=2, pair correction=1, adjacent differences are all zero.
    s=profile_scores(e,y);assert s['energy_score']==1 and s['variogram_score']==0
    # A T/S offset is not an adjacent-depth pair.
    e=np.array([[[0],[0],[100],[100]]]);y=np.zeros((4,1))
    assert profile_scores(e,y)['variogram_score']==0
    with pytest.raises(FloatingPointError):profile_scores(e*np.nan,y)


def paired_rows(seeds=(1,2)):
    a=[];b=[]
    for s in seeds:
        for i in range(1,15):
            r=dict(season='winter',seed=s,date=f'2019-01-{i:02}',lead=i,count=2,weight=2.,support_digest='abc')
            a.append(dict(r,value=1.));b.append(dict(r,value=4.))
    return a,b


def test_paired_bootstrap_seed_nesting_and_missing_support():
    a,b=paired_rows();r=paired_blocks(a,b,7,20,root=True)
    assert r['difference_left_minus_right']==-1 and r['conditional_95_interval']==[-1,-1]
    c,d=paired_rows((1,2,3,4));assert paired_blocks(c,d,7,20,root=True)==r
    with pytest.raises(ValueError,match='unmatched'):paired_blocks(a,b[:-1],7,20)
    b[0]['support_digest']='changed'
    with pytest.raises(ValueError,match='support'):paired_blocks(a,b,7,20)


def test_plan_counts_and_formal_source_block():
    import ems_experiments as w
    c=w.settings('configs/ems_v1_legacy.json')
    assert {p:len(w.task_list(p,c)) for p in w.PHASES}==dict(tune=56,calibrate=8,validate=64,robust=160,budget=36,final=64)
    with patch.object(w,'read',return_value={}):
        with pytest.raises(RuntimeError,match='Legacy'):w.source_check(c,formal=True)


def test_completed_hash_tamper_detected(tmp_path):
    import ems_experiments as w
    c=dict(output_root=str(tmp_path));t=dict(phase='tune',scenario='nominal',method='x',members=50,steps=50,season='winter',seed=1)
    folder=w.folder(t,c);out=folder/'attempt_001';out.mkdir(parents=True);f=out/'score.csv';f.write_text('original')
    with patch.object(w,'dependencies',return_value={}):
        (folder/'completed.json').write_text(json.dumps(dict(task=t,attempt=out.name,dependencies={},outputs={'score.csv':w.sha(f)})))
        assert w.completed(t,c)['attempt']=='attempt_001'
        f.write_text('tampered')
        with pytest.raises(RuntimeError,match='changed'):w.completed(t,c)


def test_real_cycle_compatibility_and_gaussian_without_residual_weights(tmp_path):
    import importlib.util
    root=Path(__file__).resolve().parent
    base=root if (root/'tests/test_oe_extension.py').exists() else Path('C:/Users/Administrator/Desktop/diffArgo')
    spec=importlib.util.spec_from_file_location('fixture_oe',base/'tests/test_oe_extension.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fixture.test_real_unet_cycle_replay(tmp_path)
    common=['cycle','--config',str(tmp_path/'config.yaml'),'--deterministic-checkpoint',str(tmp_path/'det.pt'),
        '--residual-checkpoint',str(tmp_path/'res.pt'),'--residual-stats',str(tmp_path/'resstats.npz'),
        '--start','2019-01-03','--days','2','--members','3','--sampling-steps','4','--sampling-eta','0',
        '--residual-scale','1','--seed','7','--save-daily','compact','--paper-holdout-root',str(tmp_path/'holdout'),
        '--individual-member-guidance','--member-mean-cycle','--paired-guidance-center']
    from oe_stable_cycle import main as old
    from ems_cycle import main as new
    for name,entry in [('stable',old),('ems',new)]:
        with patch.object(sys,'argv',common+['--output-dir',str(tmp_path/name)]),patch('torch.cuda.is_available',return_value=False):entry()
    for name in ['paper_grid_metrics.csv','paper_holdout_metrics.csv','paper_region_metrics.csv']:
        assert (tmp_path/'stable'/name).read_bytes()==(tmp_path/'ems'/name).read_bytes()
    for day in ['2019-01-04','2019-01-05']:
        with np.load(tmp_path/'stable'/f'{day}.npz') as a,np.load(tmp_path/'ems'/f'{day}.npz') as b:
            np.testing.assert_array_equal(a['analysis'],b['analysis'])
    # This verifies that the cheap baseline does not load an expensive unused network.
    args=common.copy();args[args.index('--residual-checkpoint')+1]=str(tmp_path/'intentionally_absent.pt')
    with patch.object(sys,'argv',args+['--ems-backend','diagonal_gaussian','--output-dir',str(tmp_path/'gauss')]),patch('torch.cuda.is_available',return_value=False):new()
    assert (tmp_path/'gauss/compute_cost.json').exists()
    # Full streaming wrapper: guidance on day 1, outage on day 2, actual TEOS
    # diagnostics and standardized profile scores, no global scientific claim.
    from ems_observer import main as observer
    dc=json.loads((base/'configs/oe_stability_v3.json').read_text())['diagnostics']
    unit=tmp_path/'unit.npz';np.savez_compressed(unit,inflation_factor=np.ones((2,6)))
    dc.update(expected_days=2,reference_calibration=str(unit),state_stats=str(tmp_path/'stats.npz'),profile_sample_seed=7,profile_sample_points=8)
    config=tmp_path/'observer.json';config.write_text(json.dumps(dict(config=dc,candidates=[dict(id='raw',kind='raw')])))
    stress=tmp_path/'stress.json';stress.write_text(json.dumps(dict(start_date='2019-01-03',outage_leads=[2])))
    args=common+['--argo-root',str(tmp_path/'holdout'),'--require-argo','--ems-observer',str(config),
        '--ems-observation-options',str(stress),'--output-dir',str(tmp_path/'wrapped')]
    with patch.object(sys,'argv',args),patch('torch.cuda.is_available',return_value=False):observer()
    import csv
    with (tmp_path/'wrapped/ems_multivariate.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and all(np.isfinite(float(r['energy_score'])) for r in rows)
    audit=[json.loads(r) for r in (tmp_path/'wrapped/ems_observation_audit.jsonl').read_text().splitlines()]
    assert audit[0]['retained_values']==6 and audit[1]['outage'] and audit[1]['retained_values']==0
