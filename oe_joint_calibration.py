"""Mean-preserving joint T/S calibration of sampled residual-diffusion members.

Uses forecasts, depth and positions only. Truth never enters the transform.
"""
from __future__ import annotations
import itertools
import numpy as np


def candidate_catalog(config):
    result=[dict(id='raw',kind='raw'),dict(id='marginal',kind='marginal')]
    c=config['candidates']
    for length,mix,amplitude,regional in itertools.product(c['length_m'],c['mix'],c['amplitude'],c['regional_gain']):
        name=f'joint_z{length:g}_m{mix:g}_a{amplitude:g}_r{regional:g}'
        result.append(dict(id=name,kind='joint',length_m=float(length),mix=float(mix),
            amplitude=float(amplitude),regional_gain=float(regional),
            max_renormalization=float(c['max_renormalization'])))
    return result


def regional_multiplier(latitude,longitude,regions,gain,feather_mdeg=3.):
    """Full gain inside predeclared boxes, cosine taper outside their edges."""
    lat,lon=np.broadcast_arrays(latitude,np.mod(longitude,360.))
    feather=float(feather_mdeg)
    if feather<=0 or gain<=0:raise ValueError('Invalid regional gain/feather')
    taper=np.zeros(lat.shape,float)
    for r in regions:
        south,north=r['latitude'];west,east=np.mod(r['longitude'],360.)
        span=(east-west)%360.;center=(west+span/2)%360.
        dx=np.maximum(abs((lon-center+180)%360-180)-span/2,0)
        dy=np.maximum(np.maximum(south-lat,lat-north),0)
        tx=np.where(dx<feather,np.cos(np.pi/2*np.minimum(dx/feather,1))**2,0)
        ty=np.where(dy<feather,np.cos(np.pi/2*np.minimum(dy/feather,1))**2,0)
        taper=np.maximum(taper,tx*ty)
    return 1+(gain-1)*taper


def smooth_vertical(anomaly,depth,valid,length):
    """Gaussian depth-distance filter within contiguous wet-layer segments.

    anomaly=(member,2*level,point); valid=(2*level,point).
    No coupling across missing layers or between different ocean columns.
    """
    a=np.asarray(anomaly,float);d=np.abs(np.asarray(depth,float));levels=len(d)
    if a.ndim!=3 or a.shape[1]!=2*levels:raise ValueError('Expected (member,2*level,point)')
    if length<=0:return a.copy()
    order=np.argsort(d);inverse=np.argsort(order);d=d[order]
    if not np.isfinite(d).all() or np.any(np.diff(d)<=0):raise ValueError('Depths must be finite and unique')
    out=np.zeros_like(a)
    distance=abs(d[:,None]-d[None,:]);kernel=np.exp(-.5*(distance/length)**2)
    kernel[distance>3*length]=0
    for variable in range(2):
        x=a[:,variable*levels:(variable+1)*levels][:,order]
        wet=np.asarray(valid[variable*levels:(variable+1)*levels],bool)[order]
        segment=np.cumsum(~wet,axis=0)
        filtered=np.zeros_like(x)
        for k in range(levels):
            weights=kernel[k,:,None]*wet*(segment==segment[k])
            weights*=wet[k][None]
            denom=weights.sum(axis=0)
            value=np.einsum('jn,mjn->mn',weights,x,optimize=False)
            filtered[:,k]=np.divide(value,denom[None],out=np.zeros_like(value),where=denom[None]>0)
        out[:,variable*levels:(variable+1)*levels]=filtered[:,inverse]
    return out


def bound_about_center(anomaly,center,valid,temperature_bounds=(-3.,40.),salinity_bounds=(0.,42.)):
    """Uniform per-channel/column shrinkage retains the exact T/S center."""
    a=np.where(valid[None],np.asarray(anomaly,float),0).copy()
    c=np.asarray(center,float);levels=c.shape[0]//2
    lower=np.array([temperature_bounds[0]]*levels+[salinity_bounds[0]]*levels)[:,None]
    upper=np.array([temperature_bounds[1]]*levels+[salinity_bounds[1]]*levels)[:,None]
    if np.any(valid&((c<lower)|(c>upper))):raise ValueError('Forecast center lies outside declared bounds')
    a-=a.mean(axis=0,keepdims=True)
    pos=np.maximum(a.max(axis=0),0);neg=np.maximum(-a.min(axis=0),0)
    safe=np.ones(c.shape,float)
    safe=np.minimum(safe,np.divide(upper-c,pos,out=np.ones_like(c),where=pos>0))
    safe=np.minimum(safe,np.divide(c-lower,neg,out=np.ones_like(c),where=neg>0))
    safe=np.where(valid,np.clip(safe,0,1),0)
    a*=safe[None]*(1-2e-12)
    # Float64 arithmetic keeps the ensemble mean within numerical round-off.
    a-=a.mean(axis=0,keepdims=True)
    en=np.where(valid[None],c[None]+a,np.nan)
    error=float(np.max(np.where(valid,abs(np.where(valid[None],en,0).mean(axis=0)-np.where(valid,c,0)),0),initial=0))
    if error>2e-5:raise FloatingPointError(f'Calibration moved the cycling center: {error}')
    return en,dict(mean_error=error,bound_limited_fraction=float(np.mean(safe[valid]<.999999)) if valid.any() else 0.)


def calibrated_members(ensemble,center,depth,latitude,longitude,factor,candidates,regions,feather_degrees=3.):
    """Yield variants, with reusable filtered bases for the candidate scan."""
    ensemble=np.asarray(ensemble,float);center=np.asarray(center,float)
    if ensemble.ndim!=3 or center.shape!=ensemble.shape[1:]:raise ValueError('Invalid member/center shape')
    valid=np.all(np.isfinite(ensemble),axis=0)&np.isfinite(center)
    raw=np.where(valid[None],ensemble-center[None],0)
    raw-=raw.mean(axis=0,keepdims=True)
    f=np.asarray(factor,float).reshape(-1,1)
    if len(f)!=center.shape[0] or not np.isfinite(f).all() or (f<1).any():raise ValueError('Invalid marginal factors')
    base=raw*f[None];cache={};filters={}
    for candidate in candidates:
        kind=candidate['kind']
        if kind=='raw':anomaly=raw
        elif kind=='marginal':anomaly=base
        elif kind=='joint':
            length=candidate['length_m'];mix=candidate['mix'];cap=candidate['max_renormalization']
            if not 0<=mix<=1 or cap<1 or candidate['amplitude']<=0:raise ValueError('Invalid joint parameters')
            key=(length,mix,cap)
            if key not in cache:
                if length not in filters:filters[length]=smooth_vertical(base,depth,valid,length)
                x=(1-mix)*base+mix*filters[length]
                old=np.sqrt(np.mean(base*base,axis=0));new=np.sqrt(np.mean(x*x,axis=0))
                scale=np.divide(old,np.maximum(new,1e-14),out=np.ones_like(old),where=old>1e-14)
                x=x*np.minimum(scale,cap)[None]
                cache[key]=x
            spatial=regional_multiplier(latitude,longitude,regions,candidate['regional_gain'],feather_degrees)
            anomaly=cache[key]*candidate['amplitude']*spatial[None,None]
        else:raise ValueError(f'Unknown candidate kind {kind}')
        en,diagnostic=bound_about_center(anomaly,center,valid)
        yield candidate,en,diagnostic


def lead_factors(path,lead,channels):
    with np.load(path,allow_pickle=False) as f:factor=f['inflation_factor'].astype(float)
    if factor.ndim==2:
        if not 1<=lead<=len(factor):raise ValueError('Calibration lead not available')
        factor=factor[lead-1]
    if factor.shape!=(channels,):raise ValueError('Calibration channels differ from model')
    return factor
