"""Post-sampling TEOS-10 verification. Does not alter forecasts or random state.

N2 is evaluated at adjacent pressure midpoints, never across missing layers.
Engineering events are hypothetical fixed-volume sensitivities, not device safety.
"""
from __future__ import annotations
import csv, gzip, json
from pathlib import Path
import numpy as np
import gsw
from paper_metrics import score_samples

UNITS = {'rho':'kg m-3','sigma0':'kg m-3','n2':'s-2',
         'buoyancy_per_volume':'N m-3','pycnocline_depth':'m'}
SCORE_KEYS = ['count','weight','rmse','bias','mae','rms_spread',
              'spread_skill_ratio','crps','range_coverage','coverage90']
SCORE_KEYS += [f'rank_bin_{i}' for i in range(10)]
METRIC_FIELDS = ['date','lead','domain','region','variable','unit','estimator',
                 'depth_index','depth_m','members'] + SCORE_KEYS
EVENT_FIELDS = ['date','lead','domain','region','event','reference_density_kgm3',
                'tolerance_N_m3','members','count','weight','brier',
                'forecast_frequency','observed_frequency']
EVENT_FIELDS += [f'bin_{i}_{v}' for i in range(10) for v in ('weight','probability','observed')]


def append_gzip(path, rows, fields):
    if not rows: return
    path = Path(path)
    exists = path.exists()
    with gzip.open(path, 'at', encoding='utf-8', newline='', compresslevel=6) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        w.writerows(rows)


def member_mean(a):
    """No all-NaN warnings and no variable member counts across spatial samples."""
    valid = np.all(np.isfinite(a), axis=0)
    return np.where(valid, np.where(valid[None], a, 0.).mean(axis=0), np.nan)


def derive_fields(temperature, salinity, depth, latitude, longitude,
                  temperature_kind='potential_0dbar'):
    """Inputs (member, depth, *space); latitude/longitude broadcast to *space."""
    t, s = np.asarray(temperature, float), np.asarray(salinity, float)
    if t.shape != s.shape or t.ndim < 3:
        raise ValueError('Expected matching (member, depth, *space) arrays')
    depth = np.abs(np.asarray(depth, float))
    if len(depth) != t.shape[1] or not np.all(np.isfinite(depth)):
        raise ValueError('Invalid depth vector')
    order = np.argsort(depth); d = depth[order]
    if np.any(np.diff(d) <= 0): raise ValueError('Depths must be unique')
    lat = np.broadcast_to(np.asarray(latitude, float), t.shape[2:])
    lon = np.broadcast_to(np.mod(longitude, 360.), t.shape[2:])
    if np.any(~np.isfinite(lat)) or np.any(np.abs(lat)>90) or np.any(~np.isfinite(lon)):
        raise ValueError('Invalid positions')
    la, lo = lat[None,None], lon[None,None]
    z = -d.reshape((1,len(d)) + (1,)*(t.ndim-2))
    p = gsw.p_from_z(z, la)
    t, s = t[:,order], s[:,order]
    valid = np.isfinite(t) & np.isfinite(s) & (s>=0) & (s<=42) & (t>=-3) & (t<=40)
    t, s = np.where(valid,t,np.nan), np.where(valid,s,np.nan)
    sa = gsw.SA_from_SP(s,p,lo,la)
    if temperature_kind == 'potential_0dbar': ct = gsw.CT_from_pt(sa,t)
    elif temperature_kind == 'in_situ': ct = gsw.CT_from_t(sa,t,p)
    else: raise ValueError('Unconfirmed temperature kind; physical scoring refused')
    rho, sigma = gsw.rho(sa,ct,p), gsw.sigma0(sa,ct)
    n2, _ = gsw.Nsquared(sa,ct,p,lat=la,axis=1)
    # Explicit adjacency mask: do not bridge missing/below-bottom model levels.
    pair_valid = valid[:,:-1] & valid[:,1:]
    n2 = np.where(pair_valid,n2,np.nan)
    gravity = np.broadcast_to(gsw.grav(la,p),rho.shape)
    fields = dict(rho=rho,sigma0=sigma,n2=n2,
                  buoyancy_per_volume=rho*gravity)
    funnel = np.asarray(gsw.infunnel(sa,ct,p),bool)
    finite = np.isfinite(sa) & np.isfinite(ct)
    info = dict(valid_values=int(finite.sum()),outside_funnel=int((finite & ~funnel).sum()))
    return fields, d, order, np.asarray(gravity[0]), info


def pycnocline_depth(n2, depth, bounds):
    mids = (np.asarray(depth[:-1])+depth[1:])/2
    selected = (mids>=bounds[0]) & (mids<=bounds[1])
    if not selected.any(): return np.full(n2.shape[:1]+n2.shape[2:],np.nan)
    values = n2[:,selected]
    # A missing interval inside the declared search domain invalidates the proxy.
    valid = np.all(np.isfinite(values),axis=1) & np.any(values>0,axis=1)
    index = np.where(np.isfinite(values),values,-np.inf).argmax(axis=1)
    return np.where(valid,mids[selected][index],np.nan)


def event_scores(ensemble_event, truth_event, valid, weights=None):
    valid = np.asarray(valid,bool)
    w = np.ones(valid.shape,float) if weights is None else np.broadcast_to(weights,valid.shape)
    valid &= np.isfinite(w) & (w>0)
    p = np.mean(ensemble_event,axis=0)[valid]
    y, w = np.asarray(truth_event,float)[valid], w[valid]
    total = float(w.sum())
    result = dict(count=int(valid.sum()),weight=total,brier=float('nan'),
                  forecast_frequency=float('nan'),observed_frequency=float('nan'))
    for i in range(10):
        result.update({f'bin_{i}_weight':0.,f'bin_{i}_probability':float('nan'),
                       f'bin_{i}_observed':float('nan')})
    if not total: return result
    avg = lambda a: float(np.dot(w,a)/total)
    result.update(brier=avg((p-y)**2),forecast_frequency=avg(p),observed_frequency=avg(y))
    bins = np.minimum((p*10).astype(int),9)
    for i in range(10):
        keep = bins==i; weight = float(w[keep].sum())
        result[f'bin_{i}_weight'] = weight
        if weight:
            result[f'bin_{i}_probability'] = float(np.dot(w[keep],p[keep])/weight)
            result[f'bin_{i}_observed'] = float(np.dot(w[keep],y[keep])/weight)
    return result


def region_masks(lat,lon,config):
    lat,lon = np.broadcast_arrays(lat,lon)
    masks = {'global':np.ones(lat.shape,bool)}
    for label,lo,hi in [('tropical',0,20),('midlatitude',20,50),('highlatitude',50,91)]:
        masks[label] = (np.abs(lat)>=lo) & (np.abs(lat)<hi)
    for r in config.get('regions',[]):
        west,east = np.mod(r['longitude'],360.)
        lons = np.mod(lon,360.)
        lm = (lons>=west)&(lons<=east) if west<=east else (lons>=west)|(lons<=east)
        masks[r['name']] = lm & (lat>=r['latitude'][0]) & (lat<=r['latitude'][1])
    return masks


class PhysicsObserver:
    def __init__(self, config, output_dir):
        self.cfg = config; self.out = Path(output_dir); self.days = []
        self.funnel = dict(valid_values=0,outside_funnel=0)
        self.held_skipped = []
        self.held_empty = []
        self.case_note = 'Hypothetical fixed-volume bulk density; NOT a validated device tolerance or safety benefit.'

    def score_domain(self,target,lead,domain,ens_fields,center_fields,truth_fields,
                     depths,lat,lon,gravity,weights):
        metrics, events = [], []
        masks = region_masks(lat,lon,self.cfg)
        mids = (depths[:-1]+depths[1:])/2
        ens_fields = dict(ens_fields); center_fields = dict(center_fields); truth_fields = dict(truth_fields)
        for fields in (ens_fields,center_fields,truth_fields):
            fields['pycnocline_depth'] = pycnocline_depth(fields['n2'],depths,self.cfg['pycnocline_depth_m'])
        for region,spatial in masks.items():
            for variable,en in ens_fields.items():
                if variable == 'pycnocline_depth':
                    en, tr, cp = en[:,spatial],truth_fields[variable][0][spatial],center_fields[variable][0][spatial]
                    ws = None if weights is None else np.broadcast_to(weights,spatial.shape)[spatial]
                    ds = None
                else:
                    en, tr, cp = en[:,:,spatial],truth_fields[variable][0][:,spatial],center_fields[variable][0][:,spatial]
                    ws = None if weights is None else np.broadcast_to(weights,spatial.shape)[spatial][None]
                    ds = mids if variable=='n2' else depths
                predictions = [('ensemble_mean',member_mean(en)),('cycle_center',cp)]
                indices = [-1] + (list(range(len(ds))) if region=='global' and ds is not None else [])
                for estimator,pred in predictions:
                    for k in indices:
                        if k<0: args = (pred,tr,en,ws)
                        else: args = (pred[k],tr[k],en[:,k],None if ws is None else ws[0])
                        metrics.append(dict(date=target.isoformat(),lead=lead,domain=domain,region=region,
                            variable=variable,unit=UNITS[variable],estimator=estimator,depth_index=k,
                            depth_m='all' if k<0 else float(ds[k]),members=len(en),**score_samples(*args)))
            if region in ('global',*[r['name'] for r in self.cfg.get('regions',[])]):
                en = ens_fields['n2'][:,:,spatial]; tr = truth_fields['n2'][0][:,spatial]
                valid = np.all(np.isfinite(en),axis=0) & np.isfinite(tr)
                ws = None if weights is None else np.broadcast_to(weights,spatial.shape)[spatial][None]
                stats = event_scores(en<0,tr<0,valid,ws)
                events.append(dict(date=target.isoformat(),lead=lead,domain=domain,region=region,event='negative_n2',
                    reference_density_kgm3='',tolerance_N_m3='',members=len(en),**stats))
                working = (depths>=self.cfg['working_depth_m'][0]) & (depths<=self.cfg['working_depth_m'][1])
                er = ens_fields['rho'][:,working][:,:,spatial]
                rr = truth_fields['rho'][0][working][:,spatial]
                gr = gravity[working][:,spatial]
                joint = np.all(np.isfinite(er),axis=0) & np.isfinite(rr)
                for density in self.cfg['reference_density_kgm3']:
                    for tolerance in self.cfg['force_tolerances_N_m3']:
                        event = np.abs(gr[None]*(er-density))>tolerance
                        actual = np.abs(gr*(rr-density))>tolerance
                        stats = event_scores(event,actual,joint,ws)
                        events.append(dict(date=target.isoformat(),lead=lead,domain=domain,region=region,
                            event='hypothetical_net_buoyancy_exceedance',reference_density_kgm3=density,
                            tolerance_N_m3=tolerance,members=len(er),**stats))
        append_gzip(self.out/'oe_physics_metrics.csv.gz',metrics,METRIC_FIELDS)
        append_gzip(self.out/'oe_event_metrics.csv.gz',events,EVENT_FIELDS)

    def __call__(self,dataset,target,lead,analysis,ensemble,holdout_root,output_dir,deterministic=False):
        if lead in self.days: raise RuntimeError('Physics observer called twice for the same lead')
        levels = dataset.levels
        if deterministic: ensemble = ensemble[:1]
        raw, valid = dataset.archive.read_state(target)
        depth = np.abs(dataset.archive.coordinates()['depth'])
        lat,lon = np.broadcast_arrays(dataset.lat[:,None],dataset.lon[None,:])
        ef,d,order,gravity,info = derive_fields(ensemble[:,:levels],ensemble[:,levels:],depth,lat,lon)
        cf,_,_,_,_ = derive_fields(analysis[None,:levels],analysis[None,levels:],depth,lat,lon)
        truth = np.where(valid,raw,np.nan)
        tf,_,_,_,_ = derive_fields(truth[None,:levels],truth[None,levels:],depth,lat,lon)
        self.funnel = {key:self.funnel[key]+info[key] for key in self.funnel}
        self.score_domain(target,lead,'glorys',ef,cf,tf,d,lat,lon,gravity,np.cos(np.deg2rad(lat)))
        # Compact, predeclared profile snapshots only; never full global members.
        if lead in self.cfg['snapshot_leads']:
            ii = [int(np.argmin(abs(dataset.lat-p['latitude']))) for p in self.cfg['snapshot_points']]
            jj = [int(np.argmin(abs((dataset.lon-p['longitude']+180)%360-180))) for p in self.cfg['snapshot_points']]
            np.savez_compressed(self.out/f'oe_profiles_day{lead:03d}.npz',depth_m=d,
                latitude=dataset.lat[ii],longitude=dataset.lon[jj],
                member_temperature=ensemble[:,:levels][:,order][:,:,ii,jj],
                member_salinity=ensemble[:,levels:][:,order][:,:,ii,jj],
                member_rho=ef['rho'][:,:,ii,jj],member_n2=ef['n2'][:,:,ii,jj],
                truth_rho=tf['rho'][0][:,ii,jj],truth_n2=tf['n2'][0][:,ii,jj])
        # T-S covariance and adjacent-depth correlation, with zero-spread samples excluded.
        t,s = ensemble[:,:levels][:,order],ensemble[:,levels:][:,order]
        tm,sm = member_mean(t),member_mean(s)
        ta,sa = t-tm[None],s-sm[None]
        tv,sv = np.mean(ta*ta,axis=0),np.mean(sa*sa,axis=0)
        cov = np.mean(ta*sa,axis=0)
        denom = np.sqrt(tv*sv)
        corr = np.divide(cov,denom,out=np.full_like(cov,np.nan),where=denom>1e-12)
        w = np.broadcast_to(np.cos(np.deg2rad(lat)),corr.shape)
        ok = np.isfinite(corr); total = float(w[ok].sum())
        structure = dict(date=target.isoformat(),lead=lead,count=int(ok.sum()),weight=total,
            temperature_salinity_correlation=float(np.dot(w[ok],corr[ok])/total) if total else float('nan'),
            temperature_salinity_covariance=float(np.dot(w[ok],cov[ok])/total) if total else float('nan'))
        rho=ef['rho'];ra=rho-member_mean(rho)[None]
        rv=np.mean(ra*ra,axis=0)
        denom=np.sqrt(rv[:-1]*rv[1:])
        adjacent=np.divide(np.mean(ra[:,:-1]*ra[:,1:],axis=0),denom,
            out=np.full_like(denom,np.nan),where=denom>1e-12)
        aw=np.broadcast_to(np.cos(np.deg2rad(lat)),adjacent.shape)
        ao=np.isfinite(adjacent);at=float(aw[ao].sum())
        structure.update(adjacent_density_count=int(ao.sum()),adjacent_density_weight=at,
            adjacent_density_correlation=float(np.dot(aw[ao],adjacent[ao])/at) if at else float('nan'))
        append_gzip(self.out/'oe_joint_structure.csv.gz',[structure],list(structure))
        path = Path(holdout_root)/str(target.year)/f'{target.isoformat()}.npz'
        with np.load(path,allow_pickle=False) as f:
            kind = str(f['temperature_kind']) if 'temperature_kind' in f else 'unconfirmed'
            coordinate = str(f['vertical_coordinate']) if 'vertical_coordinate' in f else 'unconfirmed'
            if kind!='potential_0dbar' or coordinate!='depth_m':
                self.held_skipped.append(target.isoformat())
            else:
                i,j = f['grid_i'].astype(int),f['grid_j'].astype(int)
                if np.any((i<0)|(i>=len(dataset.lat))|(j<0)|(j>=len(dataset.lon))):
                    raise ValueError('Invalid held-profile cell indices')
                if len(i):
                    hlat,hlon = f['profile_latitude'],f['profile_longitude']
                    he,hd,_,hg,_ = derive_fields(ensemble[:,:levels][:,:,i,j],ensemble[:,levels:][:,:,i,j],depth,hlat,hlon)
                    hc,_,_,_,_ = derive_fields(analysis[None,:levels][:,:,i,j],analysis[None,levels:][:,:,i,j],depth,hlat,hlon)
                    ht,_,_,_,_ = derive_fields(f['temperature'].T[None],f['salinity'].T[None],depth,hlat,hlon)
                    self.score_domain(target,lead,'holdout',he,hc,ht,hd,hlat,hlon,hg,None)
                else:self.held_empty.append(target.isoformat())
        self.days.append(lead)
        self.write_status()

    def write_status(self):
        status = dict(completed_leads=self.days,gsw_version=gsw.__version__,
            model_temperature_kind='potential_0dbar',salinity_kind='practical_salinity',
            depth_order='sorted positive-down for diagnostics only; forecast channels unchanged',
            n2_location='adjacent pressure midpoint; depth labels are nominal depth midpoints',
            funnel=self.funnel,holdout_physics_skipped_unconfirmed=self.held_skipped,
            holdout_physics_empty_profile_days=self.held_empty,
            engineering_interpretation=self.case_note,
            density_estimators='ensemble_mean=mean(memberwise density); cycle_center=density(cycling mean T/S)',
            full_global_members_saved=False)
        (self.out/'oe_diagnostic_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
