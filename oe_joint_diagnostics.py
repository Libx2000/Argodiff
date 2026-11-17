"""Common-support TEOS-10 diagnostics and compact calibration sufficient scores."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
import gsw
from oe_physics import member_mean,pycnocline_depth,region_masks,event_scores
from paper_metrics import score_samples
from oe_joint_calibration import calibrated_members,lead_factors

CORE=('temperature','salinity','rho','n2')
VARIABLES=CORE+('sigma0','buoyancy_per_volume','pycnocline_depth')
STATISTICS=('rmse','cycle_center_rmse','bias','mae','crps','rms_spread','coverage90','range_coverage','width90')
METRIC_FIELDS=['date','lead','candidate','domain','support','region','variable','depth_index','depth_m','members','support_digest','count','weight']+list(STATISTICS)+['spread_skill_ratio']+[f'rank_bin_{i}' for i in range(10)]
EVENT_FIELDS=['date','lead','candidate','estimator','domain','support','region','threshold_s2','support_digest','count','weight','brier','forecast_frequency','observed_frequency']


def write_gzip(path,value):
    tmp=Path(str(path)+'.pending')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=5) as f:json.dump(value,f,allow_nan=False)
    tmp.replace(path)


def append_rows(path,rows,fields):
    if not rows:return
    exists=Path(path).exists()
    with gzip.open(path,'at',encoding='utf-8',newline='',compresslevel=5) as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if not exists:writer.writeheader()
        writer.writerows(rows)


def physics_fields(state,depth,latitude,longitude,search=(10.,500.)):
    """State=(member,2*level,point). Preserve actual model depth order on input."""
    x=np.asarray(state,float);d=np.abs(np.asarray(depth,float));order=np.argsort(d);d=d[order]
    if x.ndim!=3 or x.shape[1]!=2*len(d) or len(d)<2 or not np.isfinite(d).all() or (np.diff(d)<=0).any():
        raise ValueError('Invalid state/depths')
    t=x[:,:len(d)][:,order];s=x[:,len(d):][:,order]
    lat=np.asarray(latitude,float)[None,None];lon=np.mod(longitude,360.)[None,None]
    p=gsw.p_from_z(-d[None,:,None],lat)
    # Reference hypersaline values are retained in common finite scores; funnel
    # sensitivity is a separate, truth-defined support rather than a model filter.
    valid=np.isfinite(t)&np.isfinite(s)&(t>=-3)&(t<=40)&(s>=0)&(s<=50)
    t=np.where(valid,t,np.nan);s=np.where(valid,s,np.nan)
    sa=gsw.SA_from_SP(s,p,lon,lat);ct=gsw.CT_from_pt(sa,t)
    rho=gsw.rho(sa,ct,p);n2,_=gsw.Nsquared(sa,ct,p,lat=lat,axis=1)
    n2=np.where(valid[:,:-1]&valid[:,1:],n2,np.nan)
    fields=dict(temperature=t,salinity=s,rho=rho,n2=n2,sigma0=gsw.sigma0(sa,ct),
        buoyancy_per_volume=rho*gsw.grav(lat,p),pycnocline_depth=pycnocline_depth(n2,d,search))
    funnel=np.asarray(gsw.infunnel(sa,ct,p),bool)&np.isfinite(sa)&np.isfinite(ct)
    return fields,d,funnel


class ScoreAccumulator:
    """Pool weighted squared errors, never average the reported RMSEs."""
    def __init__(self):self.rows={};self.events={}
    @staticmethod
    def windows(lead):
        result=['all','early' if lead<=30 else ('middle' if lead<=60 else 'late')]
        if lead==90:result.append('day90')
        return result
    def add(self,row,event=False):
        names=('candidate','estimator','domain','support','region','threshold_s2') if event else ('candidate','domain','support','region','variable','depth_index')
        target=self.events if event else self.rows
        if row['weight']<=0:return
        for window in self.windows(row['lead']):
            key=tuple(str(row[k]) for k in names)+(window,)
            if key not in target:
                target[key]=dict(**{k:row[k] for k in names},window=window,count=0,weight=0.,sums={})
            a=target[key];weight=float(row['weight']);a['weight']+=weight;a['count']+=int(row['count'])
            columns=('brier','forecast_frequency','observed_frequency') if event else STATISTICS+tuple(f'rank_bin_{i}' for i in range(10))
            for name in columns:
                value=float(row.get(name,0))
                if not np.isfinite(value):raise FloatingPointError(f'Nonfinite nonempty score: {name}')
                a['sums'][name]=a['sums'].get(name,0.)+weight*(value*value if name in ('rmse','rms_spread','cycle_center_rmse') else value)
    def export(self):
        def finish(rows):
            result=[]
            for r in rows.values():
                a={k:v for k,v in r.items() if k!='sums'}
                for k,v in r['sums'].items():a[k]=float(np.sqrt(max(0,v/r['weight']))) if k in ('rmse','rms_spread','cycle_center_rmse') else float(v/r['weight'])
                if 'rms_spread' in a:a['spread_skill_ratio']=a['rms_spread']/max(a['rmse'],1e-12)
                result.append(a)
            return result
        return dict(scores=finish(self.rows),events=finish(self.events))


def sample_grid(lat,lon,static_mask,config):
    la,lo=np.broadcast_arrays(np.asarray(lat)[:,None],np.asarray(lon)[None,:])
    eligible=np.all(np.asarray(static_mask,bool)[np.argsort(abs(np.asarray(config['depth'])))[0:2]],axis=0)
    regions=region_masks(la,lo,config)
    rng=np.random.default_rng(config['sample_seed']);chosen={}
    for name,count in [('global',config['global_sample_points'])]+[(r['name'],config['regional_sample_points']) for r in config['regions']]:
        indices=np.flatnonzero(eligible&regions[name])
        if not len(indices):raise ValueError(f'No eligible sampling points in {name}')
        chosen[name]=np.sort(rng.choice(indices,size=min(int(count),len(indices)),replace=False))
    union=np.unique(np.concatenate(list(chosen.values())))
    memberships={name:np.isin(union,ids) for name,ids in chosen.items()}
    return union//len(lon),union%len(lon),memberships


class JointObserver:
    def __init__(self,config,candidates,output,mode='collect'):
        self.c=config;self.candidates=candidates;self.out=Path(output);self.mode=mode
        self.acc=ScoreAccumulator();self.days=[];self.skipped=[];self.empty=[];self.mean_error=0.;self.funnels=[]
        self.coordinate_cache={};self.regions=[r['name'] for r in self.c['regions']]
        self.full=(mode!='collect')
    def diagnose(self,target,lead,domain,center,ensemble,truth,depth,lat,lon,memberships,weights,transform_lat=None,transform_lon=None):
        factor=lead_factors(self.c['reference_calibration'],lead,center.shape[0])
        point=(ensemble.shape[0]==1)
        candidates=[dict(id='point',kind='raw')] if point else self.candidates
        ef={};diagnostics={}
        for candidate,en,info in calibrated_members(ensemble,center,depth,
                lat if transform_lat is None else transform_lat,lon if transform_lon is None else transform_lon,
                factor,candidates,self.c['regions'],self.c['regional_feather_degrees']):
            ef[candidate['id']],d,fun=physics_fields(en,depth,lat,lon,self.c['pycnocline_depth_m'])
            self.mean_error=max(self.mean_error,info['mean_error']);diagnostics[candidate['id']]=info
            finite=np.isfinite(ef[candidate['id']]['rho'])
            self.funnels.append(dict(date=target.isoformat(),lead=lead,domain=domain,candidate=candidate['id'],
                valid=int(finite.sum()),outside=int((finite&~fun).sum())))
        cf,_,_=physics_fields(center[None],depth,lat,lon,self.c['pycnocline_depth_m'])
        tf,_,funnel=physics_fields(truth[None],depth,lat,lon,self.c['pycnocline_depth_m'])
        finite=np.isfinite(tf['rho'])
        self.funnels.append(dict(date=target.isoformat(),lead=lead,domain=domain,candidate='truth',valid=int(finite.sum()),outside=int((finite&~funnel).sum())))
        coords=np.ascontiguousarray(np.stack([lat,lon]),dtype=np.float64).tobytes()+np.ascontiguousarray(d).tobytes()
        coordinate_digest=hashlib.sha256(coords).digest()
        metrics=[];events=[];variables=VARIABLES if self.full else CORE
        for variable in variables:
            actual=tf[variable][0];central=cf[variable][0]
            common=np.isfinite(actual)&np.isfinite(central)
            for fields in ef.values():common &= np.all(np.isfinite(fields[variable]),axis=0)
            if variable=='n2':inside=funnel[0,:-1]&funnel[0,1:];ds=(d[:-1]+d[1:])/2
            elif variable=='pycnocline_depth':inside=np.all(funnel[0],axis=0);ds=None
            else:inside=funnel[0];ds=d
            for support in ('common','truth_funnel'):
                mask=common if support=='common' else common&inside
                for region,spatial in memberships.items():
                    keep=mask&spatial
                    ws=np.broadcast_to(weights,keep.shape)
                    depth_indices=[-1]
                    if self.full and region=='global' and variable in ('rho','n2'):depth_indices+=list(range(len(ds)))
                    for k in depth_indices:
                        mk=keep if k<0 else keep[k];w=ws if k<0 else ws[k]
                        tr=actual if k<0 else actual[k];cp=central if k<0 else central[k]
                        digest=hashlib.sha256(coordinate_digest+np.packbits(mk).tobytes()).hexdigest()
                        for name,fields in ef.items():
                            en=fields[variable] if k<0 else fields[variable][:,k]
                            pred=member_mean(en)
                            stats=score_samples(np.where(mk,pred,np.nan),tr,en,w)
                            if stats['weight']:
                                selected_w=w[mk];denom=selected_w.sum()
                                stats['cycle_center_rmse']=float(np.sqrt(np.dot(selected_w,(cp[mk]-tr[mk])**2)/denom))
                                q=np.quantile(en[:,mk],[.05,.95],axis=0)
                                stats['width90']=float(np.dot(selected_w,q[1]-q[0])/denom)
                            else:stats.update(cycle_center_rmse=float('nan'),width90=float('nan'))
                            row=dict(date=target.isoformat(),lead=lead,candidate=name,domain=domain,support=support,region=region,
                                variable=variable,depth_index=k,depth_m='all' if k<0 else float(ds[k]),members=len(en),support_digest=digest,**stats)
                            self.acc.add(row);metrics.append(row)
                    if variable=='n2':
                        digest=hashlib.sha256(coordinate_digest+np.packbits(keep).tobytes()).hexdigest()
                        for threshold in self.c['negative_n2_thresholds_s2']:
                            for name,fields in ef.items():
                                stats=event_scores(fields['n2']<threshold,actual<threshold,keep,ws)
                                row=dict(date=target.isoformat(),lead=lead,candidate=name,estimator='member_probability',domain=domain,
                                    support=support,region=region,threshold_s2=threshold,support_digest=digest,
                                    **{key:stats[key] for key in ('count','weight','brier','forecast_frequency','observed_frequency')})
                                self.acc.add(row,True);events.append(row)
                            stats=event_scores(central[None]<threshold,actual<threshold,keep,ws)
                            row=dict(date=target.isoformat(),lead=lead,candidate='common_center',estimator='cycle_center',domain=domain,
                                support=support,region=region,threshold_s2=threshold,support_digest=digest,
                                **{key:stats[key] for key in ('count','weight','brier','forecast_frequency','observed_frequency')})
                            self.acc.add(row,True);events.append(row)
        if self.full:
            append_rows(self.out/'joint_metrics.csv.gz',metrics,METRIC_FIELDS)
            append_rows(self.out/'joint_events.csv.gz',events,EVENT_FIELDS)
        return diagnostics
    def __call__(self,dataset,target,lead,analysis,ensemble,holdout_root,output_dir,deterministic=False):
        if lead in self.days:raise RuntimeError('Observer called twice on same day')
        raw,valid=dataset.archive.read_state(target);truth=np.where(valid,raw,np.nan)
        levels=dataset.levels;depth=np.abs(dataset.archive.coordinates()['depth'])
        if deterministic:ensemble=ensemble[:1]
        if self.full:
            i,j=np.indices((len(dataset.lat),len(dataset.lon)));i,j=i.ravel(),j.ravel()
            lat,lon=dataset.lat[i],dataset.lon[j]
            regions=region_masks(lat,lon,self.c)
            memberships={k:regions[k] for k in ['global']+self.regions}
        else:
            with np.load(self.c['sample_grid'],allow_pickle=False) as f:
                i,j=f['i'],f['j'];memberships={k:f['region_'+k] for k in ['global']+self.regions}
            lat,lon=dataset.lat[i],dataset.lon[j]
        diagnostic=self.diagnose(target,lead,'glorys',analysis[:,i,j],ensemble[:,:,i,j],truth[:,i,j],
            depth,lat,lon,memberships,np.cos(np.deg2rad(lat)).astype(float))
        if self.full:
            path=Path(holdout_root)/str(target.year)/f'{target}.npz'
            with np.load(path,allow_pickle=False) as f:
                kind=str(f['temperature_kind']) if 'temperature_kind' in f else 'unconfirmed'
                vertical=str(f['vertical_coordinate']) if 'vertical_coordinate' in f else 'unconfirmed'
                if kind!='potential_0dbar' or vertical!='depth_m':self.skipped.append(target.isoformat())
                elif len(f['grid_i']):
                    hi,hj=f['grid_i'].astype(int),f['grid_j'].astype(int)
                    if not np.allclose(f['depth_m'],depth) or not np.allclose(f['latitude'],dataset.lat) or not np.allclose(f['longitude'],dataset.lon):raise ValueError('Holdout geometry mismatch')
                    hlat,hlon=f['profile_latitude'],f['profile_longitude'];regions=region_masks(hlat,hlon,self.c)
                    membership={k:regions[k] for k in ['global']+self.regions}
                    self.diagnose(target,lead,'holdout',analysis[:,hi,hj],ensemble[:,:,hi,hj],
                        np.concatenate([f['temperature'].T,f['salinity'].T]),depth,hlat,hlon,membership,np.ones(len(hi)),
                        dataset.lat[hi],dataset.lon[hj])
                else:self.empty.append(target.isoformat())
        if self.full and lead in self.c['snapshot_leads']:
            ii=[int(np.argmin(abs(dataset.lat-p['latitude']))) for p in self.c['snapshot_points']]
            jj=[int(np.argmin(abs((dataset.lon-p['longitude']+180)%360-180))) for p in self.c['snapshot_points']]
            values=dict(center=analysis[:,ii,jj],truth=truth[:,ii,jj],depth_m=depth,latitude=dataset.lat[ii],longitude=dataset.lon[jj])
            selected=[dict(id='point',kind='raw')] if deterministic else self.candidates
            for candidate,en,_ in calibrated_members(ensemble[:,:,ii,jj],analysis[:,ii,jj],depth,dataset.lat[ii],dataset.lon[jj],
                    lead_factors(self.c['reference_calibration'],lead,dataset.state_channels),selected,self.c['regions'],self.c['regional_feather_degrees']):
                values['ensemble_'+candidate['id']]=en.astype(np.float32)
            np.savez_compressed(self.out/f'joint_profiles_day{lead:03d}.npz',**values)
        self.days.append(lead)
        if lead%10==0 or lead==self.c['expected_days']:
            write_gzip(self.out/'joint_pooled.json.gz',self.acc.export())
            write_gzip(self.out/'joint_funnel.json.gz',self.funnels)
        status=dict(completed_leads=self.days,mode=self.mode,candidates=[c['id'] for c in self.candidates],
            max_ensemble_center_error=self.mean_error,holdout_physics_skipped_unconfirmed=self.skipped,empty_holdout_days=self.empty,
            reference_source='GLORYS potential temperature and Practical Salinity',
            common_support='Intersection of truth, cycling center and ALL predeclared variants; identical within task',
            funnel_support='Truth-defined TEOS-10 funnel; forecast funnel counts reported separately',
            event_unit='Adjacent layer/position events, NOT unstable-profile fraction',
            ensemble_role='Post-sampling calibration; the same T/S center is used for the next daily cycle',
            full_global_members_saved=False)
        (self.out/'joint_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
