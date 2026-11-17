"""Memory-only evaluation amendment: full population, float64, bounded columns.

The model, sampling, masks, calibration transform and score definitions remain
the original ones. Only spatial summation order changes (roundoff is possible).
No global float64 member cube or member-derived field is retained here.
"""
from __future__ import annotations
import hashlib
import numpy as np
from oe_joint_diagnostics import (JointObserver as OriginalJointObserver,
    physics_fields,CORE,VARIABLES,STATISTICS,METRIC_FIELDS,EVENT_FIELDS,append_rows)
from oe_joint_calibration import calibrated_members,lead_factors
from oe_physics import member_mean,event_scores
from paper_metrics import score_samples


class DailyPool:
    def __init__(self,event=False):
        self.event=event;self.rows={}

    def add(self,row):
        names=('date','lead','candidate','estimator','domain','support','region','threshold_s2') if self.event else (
            'date','lead','candidate','domain','support','region','variable','depth_index','depth_m','members')
        key=tuple(row[k] for k in names)
        if key not in self.rows:self.rows[key]=[dict(row),{},0.,0]
        stored,sums,weight,count=self.rows[key]
        w=float(row['weight']);self.rows[key][2]+=w;self.rows[key][3]+=int(row['count'])
        if w<=0:return
        columns=('brier','forecast_frequency','observed_frequency') if self.event else STATISTICS+tuple(f'rank_bin_{i}' for i in range(10))
        for name in columns:
            value=float(row[name])
            if not np.isfinite(value):raise FloatingPointError('Nonfinite nonempty chunk score: '+name)
            sums[name]=sums.get(name,0.)+w*(value*value if name in ('rmse','rms_spread','cycle_center_rmse') else value)

    def finish(self,digest):
        out=[]
        for stored,sums,w,count in self.rows.values():
            r=dict(stored,weight=w,count=count,support_digest=digest(stored,self.event))
            if w>0:
                for name,total in sums.items():
                    r[name]=float(np.sqrt(max(0.,total/w))) if name in ('rmse','rms_spread','cycle_center_rmse') else total/w
                if not self.event:r['spread_skill_ratio']=r['rms_spread']/max(r['rmse'],1e-12)
            out.append(r)
        return out


class ChunkedJointObserver(OriginalJointObserver):
    def __init__(self,*args,chunk_columns=512,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(chunk_columns,int) or chunk_columns<1:raise ValueError('Positive chunk column count required')
        self.chunk_columns=chunk_columns

    def diagnose(self,target,lead,domain,center,ensemble,truth,depth,lat,lon,memberships,weights,
                 transform_lat=None,transform_lon=None):
        factor=lead_factors(self.c['reference_calibration'],lead,center.shape[0])
        candidates=[dict(id='point',kind='raw')] if ensemble.shape[0]==1 else self.candidates
        variables=VARIABLES if self.full else CORE
        total_points=len(lat);levels=len(depth);d=np.sort(np.abs(np.asarray(depth,float)))
        metric_pool,event_pool=DailyPool(),DailyPool(True)
        # These masks scale as depths*points, NOT members*depths*points.
        masks={}
        for variable in variables:
            shape=(total_points,) if variable=='pycnocline_depth' else ((levels-1 if variable=='n2' else levels),total_points)
            for support in ('common','truth_funnel'):masks[variable,support]=np.zeros(shape,bool)
        diagnostics={};funnel_counts={}
        for left in range(0,total_points,self.chunk_columns):
            right=min(total_points,left+self.chunk_columns);sl=slice(left,right)
            diagnostic=self._chunk(target,lead,domain,center[...,sl],ensemble[...,sl],truth[...,sl],depth,
                lat[sl],lon[sl],{name:m[sl] for name,m in memberships.items()},weights[sl],
                None if transform_lat is None else transform_lat[sl],None if transform_lon is None else transform_lon[sl],
                factor,candidates,variables,sl,masks,metric_pool,event_pool,funnel_counts)
            for name,info in diagnostic.items():
                a=diagnostics.setdefault(name,dict(mean_error=0.,limited=0.,valid=0))
                a['mean_error']=max(a['mean_error'],info['mean_error'])
                a['limited']+=info['bound_limited_fraction']*info['valid'];a['valid']+=info['valid']
        coordinate_digest=hashlib.sha256(np.ascontiguousarray(np.stack([lat,lon]),dtype=np.float64).tobytes()+
            np.ascontiguousarray(d).tobytes()).digest()
        digests={}
        def support_digest(r,event):
            variable='n2' if event else r['variable'];k=-1 if event else r['depth_index']
            key=(variable,r['support'],r['region'],k)
            if key not in digests:
                mask=masks[variable,r['support']]&memberships[r['region']]
                if k>=0:mask=mask[k]
                digests[key]=hashlib.sha256(coordinate_digest+np.packbits(mask).tobytes()).hexdigest()
            return digests[key]
        metrics=metric_pool.finish(support_digest);events=event_pool.finish(support_digest)
        for row in metrics:self.acc.add(row)
        for row in events:self.acc.add(row,True)
        if self.full:
            append_rows(self.out/'joint_metrics.csv.gz',metrics,METRIC_FIELDS)
            append_rows(self.out/'joint_events.csv.gz',events,EVENT_FIELDS)
        for name,counts in funnel_counts.items():
            self.funnels.append(dict(date=target.isoformat(),lead=lead,domain=domain,candidate=name,
                valid=counts[0],outside=counts[1]))
        return {name:dict(mean_error=a['mean_error'],bound_limited_fraction=a['limited']/a['valid'] if a['valid'] else 0.)
            for name,a in diagnostics.items()}

    def _chunk(self,target,lead,domain,center,ensemble,truth,depth,lat,lon,memberships,weights,
               transform_lat,transform_lon,factor,candidates,variables,sl,masks,metric_pool,event_pool,funnel_counts):
        ef={};diagnostics={}
        valid_count=int((np.all(np.isfinite(ensemble),axis=0)&np.isfinite(center)).sum())
        def track(name,fields,funnel):
            finite=np.isfinite(fields['rho']);counts=funnel_counts.setdefault(name,[0,0])
            counts[0]+=int(finite.sum());counts[1]+=int((finite&~funnel).sum())
        for candidate,en,info in calibrated_members(ensemble,center,depth,
            lat if transform_lat is None else transform_lat,lon if transform_lon is None else transform_lon,
            factor,candidates,self.c['regions'],self.c['regional_feather_degrees']):
            name=candidate['id'];ef[name],d,fun=physics_fields(en,depth,lat,lon,self.c['pycnocline_depth_m'])
            self.mean_error=max(self.mean_error,info['mean_error']);diagnostics[name]=dict(info,valid=valid_count)
            track(name,ef[name],fun)
        cf,_,_=physics_fields(center[None],depth,lat,lon,self.c['pycnocline_depth_m'])
        tf,_,funnel=physics_fields(truth[None],depth,lat,lon,self.c['pycnocline_depth_m']);track('truth',tf,funnel)
        for variable in variables:
            actual=tf[variable][0];central=cf[variable][0]
            common=np.isfinite(actual)&np.isfinite(central)
            for fields in ef.values():common &= np.all(np.isfinite(fields[variable]),axis=0)
            if variable=='n2':inside=funnel[0,:-1]&funnel[0,1:];ds=(d[:-1]+d[1:])/2
            elif variable=='pycnocline_depth':inside=np.all(funnel[0],axis=0);ds=None
            else:inside=funnel[0];ds=d
            for support in ('common','truth_funnel'):
                mask=common if support=='common' else common&inside
                masks[variable,support][...,sl]=mask
                for region,spatial in memberships.items():
                    keep=mask&spatial;ws=np.broadcast_to(weights,keep.shape)
                    indices=[-1]+(list(range(len(ds))) if self.full and region=='global' and variable in ('rho','n2') else [])
                    for k in indices:
                        mk=keep if k<0 else keep[k];w=ws if k<0 else ws[k]
                        tr=actual if k<0 else actual[k];cp=central if k<0 else central[k]
                        for name,fields in ef.items():
                            en=fields[variable] if k<0 else fields[variable][:,k]
                            stats=score_samples(np.where(mk,member_mean(en),np.nan),tr,en,w)
                            if stats['weight']:
                                sw=w[mk];denom=sw.sum()
                                stats['cycle_center_rmse']=float(np.sqrt(np.dot(sw,(cp[mk]-tr[mk])**2)/denom))
                                q=np.quantile(en[:,mk],[.05,.95],axis=0)
                                stats['width90']=float(np.dot(sw,q[1]-q[0])/denom)
                            else:stats.update(cycle_center_rmse=float('nan'),width90=float('nan'))
                            metric_pool.add(dict(date=target.isoformat(),lead=lead,candidate=name,domain=domain,support=support,
                                region=region,variable=variable,depth_index=k,depth_m='all' if k<0 else float(ds[k]),
                                members=len(en),support_digest='',**stats))
                    if variable=='n2':
                        for threshold in self.c['negative_n2_thresholds_s2']:
                            for name,fields in ef.items():
                                stats=event_scores(fields['n2']<threshold,actual<threshold,keep,ws)
                                event_pool.add(dict(date=target.isoformat(),lead=lead,candidate=name,estimator='member_probability',
                                    domain=domain,support=support,region=region,threshold_s2=threshold,support_digest='',
                                    **{key:stats[key] for key in ('count','weight','brier','forecast_frequency','observed_frequency')}))
                            stats=event_scores(central[None]<threshold,actual<threshold,keep,ws)
                            event_pool.add(dict(date=target.isoformat(),lead=lead,candidate='common_center',estimator='cycle_center',
                                domain=domain,support=support,region=region,threshold_s2=threshold,support_digest='',
                                **{key:stats[key] for key in ('count','weight','brier','forecast_frequency','observed_frequency')}))
        return diagnostics
