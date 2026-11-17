"""New, declared TEOS-10 mapping; never rewrites the historical split.

Policies: disclosed raw export or native QC with adjusted A/D values.
Confirmation of source definitions is required BEFORE any new data are written.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from datetime import date,timedelta
from pathlib import Path
import numpy as np
import netCDF4
import gsw
from argo_common import (resolve_name,masked_array,orient_profile_level,parse_filename_date,
    TEMP_NAMES,PSAL_NAMES,PRES_NAMES,LAT_NAMES,LON_NAMES,TIME_NAMES)
from prepare_paper_argo import spatial_partition
from map_argo_to_glorys import load_grid,find_glorys_file,nearest_indices,nearest_longitude_indices,aggregate_profiles

def validate_declaration(c):
    if c.get('human_confirmed') is not True:
        raise ValueError('Source definitions unconfirmed. Review audit source_metadata.json and upstream documentation, then complete configs/oe_source_declaration.json.')
    if c.get('vertical_coordinate') not in ('pressure_dbar','depth_m','depth_negative_m'):
        raise ValueError('Declare pressure_dbar / depth_m / depth_negative_m explicitly')
    if c.get('temperature_kind') not in ('in_situ','potential_0dbar'):
        raise ValueError('Declare in_situ / potential_0dbar explicitly')
    if c.get('salinity_kind')!='practical_salinity':raise ValueError('This mapper expects Practical Salinity, not Absolute Salinity')
    if c.get('quality_policy') not in ('export_disclosed','native_adjusted'):
        raise ValueError('Unknown QC policy')
    if not str(c.get('quality_provenance','')).strip():
        raise ValueError('Record upstream quality/export provenance; missing QC is not certification')

def char_values(variable):
    a=np.ma.asarray(variable[:])
    fill='' if a.dtype.kind=='U' else (b'' if a.dtype.kind=='S' else 0)
    return np.asarray(a.filled(fill),dtype='U1')

def numeric(variable,n):
    return orient_profile_level(masked_array(variable),n).astype(float)

def flags(ds,name,shape,required):
    qc=resolve_name(ds,(name+'_QC',),False)
    if qc is None:
        if required:raise ValueError(f'Native QC required but {name}_QC absent')
        return np.ones(shape,bool),False
    a=orient_profile_level(char_values(ds.variables[qc]),shape[0])
    if a.shape!=shape:raise ValueError('QC dimensions differ from their variable')
    return np.isin(a,('1','2')),True

def read_source(path,c):
    validate_declaration(c)
    strict=c['quality_policy']=='native_adjusted'
    with netCDF4.Dataset(path) as ds:
        la=resolve_name(ds,LAT_NAMES);lo=resolve_name(ds,LON_NAMES)
        lat=masked_array(ds.variables[la]).reshape(-1).astype(float)
        lon=masked_array(ds.variables[lo]).reshape(-1).astype(float)
        n=len(lat)
        if len(lon)!=n:raise ValueError('Position lengths differ')
        position=np.isfinite(lat)&np.isfinite(lon)&(abs(lat)<=90)&(abs(lon)<=360)
        if strict:
            if 'POSITION_QC' not in ds.variables:raise ValueError('Native POSITION_QC absent')
            q=char_values(ds.variables['POSITION_QC']).reshape(-1)
            if len(q)!=n:raise ValueError('POSITION_QC shape differs')
            position &= np.isin(q,('1','2'))
            if 'DATA_MODE' not in ds.variables:raise ValueError('Native DATA_MODE absent')
            mode=char_values(ds.variables['DATA_MODE']).reshape(-1)
            if len(mode)==1:mode=np.full(n,mode[0])
            if len(mode)!=n:raise ValueError('DATA_MODE shape differs')
            position &= np.isin(mode,('R','A','D'))
        else:mode=np.full(n,'R')
        fields={};qc_present={};selected={}
        for key,candidates in (('temperature',TEMP_NAMES),('salinity',PSAL_NAMES),('vertical',PRES_NAMES)):
            name=resolve_name(ds,candidates)
            values=numeric(ds.variables[name],n)
            mask,present=flags(ds,name,values.shape,strict)
            qc_present[key]=present;selected[key]=name
            if strict and np.isin(mode,('A','D')).any():
                adjusted=resolve_name(ds,(name+'_ADJUSTED',),False)
                if adjusted is None:raise ValueError(f'A/D profile has no {name}_ADJUSTED')
                v=numeric(ds.variables[adjusted],n)
                q,_=flags(ds,adjusted,v.shape,True)
                if v.shape!=values.shape:raise ValueError('Adjusted values shape differs')
                use=np.isin(mode,('A','D'))[:,None]
                values=np.where(use,v,values);mask=np.where(use,q,mask)
            fields[key]=values;fields['valid_'+key]=mask & np.isfinite(values)
        if fields['temperature'].shape!=fields['salinity'].shape or fields['vertical'].shape!=fields['temperature'].shape:
            raise ValueError('Profile value dimensions differ')
        # Reject known contradictory metadata rather than guessing a unit conversion.
        tu=str(getattr(ds.variables[selected['temperature']],'units','')).lower()
        tn=str(getattr(ds.variables[selected['temperature']],'standard_name','')).lower()
        vu=str(getattr(ds.variables[selected['vertical']],'units','')).lower()
        sn=str(getattr(ds.variables[selected['salinity']],'standard_name','')).lower()
        if 'kelvin' in tu or tu.strip()=='k':raise ValueError('Temperature in Kelvin is unsupported; document and convert upstream')
        if c['vertical_coordinate']=='pressure_dbar' and vu.strip() in ('pa','pascal','pascals','bar'):
            raise ValueError('Declared dbar conflicts with pressure metadata')
        if c['vertical_coordinate']=='pressure_dbar' and vu.strip() in ('m','meter','meters','metre','metres'):
            raise ValueError('Declared pressure conflicts with depth metadata')
        if c['vertical_coordinate'].startswith('depth') and vu.strip() in ('dbar','decibar','decibars','pa','bar'):
            raise ValueError('Declared depth conflicts with pressure metadata')
        if tn=='sea_water_potential_temperature' and c['temperature_kind']!='potential_0dbar':
            raise ValueError('Declared in-situ temperature conflicts with potential-temperature metadata')
        if tn=='sea_water_temperature' and c['temperature_kind']!='in_situ':
            raise ValueError('Declared potential temperature conflicts with in-situ metadata')
        if 'absolute_salinity' in sn:raise ValueError('Absolute Salinity is not Practical Salinity')
        days=np.full(n,'',dtype='U10')
        time_name=resolve_name(ds,TIME_NAMES,False)
        if time_name is not None and getattr(ds.variables[time_name],'units',None):
            v=ds.variables[time_name];raw=np.ma.asarray(v[:],dtype=float).reshape(-1)
            if len(raw)==1:raw=np.repeat(raw,n)
            if len(raw)!=n:raise ValueError('Cannot associate time to profiles')
            keep=~np.ma.getmaskarray(raw)&np.isfinite(raw.filled(np.nan))
            if strict:
                time_qc=resolve_name(ds,(time_name+'_QC',),False)
                if time_qc is None:raise ValueError('Native time QC absent')
                tq=char_values(ds.variables[time_qc]).reshape(-1)
                if len(tq)==1:tq=np.repeat(tq,n)
                if len(tq)!=n:raise ValueError('Time QC dimensions differ')
                keep &= np.isin(tq,('1','2'))
            if keep.any():
                decoded=netCDF4.num2date(raw.data[keep],v.units,calendar=getattr(v,'calendar','standard'))
                days[keep]=[f'{x.year:04d}-{x.month:02d}-{x.day:02d}' for x in decoded]
        elif c.get('allow_filename_date'):
            d=parse_filename_date(path)
            if d is None:raise ValueError('No usable profile date')
            days[:]=d.isoformat()
        else:raise ValueError('No profile time units; filename date requires explicit daily-export declaration')
    vertical=fields['vertical'];valid_p=fields['valid_vertical']
    if c['vertical_coordinate']=='depth_negative_m':depth=-vertical
    elif c['vertical_coordinate']=='depth_m':depth=vertical
    else:depth=-gsw.z_from_p(np.where(valid_p & (vertical>=0),vertical,np.nan),lat[:,None])
    valid_p &= np.isfinite(depth)&(depth>=0)
    pressure=vertical if c['vertical_coordinate']=='pressure_dbar' else gsw.p_from_z(-depth,lat[:,None])
    t,s=fields['temperature'],fields['salinity']
    vt=fields['valid_temperature']&valid_p&(t>=-3)&(t<=40)
    vs=fields['valid_salinity']&valid_p&(s>=0)&(s<=42)
    if c['temperature_kind']=='in_situ':
        joint=vt&vs
        sa=gsw.SA_from_SP(np.where(joint,s,np.nan),pressure,np.mod(lon[:,None],360),lat[:,None])
        t=gsw.pt0_from_t(sa,np.where(joint,t,np.nan),pressure)
        vt=joint&np.isfinite(t)
    return dict(latitude=lat,longitude=lon,position=position,dates=days,temperature=t,salinity=s,
        valid_temperature=vt,valid_salinity=vs,depth=depth,qc_present=qc_present,n=n)

def interpolate(depth,values,valid,targets,max_gap_m=200.):
    ok=valid&np.isfinite(depth)&np.isfinite(values)
    if ok.sum()<2:return np.full(len(targets),np.nan,np.float32)
    d,v=depth[ok],values[ok];order=np.argsort(d);d,v=d[order],v[order]
    unique,index=np.unique(d,return_inverse=True)
    total=np.bincount(index,weights=v);count=np.bincount(index)
    d,v=unique,total/count
    if len(d)<2:return np.full(len(targets),np.nan,np.float32)
    result=np.interp(targets,d,v,left=np.nan,right=np.nan)
    if max_gap_m>0:
        right=np.searchsorted(d,targets,side='left');bounded=(right>0)&(right<len(d))
        idx=np.clip(right,0,len(d)-1)
        exact=np.isclose(d[idx],targets,rtol=0,atol=1e-6)
        gaps=d[np.clip(right,0,len(d)-1)]-d[np.clip(right-1,0,len(d)-1)]
        result[bounded&~exact&(gaps>max_gap_m)]=np.nan
    return result.astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',required=True);p.add_argument('--glorys-root',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--declaration',default='configs/oe_source_declaration.json')
    p.add_argument('--years',nargs='+',type=int,default=[2018,2019,2021]);p.add_argument('--max-gap-m',type=float,default=200.)
    a=p.parse_args();c=json.loads(Path(a.declaration).read_text(encoding='utf-8'));validate_declaration(c)
    if a.max_gap_m<0:p.error('max-gap-m must be nonnegative; zero disables the gap limit')
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Use a NEW split directory; prior partial/completed mappings are preserved')
    lat,lon,depth=load_grid(find_glorys_file(Path(a.glorys_root)))
    guide,held=spatial_partition(lat,lon,9.,.30,314159,1)
    by_day=defaultdict(list);source_hashes={};source_counts=defaultdict(int)
    files=sorted(Path(a.raw_root).rglob('*.nc'))
    for file_index,path in enumerate(files):
        # Profile time is authoritative, even when a filename fallback was permitted.
        data=read_source(path,c)
        selected=data['position'] & np.isin(np.array([x[:4] for x in data['dates']]),[str(y) for y in a.years])
        ids=np.flatnonzero(selected)
        if not len(ids):continue
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
        source_hashes[str(path)]=digest.hexdigest()
        ii=nearest_indices(lat,data['latitude'][ids]);jj=nearest_longitude_indices(lon,data['longitude'][ids])
        for k,i,j in zip(ids,ii,jj):
            text=str(data['dates'][k])
            t=interpolate(data['depth'][k],data['temperature'][k],data['valid_temperature'][k],depth,a.max_gap_m)
            s=interpolate(data['depth'][k],data['salinity'][k],data['valid_salinity'][k],depth,a.max_gap_m)
            by_day[text].append((int(i),int(j),t,s,float(data['latitude'][k]),float(data['longitude'][k]),f'{path.as_posix()}#{k}'))
        for key,present in data['qc_present'].items():source_counts[key+'_qc_missing_files']+=int(not present)
        source_counts['files']+=1;source_counts['profiles']+=len(ids)
        print(f'{file_index+1}/{len(files)}: {path.name} selected={len(ids)}',flush=True)
    if not by_day:raise ValueError('No profiles in requested years; no output written')
    out.mkdir(parents=True)
    np.savez_compressed(out/'spatial_partition.npz',guide=guide,holdout=held,latitude=lat,longitude=lon,depth_m=depth)
    counts=[]
    metadata=dict(latitude=lat,longitude=lon,depth_m=depth,
        temperature_kind=np.asarray('potential_0dbar'),vertical_coordinate=np.asarray('depth_m'),
        quality_policy=np.asarray(c['quality_policy']))
    for year in sorted(set(a.years)):
        day=date(year,1,1)
        while day.year==year:
            text=day.isoformat();records=by_day.get(text,[])
            ii=np.array([r[0] for r in records],np.int16);jj=np.array([r[1] for r in records],np.int16)
            t=np.array([r[2] for r in records],np.float32).reshape(-1,len(depth))
            s=np.array([r[3] for r in records],np.float32).reshape(-1,len(depth))
            g,h=guide[ii,jj],held[ii,jj]
            mapped=aggregate_profiles(ii[g].astype(np.int64)*len(lon)+jj[g],t[g],s[g],depth)
            for section in ('guidance','holdout'):(out/section/str(year)).mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'guidance'/str(year)/f'{text}.npz',**mapped,**metadata)
            np.savez_compressed(out/'holdout'/str(year)/f'{text}.npz',grid_i=ii[h],grid_j=jj[h],temperature=t[h],salinity=s[h],
                profile_latitude=np.array([r[4] for r in records],np.float32)[h],profile_longitude=np.array([r[5] for r in records],np.float32)[h],
                profile_id=np.array([r[6] for r in records],str)[h],**metadata)
            counts.append(dict(date=text,profiles=len(records),guidance_profiles=int(g.sum()),holdout_profiles=int(h.sum()),
                buffer_profiles=int((~g&~h).sum()),missing_source_day=not bool(records)))
            day+=timedelta(days=1)
    with (out/'split_counts.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
    manifest=dict(years=sorted(set(a.years)),temperature_kind='potential_0dbar',vertical_coordinate='depth_m',
        source_declaration=c,source_sha256=source_hashes,source_counts=dict(source_counts),max_interpolation_gap_m=a.max_gap_m,
        partition_sha256=hashlib.sha256(guide.tobytes()+held.tobytes()).hexdigest(),
        missing_source_days=[r['date'] for r in counts if r['missing_source_day']],
        gsw_version=gsw.__version__,note='New sensitivity mapping, NOT certification; same spatial partition, no extrapolation, observations converted before aggregation.')
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Saved new mapping: {out}; missing_source_days={len(manifest["missing_source_days"])}',flush=True)

if __name__=='__main__':main()
