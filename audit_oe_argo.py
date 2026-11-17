"""Read-only Argo QC/variable/mapping audit; never selects data using forecast errors."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import Counter
import numpy as np
import netCDF4
from argo_common import read_profile_arrays,parse_filename_date,orient_profile_level

def text_values(v):
    a=np.ma.asarray(v[:]);mask=np.ma.getmaskarray(a)
    return np.where(mask,"missing",a.data.astype("U")).reshape(-1)

def audit(root,split,out,gsw_module=None,years=(2018,2019,2021)):
    out.mkdir(parents=True,exist_ok=True)
    if (out/"summary.json").exists():raise FileExistsError("Audit exists; preserve it and choose a new output-dir")
    counts=Counter();flags=Counter();errors=[];rows=[];mapped=[]
    for path in sorted(root.rglob("*.nc")):
        day=parse_filename_date(path)
        if day is not None and day.year not in years:continue
        try:
            a=read_profile_arrays(path)
            if a["date"] is None:raise ValueError("Profile date unavailable")
            if a["date"].year not in years:continue
            from prepare_paper_argo import profile_day_mask
            profile_day_mask(path,a["date"].isoformat(),a["n_profiles"])
            counts["files"]+=1;counts["profiles"]+=a["n_profiles"]
            counts["invalid_position_profiles"]+=int((~np.isfinite(a["latitude"]) |
                ~np.isfinite(a["longitude"]) | (np.abs(a["latitude"])>90)).sum())
            r=dict(path=str(path),date=a["date"].isoformat(),profiles=a["n_profiles"],
                temperature_name=a["temperature_name"],salinity_name=a["salinity_name"],
                pressure_name=a["pressure_name"],temperature_qc_present=a["temperature_qc_present"],
                salinity_qc_present=a["salinity_qc_present"],adjusted_salinity_available=False,
                adjusted_salinity_difference_rms="",teos_pressure_depth_difference_rms="",teos_insitu_potential_difference_rms="")
            for key in ("temperature","salinity"):
                valid=a["valid_"+key];v=a[key][valid]
                counts[key+"_valid"]+=len(v)
                counts[key+"_missing_qc_files"]+=not a[key+"_qc_present"]
                if key=="salinity":
                    counts["salinity_below20"]+=int((v<20).sum());counts["salinity_zero"]+=int((v==0).sum())
                    counts["salinity_above42"]+=int((v>42).sum())
            with netCDF4.Dataset(path) as ds:
                for name in ("DATA_MODE","PSAL_QC","PSAL_ADJUSTED_QC","TEMP_QC","PRES_QC"):
                    if name in ds.variables:
                        flags.update({name+":"+k:int(v) for k,v in Counter(text_values(ds.variables[name])).items()})
                if "PSAL_ADJUSTED" in ds.variables:
                    r["adjusted_salinity_available"]=True
                    b=orient_profile_level(np.ma.asarray(ds.variables["PSAL_ADJUSTED"][:]).filled(np.nan),a["n_profiles"])
                    v=a["valid_salinity"] & np.isfinite(b) & (b>=0) & (b<=42)
                    if v.any():r["adjusted_salinity_difference_rms"]=float(np.sqrt(np.mean((b[v]-a["salinity"][v])**2)))
                # Audit only; preserve original mapping so existing results remain traceable.
                counts[f'files_{a["date"].year}']+=1
                if gsw_module is not None and counts[f'files_{a["date"].year}']<=100:
                    p=a["pressure"].astype(float);t=a["temperature"].astype(float);s=a["salinity"].astype(float)
                    lat=a["latitude"][:,None];lon=a["longitude"][:,None]
                    valid=a["valid_temperature"] & a["valid_salinity"] & (p>=0) & (s>=0) & (s<=42)
                    valid &= np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat)<=90)
                    if valid.any():
                        pp=np.where(valid,p,np.nan);ss=np.where(valid,s,np.nan);tt=np.where(valid,t,np.nan)
                        depth=-gsw_module.z_from_p(pp,lat)
                        sa=gsw_module.SA_from_SP(ss,pp,lon,lat)
                        pt=gsw_module.pt0_from_t(sa,tt,pp)
                        r["teos_pressure_depth_difference_rms"]=float(np.sqrt(np.nanmean((depth[valid]-pp[valid])**2)))
                        r["teos_insitu_potential_difference_rms"]=float(np.sqrt(np.nanmean((pt[valid]-tt[valid])**2)))
            rows.append(r)
        except Exception as e:errors.append(dict(path=str(path),error=repr(e)))
    if rows:
        with (out/"raw_files.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for section in ("guidance","holdout"):
        for year in years:
            for path in sorted((split/section/str(year)).glob("*.npz")):
                try:
                    with np.load(path,allow_pickle=False) as f:
                        for variable in ("temperature","salinity"):
                            values=f[variable];v=values[np.isfinite(values)]
                            mapped.append(dict(path=str(path),section=section,variable=variable,count=len(v),
                                below20=int((v<20).sum()) if variable=="salinity" else 0,
                                zero=int((v==0).sum()) if variable=="salinity" else 0,
                                minimum=float(v.min()) if len(v) else "",maximum=float(v.max()) if len(v) else "",
                                quantile01=float(np.quantile(v,.01)) if len(v) else ""))
                except Exception as e:errors.append(dict(path=str(path),error=repr(e)))
    if mapped:
        with (out/"mapped_daily.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(mapped[0]));w.writeheader();w.writerows(mapped)
    if not rows:errors.append(dict(path=str(root),error="No readable requested-year raw profiles"))
    if not mapped:errors.append(dict(path=str(split),error="No mapped observations"))
    metadata=[]
    for r in rows:
        with netCDF4.Dataset(r["path"]) as ds:
            selected={key:r[key+"_name"] for key in ("temperature","salinity","pressure")}
            metadata.append(dict(path=r["path"],date=r["date"],variables={
                key:dict(name=name,units=str(getattr(ds.variables[name],"units","")),
                    standard_name=str(getattr(ds.variables[name],"standard_name","")),
                    long_name=str(getattr(ds.variables[name],"long_name",""))) for key,name in selected.items()},
                provenance={key:str(getattr(ds,key,"")) for key in ("source","history","institution","title")},
                data_mode_present="DATA_MODE" in ds.variables))
    (out/"source_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    summary=dict(counts=dict(counts),qc_flags=dict(flags),file_errors=errors,gsw_available=gsw_module is not None,
        raw_root=str(root),split_root=str(split),years=list(years),
        warnings=["Missing QC is NOT proof of good data. Existing reader accepts missing QC.",
          "Existing reader prefers raw TEMP/PSAL; inspect DATA_MODE and adjusted availability.",
          "Pressure dbar is not exactly depth m; Argo TEMP is not necessarily GLORYS potential temperature.",
          "Low salinity can be real: never exclude observations because GLORYS/model disagrees.",
          "TEOS sample uses first 100 sorted files per requested year; not a global correction or representative estimate.",
          "If remapping is needed, create a new split version BEFORE freezing and rerun all paired experiments."])
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument("--raw-root",required=True);p.add_argument("--split-root",required=True)
    p.add_argument("--years",nargs="+",type=int,default=[2018,2019,2021]);p.add_argument("--output-dir",default="runs/oe_physics_v1/argo_audit");p.add_argument("--require-gsw",action="store_true")
    a=p.parse_args()
    try:import gsw
    except ImportError:
        if a.require_gsw:raise
        gsw=None
    audit(Path(a.raw_root),Path(a.split_root),Path(a.output_dir),gsw,tuple(a.years))

if __name__=="__main__":main()
