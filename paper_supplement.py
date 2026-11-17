"""Ordinary-job, paired/block supplementary experiments; never submits jobs itself."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, subprocess, sys
import shutil
from pathlib import Path
from datetime import date, timedelta, datetime, timezone
from collections import defaultdict
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/paper_supplement_v1"
SEASONS = ("winter", "spring", "summer", "autumn")
SEEDS = (20261006, 20261017)
METHODS = ("deterministic", "old_paired", "fine_paired", "fine_calibrated")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()

@lru_cache(maxsize=128)
def archive_sha(path,size,mtime_ns):
    # Cache within this process only; any metadata change invalidates the key.
    return sha(path)

def protocol():
    c = json.loads((ROOT/"configs/paper_protocol.json").read_text(encoding="utf-8"))
    c.update(json.loads((ROOT/"configs/paper_supplement.json").read_text(encoding="utf-8")))
    return c

def tasks(phase):
    if phase == "attribution":
        recipes = [(m,s,None) for m in ("old_paired","fine_paired") for s in (.25,1.)]
        year = 2019
    elif phase == "calibrate":
        recipes, year = [("fine_paired",1.,None)], 2018
    elif phase == "validate":
        recipes, year = [(m,1.,None) for m in METHODS], 2019
    elif phase == "free":
        recipes, year = [("fine_paired",1.,n) for n in (0,7,30,None)], 2019
    elif phase in ("test2021","test2022"):
        recipes, year = [(m,1.,None) for m in METHODS], int(phase[-4:])
    else: raise ValueError(phase)
    return [dict(phase=phase,year=year,method=m,scale=s,stop=n,season=season,
                 month=1+3*i,seed=seed) for m,s,n in recipes
            for i,season in enumerate(SEASONS) for seed in SEEDS]

def destination(t):
    stop = "continuous" if t["stop"] is None else f"stop{t['stop']}"
    return OUT/t["phase"]/f"{t['method']}_scale{t['scale']:g}_{stop}"/t["season"]/str(t["seed"])

def fingerprints(c):
    paths = [c[k] for k in ("config","deterministic_checkpoint","residual_checkpoint",
                            "fine_checkpoint","residual_stats","state_stats")]
    paths += [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")
              if not any(x in p.relative_to(ROOT).parts for x in ("runs",".git","__pycache__"))]
    paths += ["configs/paper_protocol.json","configs/paper_supplement.json"]
    cal = OUT/"calibration_2018.npz"
    if cal.exists(): paths.append(str(cal.relative_to(ROOT)))
    split = Path(c["argo_split_root"])
    paths += [str(split/"split_manifest.json"),str(split/"spatial_partition.npz")]
    return {p:sha(ROOT/p if not Path(p).is_absolute() else p) for p in sorted(set(paths))}

def input_hashes(c,t):
    root = Path(c["argo_split_root"])
    first = date(t["year"],t["month"],1)
    paths = [root/section/str(d.year)/f"{d.isoformat()}.npz"
             for section in ("guidance","holdout")
             for d in (first+timedelta(days=k) for k in range(c["days"]))]
    # Hash bytes only, do not inspect future test labels or performance before freeze.
    hashes={}
    for path in paths:
        stat=path.stat()
        hashes[str(path)]=archive_sha(str(path),stat.st_size,stat.st_mtime_ns)
    import yaml
    config=yaml.safe_load((ROOT/c["config"]).read_text(encoding="utf-8"))["data"]
    data_root=Path(config["root"])
    if not data_root.is_absolute():data_root=ROOT/data_root
    last=first+timedelta(days=c["days"]-1)
    initial=first-timedelta(days=config.get("context_days",3))
    for year in range(initial.year,last.year+1):
        for variable in ("temperature","salinity"):
            path=data_root/config[variable+"_subdir"]/config[variable+"_pattern"].format(year=year)
            stat=path.stat()
            hashes[str(path)]=archive_sha(str(path),stat.st_size,stat.st_mtime_ns)
    return hashes

def run(t,c):
    fp = fingerprints(c)
    data_fp = input_hashes(c,t)
    if t["phase"].startswith("test"):
        frozen = json.loads((OUT/"frozen_protocol.json").read_text(encoding="utf-8"))
        if sha(OUT/"argo_audit/summary.json") != frozen["audit_sha256"]:
            raise RuntimeError("Argo audit changed after freeze")
        if t["phase"] not in frozen["test_data"]:
            raise RuntimeError("This test year was not included BEFORE freezing")
        if frozen["fingerprints"] != fp or frozen["protocol"] != c:
            raise RuntimeError("Frozen code/models/parameters changed. Do not silently redefine a blind test.")
        if frozen["test_data"][t["phase"]][str(t["month"])] != data_fp:
            raise RuntimeError("Frozen test observation files changed")
    out = destination(t)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"recipe.json").exists():
        raise FileExistsError(f"Preserving existing run: {out}; use a NEW version directory for reruns")
    start = date(t["year"],t["month"],1)-timedelta(days=1)
    checkpoint = c["residual_checkpoint"] if t["method"] == "old_paired" else c["fine_checkpoint"]
    cmd = [sys.executable,"-u",str(ROOT/"argo_guided_cycle.py"),"--config",c["config"],
           "--deterministic-checkpoint",c["deterministic_checkpoint"],
           "--residual-checkpoint",checkpoint,"--residual-stats",c["residual_stats"],
           "--start",start.isoformat(),"--days",str(c["days"]),"--members",str(c["members"]),
           "--sampling-steps",str(c["sampling_steps"]),"--sampling-eta","0",
           "--residual-scale",str(t["scale"]),"--seed",str(t["seed"]),
           "--paper-holdout-root",str(Path(c["argo_split_root"])/"holdout"),
           "--save-daily","none","--output-dir",str(out)]
    if t["method"] == "deterministic": cmd += ["--deterministic-only"]
    else:
        cmd += ["--argo-root",str(Path(c["argo_split_root"])/"guidance"),"--require-argo",
                "--individual-member-guidance","--member-mean-cycle","--paired-guidance-center",
                "--temperature-guidance",str(c["temperature_guidance"]),
                "--salinity-guidance",str(c["salinity_guidance"]),
                "--guidance-start",str(c["guidance_start"])]
    if t["stop"] is not None: cmd += ["--argo-stop-after",str(t["stop"])]
    if t["method"] == "fine_calibrated":
        cmd += ["--spread-calibration",str(OUT/"calibration_2018.npz"),"--strict-calibration-leads"]
    if t["phase"] == "calibrate":
        cmd += ["--calibration-statistics-output",str(out/"calibration_statistics.npz")]
    recipe = dict(task=t,protocol=c,fingerprints=fp,data_fingerprints=data_fp,command=cmd)
    (out/"recipe.json").write_text(json.dumps(recipe,indent=2),encoding="utf-8")
    subprocess.run(cmd,cwd=ROOT,check=True)
    # Only successful, fully scored 90-day jobs are marked complete.
    for name in ("paper_grid_metrics.csv","paper_holdout_metrics.csv"):
        with (out/name).open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
        if {int(r["lead"]) for r in rows} != set(range(1,c["days"]+1)):
            raise RuntimeError("Incomplete verification")
    (out/"completed.json").write_text(json.dumps(dict(task=t,utc=datetime.now(timezone.utc).isoformat())),encoding="utf-8")

def pool(rows):
    rows=[r for r in rows if float(r["weight"])>0 and math.isfinite(float(r["rmse"]))]
    w=sum(float(r["weight"]) for r in rows)
    if w == 0: return dict(weight=0,rmse=None,crps=None,coverage90=None,spread=None)
    avg=lambda key: sum(float(r["weight"])*float(r[key]) for r in rows)/w
    return dict(weight=w,rmse=math.sqrt(sum(float(r["weight"])*float(r["rmse"])**2 for r in rows)/w),
                crps=avg("crps"),coverage90=avg("coverage90"),
                spread=math.sqrt(sum(float(r["weight"])*float(r["rms_spread"])**2 for r in rows)/w))

def summarize(phase,c, require_complete=False):
    groups=defaultdict(list); costs=[]; missing=[]
    current=fingerprints(c)
    expected_data={}
    for t in tasks(phase):
        out=destination(t)
        if not (out/"completed.json").exists(): missing.append(t);continue
        saved=json.loads((out/"recipe.json").read_text(encoding="utf-8"))
        # All comparisons must use identical code, backbone, observations and settings.
        if saved["protocol"] != c: raise RuntimeError(f"Mixed protocol: {out}")
        data_key=(t["year"],t["month"])
        if data_key not in expected_data:expected_data[data_key]=input_hashes(c,t)
        if saved["data_fingerprints"] != expected_data[data_key]:
            raise RuntimeError(f"Observation/reference data changed: {out}")
        for key,value in saved["fingerprints"].items():
            if key.endswith("calibration_2018.npz"): continue
            if current.get(key)!=value: raise RuntimeError(f"Changed artifact {key}: {out}")
        for name in ("paper_grid_metrics.csv","paper_holdout_metrics.csv","paper_region_metrics.csv"):
            with (out/name).open(encoding="utf-8",newline="") as f:
                for r in csv.DictReader(f):
                    key=(t["method"],t["scale"],t["stop"],t["season"],t["seed"],r["domain"],r["variable"],r["depth_index"])
                    groups[key].append(r)
        costs.append(dict(task=t,**json.loads((out/"compute_cost.json").read_text(encoding="utf-8"))))
    if require_complete and missing: raise RuntimeError(f"{len(missing)} incomplete jobs in {phase}")
    summary=[dict(method=k[0],scale=k[1],stop=k[2],season=k[3],seed=k[4],domain=k[5],variable=k[6],
                  depth_index=k[7],**pool(v),last30=pool([r for r in v if int(r["lead"])>60]),
                  final_day=pool([r for r in v if int(r["lead"])==c["days"]])) for k,v in groups.items()]
    result=dict(phase=phase,complete=not missing,missing=missing,scores=summary,costs=costs)
    # Descriptive paired comparisons, repeated seeds are NOT independent ocean cases.
    paired=[]
    for k,rows in groups.items():
        method,scale,stop,season,seed,domain,variable,depth=k
        if depth != "-1" or method=="deterministic":continue
        reference=groups.get(("deterministic",1.,None,season,seed,domain,variable,depth))
        if reference is None:continue
        lookup={r["date"]:r for r in reference}
        if any(r["date"] not in lookup or r["count"]!=lookup[r["date"]]["count"] or
               r["weight"]!=lookup[r["date"]]["weight"] for r in rows):
            raise RuntimeError("Unmatched samples in paired comparison")
        cand,base=pool(rows),pool(reference)
        if not cand["weight"] or not base["rmse"]:continue
        paired.append(dict(method=method,scale=scale,stop=stop,season=season,seed=seed,
            domain=domain,variable=variable,rmse_gain_percent=100*(1-cand["rmse"]/base["rmse"]),
            crps_change=cand["crps"]-base["crps"],coverage90=cand["coverage90"]))
    result["paired_vs_deterministic"]=paired
    result["inference_note"]="Descriptive blocked comparisons. Do not treat days, grid cells or sampling seeds as independent replicates. Four starts alone do not establish statistical significance."
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f"summary_{phase}.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("snapshot","plan","run","fit","report","freeze"))
    p.add_argument("--phase",default="attribution",choices=("attribution","calibrate","validate","free","test2021","test2022"))
    p.add_argument("--task",type=int,default=0)
    p.add_argument("--audit-reviewed",action="store_true")
    a=p.parse_args();c=protocol();OUT.mkdir(parents=True,exist_ok=True)
    if a.action=="snapshot":
        folder=OUT/"previous_stage_snapshot"
        if folder.exists():raise FileExistsError("Snapshot already preserved")
        folder.mkdir()
        files=[p for base in (ROOT/"runs/residual_rollout_v1",ROOT/"runs/residual_rollout_v1_validation90")
               for p in base.rglob("*") if p.is_file() and p.suffix in (".json",".csv",".md") and p.stat().st_size<5_000_000]
        manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
        for p in files:
            target=folder/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        weights={c[k]:sha(ROOT/c[k]) for k in ("fine_checkpoint","residual_checkpoint","deterministic_checkpoint")}
        (folder/"manifest.json").write_text(json.dumps(dict(files=manifest,checkpoint_hashes=weights,
            note="Checkpoints kept at original paths; no duplicate large weight files."),indent=2),encoding="utf-8")
    elif a.action=="plan":
        rows=[dict(task=i,**t,output=str(destination(t))) for i,t in enumerate(tasks(a.phase))]
        with (OUT/f"plan_{a.phase}.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        for r in rows:print(r["task"],r["method"],r["scale"],r["stop"],r["season"],r["seed"])
    elif a.action=="run":
        ts=tasks(a.phase)
        if not 0<=a.task<len(ts): p.error(f"task range 0..{len(ts)-1}")
        run(ts[a.task],c)
    elif a.action=="fit":
        summarize("calibrate",c,True)
        output=OUT/"calibration_2018.npz"
        if output.exists():raise FileExistsError("Calibration already exists; preserve it")
        subprocess.run([sys.executable,str(ROOT/"fit_spread_calibration.py"),"--statistics",
            *[str(destination(t)/"calibration_statistics.npz") for t in tasks("calibrate")],
            "--output",str(output),"--lead-dependent","--monotonic-lead",
            "--max-factor",str(c["max_factor"]),"--log-shrinkage",str(c["log_shrinkage"])],cwd=ROOT,check=True)
    elif a.action=="report":summarize(a.phase,c)
    else:
        if not a.audit_reviewed: p.error("Review audit_argo first, then explicitly pass --audit-reviewed")
        if (OUT/"frozen_protocol.json").exists():raise FileExistsError("Already frozen; do not overwrite a blind-test protocol")
        reports={phase:summarize(phase,c,True) for phase in ("attribution","calibrate","validate","free")}
        audit=OUT/"argo_audit/summary.json"
        audit_result=json.loads(audit.read_text(encoding="utf-8"))
        if audit_result["file_errors"]:raise RuntimeError("Resolve Argo read errors before freezing")
        if not audit_result["gsw_available"]:
            raise RuntimeError("Run the pressure/temperature audit with GSW before freezing")
        if not (OUT/"calibration_2018.npz").exists():raise FileNotFoundError("2018 calibration missing")
        frozen=dict(protocol=c,fingerprints=fingerprints(c),audit_sha256=sha(audit),audit_reviewed=True,
            utc=datetime.now(timezone.utc).isoformat(),validation_reports={k:sha(OUT/f"summary_{k}.json") for k in reports},
            test_data={f"test{year}":{str(t["month"]):input_hashes(c,t) for t in tasks(f"test{year}")} for year in c["blind_years"]},
            claim="Frozen evaluation, NOT a guarantee of skill. No test-informed parameter changes.")
        (OUT/"frozen_protocol.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
        print("Frozen. Run test2021 first; test2022 is optional independent replication.")

if __name__=="__main__":main()
