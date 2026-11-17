"""Auditable resource-only amendment without editing frozen EM&S v1 sources.

All subsequent commands use this entry point. Old completed cases keep their
original strict dependency checks; new cases additionally bind this amendment.
"""
from __future__ import annotations
import argparse
from collections import deque
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
AMENDMENT='runtime_memory_v1.json'
FILES=('ems_lowmem.py','ems_chunked_diagnostics.py',
       'ems_resource_probe/detect_resources.py','ems_resource_probe/_common.py',
       'submit_ems_lowmem_batch.slurm','submit_ems_lowmem_tools.slurm')


def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def code_hashes():return {name:file_hash(ROOT/name) for name in FILES}


def checked_record(filename):
    p=Path(filename)
    record=json.loads(p.read_text(encoding='utf-8'))
    if record['schema']!='ems-memory-v1' or record['files']!=code_hashes():
        raise RuntimeError('Memory amendment code changed; preserve it and create an explicitly versioned revision')
    if file_hash(p.parent/'protocol.json')!=record['base_protocol_sha256']:
        raise RuntimeError('Original frozen protocol changed; no memory-only override permitted')
    if not isinstance(record['chunk_columns'],int) or record['chunk_columns']<1:
        raise RuntimeError('Invalid memory amendment chunk size')
    return record


def initialize(c,columns):
    import ems_experiments as w
    w.verify(c)
    if not 1<=columns<=4096:raise ValueError('Use 1..4096 columns per diagnostic block')
    p=w.path(c['output_root'])/AMENDMENT
    if p.exists():
        r=checked_record(p)
        if r['chunk_columns']!=columns:raise RuntimeError('Amendment chunk size already frozen; do not overwrite')
        print('Existing memory amendment verified:',p,flush=True);return
    w.write_new(p,dict(schema='ems-memory-v1',created_utc=w.stamp(),chunk_columns=columns,
        base_protocol_sha256=file_hash(p.parent/'protocol.json'),files=code_hashes(),
        cause='NumPy float64 allocation failure during post-sampling physical verification',
        modification='Full spatial population processed in bounded column blocks; weighted sufficient sums pooled before roots',
        unchanged=['trained weights','forecast and denoising','50-member production settings','QC and source restrictions',
            'masks and support digests','float64 thermodynamics','score formulas','calibration fitting and selection gates'],
        precision='Only floating-point spatial reduction order can differ; not bitwise equivalence',
        resource_policy=dict(native_cpu_threads=1,malloc_arena_max_in_slurm=2,requested_node_memory='128G; actual limits must be checked in job log'),
        local_test_scope='Point, raw, marginal and joint transforms, missing layers, empty regions, 50 members/15 levels; not global HPC validation'))
    print('Memory amendment created; original protocol untouched:',p,flush=True)


def resource_info():
    probe=ROOT/'ems_resource_probe/detect_resources.py'
    print('BEGIN effective resources (read-only; identifiers redacted)',flush=True)
    result=subprocess.run([sys.executable,str(probe),'--skip-accelerators'],cwd=ROOT,check=False)
    if result.returncode:print('Resource probe partial/unavailable; do not infer unlimited memory',flush=True)
    try:
        import resource
        limits={}
        for name in ('RLIMIT_AS','RLIMIT_DATA','RLIMIT_STACK'):
            if hasattr(resource,name):
                values=resource.getrlimit(getattr(resource,name))
                limits[name]=['infinity' if x==resource.RLIM_INFINITY else x for x in values]
        print('Read-only process limits [soft,hard], bytes:',json.dumps(limits),flush=True)
    except (ImportError,OSError,ValueError):print('Process address-space limits unavailable',flush=True)
    print('END effective resources',flush=True)


def install_workflow(c):
    """Return a restore function, useful in tests and embedded diagnostics."""
    import ems_experiments as w
    root=w.path(c['output_root']);amend=root/AMENDMENT
    checked_record(amend);digest=file_hash(amend)
    old_dependencies,old_command,old_run=w.dependencies,w.build_command,w.run_case
    def check():
        if file_hash(amend)!=digest:raise RuntimeError('Memory amendment changed during execution')
        checked_record(amend)
    def dependencies(t,config):
        check()
        if w.path(config['output_root']).resolve()!=root.resolve():raise RuntimeError('Memory amendment output-root mismatch')
        result=old_dependencies(t,config)
        mark=w.folder(t,config)/'completed.json'
        if not mark.exists():
            smoke_mark=w.folder(t,config,True)/'completed.json'
            if smoke_mark.exists():mark=smoke_mark
        if mark.exists():
            prior=w.read(mark)['dependencies']
            if AMENDMENT not in prior:return result # genuine pre-amendment completion, still hash-verified by original code
        return dict(result,**{AMENDMENT:digest})
    def command(t,config,out,smoke=False):
        cmd=old_command(t,config,out,smoke)
        if cmd[:3]!=[sys.executable,'-u','ems_observer.py']:raise RuntimeError('Unexpected original child command')
        return [sys.executable,'-u','ems_lowmem.py','child','--memory-amendment',str(amend),*cmd[3:]]
    def run(t,config,smoke=False):
        try:return old_run(t,config,smoke)
        except RuntimeError:
            task=dict(t,days=2,members=4,steps=4) if smoke else t
            attempts=sorted(w.folder(task,config,smoke).glob('attempt_*/execution.log'))
            if attempts:
                print('Child failure, last 60 log lines:',attempts[-1],file=sys.stderr,flush=True)
                with attempts[-1].open(encoding='utf-8',errors='replace') as f:
                    for line in deque(f,maxlen=60):print(line,end='',file=sys.stderr)
            raise
    w.dependencies,w.build_command,w.run_case=dependencies,command,run
    def restore():w.dependencies,w.build_command,w.run_case=old_dependencies,old_command,old_run
    return restore


def child(arguments):
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--memory-amendment',required=True)
    a,rest=p.parse_known_args(arguments);record=checked_record(a.memory_amendment)
    resource_info()
    q=argparse.ArgumentParser(add_help=False);q.add_argument('--output-dir',required=True)
    target,_=q.parse_known_args(rest);out=Path(target.output_dir)
    # This file is covered by the original completion output hashes.
    with (out/'memory_runtime.json').open('x',encoding='utf-8') as f:
        json.dump(dict(amendment=str(a.memory_amendment),amendment_sha256=file_hash(a.memory_amendment),
            chunk_columns=record['chunk_columns'],files=record['files'],
            global_points_not_subsampled=True,forecast_unchanged=True),f,indent=2)
    import oe_joint_diagnostics as diagnostics
    from ems_chunked_diagnostics import ChunkedJointObserver
    from ems_observer import main as observed
    original=diagnostics.JointObserver;saved=sys.argv
    diagnostics.JointObserver=partial(ChunkedJointObserver,chunk_columns=record['chunk_columns'])
    sys.argv=['ems_observer.py',*rest]
    try:observed()
    finally:diagnostics.JointObserver=original;sys.argv=saved


def main():
    # Apply before numpy/torch imports. This is a resource policy, not a new
    # forecast setting. Native CPU reduction roundoff is covered by parity tests.
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    if len(sys.argv)>1 and sys.argv[1]=='child':child(sys.argv[2:]);return
    import ems_experiments as w
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('action',choices=('init','preflight','status','plan','run','batch','report','select','fit','decide','evaluate','freeze-final'))
    p.add_argument('--config',default='configs/ems_v1_legacy.json')
    p.add_argument('--chunk-columns',type=int,default=512)
    a,rest=p.parse_known_args();c=w.settings(a.config)
    if a.action=='init':initialize(c,a.chunk_columns);return
    if a.action=='plan':
        saved=sys.argv;sys.argv=['ems_experiments.py','plan','--config',a.config,*rest]
        try:w.main()
        finally:sys.argv=saved
        return
    w.verify(c);record=checked_record(w.path(c['output_root'])/AMENDMENT)
    print('Memory-only amendment verified; columns/block:',record['chunk_columns'],flush=True)
    restore=install_workflow(c);saved=sys.argv
    try:
        if a.action=='evaluate':
            import ems_evaluate
            sys.argv=['ems_evaluate.py','--config',a.config,*rest];ems_evaluate.main()
        else:
            sys.argv=['ems_experiments.py',a.action,'--config',a.config,*rest];w.main()
    finally:restore();sys.argv=saved


if __name__=='__main__':main()
