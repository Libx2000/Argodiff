"""Stream hydrographic, exact GSW and normalized-buoyancy diagnostics."""
import argparse, json, sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--stable-observer',required=True)
    args,rest=p.parse_known_args()
    definition=json.loads(Path(args.stable_observer).read_text(encoding='utf-8'))
    q=argparse.ArgumentParser(add_help=False);q.add_argument('--output-dir',required=True)
    out,_=q.parse_known_args(rest)
    import paper_metrics
    from oe_joint_diagnostics import JointObserver
    from oe_buoyancy import BuoyancyObserver
    from oe_stable_cycle import main as cycle
    observer=JointObserver(definition['config'],definition['candidates'],out.output_dir,'validate')
    buoyancy=BuoyancyObserver(definition['config'],out.output_dir,definition['candidates'])
    original=paper_metrics.evaluate_day
    def evaluate(dataset,target,lead,analysis,ensemble,holdout_root,output_dir,deterministic=False):
        original(dataset,target,lead,analysis,ensemble,holdout_root,output_dir,deterministic)
        observer(dataset,target,lead,analysis,ensemble,holdout_root,output_dir,deterministic)
        buoyancy(dataset,target,lead,analysis,ensemble,deterministic)
    saved=sys.argv;paper_metrics.evaluate_day=evaluate
    sys.argv=[sys.argv[0]]+rest
    try:cycle()
    finally:paper_metrics.evaluate_day=original;sys.argv=saved


if __name__=='__main__':main()
