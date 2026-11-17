"""Add a post-sampling observer without editing the frozen cycling engine."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--oe-config',required=True)
    config_args,cycle_args=parser.parse_known_args()
    import paper_metrics
    from oe_physics import PhysicsObserver
    from argo_guided_cycle import main as original_cycle
    out_parser=argparse.ArgumentParser(add_help=False)
    out_parser.add_argument('--output-dir',required=True)
    out,_=out_parser.parse_known_args(cycle_args)
    config=json.loads(Path(config_args.oe_config).read_text(encoding='utf-8'))
    observer=PhysicsObserver(config,Path(out.output_dir))
    evaluate_original=paper_metrics.evaluate_day
    def evaluate_with_physics(*args,**kwargs):
        evaluate_original(*args,**kwargs)
        observer(*args,**kwargs)
    paper_metrics.evaluate_day=evaluate_with_physics
    saved_argv=sys.argv
    sys.argv=[sys.argv[0]]+cycle_args
    try: original_cycle()
    finally:
        paper_metrics.evaluate_day=evaluate_original
        sys.argv=saved_argv

if __name__=='__main__':main()
