"""Prepare a relocated 90-day archived recipe; execute only when --run is explicit."""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD_PROJECT='/fs6/home/yangjh16/zs_yangjh16/diffArgo/'
OLD_ARGO='/fs6/home/yangjh16/zs_yangjh16/Argo'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--argo-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    data=a.data_root.resolve();src=a.recipe.resolve();output=a.output_dir.resolve()
    recipe=json.loads(src.read_text());old=recipe['command']
    if 'ems_quick_observer.py' in old or '--quick-protocol' in old:
        raise ValueError('Quick-screen recipes are protocol-bound; initialize a fresh quick protocol instead of bypassing its hashes.')
    if output.exists():raise FileExistsError('Choose a new output directory')
    output.mkdir(parents=True)
    command=[sys.executable,*old[1:]]
    def relocated(value):
        if value.startswith(OLD_ARGO):return str(a.argo_root.resolve())+value[len(OLD_ARGO):]
        if value.startswith(OLD_PROJECT):
            relative=Path(value[len(OLD_PROJECT):])
            if relative.name in ('observer.json','observation_options.json','stability_options.json'):
                return str(output/relative.name)
            return str(ROOT/relative)
        return value
    for i,value in enumerate(command):command[i]=relocated(value)
    command[command.index('--output-dir')+1]=str(output)
    for name in ('observer.json','observation_options.json','stability_options.json'):
        f=src.parent/name
        if f.exists():
            obj=json.loads(f.read_text())
            def walk(x):
                if isinstance(x,str):return relocated(x)
                if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
                if isinstance(x,list):return [walk(v) for v in x]
                return x
            (output/name).write_text(json.dumps(walk(obj),indent=2)+'\n')
    record={'source_recipe':str(src),'task':recipe['task'],'command':command,
            'relocation_only':True,'historical_protocol_revalidated':False,'executed':a.run}
    (output/'relocated_recipe.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if a.run:subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
