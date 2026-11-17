"""Configure a new checkout and optionally install trusted released model artifacts."""
import argparse, hashlib, json, shutil
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--glorys-root',type=Path);p.add_argument('--argo-root',type=Path)
    p.add_argument('--install-models',action='store_true');a=p.parse_args()
    data=a.data_root.resolve()
    if not (data/'models').is_dir():raise FileNotFoundError(data/'models')
    for f in (ROOT/'configs').glob('*.yaml'):
        c=yaml.safe_load(f.read_text())
        if a.glorys_root and 'data' in c:c['data']['root']=str(a.glorys_root.resolve())
        f.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
    for f in (ROOT/'configs').glob('*.json'):
        c=json.loads(f.read_text())
        if a.argo_root:
            for key in ('raw_argo_root','split_root','argo_split_root'):
                if key in c:c[key]=str(a.argo_root.resolve() if key=='raw_argo_root' else a.argo_root.resolve()/'argo_derived'/'paper_split_v1')
        f.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    if a.install_models:
        for src in (data/'models').rglob('*'):
            if not src.is_file():continue
            dst=ROOT/src.relative_to(data/'models');dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists() and sha(dst)!=sha(src):raise FileExistsError('Refusing to overwrite '+str(dst))
            if not dst.exists():shutil.copy2(src,dst)
    (ROOT/'release_paths.json').write_text(json.dumps({'data_root':str(data),
      'glorys_root':str(a.glorys_root.resolve()) if a.glorys_root else None,
      'argo_root':str(a.argo_root.resolve()) if a.argo_root else None},indent=2)+'\n')
    print('Configured checkout. Historical protocol hashes were not modified; use a new output directory for reruns.')


if __name__=='__main__':main()
