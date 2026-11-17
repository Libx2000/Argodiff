"""Verify the release SHA-256 inventory without importing scientific libraries."""
import argparse, hashlib, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args()
    root=a.bundle.resolve();rows=json.loads((root/'file_manifest.json').read_text(encoding='utf-8'))
    errors=[]
    for row in rows:
        f=(root/row['path']).resolve()
        if not f.is_relative_to(root):raise ValueError('Manifest path escapes bundle')
        if not f.is_file():errors.append({'file':row['path'],'error':'missing'});continue
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=row['sha256']:errors.append({'file':row['path'],'error':'checksum'})
    print(json.dumps({'checked':len(rows),'errors':errors},indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
