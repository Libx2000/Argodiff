"""Run archived figure scripts in a new workspace, leaving the data bundle untouched."""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    source=a.data_root.resolve()/'paper';out=a.output_dir.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory: '+str(out))
    out.mkdir(parents=True)
    for pattern in ('*.csv','*.json'):
        for src in source.glob(pattern):shutil.copy2(src,out/src.name)
    for name in ('sections_20261112','availability_20261116'):
        shutil.copytree(source/'source_data'/name,out/'source_data'/name)
    for src in (ROOT/'figures').glob('*.py'):shutil.copy2(src,out/src.name)
    shutil.copytree(ROOT/'figures'/'figure8_support',out/'figure8_support')
    jobs=[['plot_figure2_zoom.py','--force'],['plot_figure3.py','--force'],['plot_figure4.py'],
          ['plot_figure5.py'],['plot_sections_ngeo.py','--force'],['plot_figure8.py']]
    env=dict(os.environ,MPLBACKEND='Agg')
    for job in jobs:subprocess.run([sys.executable,*job],cwd=out,env=env,check=True)
    missing=[str(i) for i in range(2,9) if not (out/f'Figure{i}.pdf').is_file()]
    if missing:raise RuntimeError('Missing reconstructed figures: '+', '.join(missing))
    print('Reconstructed Figures 2-8 in',out)
    print('Figure 1 is a supplied schematic, not a numerical experiment.')


if __name__=='__main__':main()
