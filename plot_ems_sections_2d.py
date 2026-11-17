"""Plot genuine longitude-depth and latitude-depth T/S sections.

Accept only verified, completed section exports. No interpolation across land,
quantile color clipping, missing-value imputation, or performance-based selection.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

LABEL={'deterministic':'Deterministic','selected':'SA-Diff','guided_raw':'Argo-Diff'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_verified(root,method,lead):
    candidates=sorted((root/method).glob('attempt_*'),reverse=True)
    for folder in candidates:
        complete=folder/'section_complete.json'
        if not complete.is_file():continue
        record=json.loads(complete.read_text(encoding='utf-8'))
        name=f'lead{lead:03d}_sections.npz'
        entries=[r for r in record['files'] if r['path']==name]
        if len(entries)!=1:continue
        file=folder/name
        if sha(file)!=entries[0]['sha256']:raise ValueError('Section checksum mismatch: '+str(file))
        with np.load(file,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        if not bool(a['replay_verified']) or str(a['method'])!=method or int(a['lead'])!=lead:
            raise ValueError('Unverified or mismatched export')
        recipe=json.loads((folder/'section_recipe.json').read_text(encoding='utf-8'))
        if str(a['source_recipe_sha256'])!=recipe['source_recipe_sha256']:
            raise ValueError('Recipe binding mismatch')
        return a,file,recipe
    raise FileNotFoundError(f'No verified completed section for {method}, lead {lead}')

def check_pair(arrays,recipes):
    base=arrays[0];task=recipes[0]['task']
    for a,r in zip(arrays[1:],recipes[1:]):
        for key in ('date','lead','depth_m','zonal_longitude','zonal_latitude',
                    'meridional_latitude','meridional_longitude','zonal_valid','meridional_valid'):
            np.testing.assert_array_equal(a[key],base[key],err_msg='Mismatched '+key)
        for key in ('zonal_truth','meridional_truth'):
            np.testing.assert_array_equal(a[key],base[key],err_msg='Mismatched reference')
        for key in ('phase','year','scenario','season','seed','members','steps'):
            if r['task'][key]!=task[key]:raise ValueError('Unpaired case: '+key)
    for a in arrays:
        for section in ('zonal','meridional'):
            mask=a[section+'_valid']
            if not np.isfinite(a[section+'_prediction'][mask]).all():
                raise ValueError('Invalid forecast on reference support')

def edges(x):
    x=np.asarray(x,float)
    if len(x)<2 or np.any(np.diff(x)<=0):raise ValueError('Coordinate must increase')
    return np.r_[x[0]-(x[1]-x[0])/2,(x[1:]+x[:-1])/2,x[-1]+(x[-1]-x[-2])/2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='runs/ems_sections_2d_v1')
    p.add_argument('--methods',nargs='+',default=['deterministic','selected'],choices=LABEL)
    p.add_argument('--lead',type=int,default=90)
    p.add_argument('--output',help='New output directory; refuses overwrite')
    args=p.parse_args()
    if args.methods[0]!='deterministic' or len(set(args.methods))!=len(args.methods):
        p.error('Use deterministic first and list each method once')
    root=Path(args.root)
    loaded=[load_verified(root,m,args.lead) for m in args.methods]
    arrays=[r[0] for r in loaded];recipes=[r[2] for r in loaded]
    check_pair(arrays,recipes)
    out=Path(args.output) if args.output else root/f'figures_day{args.lead:03d}'
    out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
        'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    nrows=1+len(arrays);outputs=[];scores=[];norms={}
    for section,axis in (('zonal','zonal_longitude'),('meridional','meridional_latitude')):
        x=arrays[0][axis];depth=arrays[0]['depth_m'];xe=edges(x);de=edges(depth)
        de[0]=max(0.,de[0]);de[-1]=depth[-1]
        truth=arrays[0][section+'_truth'];valid=arrays[0][section+'_valid']
        coordinate=float(arrays[0]['zonal_latitude' if section=='zonal' else 'meridional_longitude'])
        error_ranges=[];field_ranges=[]
        for v in range(2):
            all_fields=np.concatenate([truth[v][valid[v]]]+[a[section+'_prediction'][v][valid[v]] for a in arrays])
            if not len(all_fields):raise ValueError('Section has no valid reference cells')
            lo,hi=float(all_fields.min()),float(all_fields.max())
            if hi==lo:hi=lo+1e-6
            field_ranges.append((lo,hi))
            limit=max(float(np.max(abs(a[section+'_prediction'][v][valid[v]]-truth[v][valid[v]]))) for a in arrays)
            error_ranges.append(max(limit,1e-6))
        norms[section]=dict(fields=field_ranges,symmetric_error_limits=error_ranges)
        for mode in ('fields','reference_and_errors'):
            fig=plt.figure(figsize=(7.2,2.0*nrows+.6),layout='constrained')
            gs=fig.add_gridspec(nrows,4,width_ratios=[1,.04,1,.04])
            for v,var in enumerate(('Temperature','Salinity')):
                images=[];axes=[]
                for row in range(nrows):
                    ax=fig.add_subplot(gs[row,2*v]);axes.append(ax)
                    if row==0:values=truth[v];name='GLORYS'
                    else:
                        values=arrays[row-1][section+'_prediction'][v]
                        name=LABEL[args.methods[row-1]]
                        if mode=='reference_and_errors':values=values-truth[v];name+=' minus GLORYS'
                    signed=mode=='reference_and_errors' and row>0
                    cmap=plt.get_cmap('RdBu_r' if signed else ('plasma' if v==0 else 'viridis')).with_extremes(bad='#D8D8D8')
                    norm=TwoSlopeNorm(vmin=-error_ranges[v],vcenter=0,vmax=error_ranges[v]) if signed else Normalize(*field_ranges[v])
                    masked=np.ma.array(values,mask=~valid[v])
                    image=ax.pcolormesh(xe,de,masked,cmap=cmap,norm=norm,shading='flat')
                    images.append(image)
                    ax.set_ylim(depth[-1],0);ax.set_xlim(x[0],x[-1])
                    ax.set_title(f'{chr(97+2*row+v)}  {name}: {var.lower()}',loc='left')
                    if v==0:ax.set_ylabel('Depth (m)')
                    if row==nrows-1:ax.set_xlabel('Longitude (°E)' if section=='zonal' else 'Latitude (°N)')
                    else:ax.tick_params(labelbottom=False)
                    ax.set_facecolor('#D8D8D8')
                label='Temperature (°C)' if v==0 else 'Practical Salinity'
                if mode=='fields':
                    cax=fig.add_subplot(gs[:,2*v+1]);fig.colorbar(images[0],cax=cax,label=label)
                else:
                    cax=fig.add_subplot(gs[0,2*v+1]);fig.colorbar(images[0],cax=cax,label=label)
                    cax=fig.add_subplot(gs[1:,2*v+1]);fig.colorbar(images[1],cax=cax,label=('T error (°C)' if v==0 else 'S error'))
            line=f'{coordinate:g}°N' if section=='zonal' else f'{coordinate:g}°E'
            fig.suptitle(f'{section.capitalize()} section at {line} | {str(arrays[0]["date"])} | cycle day {args.lead}',fontsize=10)
            stem=out/f'{section}_{mode}_day{args.lead:03d}'
            fig.savefig(stem.with_suffix('.pdf'))
            fig.savefig(stem.with_suffix('.png'),dpi=500)
            plt.close(fig)
            outputs.extend([stem.with_suffix('.pdf'),stem.with_suffix('.png')])
        for a,method in zip(arrays,args.methods):
            for v,var in enumerate(('temperature','salinity')):
                diff=(a[section+'_prediction'][v]-truth[v])[valid[v]].astype(float)
                scores.append(dict(section=section,method=method,variable=var,lead=args.lead,
                    date=str(a['date']),count=len(diff),rmse=float(np.sqrt(np.mean(diff**2))),
                    bias=float(np.mean(diff)),mae=float(np.mean(abs(diff)))))
    with (out/'section_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
    manifest=dict(inputs=[dict(path=str(item[1]),sha256=sha(item[1])) for item in loaded],
        outputs=[dict(path=p.name,sha256=sha(p)) for p in outputs],
        plotting_code_sha256=sha(__file__),methods=args.methods,lead=args.lead,normalization=norms,
        interpolation='None. Native grid cells are displayed; gray denotes land/missing reference.',
        error_definition='Prediction minus GLORYS; symmetric full-range color limits shared by methods.',
        field_limits='Full range across reference and all methods, shared for each variable and section.',
        score_definition='Unweighted over stored valid section grid points, not volume-weighted or global RMSE.',
        units='GLORYS potential temperature (degrees C) and Practical Salinity.',
        geographic_selection='Fixed 17.25 N zonal line and 135 E meridional line by default; actual native coordinates in exports.',
        human_verified=False)
    (out/'section_figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('REAL SECTION FIGURES:',out,flush=True)

if __name__=='__main__':main()
