"""Compute training-only deterministic forecast residual statistics."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffargo.config import load_config
from diffargo.data import GlorysSequenceDataset, load_stats
from diffargo.model import build_model
from diffargo.utils import EMA, seed_everything


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--deterministic-checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--num-workers',type=int,default=4); args=p.parse_args()
    cfg=load_config(args.config); seed_everything(int(cfg['seed'])); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats=load_stats(cfg['data']['stats_path']); ds=GlorysSequenceDataset(cfg['data'],'train',stats)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=device.type=='cuda',persistent_workers=args.num_workers>0)
    ckpt=torch.load(args.deterministic_checkpoint,map_location=device,weights_only=False)
    model=build_model(cfg,ds.state_channels,ds.condition_channels).to(device); model.load_state_dict(ckpt['model'])
    if 'ema' in ckpt:
        ema=EMA(model,float(cfg['train'].get('ema_decay',0.999))); ema.load_state_dict(ckpt['ema']); ema.copy_to(model)
    model.eval(); inc_mean=torch.as_tensor(stats['increment_mean'],device=device).view(1,-1,1,1); inc_std=torch.as_tensor(stats['increment_std'],device=device).view(1,-1,1,1)
    count=torch.zeros(ds.state_channels,device=device,dtype=torch.float64); total=count.clone(); square=count.clone()
    with torch.inference_mode():
        for batch in tqdm(loader,desc='deterministic residual statistics'):
            condition=batch['condition'].to(device,non_blocking=True); target=batch['target'].to(device,non_blocking=True); mask=batch['mask'].to(device,non_blocking=True).bool()
            zeros=condition.new_zeros((1,ds.state_channels,condition.shape[-2],condition.shape[-1])); timestep=torch.zeros(1,device=device,dtype=torch.long)
            persistence=-inc_mean/inc_std; det_norm=persistence+model(zeros,condition,timestep)
            residual=(target-det_norm)*inc_std
            safe=torch.where(mask,residual,0.0).double(); count+=mask.sum((0,2,3)); total+=safe.sum((0,2,3)); square+=safe.square().sum((0,2,3))
    mean=total/count.clamp_min(1); var=square/count.clamp_min(1)-mean.square(); std=torch.sqrt(var.clamp_min(1e-10))
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,residual_mean=mean.cpu().numpy().astype(np.float32),residual_std=std.cpu().numpy().astype(np.float32),count=count.cpu().numpy().astype(np.int64),source_checkpoint=str(Path(args.deterministic_checkpoint).resolve()),train_years=np.asarray(cfg['data']['train_years'],dtype=np.int32))
    print(f'saved: {output}',flush=True); print(f'residual std range: {std.min().item():.6g} .. {std.max().item():.6g}',flush=True)

if __name__=='__main__': main()
