"""Fine-tune the existing diffusion on its own free-running contexts.

Frozen deterministic backbone, denoising + differentiable ensemble-mean loss,
truncated day-to-day gradients, and independent deterministic rollout scoring.
Training uses 1993-2016; checkpoint selection uses a fixed 2019 short-cycle set.
No test-year data or Argo holdout labels enter training.
"""
from __future__ import annotations
import argparse,csv,json,random,time
from pathlib import Path
from datetime import date,timedelta
import numpy as np
import torch

from diffargo.config import load_config
from diffargo.data import GlorysSequenceDataset,load_stats,inclusive_years
from diffargo.diffusion import build_diffusion,masked_weighted_mse
from diffargo.model import build_model
from diffargo.utils import EMA,atomic_torch_save,seed_everything
from diffargo.residual_rollout import (deterministic_step,differentiable_ddim,
    clamp_physical,mean_training_loss,tensors_for)


def init_model(cfg,dataset,path,device,trainable):
    model=build_model(cfg,dataset.state_channels,dataset.condition_channels).to(device)
    payload=torch.load(path,map_location=device,weights_only=False)
    if trainable and payload.get('objective')!='residual_diffusion':
        raise ValueError('Expected a residual_diffusion source checkpoint')
    model.load_state_dict(payload['model'],strict=True)
    if 'ema' in payload:
        source_ema=EMA(model,float(cfg['train'].get('ema_decay',.999)))
        source_ema.load_state_dict(payload['ema']);source_ema.copy_to(model)
    for p in model.parameters():p.requires_grad_(trainable)
    model.eval()
    return model


def history_for(dataset,anchor,device):
    states,masks=[],[]
    for offset in reversed(range(dataset.context_days)):
        x,m=dataset.archive.read_state(anchor-timedelta(days=offset))
        states.append(torch.from_numpy(np.where(m,x,0).astype(np.float32)).unsqueeze(0).to(device))
        masks.append(torch.from_numpy(m).unsqueeze(0).to(device))
    return states,masks


def append_history(states,masks,forecast):
    return (states+[torch.where(masks[-1],forecast.detach(),0.0)])[1:],(masks+[masks[-1]])[1:]


def validation(model,det,diffusion,dataset,tensors,args,device):
    model.eval();seed_everything(args.seed+8191)
    sums=np.zeros((args.val_days,2,3),dtype=np.float64)
    with torch.no_grad():
        for month in (1,4,7,10):
            anchor=date(2019,month,1)
            states,masks=history_for(dataset,anchor,device)
            reference=[x.clone() for x in states];ref_masks=[m.clone() for m in masks]
            for lead in range(1,args.val_days+1):
                ordinal=anchor.toordinal()+lead-1
                bg,condition=deterministic_step(det,states,masks,ordinal,tensors,dataset.levels)
                ref,_=deterministic_step(det,reference,ref_masks,ordinal,tensors,dataset.levels)
                z=differentiable_ddim(model,diffusion,condition,dataset.state_channels,masks[-1],
                    args.val_members,args.val_steps,checkpoint_model=False,antithetic=False)
                residual=z*tensors['residual_std']+tensors['residual_mean']
                forecast=clamp_physical(bg+args.residual_scale*residual.mean(0,keepdim=True),dataset.levels)
                truth_np,valid_np=dataset.archive.read_state(anchor+timedelta(days=lead))
                truth=torch.from_numpy(np.where(valid_np,truth_np,0).astype(np.float32)).unsqueeze(0).to(device)
                valid=torch.from_numpy(valid_np).unsqueeze(0).to(device)&masks[-1]&ref_masks[-1]
                for vi,sl in enumerate((slice(0,dataset.levels),slice(dataset.levels,None))):
                    w=valid[:,sl].float()*tensors['latitude_weight']
                    err=torch.where(valid[:,sl],forecast[:,sl]-truth[:,sl],0.0)
                    referr=torch.where(valid[:,sl],ref[:,sl]-truth[:,sl],0.0)
                    sums[lead-1,vi]+=[(err.square()*w).sum().item(),(referr.square()*w).sum().item(),w.sum().item()]
                if not torch.isfinite(forecast[masks[-1]]).all():raise FloatingPointError('Nonfinite validation forecast')
                states,masks=append_history(states,masks,forecast)
                reference,ref_masks=append_history(reference,ref_masks,ref)
    pooled=sums.sum(axis=0)
    candidate=np.sqrt(pooled[:,0]/pooled[:,2]);baseline=np.sqrt(pooled[:,1]/pooled[:,2])
    ratio=candidate/baseline
    return dict(temperature_rmse=float(candidate[0]),salinity_rmse=float(candidate[1]),
        deterministic_temperature_rmse=float(baseline[0]),deterministic_salinity_rmse=float(baseline[1]),
        temperature_ratio=float(ratio[0]),salinity_ratio=float(ratio[1]),
        score=float(np.mean(ratio)),passed=bool(np.all(ratio<1-args.min_improvement))),sums


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/residual_diffusion.yaml')
    p.add_argument('--deterministic-checkpoint',default='runs/hpc_rollout_h7/checkpoint_best.pt')
    p.add_argument('--source-checkpoint',default='runs/residual_diffusion_guided/checkpoint_best.pt')
    p.add_argument('--residual-stats',default='artifacts/residual_stats_rollout_h7.npz')
    p.add_argument('--output-dir',default='runs/residual_rollout_v1')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--train-windows',type=int,default=16)
    p.add_argument('--rollout-days',type=int,default=30)
    p.add_argument('--members',type=int,default=2)
    p.add_argument('--mean-steps',type=int,default=4)
    p.add_argument('--val-days',type=int,default=30)
    p.add_argument('--val-members',type=int,default=8)
    p.add_argument('--val-steps',type=int,default=20)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--residual-scale',type=float,default=1.0)
    p.add_argument('--mean-weight',type=float,default=.25)
    p.add_argument('--guard-weight',type=float,default=.5)
    p.add_argument('--min-improvement',type=float,default=.005)
    p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke:a.epochs=1;a.train_windows=1;a.rollout_days=2;a.val_days=2;a.val_members=2;a.val_steps=4
    if min(a.epochs,a.train_windows,a.rollout_days,a.val_days)<1 or min(a.members,a.val_members,a.mean_steps,a.val_steps)<2:
        p.error('Positive window/horizon counts and >=2 members/sampling steps required')
    if a.residual_scale<=0 or not 0<=a.min_improvement<1:p.error('Invalid residual scale or improvement gate')
    cfg=load_config(a.config);seed_everything(a.seed)
    # Keep optimization strictly within the original training years.
    if cfg['data']['train_years'][1]>2016:raise ValueError('Fine-tuning must not use validation/test years')
    stats=load_stats(cfg['data']['stats_path']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=GlorysSequenceDataset(cfg['data'],'train',stats)
    with np.load(a.residual_stats,allow_pickle=False) as f:mean=f['residual_mean'].astype(np.float32);std=f['residual_std'].astype(np.float32)
    if mean.shape!=(dataset.state_channels,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Invalid residual normalization')
    tensors=tensors_for(dataset,stats,mean,std,device)
    det=init_model(cfg,dataset,a.deterministic_checkpoint,device,False)
    model=init_model(cfg,dataset,a.source_checkpoint,device,True)
    diffusion=build_diffusion(cfg).to(device);ema=EMA(model,.99)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    last=output/'checkpoint_last.pt';start_epoch=0;best=float('inf');best_passed=float('inf');rows=[]
    if a.resume:
        payload=torch.load(last,map_location=device,weights_only=False)
        previous=payload['rollout_training']
        if any(previous[k]!=vars(a)[k] for k in vars(a) if k not in ('resume','output_dir')):
            raise ValueError('Resume arguments differ; preserve the same training recipe')
        model.load_state_dict(payload['model']);ema.load_state_dict(payload['ema']);optimizer.load_state_dict(payload['optimizer'])
        start_epoch=payload['epoch']+1;best=payload['best_score'];best_passed=payload['best_passed_score'];rows=payload['history']
    elif last.exists():raise FileExistsError('Run exists; use --resume or choose another output-dir')
    split=set(inclusive_years(cfg['data']['train_years']))
    anchors=[anchor for anchor in dataset.anchors if all((anchor+timedelta(days=d)) in dataset.archive.date_index
             and (anchor+timedelta(days=d)).year in split for d in range(1,a.rollout_days+1))]
    if not anchors:raise ValueError('No complete training trajectories')
    for month in (1,4,7,10):
        anchor=date(2019,month,1)
        if any(anchor+timedelta(days=d) not in dataset.archive.date_index for d in range(-dataset.context_days+1,a.val_days+1)):
            raise ValueError('Missing 2019 validation dates')
    (output/'recipe.json').write_text(json.dumps(vars(a)|{'train_years':cfg['data']['train_years'],'validation_year':2019,'test_used':False},indent=2))
    for epoch in range(start_epoch,a.epochs):
        seed_everything(a.seed+epoch)
        # Curriculum: 7, 14, then 30-day free-running contexts.
        if epoch<a.epochs//3:horizon=min(7,a.rollout_days)
        elif epoch<2*a.epochs//3:horizon=min(14,a.rollout_days)
        else:horizon=a.rollout_days
        selected=random.Random(a.seed+epoch).sample(anchors,min(a.train_windows,len(anchors)))
        totals=np.zeros(3);updates=0;started=time.monotonic()
        for wi,anchor in enumerate(selected):
            states,masks=history_for(dataset,anchor,device)
            reference=[x.clone() for x in states];ref_masks=[m.clone() for m in masks]
            for lead in range(1,horizon+1):
                ordinal=anchor.toordinal()+lead-1
                with torch.no_grad():
                    bg,condition=deterministic_step(det,states,masks,ordinal,tensors,dataset.levels)
                    ref,_=deterministic_step(det,reference,ref_masks,ordinal,tensors,dataset.levels)
                    truth_np,truth_mask_np=dataset.archive.read_state(anchor+timedelta(days=lead))
                    truth=torch.from_numpy(np.where(truth_mask_np,truth_np,0).astype(np.float32)).unsqueeze(0).to(device)
                    valid=torch.from_numpy(truth_mask_np).unsqueeze(0).to(device)&masks[-1]&ref_masks[-1]
                    target=((truth-bg)/a.residual_scale-tensors['residual_mean'])/tensors['residual_std']
                    target=torch.where(valid,target,0.0)
                    t=torch.randint(0,diffusion.timesteps,(1,),device=device)
                    noisy,noise=diffusion.q_sample(target,t)
                model.eval()  # Parameter gradients stay enabled; dropout matches inference.
                optimizer.zero_grad(set_to_none=True)
                predicted=model(noisy,condition,t)
                noise_loss=masked_weighted_mse(predicted,noise,valid.float(),tensors['latitude_weight'])
                z=differentiable_ddim(model,diffusion,condition,dataset.state_channels,masks[-1],a.members,a.mean_steps)
                residual=z*tensors['residual_std']+tensors['residual_mean']
                forecast=clamp_physical(bg+a.residual_scale*residual.mean(0,keepdim=True),dataset.levels)
                mean_loss,guard_loss=mean_training_loss(forecast,truth,ref,valid,tensors['latitude_weight'],dataset.levels)
                loss=noise_loss+a.mean_weight*mean_loss+a.guard_weight*guard_loss
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fine-tuning loss; last completed epoch is preserved')
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if not torch.isfinite(grad_norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();ema.update(model)
                totals+=[noise_loss.item(),mean_loss.item(),guard_loss.item()];updates+=1
                states,masks=append_history(states,masks,forecast)
                reference,ref_masks=append_history(reference,ref_masks,ref)
            print(f'epoch={epoch+1} window={wi+1}/{len(selected)} horizon={horizon} loss={loss.item():.5f}',flush=True)
        backup={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ema.copy_to(model)
        metrics,sums=validation(model,det,diffusion,dataset,tensors,a,device)
        model.load_state_dict(backup)
        row=dict(epoch=epoch+1,horizon=horizon,updates=updates,noise_loss=float(totals[0]/updates),
            mean_loss=float(totals[1]/updates),guard_loss=float(totals[2]/updates),seconds=time.monotonic()-started,**metrics)
        rows.append(row);best=min(best,metrics['score'])
        improved_pass=metrics['passed'] and metrics['score']<best_passed
        if improved_pass:best_passed=metrics['score']
        payload=dict(epoch=epoch,model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),config=cfg,
            objective='residual_diffusion',rollout_training=vars(a),history=rows,best_score=best,best_passed_score=best_passed,
            validation=metrics,source_checkpoint=a.source_checkpoint,source_deterministic=a.deterministic_checkpoint,
            inference_residual_scale=a.residual_scale)
        atomic_torch_save(payload,last)
        inference_payload={k:v for k,v in payload.items() if k not in ('optimizer','history')}
        if metrics['score']<=best:atomic_torch_save(inference_payload,output/'checkpoint_best.pt')
        if improved_pass:atomic_torch_save(inference_payload,output/'checkpoint_passed_short_rollout.pt')
        np.savez_compressed(output/'validation_latest.npz',sums=sums)
        with (output/'training_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
        print(json.dumps(row),flush=True)
    dataset.archive.close()
    print('Fine-tuning finished. Short-cycle pass is preliminary; require four-season 90-day validation before using final test.',flush=True)


if __name__=='__main__':main()
