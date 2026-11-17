"""Differentiable residual DDIM and free-running fine-tuning utilities."""
from __future__ import annotations
import math
from datetime import date
import torch
from torch.utils.checkpoint import checkpoint
from .diffusion import extract


def make_condition(states, masks, ordinal, tensors):
    normalized=[torch.where(m,(x-tensors['state_mean'])/tensors['state_std'],0.0)
                for x,m in zip(states,masks)]
    anchor=date.fromordinal(int(ordinal))
    angle=2*math.pi*(anchor.timetuple().tm_yday-1)/365.2425
    h,w=states[-1].shape[-2:]
    seasonal=states[-1].new_tensor([math.sin(angle),math.cos(angle)]).view(1,2,1,1).expand(1,2,h,w)
    return torch.cat(normalized+[tensors['static_mask'],tensors['spatial_features'],seasonal],dim=1)


def clamp_physical(x, levels):
    return torch.cat((x[:,:levels].clamp(-3,40),x[:,levels:].clamp(0,42)),dim=1)


def deterministic_step(model, states, masks, ordinal, tensors, levels):
    condition=make_condition(states,masks,ordinal,tensors)
    shape=states[-1].shape
    zeros=condition.new_zeros(shape)
    t=torch.zeros(1,device=condition.device,dtype=torch.long)
    # The persistence term cancels increment_mean in the existing direct model.
    forecast=states[-1]+model(zeros,condition,t).float()*tensors['increment_std']
    return torch.where(masks[-1],clamp_physical(forecast,levels),0.0),condition


def differentiable_ddim(model,diffusion,condition,channels,mask,members,steps,
                        checkpoint_model=True,initial_noise=None,antithetic=True):
    """Eta=0 sampler with parameter gradients; same clean clamp as production.

    Antithetic pairs reduce the training mean estimator's noise. Validation
    instead uses independent draws to match production sampling.
    """
    if condition.shape[0]!=1 or members<2 or steps<2:
        raise ValueError('Require batch=1, members>=2, sampling_steps>=2')
    cond=condition.expand(members,-1,-1,-1)
    shape=(members,channels,*condition.shape[-2:])
    if initial_noise is not None:
        if tuple(initial_noise.shape)!=shape:raise ValueError('Wrong initial_noise shape')
        x=initial_noise.clone()
    elif antithetic:
        noise=torch.randn(((members+1)//2,channels,*condition.shape[-2:]),device=condition.device)
        x=torch.cat((noise,-noise),dim=0)[:members]
    else:
        x=torch.randn(shape,device=condition.device)
    schedule=torch.linspace(diffusion.timesteps-1,0,min(steps,diffusion.timesteps),device=x.device).round().long()
    schedule=torch.unique_consecutive(schedule)
    for index,value in enumerate(schedule):
        t=torch.full((members,),int(value),device=x.device,dtype=torch.long)
        if checkpoint_model and torch.is_grad_enabled():
            predicted=checkpoint(model,x,cond,t,use_reentrant=False).float()
        else:
            predicted=model(x,cond,t).float()
        alpha=extract(diffusion.alpha_bar,t,x.ndim)
        clean=((x-torch.sqrt(1-alpha)*predicted)/torch.sqrt(alpha)).clamp(-8,8)
        if index==len(schedule)-1:
            x=clean
        else:
            next_t=torch.full_like(t,int(schedule[index+1]))
            next_alpha=extract(diffusion.alpha_bar,next_t,x.ndim)
            x=torch.sqrt(next_alpha)*clean+torch.sqrt(1-next_alpha)*predicted
    return x*mask


def variable_mse(prediction,truth,valid,latitude_weights,levels):
    error=torch.where(valid,prediction-truth,0.0)
    result=[]
    for sl in (slice(0,levels),slice(levels,2*levels)):
        w=valid[:,sl].float()*latitude_weights
        result.append((error[:,sl].square()*w).sum()/w.sum().clamp_min(1.0))
    return torch.stack(result)


def mean_training_loss(forecast,truth,reference,valid,latitude_weights,levels,
                       temperature_scale=0.5,salinity_scale=0.1):
    scales=forecast.new_tensor([temperature_scale,salinity_scale])**2
    error=variable_mse(forecast,truth,valid,latitude_weights,levels)
    ref_error=variable_mse(reference,truth,valid,latitude_weights,levels).detach()
    mean_loss=(error/scales).mean()
    # Compare with a separately rolled deterministic trajectory, not a
    # deterministic step fed the candidate's already changed history.
    non_degradation=(torch.relu(error-ref_error)/scales).mean()
    return mean_loss,non_degradation


def tensors_for(dataset,stats,residual_mean,residual_std,device):
    result={key:torch.as_tensor(stats[key],device=device,dtype=torch.float32).view(1,-1,1,1)
            for key in ('state_mean','state_std','increment_std')}
    result.update(residual_mean=torch.as_tensor(residual_mean,device=device).view(1,-1,1,1),
        residual_std=torch.as_tensor(residual_std,device=device).view(1,-1,1,1),
        static_mask=torch.as_tensor(dataset.static_mask,device=device).unsqueeze(0),
        spatial_features=torch.as_tensor(dataset.spatial_features,device=device).unsqueeze(0),
        latitude_weight=torch.as_tensor(dataset.latitude_weights,device=device).unsqueeze(0))
    return result
