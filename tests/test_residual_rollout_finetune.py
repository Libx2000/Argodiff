"""CPU tests for sampler gradients, inference parity, and strict score gates."""
import csv,tempfile,unittest,json
from pathlib import Path
import torch
from torch import nn
from diffargo.diffusion import GaussianDiffusion
from diffargo.residual_rollout import differentiable_ddim,mean_training_loss
from residual_rollout_experiment import evaluate_gate,MODES,SEASONS


class TinyNoise(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.01))
    def forward(self,x,condition,t):
        return x*self.weight+.0001*condition[:,:x.shape[1]]


class FineTuneTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_sampler_matches_existing_inference_ddim(self):
        model=TinyNoise();diffusion=GaussianDiffusion(16)
        condition=torch.randn(1,3,4,4);mask=torch.ones(1,2,4,4)
        noise=torch.randn(2,2,4,4)*.001
        with torch.no_grad():
            expected=diffusion.ddim_sample(model,condition.expand(2,-1,-1,-1),2,
                         mask=mask,sampling_steps=4,eta=0,initial_noise=noise)
            actual=differentiable_ddim(model,diffusion,condition,2,mask,2,4,initial_noise=noise)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_sampler_backpropagates_parameter_gradient(self):
        model=TinyNoise();diffusion=GaussianDiffusion(16)
        condition=torch.ones(1,3,4,4);mask=torch.ones(1,2,4,4)
        noise=torch.full((2,2,4,4),.001)
        result=differentiable_ddim(model,diffusion,condition,2,mask,2,4,initial_noise=noise)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(model.weight.grad))
        self.assertGreater(abs(model.weight.grad.item()),0)

    def test_guard_detects_degradation_and_ignores_masked_nan(self):
        truth=torch.zeros(1,2,2,2);reference=truth+.1;forecast=truth+.2
        valid=torch.ones_like(truth,dtype=torch.bool);valid[:,:,0,0]=False
        forecast[:,:,0,0]=float('nan')
        mean,guard=mean_training_loss(forecast,truth,reference,valid,torch.ones(1,1,2,1),1)
        self.assertTrue(torch.isfinite(mean));self.assertGreater(guard.item(),0)

    def test_gate_rejects_when_only_one_variable_improves(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            fields=['date','lead','variable','depth_index','count','weight','rmse','crps']
            for mode in MODES:
                for season in SEASONS:
                    out=base/mode/season;out.mkdir(parents=True)
                    (out/'recipe.json').write_text(json.dumps({'scale':1,'fingerprints':{'checkpoint':'synthetic'}}))
                    for name in ['paper_grid_metrics.csv','paper_holdout_metrics.csv']:
                        with (out/name).open('w',newline='') as f:
                            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                            for lead in range(1,91):
                                for variable in ['temperature','salinity']:
                                    error=1.0 if mode=='deterministic' else (.9 if variable=='temperature' else 1.1)
                                    for depth in range(-1,15):
                                        writer.writerow(dict(date=f'{season}-{lead}',lead=lead,variable=variable,
                                            depth_index=depth,count=10,weight=10,rmse=error,crps=.5))
            gate=evaluate_gate(base)
            self.assertFalse(gate['any_accepted'])
            for result in gate['methods'].values():
                self.assertTrue(result['details']['glorys']['temperature']['passed'])
                self.assertFalse(result['details']['glorys']['salinity']['passed'])
            (base/'guided'/'autumn'/'recipe.json').write_text(json.dumps({'scale':1,'fingerprints':{'checkpoint':'different'}}))
            with self.assertRaisesRegex(ValueError,'mixed results'):
                evaluate_gate(base)


if __name__=='__main__':unittest.main()
