"""Cheap diagonal-Gaussian residual-conditioning baseline; NOT the main method.

This uses an explicit diagonal covariance, not an EnKF ensemble covariance.
Training residual statistics define the prior scale; R is tuned on development
observations only. No reference state or held-out profile is supplied here.
"""
import torch
import math


class DiagonalGaussianResidual(torch.nn.Module):
    def __init__(self, temperature_error=.5, salinity_error=.1):
        super().__init__()
        if not all(math.isfinite(x) and x > 0 for x in (temperature_error,salinity_error)):
            raise ValueError('Observation-error assumptions must be positive')
        self.temperature_error = float(temperature_error)
        self.salinity_error = float(salinity_error)

    def ddim_sample(self, model, condition, output_channels, mask=None,
                    sampling_steps=None, eta=0., initial_noise=None, **kwargs):
        shape = (len(condition), output_channels, *condition.shape[-2:])
        noise = torch.randn(shape, device=condition.device, dtype=condition.dtype) if initial_noise is None else initial_noise.clone()
        return noise if mask is None else torch.where(mask.bool(), noise, torch.zeros_like(noise))

    def ddim_sample_observation_guided(self, model, condition, output_channels,
            background_state, residual_mean, residual_std, observation,
            observation_mask, levels, residual_scale=1., mask=None,
            initial_noise=None, **kwargs):
        noise = self.ddim_sample(model, condition, output_channels, mask=mask, initial_noise=initial_noise)
        if residual_scale == 0:
            return noise
        scale = residual_scale * residual_std
        r = torch.empty_like(background_state)
        r[:, :levels] = self.temperature_error**2
        r[:, levels:] = self.salinity_error**2
        k = scale.square() / (scale.square() + r)
        # Conditional normal draw with the same initialization as the prior.
        prior_mean = background_state + residual_scale * residual_mean
        posterior_z = (k * (observation - prior_mean) / scale.clamp_min(1e-12)
                       + torch.sqrt(1-k) * noise)
        updated = torch.where(observation_mask.bool(), posterior_z, noise)
        return updated if mask is None else torch.where(mask.bool(), updated, torch.zeros_like(updated))
