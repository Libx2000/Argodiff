from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + offset) / (1 + offset) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999).float()


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000.0 / timesteps
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps).clamp(max=0.999)


def extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (ndim - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine") -> None:
        super().__init__()
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_previous = torch.cat((torch.ones(1), alpha_bar[:-1]))
        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_previous", alpha_bar_previous)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        noisy = (
            extract(self.sqrt_alpha_bar, timesteps, clean.ndim) * clean
            + extract(self.sqrt_one_minus_alpha_bar, timesteps, clean.ndim) * noise
        )
        return noisy, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        output_channels: int,
        mask: torch.Tensor | None = None,
        sampling_steps: int = 50,
        eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate normalized increments using DDIM sampling."""
        batch, _, height, width = condition.shape
        shape = (batch, output_channels, height, width)
        x = torch.randn(shape, device=condition.device) if initial_noise is None else initial_noise
        if x.shape != shape:
            raise ValueError(f"Initial noise shape {x.shape} does not match {shape}")

        sampling_steps = min(int(sampling_steps), self.timesteps)
        schedule = torch.linspace(
            self.timesteps - 1, 0, sampling_steps, device=condition.device
        ).round().long()
        schedule = torch.unique_consecutive(schedule)

        for step_index, timestep_value in enumerate(schedule):
            timestep = torch.full(
                (batch,), int(timestep_value.item()), device=condition.device, dtype=torch.long
            )
            predicted_noise = model(x, condition, timestep)
            alpha_t = extract(self.alpha_bar, timestep, x.ndim)
            predicted_clean = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            predicted_clean = predicted_clean.clamp(-8.0, 8.0)

            if step_index == len(schedule) - 1:
                x = predicted_clean
                break

            next_value = schedule[step_index + 1]
            next_timestep = torch.full(
                (batch,), int(next_value.item()), device=condition.device, dtype=torch.long
            )
            alpha_next = extract(self.alpha_bar, next_timestep, x.ndim)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_next) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_next)
            ).clamp_min(0.0)
            direction = torch.sqrt((1.0 - alpha_next - sigma**2).clamp_min(0.0)) * predicted_noise
            x = torch.sqrt(alpha_next) * predicted_clean + direction
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

        if mask is not None:
            x = x * mask
        return x

    def ddim_sample_observation_guided(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        output_channels: int,
        background_state: torch.Tensor,
        residual_mean: torch.Tensor,
        residual_std: torch.Tensor,
        observation: torch.Tensor,
        observation_mask: torch.Tensor,
        levels: int,
        temperature_guidance: float = 0.10,
        salinity_guidance: float = 0.02,
        guidance_start: float = 0.40,
        residual_scale: float = 1.0,
        mask: torch.Tensor | None = None,
        sampling_steps: int = 50,
        eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDIM posterior sampling with Argo loss gradients at every late denoising step.

        The observation gradient is applied to the predicted clean normalized
        residual, not as a post-processing nudging update. Temperature and
        salinity gradients are normalized separately over observed elements so
        sparse-day observation counts do not change the effective step size.
        """
        batch, _, height, width = condition.shape
        shape = (batch, output_channels, height, width)
        x = torch.randn(shape, device=condition.device) if initial_noise is None else initial_noise.clone()
        if x.shape != shape:
            raise ValueError(f"Initial noise shape {x.shape} does not match {shape}")
        if not 0.0 <= guidance_start < 1.0:
            raise ValueError("guidance_start must be in [0, 1)")

        sampling_steps = min(int(sampling_steps), self.timesteps)
        schedule = torch.linspace(
            self.timesteps - 1, 0, sampling_steps, device=condition.device
        ).round().long()
        schedule = torch.unique_consecutive(schedule)
        total_steps = max(len(schedule) - 1, 1)

        for step_index, timestep_value in enumerate(schedule):
            timestep = torch.full(
                (batch,), int(timestep_value.item()), device=condition.device, dtype=torch.long
            )
            with torch.no_grad():
                predicted_noise = model(x, condition, timestep)
                if not torch.isfinite(predicted_noise).all():
                    raise FloatingPointError(
                        f"Residual model produced NaN/Inf at DDIM step {step_index} "
                        f"(t={int(timestep_value.item())})"
                    )
                alpha_t = extract(self.alpha_bar, timestep, x.ndim)
                predicted_clean = (
                    x - torch.sqrt(1.0 - alpha_t) * predicted_noise
                ) / torch.sqrt(alpha_t)
                predicted_clean = predicted_clean.clamp(-8.0, 8.0)

            progress = step_index / total_steps
            has_observations = bool(observation_mask.any().item())
            if progress >= guidance_start and has_observations:
                ramp = min((progress - guidance_start) / max(1.0 - guidance_start, 1e-6), 1.0)
                clean_var = predicted_clean.detach().requires_grad_(True)
                physical = background_state + residual_scale * (
                    clean_var * residual_std + residual_mean
                )
                tmask = observation_mask[:, :levels]
                smask = observation_mask[:, levels:]
                has_temperature = bool(tmask.any().item())
                has_salinity = bool(smask.any().item())
                zero = clean_var.new_zeros(())
                # Smooth-L1 is deliberately used instead of squared error: a
                # surviving bad observation cannot overflow float32 and poison
                # the complete reverse trajectory.
                # Mask before evaluating the loss. Background land points are
                # NaN by design; masking after the loss would allow NaN*0 to
                # contaminate autograd.
                temperature_difference = torch.where(
                    tmask.bool(),
                    physical[:, :levels] - observation[:, :levels],
                    torch.zeros_like(physical[:, :levels]),
                )
                salinity_difference = torch.where(
                    smask.bool(),
                    physical[:, levels:] - observation[:, levels:],
                    torch.zeros_like(physical[:, levels:]),
                )
                temperature_error = torch.nn.functional.smooth_l1_loss(
                    temperature_difference, torch.zeros_like(temperature_difference),
                    reduction="none", beta=0.5,
                )
                salinity_error = torch.nn.functional.smooth_l1_loss(
                    salinity_difference, torch.zeros_like(salinity_difference),
                    reduction="none", beta=0.1,
                )
                tloss = (
                    torch.where(tmask.bool(), temperature_error, zero).sum()
                    / tmask.sum().clamp_min(1.0)
                    if has_temperature else zero
                )
                sloss = (
                    torch.where(smask.bool(), salinity_error, zero).sum()
                    / smask.sum().clamp_min(1.0)
                    if has_salinity else zero
                )
                gradients = clean_var.new_zeros(clean_var.shape)
                if has_temperature and temperature_guidance > 0:
                    tg = torch.autograd.grad(
                        tloss, clean_var,
                        retain_graph=has_salinity and salinity_guidance > 0,
                    )[0]
                    tg = torch.nan_to_num(tg, nan=0.0, posinf=0.0, neginf=0.0)
                    trms = torch.sqrt((tg[:, :levels].square() * tmask).sum() / tmask.sum().clamp_min(1.0))
                    gradients[:, :levels] += temperature_guidance * tg[:, :levels] / trms.clamp_min(1e-8)
                if has_salinity and salinity_guidance > 0:
                    sg = torch.autograd.grad(sloss, clean_var)[0]
                    sg = torch.nan_to_num(sg, nan=0.0, posinf=0.0, neginf=0.0)
                    srms = torch.sqrt((sg[:, levels:].square() * smask).sum() / smask.sum().clamp_min(1.0))
                    gradients[:, levels:] += salinity_guidance * sg[:, levels:] / srms.clamp_min(1e-8)
                predicted_clean = (clean_var - ramp * gradients).detach().clamp(-8.0, 8.0)
                if not torch.isfinite(predicted_clean).all():
                    raise FloatingPointError(
                        f"Argo guidance produced NaN/Inf at DDIM step {step_index} "
                        f"(t={int(timestep_value.item())})"
                    )

            # New version only: Argo correction above remains inside denoising.
            # The GSW gradient is an additional local physical search, not EnKF.
            hook = getattr(self, 'stability_operator', None)
            if hook is not None:
                predicted_clean = hook.denoise(
                    predicted_clean, background_state, residual_mean, residual_std,
                    residual_scale, mask, observation, observation_mask,
                    step_index, progress, step_index == len(schedule)-1,
                )

            if step_index == len(schedule) - 1:
                x = predicted_clean
                break
            next_value = schedule[step_index + 1]
            next_timestep = torch.full(
                (batch,), int(next_value.item()), device=condition.device, dtype=torch.long
            )
            alpha_next = extract(self.alpha_bar, next_timestep, x.ndim)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_next) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_next)
            ).clamp_min(0.0)
            direction = torch.sqrt((1.0 - alpha_next - sigma**2).clamp_min(0.0)) * predicted_noise
            x = torch.sqrt(alpha_next) * predicted_clean + direction
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

        if mask is not None:
            x = x * mask
        return x


def build_diffusion(config: dict) -> GaussianDiffusion:
    diffusion_cfg = config["diffusion"]
    return GaussianDiffusion(
        timesteps=int(diffusion_cfg["timesteps"]),
        beta_schedule=str(diffusion_cfg["beta_schedule"]),
    )


def masked_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitude_weight: torch.Tensor,
) -> torch.Tensor:
    weight = mask * latitude_weight
    squared_error = (prediction - target).square() * weight
    return squared_error.sum() / weight.sum().clamp_min(1.0)
