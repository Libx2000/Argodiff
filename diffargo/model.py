from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for value in range(min(preferred, channels), 0, -1):
        if channels % value == 0:
            return value
    return 1


class PeriodicConv2d(nn.Module):
    """Convolution with circular longitude and replicated latitude padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("PeriodicConv2d requires an odd kernel size")
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.padding
        if p:
            x = F.pad(x, (p, p, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, p, p), mode="replicate")
        return self.conv(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = PeriodicConv2d(in_channels, out_channels)
        self.time_projection = nn.Linear(time_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = PeriodicConv2d(out_channels, out_channels)
        nn.init.zeros_(self.conv2.conv.weight)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_projection(F.silu(time_embedding)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normalized = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return x + attended.transpose(1, 2).reshape(b, c, h, w)


class ConditionalUNet(nn.Module):
    """Noise predictor for a conditional 2-D ocean-state diffusion model."""

    def __init__(
        self,
        target_channels: int,
        condition_channels: int,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 6),
        time_embedding_dim: int = 256,
        dropout: float = 0.05,
        attention_at_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        self.target_channels = int(target_channels)
        self.condition_channels = int(condition_channels)
        channels = [base_channels * int(multiplier) for multiplier in channel_multipliers]
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, 4 * time_embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * time_embedding_dim, time_embedding_dim),
        )
        self.input_conv = PeriodicConv2d(target_channels + condition_channels, channels[0])

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        for level, output in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleList(
                    (
                        ResidualBlock(current, output, time_embedding_dim, dropout),
                        ResidualBlock(output, output, time_embedding_dim, dropout),
                    )
                )
            )
            current = output
            if level < len(channels) - 1:
                self.downsamples.append(PeriodicConv2d(current, channels[level + 1], stride=2))
                current = channels[level + 1]

        self.middle1 = ResidualBlock(current, current, time_embedding_dim, dropout)
        self.middle_attention = SpatialAttention(current) if attention_at_bottleneck else nn.Identity()
        self.middle2 = ResidualBlock(current, current, time_embedding_dim, dropout)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for skip_channels in reversed(channels[:-1]):
            self.upsamples.append(PeriodicConv2d(current, skip_channels))
            self.up_blocks.append(
                nn.ModuleList(
                    (
                        ResidualBlock(2 * skip_channels, skip_channels, time_embedding_dim, dropout),
                        ResidualBlock(skip_channels, skip_channels, time_embedding_dim, dropout),
                    )
                )
            )
            current = skip_channels

        self.output_norm = nn.GroupNorm(_groups(current), current)
        self.output_conv = PeriodicConv2d(current, target_channels)
        nn.init.zeros_(self.output_conv.conv.weight)

    def forward(
        self,
        noisy_target: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.shape[0] != condition.shape[0]:
            raise ValueError("Target and condition batch sizes differ")
        time_embedding = self.time_mlp(timestep)

        h = self.input_conv(torch.cat((noisy_target, condition), dim=1))
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down_blocks):
            h = blocks[0](h, time_embedding)
            h = blocks[1](h, time_embedding)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.middle1(h, time_embedding)
        h = self.middle_attention(h)
        h = self.middle2(h, time_embedding)

        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = upsample(h)
            h = torch.cat((h, skip), dim=1)
            h = blocks[0](h, time_embedding)
            h = blocks[1](h, time_embedding)

        return self.output_conv(F.silu(self.output_norm(h)))


def build_model(config: dict, target_channels: int, condition_channels: int) -> ConditionalUNet:
    model_cfg = config["model"]
    return ConditionalUNet(
        target_channels=target_channels,
        condition_channels=condition_channels,
        base_channels=int(model_cfg["base_channels"]),
        channel_multipliers=model_cfg["channel_multipliers"],
        time_embedding_dim=int(model_cfg["time_embedding_dim"]),
        dropout=float(model_cfg["dropout"]),
        attention_at_bottleneck=bool(model_cfg["attention_at_bottleneck"]),
    )
