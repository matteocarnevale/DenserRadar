"""Basic 3D convolution building blocks for the DenserRadar U-Net.

All convolutions use anisotropic 5x3x3 kernels (paper Sec. III-C):
larger in the Range dimension because range resolution is typically
better than elevation/azimuth resolution on 4D mmWave radars.
"""
from __future__ import annotations

import torch
import torch.nn as nn


KERNEL_REA = (5, 3, 3)   # (Range, Elevation, Azimuth) — paper default


def make_activation(name: str) -> nn.Module:
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "gelu": nn.GELU,
        "silu": lambda: nn.SiLU(inplace=True),
    }
    factory = activations.get(name.lower())
    if factory is None:
        raise ValueError(f"Unsupported activation: {name}")
    return factory()


class ConvNormAct3D(nn.Module):
    """Conv3D -> GroupNorm -> Activation -> (optional Dropout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=KERNEL_REA,
        stride: int = 1,
        norm_groups: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(norm_groups, out_channels), num_channels=out_channels),
            make_activation(activation),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock3D(nn.Module):
    """Two ConvNormAct layers with a residual (skip) connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=KERNEL_REA,
        norm_groups: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = ConvNormAct3D(in_channels, out_channels, kernel_size, 1, norm_groups, activation, dropout)
        self.conv2 = ConvNormAct3D(out_channels, out_channels, kernel_size, 1, norm_groups, activation, dropout)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class DownsampleBlock3D(nn.Module):
    """Halve spatial dims with stride-2 conv, then refine with a residual block."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8, activation: str = "silu", dropout: float = 0.0):
        super().__init__()
        self.downsample = ConvNormAct3D(in_channels, out_channels, KERNEL_REA, stride=2,
                                        norm_groups=norm_groups, activation=activation, dropout=dropout)
        self.refine = ResidualBlock3D(out_channels, out_channels, norm_groups=norm_groups, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.downsample(x))


class UpsampleBlock3D(nn.Module):
    """Double spatial dims with transposed conv, then refine with a residual block."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8, activation: str = "silu", dropout: float = 0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = ResidualBlock3D(out_channels, out_channels, norm_groups=norm_groups, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.upsample(x))
