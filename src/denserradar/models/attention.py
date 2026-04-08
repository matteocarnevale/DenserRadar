"""Cross-attention gate for 3D U-Net skip connections (paper Eq. 7).

The decoder and encoder feature maps are projected into a shared
bottleneck, summed (additive attention), and passed through a sigmoid
gate.  The gate modulates *which* encoder information is useful at
each spatial position before fusing it back into the decoder path.

This is much cheaper than flattened spatial attention on 3D volumes
while being more expressive than element-wise multiplicative gating.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttention3D(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        bottleneck = max(channels // 2, 1)

        self.project_decoder = nn.Conv3d(channels, bottleneck, kernel_size=1, bias=False)
        self.project_encoder = nn.Conv3d(channels, bottleneck, kernel_size=1, bias=False)

        self.attention_gate = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv3d(bottleneck, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, decoder_feat: torch.Tensor, encoder_feat: torch.Tensor) -> torch.Tensor:
        combined = self.project_decoder(decoder_feat) + self.project_encoder(encoder_feat)
        gate_weights = self.attention_gate(combined)

        gated_encoder = gate_weights * encoder_feat
        return decoder_feat + gated_encoder
