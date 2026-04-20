"""DenserRadar-style 3D U-Net variant for cartesian radar point inputs.

This model reuses the same encoder/decoder topology as `DenserRadarNet` but
accepts a **cartesian voxel volume** as input instead of a raw 4D radar tensor.

Input:
  radar_volume: [B, 1, Z, Y, X]

Outputs:
  pred_hr  : [B, 1, 2Z, 2Y, 2X]
  pred_ds1 : [B, 1, Z,  Y,  X ]
  pred_ds2 : [B, 1, Z/2,Y/2,X/2]
  pred_ds4 : [B, 1, Z/4,Y/4,X/4]
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from .attention import CrossAttention3D
from .blocks import ResidualBlock3D, DownsampleBlock3D, UpsampleBlock3D


class RadarPointsUNet(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        base_ch = int(model_cfg.get("base_channels", 16))
        multipliers = list(model_cfg.get("channel_multipliers", [1, 2, 4, 8]))
        groups = int(model_cfg.get("norm_groups", 8))
        act = str(model_cfg.get("activation", "silu"))
        drop = float(model_cfg.get("dropout_p", 0.0))
        self.use_cross_attention = bool(model_cfg.get("use_cross_attention", True))

        ch = [base_ch * m for m in multipliers]
        assert len(ch) >= 4, "Need at least 4 channel stages"

        # 1-channel input volume (cartesian)
        self.input_stem = nn.Conv3d(1, ch[0], kernel_size=1)

        self.encoder_full = ResidualBlock3D(ch[0], ch[0], norm_groups=groups, activation=act, dropout=drop)
        self.encoder_half = DownsampleBlock3D(ch[0], ch[1], norm_groups=groups, activation=act, dropout=drop)
        self.encoder_quarter = DownsampleBlock3D(ch[1], ch[2], norm_groups=groups, activation=act, dropout=drop)
        self.encoder_eighth = DownsampleBlock3D(ch[2], ch[3], norm_groups=groups, activation=act, dropout=drop)

        self.bottleneck = ResidualBlock3D(ch[3], ch[3], norm_groups=groups, activation=act, dropout=drop)

        self.decoder_to_quarter = UpsampleBlock3D(ch[3], ch[2], norm_groups=groups, activation=act, dropout=drop)
        self.decoder_to_half = UpsampleBlock3D(ch[2], ch[1], norm_groups=groups, activation=act, dropout=drop)
        self.decoder_to_full = UpsampleBlock3D(ch[1], ch[0], norm_groups=groups, activation=act, dropout=drop)

        self.skip_quarter = CrossAttention3D(ch[2]) if self.use_cross_attention else nn.Identity()
        self.skip_half = CrossAttention3D(ch[1]) if self.use_cross_attention else nn.Identity()
        self.skip_full = CrossAttention3D(ch[0]) if self.use_cross_attention else nn.Identity()

        self.ds_head_quarter = nn.Sequential(nn.Conv3d(ch[2], 1, kernel_size=1), nn.Sigmoid())
        self.ds_head_half = nn.Sequential(nn.Conv3d(ch[1], 1, kernel_size=1), nn.Sigmoid())
        self.ds_head_full = nn.Sequential(nn.Conv3d(ch[0], 1, kernel_size=1), nn.Sigmoid())

        self.superres_upsample = nn.ConvTranspose3d(ch[0], ch[0] // 2, kernel_size=2, stride=2)
        self.superres_head = nn.Sequential(nn.Conv3d(ch[0] // 2, 1, kernel_size=1), nn.Sigmoid())

    @staticmethod
    def _crop_to_match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if source.shape[2:] == reference.shape[2:]:
            return source
        return source[:, :, : reference.shape[2], : reference.shape[3], : reference.shape[4]]

    def _merge_skip(self, decoder_feat: torch.Tensor, encoder_feat: torch.Tensor, attention: nn.Module) -> torch.Tensor:
        decoder_feat = self._crop_to_match(decoder_feat, encoder_feat)
        if isinstance(attention, nn.Identity):
            return decoder_feat + encoder_feat
        return attention(decoder_feat, encoder_feat)

    def forward(self, radar_volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.input_stem(radar_volume)

        enc_full = self.encoder_full(features)
        enc_half = self.encoder_half(enc_full)
        enc_quarter = self.encoder_quarter(enc_half)
        enc_eighth = self.encoder_eighth(enc_quarter)

        bottle = self.bottleneck(enc_eighth)

        dec_quarter = self.decoder_to_quarter(bottle)
        dec_quarter = self._merge_skip(dec_quarter, enc_quarter, self.skip_quarter)
        pred_ds4 = self.ds_head_quarter(dec_quarter)

        dec_half = self.decoder_to_half(dec_quarter)
        dec_half = self._merge_skip(dec_half, enc_half, self.skip_half)
        pred_ds2 = self.ds_head_half(dec_half)

        dec_full = self.decoder_to_full(dec_half)
        dec_full = self._merge_skip(dec_full, enc_full, self.skip_full)
        pred_ds1 = self.ds_head_full(dec_full)

        upsampled = self.superres_upsample(dec_full)
        pred_hr = self.superres_head(upsampled)

        return {"pred_hr": pred_hr, "pred_ds1": pred_ds1, "pred_ds2": pred_ds2, "pred_ds4": pred_ds4}

