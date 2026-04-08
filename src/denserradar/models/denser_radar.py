"""DenserRadar network — the main model (paper Fig. 3 / Sec. III-C).

Architecture overview
---------------------
1. **Input stem**: 1x1x1 conv that reinterprets Doppler bins as feature channels.
2. **3D U-Net encoder** (3 downsample stages): extracts multi-scale features
   from the radar tensor in spherical (R, E, A) coordinates.
3. **Bottleneck**: residual block at the coarsest resolution.
4. **3D U-Net decoder** (3 upsample stages): reconstructs occupancy features.
   Each stage fuses encoder features via a cross-attention skip connection.
5. **Deep-supervision heads**: 1x1 conv + sigmoid at each decoder scale,
   providing auxiliary losses for stable gradient flow (paper Eq. 8).
6. **Super-resolution head**: transposed conv that doubles the spatial
   resolution, producing the final HR occupancy at 2R x 2E x 2A.
"""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from .attention import CrossAttention3D
from .blocks import ResidualBlock3D, DownsampleBlock3D, UpsampleBlock3D


class DenserRadarNet(nn.Module):

    def __init__(self, radar_cfg: dict, model_cfg: dict):
        super().__init__()

        doppler_bins = int(radar_cfg["doppler_bins"])
        base_ch = int(model_cfg.get("base_channels", 16))
        multipliers = list(model_cfg.get("channel_multipliers", [1, 2, 4, 8]))
        groups = int(model_cfg.get("norm_groups", 8))
        act = str(model_cfg.get("activation", "silu"))
        drop = float(model_cfg.get("dropout_p", 0.0))
        self.use_cross_attention = bool(model_cfg.get("use_cross_attention", True))

        # Channel widths at each U-Net level, e.g. [16, 32, 64, 128]
        ch = [base_ch * m for m in multipliers]
        assert len(ch) >= 4, "Need at least 4 channel stages"

        # ── 1. Input stem: Doppler → feature channels (paper Sec. III-C.1) ──
        self.input_stem = nn.Conv3d(doppler_bins, ch[0], kernel_size=1)

        # ── 2. Encoder: progressively halve spatial dims ──────────────────
        self.encoder_full = ResidualBlock3D(ch[0], ch[0], norm_groups=groups, activation=act, dropout=drop)
        self.encoder_half = DownsampleBlock3D(ch[0], ch[1], norm_groups=groups, activation=act, dropout=drop)
        self.encoder_quarter = DownsampleBlock3D(ch[1], ch[2], norm_groups=groups, activation=act, dropout=drop)
        self.encoder_eighth = DownsampleBlock3D(ch[2], ch[3], norm_groups=groups, activation=act, dropout=drop)

        # ── 3. Bottleneck ─────────────────────────────────────────────────
        self.bottleneck = ResidualBlock3D(ch[3], ch[3], norm_groups=groups, activation=act, dropout=drop)

        # ── 4. Decoder: progressively double spatial dims ─────────────────
        self.decoder_to_quarter = UpsampleBlock3D(ch[3], ch[2], norm_groups=groups, activation=act, dropout=drop)
        self.decoder_to_half = UpsampleBlock3D(ch[2], ch[1], norm_groups=groups, activation=act, dropout=drop)
        self.decoder_to_full = UpsampleBlock3D(ch[1], ch[0], norm_groups=groups, activation=act, dropout=drop)

        # ── Skip connections: cross-attention gates (paper Eq. 7) ─────────
        self.skip_quarter = CrossAttention3D(ch[2]) if self.use_cross_attention else nn.Identity()
        self.skip_half = CrossAttention3D(ch[1]) if self.use_cross_attention else nn.Identity()
        self.skip_full = CrossAttention3D(ch[0]) if self.use_cross_attention else nn.Identity()

        # ── 5. Deep supervision heads at each decoder scale (paper Eq. 8) ─
        self.ds_head_quarter = nn.Sequential(nn.Conv3d(ch[2], 1, kernel_size=1), nn.Sigmoid())
        self.ds_head_half = nn.Sequential(nn.Conv3d(ch[1], 1, kernel_size=1), nn.Sigmoid())
        self.ds_head_full = nn.Sequential(nn.Conv3d(ch[0], 1, kernel_size=1), nn.Sigmoid())

        # ── 6. Super-resolution head: native → 2x resolution (paper Sec. III-C.3)
        self.superres_upsample = nn.ConvTranspose3d(ch[0], ch[0] // 2, kernel_size=2, stride=2)
        self.superres_head = nn.Sequential(
            nn.Conv3d(ch[0] // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_to_match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Crop *source* spatial dims to match *reference* (handles odd dims)."""
        if source.shape[2:] == reference.shape[2:]:
            return source
        return source[:, :, :reference.shape[2], :reference.shape[3], :reference.shape[4]]

    def _merge_skip(self, decoder_feat: torch.Tensor, encoder_feat: torch.Tensor, attention: nn.Module) -> torch.Tensor:
        """Crop decoder to match encoder, then fuse via attention (or simple add)."""
        decoder_feat = self._crop_to_match(decoder_feat, encoder_feat)
        if isinstance(attention, nn.Identity):
            return decoder_feat + encoder_feat
        return attention(decoder_feat, encoder_feat)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, radar_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        radar_tensor : [B, Doppler, Range, Elevation, Azimuth]

        Returns
        -------
        dict with keys: pred_hr, pred_ds1, pred_ds2, pred_ds4
        """

        # --- Stem: Doppler bins become feature channels ---
        features = self.input_stem(radar_tensor)

        # --- Encoder path ---
        enc_full = self.encoder_full(features)       # R   x E   x A
        enc_half = self.encoder_half(enc_full)        # R/2 x E/2 x A/2
        enc_quarter = self.encoder_quarter(enc_half)  # R/4 x E/4 x A/4
        enc_eighth = self.encoder_eighth(enc_quarter) # R/8 x E/8 x A/8

        bottle = self.bottleneck(enc_eighth)

        # --- Decoder path (with skip connections) ---
        dec_quarter = self.decoder_to_quarter(bottle)
        dec_quarter = self._merge_skip(dec_quarter, enc_quarter, self.skip_quarter)
        pred_ds4 = self.ds_head_quarter(dec_quarter)

        dec_half = self.decoder_to_half(dec_quarter)
        dec_half = self._merge_skip(dec_half, enc_half, self.skip_half)
        pred_ds2 = self.ds_head_half(dec_half)

        dec_full = self.decoder_to_full(dec_half)
        dec_full = self._merge_skip(dec_full, enc_full, self.skip_full)
        pred_ds1 = self.ds_head_full(dec_full)

        # --- Super-resolution head: 2x the native radar resolution ---
        upsampled = self.superres_upsample(dec_full)
        pred_hr = self.superres_head(upsampled)

        return {
            "pred_hr": pred_hr,     # [B, 1, 2R, 2E, 2A]
            "pred_ds1": pred_ds1,   # [B, 1, R,  E,  A ]
            "pred_ds2": pred_ds2,   # [B, 1, R/2,E/2,A/2]
            "pred_ds4": pred_ds4,   # [B, 1, R/4,E/4,A/4]
        }
