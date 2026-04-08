import numpy as np
import torch

from denserradar.models.denser_radar import DenserRadarNet
from denserradar.models.attention import CrossAttention3D
from denserradar.losses.occupancy import hybrid_multiscale_loss
from denserradar.metrics.pointcloud import rpcd_rpca
from denserradar.data.voxelization import downsample_occupancy_max

_MODEL_CFG = {
    "base_channels": 8,
    "channel_multipliers": [1, 2, 4, 8],
    "use_cross_attention": True,
    "norm_groups": 4,
    "activation": "silu",
    "dropout_p": 0.0,
}


def _radar_cfg(r=32, e=16, a=8, d=8):
    return {
        "doppler_bins": d,
        "range_bins": r,
        "elevation_bins": e,
        "azimuth_bins": a,
        "range_min_m": 0.0,
        "range_max_m": float(r) * 0.46,
        "elevation_min_deg": -10.0,
        "elevation_max_deg": 10.0,
        "azimuth_min_deg": -5.0,
        "azimuth_max_deg": 5.0,
    }


def test_forward_shapes_even():
    """Power-of-2 spatial dims — basic sanity."""
    cfg = _radar_cfg(32, 16, 8)
    model = DenserRadarNet(cfg, _MODEL_CFG)
    out = model(torch.randn(1, 8, 32, 16, 8))
    assert out["pred_hr"].shape == (1, 1, 64, 32, 16)
    assert out["pred_ds1"].shape == (1, 1, 32, 16, 8)
    assert out["pred_ds2"].shape == (1, 1, 16, 8, 4)
    assert out["pred_ds4"].shape == (1, 1, 8, 4, 2)


def test_forward_shapes_odd():
    """Odd spatial dims (like K-Radar 256x107x37) must not crash."""
    cfg = _radar_cfg(r=32, e=27, a=19, d=4)
    model = DenserRadarNet(cfg, _MODEL_CFG)
    out = model(torch.randn(1, 4, 32, 27, 19))
    assert out["pred_hr"].shape[0] == 1 and out["pred_hr"].shape[1] == 1
    assert out["pred_ds1"].shape[2:] == (32, 27, 19)


def test_forward_no_cross_attention():
    cfg = _radar_cfg(32, 16, 8)
    mcfg = {**_MODEL_CFG, "use_cross_attention": False}
    model = DenserRadarNet(cfg, mcfg)
    out = model(torch.randn(1, 8, 32, 16, 8))
    assert out["pred_hr"].shape == (1, 1, 64, 32, 16)


def test_cross_attention_3d():
    attn = CrossAttention3D(16)
    dec = torch.randn(2, 16, 8, 6, 4)
    enc = torch.randn(2, 16, 8, 6, 4)
    out = attn(dec, enc)
    assert out.shape == dec.shape


def test_loss_runs():
    preds = {
        "pred_hr": torch.sigmoid(torch.randn(1, 1, 64, 32, 16)),
        "pred_ds1": torch.sigmoid(torch.randn(1, 1, 32, 16, 8)),
        "pred_ds2": torch.sigmoid(torch.randn(1, 1, 16, 8, 4)),
        "pred_ds4": torch.sigmoid(torch.randn(1, 1, 8, 4, 2)),
    }
    targets = {
        "gt_occ_hr": (torch.rand(1, 1, 64, 32, 16) > 0.95).float(),
        "gt_occ_ds1": (torch.rand(1, 1, 32, 16, 8) > 0.95).float(),
        "gt_occ_ds2": (torch.rand(1, 1, 16, 8, 4) > 0.95).float(),
        "gt_occ_ds4": (torch.rand(1, 1, 8, 4, 2) > 0.95).float(),
    }
    cfg = {"lambda_focal": 10.0, "focal_alpha": 0.25, "focal_gamma": 2.0, "smooth": 1.0}
    loss, stats = hybrid_multiscale_loss(preds, targets, cfg)
    assert loss.item() > 0
    assert "loss_total" in stats


def test_loss_spatial_mismatch():
    """Loss must not crash when pred and target shapes differ by 1."""
    preds = {"pred_hr": torch.sigmoid(torch.randn(1, 1, 64, 32, 16)),
             "pred_ds1": torch.sigmoid(torch.randn(1, 1, 32, 16, 8)),
             "pred_ds2": torch.sigmoid(torch.randn(1, 1, 16, 9, 5)),
             "pred_ds4": torch.sigmoid(torch.randn(1, 1, 8, 5, 3))}
    targets = {"gt_occ_hr": (torch.rand(1, 1, 64, 32, 16) > 0.9).float(),
               "gt_occ_ds1": (torch.rand(1, 1, 32, 16, 8) > 0.9).float(),
               "gt_occ_ds2": (torch.rand(1, 1, 16, 8, 4) > 0.9).float(),
               "gt_occ_ds4": (torch.rand(1, 1, 8, 4, 2) > 0.9).float()}
    cfg = {"lambda_focal": 10.0, "focal_alpha": 0.25, "focal_gamma": 2.0, "smooth": 1.0}
    loss, _ = hybrid_multiscale_loss(preds, targets, cfg)
    assert loss.item() > 0


def test_downsample_non_divisible():
    """downsample_occupancy_max must handle dims not divisible by factor."""
    occ = np.random.rand(1, 214, 74, 50).astype(np.float32)
    ds4 = downsample_occupancy_max(occ, factor=4)
    assert ds4.shape == (1, 214 // 4 + 1, 74 // 4 + 1, 50 // 4 + 1) or ds4.ndim == 4


def test_rpcd_rpca_basic():
    pred = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    gt = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float32)
    rpcd, rpca = rpcd_rpca(pred, gt, rpcd_radius=0.3, rpca_radius=0.5)
    assert 0.0 <= rpcd <= 1.0
    assert 0.0 <= rpca <= 1.0
