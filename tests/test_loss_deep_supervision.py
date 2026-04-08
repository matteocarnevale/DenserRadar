"""Smoke-tests for the weighted hybrid loss with deep supervision (paper Eq. 8).

Verifies:
  L = Σ_{i=0}^{N} 1/2^i · (L^Dice_i + λ_F · L^Focal_i)

Tested properties:
  1. All decoder scales contribute to the total loss (deep supervision)
  2. Weights follow the 1/2^i geometric decay
  3. Both Dice and Focal components are present at every scale
  4. Dice handles extreme class imbalance (sparse occupancy)
  5. Focal down-weights easy-to-classify voxels
  6. End-to-end backward pass produces gradients in ALL layers,
     including early encoder stages (the core benefit of deep supervision)
  7. Loss is lower for better predictions than for random ones
  8. λ_F correctly scales the focal contribution
"""
from __future__ import annotations

import math

import torch

from denserradar.losses.occupancy import (
    dice_loss,
    focal_loss,
    hybrid_multiscale_loss,
)
from denserradar.models.denser_radar import DenserRadarNet


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_R, _E, _A = 32, 16, 8  # native radar grid (power-of-2 for simplicity)
_HR = 2  # high-res factor


def _make_targets(sparsity: float = 0.95) -> dict[str, torch.Tensor]:
    """Return GT occupancy grids at all four supervision scales."""
    return {
        "gt_occ_hr": (torch.rand(1, 1, _R * _HR, _E * _HR, _A * _HR) > sparsity).float(),
        "gt_occ_ds1": (torch.rand(1, 1, _R, _E, _A) > sparsity).float(),
        "gt_occ_ds2": (torch.rand(1, 1, _R // 2, _E // 2, _A // 2) > sparsity).float(),
        "gt_occ_ds4": (torch.rand(1, 1, _R // 4, _E // 4, _A // 4) > sparsity).float(),
    }


def _make_preds_from_targets(targets: dict[str, torch.Tensor],
                             noise: float = 0.05) -> dict[str, torch.Tensor]:
    """Near-perfect predictions — targets + small noise, clamped to (0,1)."""
    return {
        "pred_hr": (targets["gt_occ_hr"] + noise * torch.randn_like(targets["gt_occ_hr"])).clamp(0.01, 0.99),
        "pred_ds1": (targets["gt_occ_ds1"] + noise * torch.randn_like(targets["gt_occ_ds1"])).clamp(0.01, 0.99),
        "pred_ds2": (targets["gt_occ_ds2"] + noise * torch.randn_like(targets["gt_occ_ds2"])).clamp(0.01, 0.99),
        "pred_ds4": (targets["gt_occ_ds4"] + noise * torch.randn_like(targets["gt_occ_ds4"])).clamp(0.01, 0.99),
    }


def _random_preds() -> dict[str, torch.Tensor]:
    return {
        "pred_hr": torch.sigmoid(torch.randn(1, 1, _R * _HR, _E * _HR, _A * _HR)),
        "pred_ds1": torch.sigmoid(torch.randn(1, 1, _R, _E, _A)),
        "pred_ds2": torch.sigmoid(torch.randn(1, 1, _R // 2, _E // 2, _A // 2)),
        "pred_ds4": torch.sigmoid(torch.randn(1, 1, _R // 4, _E // 4, _A // 4)),
    }


_LOSS_CFG = {
    "lambda_focal": 700.0,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "smooth": 1.0,
}


# ===================================================================
# 1. Deep supervision: all four scales contribute
# ===================================================================

def test_all_scales_produce_stats():
    """Every decoder scale must appear in the stats dict with both dice & focal."""
    targets = _make_targets()
    preds = _random_preds()
    _, stats = hybrid_multiscale_loss(preds, targets, _LOSS_CFG)

    for key in ("pred_hr", "pred_ds1", "pred_ds2", "pred_ds4"):
        assert f"{key}_dice" in stats, f"Missing dice stat for {key}"
        assert f"{key}_focal" in stats, f"Missing focal stat for {key}"
        assert stats[f"{key}_dice"] > 0, f"Dice for {key} should be > 0"
        assert stats[f"{key}_focal"] > 0, f"Focal for {key} should be > 0"


# ===================================================================
# 2. Weights follow 1/2^i geometric decay
# ===================================================================

def test_deep_supervision_weights_geometric():
    """Verify weight_i = 1/2^i by isolating each scale's contribution.

    Feed identical (pred, target) at every scale so dice and focal
    values are the same everywhere.  The per-scale contribution should
    then differ only by the weight factor.
    """
    shared = torch.sigmoid(torch.randn(1, 1, 4, 4, 4))
    gt = (torch.rand(1, 1, 4, 4, 4) > 0.9).float()

    preds = {k: shared.clone() for k in ("pred_hr", "pred_ds1", "pred_ds2", "pred_ds4")}
    targets = {k: gt.clone() for k in ("gt_occ_hr", "gt_occ_ds1", "gt_occ_ds2", "gt_occ_ds4")}

    _, stats = hybrid_multiscale_loss(preds, targets, _LOSS_CFG)

    d = stats["pred_hr_dice"]
    f = stats["pred_hr_focal"]
    per_scale_raw = d + _LOSS_CFG["lambda_focal"] * f
    expected_weights = [1.0, 0.5, 0.25, 0.125]
    expected_total = sum(w * per_scale_raw for w in expected_weights)

    assert math.isclose(stats["loss_total"], expected_total, rel_tol=1e-4), (
        f"Total {stats['loss_total']:.6f} != expected {expected_total:.6f}"
    )


# ===================================================================
# 3. Dice loss handles extreme class imbalance
# ===================================================================

def test_dice_loss_sparse_target():
    """With < 1% occupancy the dice loss should stay in (0, 1]."""
    pred = torch.sigmoid(torch.randn(2, 1, 16, 16, 16))
    target = (torch.rand(2, 1, 16, 16, 16) > 0.995).float()
    d = dice_loss(pred, target)
    assert 0.0 < d.item() <= 1.0


def test_dice_loss_perfect_prediction():
    target = (torch.rand(1, 1, 8, 8, 8) > 0.9).float()
    pred = target.clone()
    d = dice_loss(pred, target)
    assert d.item() < 0.05, "Dice loss for perfect prediction should be near 0"


# ===================================================================
# 4. Focal loss properties
# ===================================================================

def test_focal_loss_down_weights_easy_examples():
    """Higher gamma should reduce the focal loss on easy (well-classified) examples."""
    pred = torch.full((1, 1, 8, 8, 8), 0.95)
    target = torch.ones(1, 1, 8, 8, 8)

    f_gamma1 = focal_loss(pred, target, gamma=1.0)
    f_gamma3 = focal_loss(pred, target, gamma=3.0)
    assert f_gamma3.item() < f_gamma1.item(), (
        "Higher gamma should produce lower loss on confident correct predictions"
    )


def test_focal_loss_hard_examples_higher():
    """A wrong-side prediction (pred ≈ 0 when target = 1) must be penalised more."""
    target = torch.ones(1, 1, 8, 8, 8)
    easy = torch.full_like(target, 0.9)
    hard = torch.full_like(target, 0.1)

    f_easy = focal_loss(easy, target)
    f_hard = focal_loss(hard, target)
    assert f_hard.item() > f_easy.item()


# ===================================================================
# 5. λ_F correctly scales the focal contribution
# ===================================================================

def test_lambda_focal_scaling():
    """Doubling lambda_focal should roughly double the focal-driven part."""
    targets = _make_targets()
    preds = _random_preds()

    cfg_lo = {**_LOSS_CFG, "lambda_focal": 100.0}
    cfg_hi = {**_LOSS_CFG, "lambda_focal": 200.0}

    loss_lo, stats_lo = hybrid_multiscale_loss(preds, targets, cfg_lo)
    loss_hi, stats_hi = hybrid_multiscale_loss(preds, targets, cfg_hi)

    assert loss_hi.item() > loss_lo.item(), (
        "Higher λ_F must increase the total loss"
    )

    focal_sum_lo = sum(stats_lo[f"{k}_focal"] for k in ("pred_hr", "pred_ds1", "pred_ds2", "pred_ds4"))
    focal_sum_hi = sum(stats_hi[f"{k}_focal"] for k in ("pred_hr", "pred_ds1", "pred_ds2", "pred_ds4"))
    assert math.isclose(focal_sum_lo, focal_sum_hi, rel_tol=1e-5), (
        "Raw focal values should be identical regardless of λ_F"
    )


# ===================================================================
# 6. Loss is lower for better predictions
# ===================================================================

def test_near_perfect_beats_random():
    targets = _make_targets(sparsity=0.9)
    good_preds = _make_preds_from_targets(targets, noise=0.05)
    bad_preds = _random_preds()

    loss_good, _ = hybrid_multiscale_loss(good_preds, targets, _LOSS_CFG)
    loss_bad, _ = hybrid_multiscale_loss(bad_preds, targets, _LOSS_CFG)

    assert loss_good.item() < loss_bad.item(), (
        "Near-perfect predictions must yield lower loss than random"
    )


# ===================================================================
# 7. End-to-end: model → loss → backward produces gradients in ALL
#    layers, including early encoder (the purpose of deep supervision)
# ===================================================================

def _build_small_model():
    radar_cfg = {
        "doppler_bins": 4,
        "range_bins": _R,
        "elevation_bins": _E,
        "azimuth_bins": _A,
        "range_min_m": 0.0,
        "range_max_m": 14.72,
        "elevation_min_deg": -10.0,
        "elevation_max_deg": 10.0,
        "azimuth_min_deg": -5.0,
        "azimuth_max_deg": 5.0,
    }
    model_cfg = {
        "base_channels": 8,
        "channel_multipliers": [1, 2, 4, 8],
        "use_cross_attention": True,
        "norm_groups": 4,
        "activation": "silu",
        "dropout_p": 0.0,
    }
    return DenserRadarNet(radar_cfg, model_cfg)


def test_end_to_end_backward():
    """Forward + loss + backward must work without errors."""
    model = _build_small_model()
    model.train()

    x = torch.randn(1, 4, _R, _E, _A)
    preds = model(x)
    targets = _make_targets(sparsity=0.9)

    loss, stats = hybrid_multiscale_loss(preds, targets, _LOSS_CFG)
    assert loss.requires_grad
    loss.backward()

    assert stats["loss_total"] > 0


def test_deep_supervision_gradients_reach_all_layers():
    """Deep supervision must push gradients into *every* learnable layer.

    Without deep supervision the bottleneck / early encoder gradients
    can vanish.  With losses at each decoder scale, even the deepest
    encoder layer receives a shorter gradient path.
    """
    model = _build_small_model()
    model.train()

    x = torch.randn(1, 4, _R, _E, _A)
    preds = model(x)
    targets = _make_targets(sparsity=0.9)
    loss, _ = hybrid_multiscale_loss(preds, targets, _LOSS_CFG)
    loss.backward()

    layers_to_check = [
        ("input_stem", model.input_stem),
        ("encoder_full", model.encoder_full),
        ("encoder_half", model.encoder_half),
        ("encoder_quarter", model.encoder_quarter),
        ("encoder_eighth", model.encoder_eighth),
        ("bottleneck", model.bottleneck),
        ("decoder_to_quarter", model.decoder_to_quarter),
        ("decoder_to_half", model.decoder_to_half),
        ("decoder_to_full", model.decoder_to_full),
        ("ds_head_quarter", model.ds_head_quarter),
        ("ds_head_half", model.ds_head_half),
        ("ds_head_full", model.ds_head_full),
        ("superres_upsample", model.superres_upsample),
        ("superres_head", model.superres_head),
    ]
    for name, module in layers_to_check:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in module.parameters() if p.requires_grad
        )
        assert has_grad, f"No gradient reached layer '{name}'"


def test_ds4_head_gets_gradient_independently():
    """The ds4 head sits at the coarsest decoder layer.
    Even if we zero out the HR / ds1 / ds2 losses, the ds4 loss alone
    must still produce gradients for the bottleneck and encoder."""
    model = _build_small_model()
    model.train()

    x = torch.randn(1, 4, _R, _E, _A)
    preds = model(x)
    targets = _make_targets(sparsity=0.9)

    cfg_ds4_only = {**_LOSS_CFG}
    preds_zeroed = {
        "pred_hr": preds["pred_hr"].detach(),
        "pred_ds1": preds["pred_ds1"].detach(),
        "pred_ds2": preds["pred_ds2"].detach(),
        "pred_ds4": preds["pred_ds4"],
    }
    loss, _ = hybrid_multiscale_loss(preds_zeroed, targets, cfg_ds4_only)
    loss.backward()

    for name in ("encoder_eighth", "bottleneck", "decoder_to_quarter"):
        module = getattr(model, name)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in module.parameters() if p.requires_grad
        )
        assert has_grad, (
            f"ds4-only loss did not produce gradient in '{name}'"
        )


# ===================================================================
# 8. Differentiability — loss must be smooth
# ===================================================================

def test_loss_is_differentiable():
    """Pred tensors with requires_grad=True must get non-None gradients."""
    targets = _make_targets()
    preds = {}
    for k, v in zip(
        ("pred_hr", "pred_ds1", "pred_ds2", "pred_ds4"),
        (targets["gt_occ_hr"], targets["gt_occ_ds1"],
         targets["gt_occ_ds2"], targets["gt_occ_ds4"]),
    ):
        t = torch.sigmoid(torch.randn_like(v, requires_grad=True))
        t.retain_grad()
        preds[k] = t

    loss, _ = hybrid_multiscale_loss(preds, targets, _LOSS_CFG)
    loss.backward()

    for key in preds:
        assert preds[key].grad is not None, f"No grad for {key}"
        assert preds[key].grad.abs().sum().item() > 0, f"Zero grad for {key}"
