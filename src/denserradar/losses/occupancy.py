"""Weighted hybrid loss with deep supervision (paper Eq. 8).

    L = sum_i  1/2^i * ( Dice_i  +  lambda_F * Focal_i )

Dice loss  — handles the extreme class imbalance between occupied
             and empty voxels in occupancy grids.
Focal loss — down-weights easy-to-classify (background) voxels so
             the network focuses on hard positives near surfaces.

Both are computed at every decoder scale (HR, ds1, ds2, ds4) with
geometrically decaying weights to provide shorter gradient paths
to the deeper encoder layers.
"""
from __future__ import annotations

from typing import Dict, Tuple
import torch


# ─── Individual losses ────────────────────────────────────────────────────


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss.  Returns 1 - Dice coefficient (lower is better)."""
    pred_flat = pred.contiguous().view(pred.shape[0], -1)
    target_flat = target.contiguous().view(target.shape[0], -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice_score.mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal loss — reduces contribution of easy negatives."""
    pred_safe = pred.clamp(eps, 1.0 - eps)

    # Standard binary cross-entropy (unreduced)
    bce = -(target * torch.log(pred_safe) + (1.0 - target) * torch.log(1.0 - pred_safe))

    # p_t = model's estimated probability for the *true* class
    p_t = torch.where(target > 0.5, pred_safe, 1.0 - pred_safe)

    # alpha weighting: alpha for positives, (1-alpha) for negatives
    alpha_weight = torch.where(target > 0.5, alpha, 1.0 - alpha)

    # Focal modulation: (1 - p_t)^gamma  shrinks loss for easy examples
    focal_weight = (1.0 - p_t) ** gamma

    return (alpha_weight * focal_weight * bce).mean()


# ─── Spatial safety ───────────────────────────────────────────────────────


def _match_spatial(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop both tensors to the smaller spatial extent (handles odd dims)."""
    if pred.shape == target.shape:
        return pred, target
    min_r = min(pred.shape[2], target.shape[2])
    min_e = min(pred.shape[3], target.shape[3])
    min_a = min(pred.shape[4], target.shape[4])
    return pred[:, :, :min_r, :min_e, :min_a], target[:, :, :min_r, :min_e, :min_a]


# ─── Combined multi-scale loss ────────────────────────────────────────────

# Scale name → (pred key, target key, weight following 1/2^i)
DEEP_SUPERVISION_SCALES = [
    ("pred_hr",  "gt_occ_hr",  1.0),     # i=0  — highest resolution
    ("pred_ds1", "gt_occ_ds1", 0.5),     # i=1  — native radar resolution
    ("pred_ds2", "gt_occ_ds2", 0.25),    # i=2  — half
    ("pred_ds4", "gt_occ_ds4", 0.125),   # i=3  — quarter
]


def hybrid_multiscale_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    cfg: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the weighted hybrid loss across all supervision scales."""

    lambda_focal = float(cfg.get("lambda_focal", 700.0))
    alpha = float(cfg.get("focal_alpha", 0.25))
    gamma = float(cfg.get("focal_gamma", 2.0))
    smooth = float(cfg.get("smooth", 1.0))

    total_loss = preds["pred_hr"].new_tensor(0.0)
    stats: Dict[str, float] = {}

    for pred_key, target_key, scale_weight in DEEP_SUPERVISION_SCALES:
        pred, target = _match_spatial(preds[pred_key], targets[target_key])

        dice_val = dice_loss(pred, target, smooth=smooth)
        focal_val = focal_loss(pred, target, alpha=alpha, gamma=gamma)

        scale_loss = scale_weight * (dice_val + lambda_focal * focal_val)
        total_loss = total_loss + scale_loss

        stats[f"{pred_key}_dice"] = float(dice_val.detach().cpu())
        stats[f"{pred_key}_focal"] = float(focal_val.detach().cpu())

    stats["loss_total"] = float(total_loss.detach().cpu())
    return total_loss, stats
