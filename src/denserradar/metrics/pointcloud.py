"""RP-CD and RP-CA evaluation metrics (paper Sec. IV-B).

RP-CD (Radar Point Cloud Density):
    Fraction of ground-truth LiDAR points that have at least one
    predicted radar point within delta_d metres.
    Higher = the radar PC covers more of the scene.

RP-CA (Radar Point Cloud Accuracy):
    Fraction of predicted radar points that have at least one
    ground-truth LiDAR point within delta_a metres.
    Higher = fewer false positives in the radar PC.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
from scipy.spatial import cKDTree

from denserradar.data.geometry import spherical_to_cartesian


def _hit_rate(tree: cKDTree, query_points: np.ndarray, radius: float) -> float:
    """What fraction of *query_points* have a nearest neighbour in *tree* within *radius*?"""
    if query_points.shape[0] == 0 or tree.n == 0:
        return 0.0
    nearest_distances, _ = tree.query(query_points, k=1, workers=-1)
    return float(np.mean(nearest_distances <= radius))


def occupancy_to_points_xyz(
    occupancy: torch.Tensor,
    radar_cfg: dict,
    threshold: float = 0.5,
    high_res_factor: int = 2,
    max_points: Optional[int] = None,
) -> np.ndarray:
    """Convert a predicted occupancy grid back to Cartesian (x, y, z) points.

    1. Find voxels above *threshold*.
    2. Optionally keep only the top-*max_points* by confidence.
    3. Map voxel indices → spherical coords → Cartesian.
    """
    occ_np = occupancy.detach().cpu().numpy()
    while occ_np.ndim > 3:
        occ_np = occ_np[0]

    active_voxels = np.argwhere(occ_np >= threshold)
    if active_voxels.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Keep only top-k if requested
    if max_points is not None and active_voxels.shape[0] > max_points:
        scores = occ_np[active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]]
        top_indices = np.argsort(scores)[-max_points:]
        active_voxels = active_voxels[top_indices]

    # Grid dimensions
    range_bins = radar_cfg["range_bins"] * high_res_factor
    elev_bins = radar_cfg["elevation_bins"] * high_res_factor
    azim_bins = radar_cfg["azimuth_bins"] * high_res_factor

    # Physical extents
    range_min, range_max = radar_cfg["range_min_m"], radar_cfg["range_max_m"]
    elev_min = np.deg2rad(radar_cfg["elevation_min_deg"])
    elev_max = np.deg2rad(radar_cfg["elevation_max_deg"])
    azim_min = np.deg2rad(radar_cfg["azimuth_min_deg"])
    azim_max = np.deg2rad(radar_cfg["azimuth_max_deg"])

    # Voxel index → physical spherical coordinate (use voxel center)
    range_vals = range_min + (active_voxels[:, 0] + 0.5) / range_bins * (range_max - range_min)
    elev_vals = elev_min + (active_voxels[:, 1] + 0.5) / elev_bins * (elev_max - elev_min)
    azim_vals = azim_min + (active_voxels[:, 2] + 0.5) / azim_bins * (azim_max - azim_min)

    points_spherical = np.stack([range_vals, elev_vals, azim_vals], axis=1).astype(np.float32)
    return spherical_to_cartesian(points_spherical)


def occupancy_to_points_xyz_cartesian(
    occupancy: torch.Tensor,
    point_cloud_range: list[float],
    threshold: float = 0.5,
    max_points: Optional[int] = None,
) -> np.ndarray:
    """Convert a cartesian occupancy grid (Z,Y,X) to Cartesian (x,y,z) points (voxel centers)."""
    occ_np = occupancy.detach().cpu().numpy()
    while occ_np.ndim > 3:
        occ_np = occ_np[0]

    active = np.argwhere(occ_np >= threshold)
    if active.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if max_points is not None and active.shape[0] > max_points:
        scores = occ_np[active[:, 0], active[:, 1], active[:, 2]]
        top = np.argsort(scores)[-max_points:]
        active = active[top]

    nz, ny, nx = occ_np.shape
    pc_min = np.asarray(point_cloud_range[:3], dtype=np.float32)
    pc_max = np.asarray(point_cloud_range[3:], dtype=np.float32)
    ext = pc_max - pc_min
    vx = float(ext[0] / nx)
    vy = float(ext[1] / ny)
    vz = float(ext[2] / nz)

    # active is (z,y,x)
    x = pc_min[0] + (active[:, 2].astype(np.float32) + 0.5) * vx
    y = pc_min[1] + (active[:, 1].astype(np.float32) + 0.5) * vy
    z = pc_min[2] + (active[:, 0].astype(np.float32) + 0.5) * vz
    return np.stack([x, y, z], axis=1).astype(np.float32)


def rpcd_rpca(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    rpcd_radius: float,
    rpca_radius: float,
) -> Tuple[float, float]:
    """Compute both RP-CD and RP-CA in one call."""
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return 0.0, 0.0

    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)

    rpcd = _hit_rate(pred_tree, gt_points, rpcd_radius)
    rpca = _hit_rate(gt_tree, pred_points, rpca_radius)
    return float(rpcd), float(rpca)
