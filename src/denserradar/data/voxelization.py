"""Spherical voxelization and radar intensity gating for GT generation.

The pipeline converts a dense spherical point cloud into a 3D occupancy
grid at *high_res_factor* times the native radar resolution, then removes
voxels where the actual radar signal is too weak (intensity gating).
"""
from __future__ import annotations

import math
import numpy as np

from .geometry import quantize_spherical


# ═══════════════════════════════════════════════════════════════════════════
#  Voxelization
# ═══════════════════════════════════════════════════════════════════════════


def hard_voxelize_spherical(
    points_sph: np.ndarray, radar_cfg: dict, high_res_factor: int = 2,
) -> np.ndarray:
    """Binary voxelization: each occupied voxel = 1.0, empty = 0.0."""
    n_range = radar_cfg["range_bins"] * high_res_factor
    n_elev = radar_cfg["elevation_bins"] * high_res_factor
    n_azim = radar_cfg["azimuth_bins"] * high_res_factor

    occupancy = np.zeros((1, n_range, n_elev, n_azim), dtype=np.float32)
    if points_sph.size == 0:
        return occupancy

    voxel_indices = quantize_spherical(points_sph, radar_cfg, high_res_factor)
    occupancy[0, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0
    return occupancy


def soft_voxelize_spherical(
    points_sph: np.ndarray,
    radar_cfg: dict,
    high_res_factor: int = 2,
    sigma_voxels: float = 0.8,
    truncation_voxels: int = 2,
) -> np.ndarray:
    """Gaussian-weighted voxelization: each point spreads a soft blob."""
    n_range = radar_cfg["range_bins"] * high_res_factor
    n_elev = radar_cfg["elevation_bins"] * high_res_factor
    n_azim = radar_cfg["azimuth_bins"] * high_res_factor

    occupancy = np.zeros((1, n_range, n_elev, n_azim), dtype=np.float32)
    if points_sph.size == 0:
        return occupancy

    voxel_indices = quantize_spherical(points_sph, radar_cfg, high_res_factor)
    neighbor_offsets = range(-truncation_voxels, truncation_voxels + 1)
    two_sigma_sq = 2.0 * sigma_voxels ** 2

    for center_r, center_e, center_a in voxel_indices:
        for dr in neighbor_offsets:
            nr = center_r + dr
            if nr < 0 or nr >= n_range:
                continue
            for de in neighbor_offsets:
                ne = center_e + de
                if ne < 0 or ne >= n_elev:
                    continue
                for da in neighbor_offsets:
                    na = center_a + da
                    if na < 0 or na >= n_azim:
                        continue
                    squared_dist = float(dr * dr + de * de + da * da)
                    weight = math.exp(-squared_dist / two_sigma_sq)
                    if weight > occupancy[0, nr, ne, na]:
                        occupancy[0, nr, ne, na] = weight

    return occupancy


# ═══════════════════════════════════════════════════════════════════════════
#  Radar intensity gating
# ═══════════════════════════════════════════════════════════════════════════


def collapse_doppler(radar_tensor: np.ndarray, mode: str = "max") -> np.ndarray:
    """Reduce the Doppler dimension [D, R, E, A] → [R, E, A] via max/sum/mean."""
    ops = {"max": np.max, "sum": np.sum, "mean": np.mean}
    if mode not in ops:
        raise ValueError(f"Unsupported reduce mode: {mode}")
    return ops[mode](radar_tensor, axis=0)


def compute_threshold(intensity_map: np.ndarray, mode: str, value: float) -> float:
    """Derive an intensity threshold from the map (absolute or percentile)."""
    if mode == "absolute":
        return float(value)
    if mode == "percentile":
        return float(np.percentile(intensity_map, value))
    raise ValueError(f"Unsupported threshold mode: {mode}")


def gate_occupancy_with_radar(
    occupancy_hr: np.ndarray,
    radar_tensor: np.ndarray,
    radar_cfg: dict,
    high_res_factor: int = 2,
) -> np.ndarray:
    """Zero out HR occupancy voxels where the radar signal is below threshold.

    This filters the LiDAR-derived GT to only retain voxels that the radar
    actually "sees", preventing impossible supervision targets.
    """
    assert occupancy_hr.ndim == 4 and occupancy_hr.shape[0] == 1

    intensity_map = collapse_doppler(radar_tensor, radar_cfg.get("intensity_reduce", "max"))
    threshold = compute_threshold(
        intensity_map,
        radar_cfg.get("intensity_threshold_mode", "percentile"),
        radar_cfg.get("intensity_threshold_value", 85.0),
    )

    gated = occupancy_hr.copy()
    occupied_voxels = np.argwhere(occupancy_hr[0] > 0)

    for hr_r, hr_e, hr_a in occupied_voxels:
        # Map HR voxel back to native-resolution radar bin
        native_r = min(hr_r // high_res_factor, intensity_map.shape[0] - 1)
        native_e = min(hr_e // high_res_factor, intensity_map.shape[1] - 1)
        native_a = min(hr_a // high_res_factor, intensity_map.shape[2] - 1)

        if intensity_map[native_r, native_e, native_a] < threshold:
            gated[0, hr_r, hr_e, hr_a] = 0.0

    return gated


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-scale downsampling for deep supervision
# ═══════════════════════════════════════════════════════════════════════════


def downsample_occupancy_max(occupancy_hr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample via block max-pooling, with zero-padding for non-divisible dims."""
    if factor == 1:
        return occupancy_hr.copy()
    assert occupancy_hr.ndim == 4 and occupancy_hr.shape[0] == 1

    _, n_r, n_e, n_a = occupancy_hr.shape

    # Pad to the next multiple of *factor* if needed
    pad_r = (factor - n_r % factor) % factor
    pad_e = (factor - n_e % factor) % factor
    pad_a = (factor - n_a % factor) % factor
    if pad_r or pad_e or pad_a:
        occupancy_hr = np.pad(occupancy_hr, ((0, 0), (0, pad_r), (0, pad_e), (0, pad_a)),
                              mode="constant", constant_values=0)
        _, n_r, n_e, n_a = occupancy_hr.shape

    # Reshape into blocks of size *factor* and take the max of each block
    blocked = occupancy_hr.reshape(
        1,
        n_r // factor, factor,
        n_e // factor, factor,
        n_a // factor, factor,
    )
    downsampled = blocked.max(axis=(2, 4, 6))
    return downsampled.astype(np.float32)
