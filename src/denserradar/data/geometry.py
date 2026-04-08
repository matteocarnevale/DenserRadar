"""3D geometry primitives: coordinate transforms, bounding boxes, FOV cropping.

Conventions
-----------
- Cartesian: (x, y, z) — x forward, y left, z up.
- Spherical: (range, elevation, azimuth) — elevation = angle from XY plane,
  azimuth = angle from X axis in the XY plane.  Both in radians.
- Transforms: 4x4 homogeneous matrices (rotation | translation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Bounding box dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Box3D:
    track_id: int
    class_name: str
    center: np.ndarray      # [x, y, z]
    size: np.ndarray         # [length, width, height]
    yaw: float               # rotation around Z axis (radians)
    velocity: np.ndarray | None = None  # [vx, vy, vz]


def parse_boxes(raw_boxes: Iterable[dict]) -> List[Box3D]:
    """Deserialize a list of JSON box dicts into Box3D objects."""
    return [
        Box3D(
            track_id=int(item.get("track_id", -1)),
            class_name=str(item.get("class_name", "unknown")),
            center=np.asarray(item["center"], dtype=np.float32),
            size=np.asarray(item["size"], dtype=np.float32),
            yaw=float(item.get("yaw", 0.0)),
            velocity=np.asarray(item.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32),
        )
        for item in raw_boxes
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  Homogeneous transforms
# ═══════════════════════════════════════════════════════════════════════════


def _to_homogeneous(points: np.ndarray) -> np.ndarray:
    """[N, 3] → [N, 4] by appending a column of ones."""
    if points.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.concatenate([points[:, :3], ones], axis=1)


def apply_transform(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an [N, 3] point cloud."""
    if points.size == 0:
        return points.copy()
    homogeneous = _to_homogeneous(points)
    transformed = (transform_4x4 @ homogeneous.T).T
    return transformed[:, :3]


def invert_transform(transform_4x4: np.ndarray) -> np.ndarray:
    return np.linalg.inv(transform_4x4)


def yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    """Rotation matrix for a Z-axis rotation by *yaw* radians."""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw,  cos_yaw, 0.0],
        [0.0,      0.0,     1.0],
    ], dtype=np.float32)


def box_to_transform(box: Box3D) -> np.ndarray:
    """Build a 4x4 transform that maps from the box's local frame to world."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = yaw_to_rotation_matrix(box.yaw)
    transform[:3, 3] = box.center.astype(np.float32)
    return transform


# ═══════════════════════════════════════════════════════════════════════════
#  Coordinate conversions
# ═══════════════════════════════════════════════════════════════════════════


def cartesian_to_spherical(points_xyz: np.ndarray) -> np.ndarray:
    """(x, y, z) → (range, elevation, azimuth).  All angles in radians."""
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]

    range_val = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    azimuth = np.arctan2(y, x)

    return np.stack([range_val, elevation, azimuth], axis=1).astype(np.float32)


def spherical_to_cartesian(points_sph: np.ndarray) -> np.ndarray:
    """(range, elevation, azimuth) → (x, y, z)."""
    if points_sph.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    r, el, az = points_sph[:, 0], points_sph[:, 1], points_sph[:, 2]

    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)

    return np.stack([x, y, z], axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Radar grid helpers
# ═══════════════════════════════════════════════════════════════════════════


def crop_points_by_range_and_fov(
    points_sph: np.ndarray,
    range_min: float, range_max: float,
    elev_min_deg: float, elev_max_deg: float,
    azim_min_deg: float, azim_max_deg: float,
) -> np.ndarray:
    """Keep only spherical points inside the radar's detection volume."""
    if points_sph.size == 0:
        return points_sph.copy()

    range_val = points_sph[:, 0]
    elevation = points_sph[:, 1]
    azimuth = points_sph[:, 2]

    inside = (
        (range_val >= range_min) & (range_val <= range_max)
        & (elevation >= np.deg2rad(elev_min_deg)) & (elevation <= np.deg2rad(elev_max_deg))
        & (azimuth >= np.deg2rad(azim_min_deg)) & (azimuth <= np.deg2rad(azim_max_deg))
    )
    return points_sph[inside]


def radar_bin_centers(config: dict, high_res_factor: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the physical center of each bin along range, elevation, azimuth."""
    n_range = config["range_bins"] * high_res_factor
    n_elev = config["elevation_bins"] * high_res_factor
    n_azim = config["azimuth_bins"] * high_res_factor

    range_centers = np.linspace(config["range_min_m"], config["range_max_m"], n_range, endpoint=False, dtype=np.float32)
    elev_centers = np.linspace(np.deg2rad(config["elevation_min_deg"]), np.deg2rad(config["elevation_max_deg"]),
                               n_elev, endpoint=False, dtype=np.float32)
    azim_centers = np.linspace(np.deg2rad(config["azimuth_min_deg"]), np.deg2rad(config["azimuth_max_deg"]),
                               n_azim, endpoint=False, dtype=np.float32)
    return range_centers, elev_centers, azim_centers


def quantize_spherical(points_sph: np.ndarray, config: dict, high_res_factor: int = 1) -> np.ndarray:
    """Map continuous spherical coordinates to integer voxel indices."""
    if points_sph.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    n_range = config["range_bins"] * high_res_factor
    n_elev = config["elevation_bins"] * high_res_factor
    n_azim = config["azimuth_bins"] * high_res_factor

    range_min, range_max = config["range_min_m"], config["range_max_m"]
    elev_min = np.deg2rad(config["elevation_min_deg"])
    elev_max = np.deg2rad(config["elevation_max_deg"])
    azim_min = np.deg2rad(config["azimuth_min_deg"])
    azim_max = np.deg2rad(config["azimuth_max_deg"])

    # Normalize to [0, 1) then multiply by bin count
    range_frac = np.clip((points_sph[:, 0] - range_min) / max(range_max - range_min, 1e-8), 0.0, 0.999999)
    elev_frac = np.clip((points_sph[:, 1] - elev_min) / max(elev_max - elev_min, 1e-8), 0.0, 0.999999)
    azim_frac = np.clip((points_sph[:, 2] - azim_min) / max(azim_max - azim_min, 1e-8), 0.0, 0.999999)

    range_idx = (range_frac * n_range).astype(np.int64)
    elev_idx = (elev_frac * n_elev).astype(np.int64)
    azim_idx = (azim_frac * n_azim).astype(np.int64)

    return np.stack([range_idx, elev_idx, azim_idx], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
#  Oriented bounding box mask
# ═══════════════════════════════════════════════════════════════════════════


def oriented_box_mask(
    points_xyz: np.ndarray,
    box: Box3D,
    scale: float = 1.0,
    z_padding_m: float = 0.0,
) -> np.ndarray:
    """Boolean mask: True for points inside the (optionally scaled) box."""
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=bool)

    rotation = yaw_to_rotation_matrix(box.yaw)
    relative_coords = points_xyz - box.center.reshape(1, 3)
    local_coords = relative_coords @ rotation  # rotate into box-local frame

    half_extents = 0.5 * box.size.astype(np.float32) * scale
    half_extents[2] += z_padding_m

    inside = (
        (np.abs(local_coords[:, 0]) <= half_extents[0])
        & (np.abs(local_coords[:, 1]) <= half_extents[1])
        & (np.abs(local_coords[:, 2]) <= half_extents[2])
    )
    return inside
