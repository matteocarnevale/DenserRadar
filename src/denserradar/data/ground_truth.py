"""Ground truth generation pipeline (paper Sec. III-B / Fig. 2).

Steps
-----
1. Split each LiDAR frame into static background and dynamic objects.
2. Remove ground plane from the static part (RANSAC).
3. Stitch static points across t neighbouring frames using ego-poses.
4. Stitch dynamic points across t neighbouring frames using box transforms.
5. Merge static + dynamic into one dense point cloud.
6. Transform from LiDAR frame to radar frame (extrinsics).
7. Convert to spherical coordinates, crop to radar FOV.
8. Voxelize into a high-resolution occupancy grid (2x radar resolution).
9. Gate with radar intensity (remove voxels the radar can't see).
10. Downsample to multiple scales for deep supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from denserradar.utils.io import load_json, load_npy
from .geometry import (
    Box3D,
    apply_transform,
    box_to_transform,
    cartesian_to_spherical,
    crop_points_by_range_and_fov,
    invert_transform,
    oriented_box_mask,
    parse_boxes,
)
from .voxelization import (
    downsample_occupancy_max,
    gate_occupancy_with_radar,
    hard_voxelize_spherical,
    soft_voxelize_spherical,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Data container for one sensor frame
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FrameData:
    sequence_id: str
    frame_id: int
    timestamp: float
    radar_tensor: np.ndarray            # [D, R, E, A]
    lidar_points: np.ndarray            # [N, 3]
    boxes: List[Box3D]
    lidar_to_radar: np.ndarray          # 4x4 extrinsic calibration
    ego_pose: np.ndarray                # 4x4 pose in world frame
    split: str
    synthetic_radar_tensor: np.ndarray | None = None


# ═══════════════════════════════════════════════════════════════════════════
#  Ground truth builder
# ═══════════════════════════════════════════════════════════════════════════


class GroundTruthBuilder:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.radar_cfg = cfg["radar"]
        self.gt_cfg = cfg["ground_truth"]

    # ── Public entry point ─────────────────────────────────────────────

    def build(
        self,
        frames_by_sequence: Dict[str, List[FrameData]],
        sequence_id: str,
        center_index: int,
    ) -> Dict[str, np.ndarray]:
        """Build multi-scale GT for the frame at *center_index* in the sequence."""

        sequence = frames_by_sequence[sequence_id]
        center_frame = sequence[center_index]

        # Step 1-2: separate static / dynamic, remove ground
        static_points, dynamic_points_by_track = self._split_static_dynamic(
            center_frame.lidar_points, center_frame.boxes,
        )
        if self.gt_cfg.get("remove_ground", True):
            static_points = self._remove_ground_ransac(static_points)

        # Step 3-4: multi-frame stitching
        temporal_window = self._get_temporal_window()
        stitched_static = self._stitch_static(sequence, center_index, static_points, temporal_window)
        stitched_dynamic = self._stitch_dynamic(sequence, center_index, dynamic_points_by_track, temporal_window)

        # Step 5: merge into one dense point cloud
        if stitched_dynamic.size > 0:
            dense_lidar = np.concatenate([stitched_static, stitched_dynamic], axis=0)
        else:
            dense_lidar = stitched_static

        # Step 6: transform to radar frame
        dense_radar_xyz = apply_transform(dense_lidar, center_frame.lidar_to_radar)

        # Step 7: convert to spherical and crop to radar FOV
        dense_radar_sph = cartesian_to_spherical(dense_radar_xyz)
        if self.gt_cfg.get("crop_to_radar_fov", True):
            dense_radar_sph = crop_points_by_range_and_fov(
                dense_radar_sph,
                self.radar_cfg["range_min_m"], self.radar_cfg["range_max_m"],
                self.radar_cfg["elevation_min_deg"], self.radar_cfg["elevation_max_deg"],
                self.radar_cfg["azimuth_min_deg"], self.radar_cfg["azimuth_max_deg"],
            )

        # Step 8: voxelize at high resolution
        hr_factor = int(self.gt_cfg.get("high_res_factor", 2))
        mode = self.gt_cfg.get("mode", "multi_frame_hard")

        if mode.endswith("soft"):
            occ_hr = soft_voxelize_spherical(
                dense_radar_sph, self.radar_cfg,
                high_res_factor=hr_factor,
                sigma_voxels=float(self.gt_cfg.get("soft_sigma_voxels", 0.8)),
            )
        else:
            occ_hr = hard_voxelize_spherical(dense_radar_sph, self.radar_cfg, high_res_factor=hr_factor)

        # Step 9: radar intensity gating
        occ_hr = gate_occupancy_with_radar(occ_hr, center_frame.radar_tensor, self.radar_cfg, hr_factor)

        # Step 10: multi-scale downsampling for deep supervision
        occ_ds1 = downsample_occupancy_max(occ_hr, factor=2)
        occ_ds2 = downsample_occupancy_max(occ_hr, factor=4)
        occ_ds4 = downsample_occupancy_max(occ_hr, factor=8)

        return {
            "gt_occ_hr": occ_hr.astype(np.float32),
            "gt_occ_ds1": occ_ds1.astype(np.float32),
            "gt_occ_ds2": occ_ds2.astype(np.float32),
            "gt_occ_ds4": occ_ds4.astype(np.float32),
            "gt_points_radar": dense_radar_xyz.astype(np.float32),
        }

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_temporal_window(self) -> int:
        mode = self.gt_cfg.get("mode", "multi_frame_hard")
        if mode.startswith("single_frame"):
            return 0
        return int(self.gt_cfg.get("temporal_window", 10))

    def _split_static_dynamic(
        self, lidar_points: np.ndarray, boxes: List[Box3D],
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Partition a LiDAR cloud into static background and per-object dynamic sets."""
        is_static = np.ones(lidar_points.shape[0], dtype=bool)
        dynamic_by_track: Dict[int, np.ndarray] = {}

        for box in boxes:
            inside = oriented_box_mask(
                lidar_points, box,
                scale=float(self.gt_cfg.get("dynamic_box_scale", 1.0)),
                z_padding_m=float(self.gt_cfg.get("box_z_padding_m", 0.2)),
            )
            if inside.any():
                dynamic_by_track[box.track_id] = lidar_points[inside]
                is_static[inside] = False

        return lidar_points[is_static], dynamic_by_track

    def _remove_ground_ransac(self, points: np.ndarray) -> np.ndarray:
        """Fit a ground plane via RANSAC, return only non-ground points."""
        if points.shape[0] < 32:
            return points

        iters = int(self.gt_cfg.get("ground_ransac_iters", 100))
        dist_threshold = float(self.gt_cfg.get("ground_distance_threshold_m", 0.15))

        best_ground_mask = None
        best_inlier_count = -1
        rng = np.random.default_rng(42)

        for _ in range(iters):
            sample = points[rng.choice(points.shape[0], size=3, replace=False)]
            p1, p2, p3 = sample

            normal = np.cross(p2 - p1, p3 - p1)
            norm_length = np.linalg.norm(normal)
            if norm_length < 1e-8:
                continue
            normal /= norm_length

            # Skip planes that aren't roughly horizontal (Z component < 0.7)
            if abs(normal[2]) < 0.7:
                continue

            plane_offset = -np.dot(normal, p1)
            distances = np.abs(points @ normal + plane_offset)
            ground_mask = distances < dist_threshold
            inlier_count = int(ground_mask.sum())

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_ground_mask = ground_mask

        if best_ground_mask is None:
            return points
        return points[~best_ground_mask]

    def _relative_pose(self, source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        """Transform that maps points from *source* frame into *target* frame."""
        return invert_transform(target_pose) @ source_pose

    def _stitch_static(
        self,
        sequence: List[FrameData],
        center_index: int,
        center_static: np.ndarray,
        temporal_window: int,
    ) -> np.ndarray:
        """Accumulate static points from neighbouring frames into the center frame."""
        accumulated = [center_static]
        center_pose = sequence[center_index].ego_pose

        for offset in range(1, temporal_window + 1):
            for direction in (-1, +1):
                neighbor_idx = center_index + direction * offset
                if neighbor_idx < 0 or neighbor_idx >= len(sequence):
                    continue

                neighbor = sequence[neighbor_idx]
                neighbor_static, _ = self._split_static_dynamic(neighbor.lidar_points, neighbor.boxes)
                if self.gt_cfg.get("remove_ground", True):
                    neighbor_static = self._remove_ground_ransac(neighbor_static)

                to_center = self._relative_pose(neighbor.ego_pose, center_pose)
                accumulated.append(apply_transform(neighbor_static, to_center))

        if not accumulated:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(accumulated, axis=0)

    def _stitch_dynamic(
        self,
        sequence: List[FrameData],
        center_index: int,
        center_dynamic: Dict[int, np.ndarray],
        temporal_window: int,
    ) -> np.ndarray:
        """Accumulate dynamic object points using box-to-box transforms."""
        center_frame = sequence[center_index]
        center_boxes = {box.track_id: box for box in center_frame.boxes}
        accumulated = []

        # Include the center frame's dynamic points as-is
        for points in center_dynamic.values():
            accumulated.append(points)

        # Stitch neighbouring frames
        for offset in range(1, temporal_window + 1):
            for direction in (-1, +1):
                neighbor_idx = center_index + direction * offset
                if neighbor_idx < 0 or neighbor_idx >= len(sequence):
                    continue

                neighbor = sequence[neighbor_idx]
                neighbor_boxes = {box.track_id: box for box in neighbor.boxes}
                _, neighbor_dynamic = self._split_static_dynamic(neighbor.lidar_points, neighbor.boxes)

                # Only stitch objects tracked in both the center and neighbor frame
                shared_tracks = set(center_boxes) & set(neighbor_boxes) & set(neighbor_dynamic)

                for track_id in shared_tracks:
                    source_transform = box_to_transform(neighbor_boxes[track_id])
                    target_transform = box_to_transform(center_boxes[track_id])
                    box_to_box = target_transform @ invert_transform(source_transform)
                    accumulated.append(apply_transform(neighbor_dynamic[track_id], box_to_box))

        if not accumulated:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(accumulated, axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Loading helpers (manifest → FrameData)
# ═══════════════════════════════════════════════════════════════════════════


def load_frame(record: dict, root: Optional[str | Path] = None) -> FrameData:
    """Load one frame's data from disk based on a manifest record."""
    base = Path(root) if root is not None else Path(".")

    radar = load_npy(base / record["radar_tensor_path"]).astype(np.float32)
    lidar = load_npy(base / record["lidar_points_path"]).astype(np.float32)
    boxes = parse_boxes(load_json(base / record["boxes_path"]))

    synthetic = None
    if record.get("synthetic_radar_tensor_path"):
        synthetic = load_npy(base / record["synthetic_radar_tensor_path"]).astype(np.float32)

    return FrameData(
        sequence_id=str(record["sequence_id"]),
        frame_id=int(record["frame_id"]),
        timestamp=float(record.get("timestamp", 0.0)),
        radar_tensor=radar,
        lidar_points=lidar,
        boxes=boxes,
        lidar_to_radar=np.asarray(record["lidar_to_radar"], dtype=np.float32),
        ego_pose=np.asarray(record["ego_pose"], dtype=np.float32),
        split=str(record.get("split", "train")),
        synthetic_radar_tensor=synthetic,
    )


def load_frames_grouped(manifest: list[dict], root: Optional[str | Path] = None) -> Dict[str, List[FrameData]]:
    """Load all frames from a manifest and group them by sequence_id, sorted by frame_id."""
    by_sequence: Dict[str, List[FrameData]] = {}
    for record in manifest:
        frame = load_frame(record, root=root)
        by_sequence.setdefault(frame.sequence_id, []).append(frame)

    for seq_id in by_sequence:
        by_sequence[seq_id].sort(key=lambda f: f.frame_id)

    return by_sequence
