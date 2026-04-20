"""VoD dataset loader (MSDNet-style bin layout) with on-the-fly occupancy GT.

Expected layout:
  root/
    lidar/<frame_id>.bin   float32 (N,4)  columns: x,y,z,intensity
    radar/<frame_id>.bin   float32 (M,5)  columns: x,y,z,intensity,velocity
    split/train.txt|test.txt|val.txt   one frame_id per line (no extension)

This loader is intended to support training the DenserRadar-style 3D U-Net on
**cartesian** voxel grids by:
  - voxelizing radar points into an input feature volume (ds1 resolution)
  - voxelizing LiDAR points into multi-scale occupancy targets (hr/ds1/ds2/ds4)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_VOD_4DRVO_TEST_SEQS = frozenset({"03", "04", "22"})


def extract_vod_sequence_id(frame_id: str) -> Optional[str]:
    """Two-digit sequence id from a VoD-style frame id, or None."""
    fid = frame_id.replace("\\", "/")
    m = re.search(r"delft_(\d+)_", fid, flags=re.I)
    if m:
        return f"{int(m.group(1)):02d}"
    m = re.match(r"^(\d{2})_\d+", fid)
    if m:
        return m.group(1)
    m = re.match(r"^(\d{2})\d{6,}$", fid)
    if m:
        return m.group(1)
    return None


def _filter_frame_ids_4drvo_net(frame_ids: List[str], split: str) -> List[str]:
    out: List[str] = []
    skipped = 0
    for fid in frame_ids:
        seq = extract_vod_sequence_id(fid)
        if seq is None:
            skipped += 1
            continue
        in_test = seq in _VOD_4DRVO_TEST_SEQS
        if split == "train" and not in_test:
            out.append(fid)
        elif split == "test" and in_test:
            out.append(fid)
        elif split not in ("train", "test"):
            if not in_test:
                out.append(fid)
    if skipped:
        print(
            f"VoD 4drvo_net filter: skipped {skipped} ids (unparsable sequence; expected e.g. delft_03_...)"
        )
    return out


def remove_ground_elevation_map(
    points: np.ndarray,
    ground_height: float = -1.5,
    grid_size: float = 0.5,
    height_threshold: float = 0.3,
) -> np.ndarray:
    """Remove ground points using a simple BEV elevation map heuristic."""
    if points.shape[0] == 0:
        return points

    xyz = points[:, :3]
    x_min, x_max = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y_min, y_max = float(xyz[:, 1].min()), float(xyz[:, 1].max())

    if x_max - x_min < 0.1 or y_max - y_min < 0.1:
        return points[xyz[:, 2] > ground_height]

    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1

    ground_heights = np.full((x_bins, y_bins), np.inf, dtype=np.float32)
    x_indices = np.clip(((xyz[:, 0] - x_min) / grid_size).astype(np.int64), 0, x_bins - 1)
    y_indices = np.clip(((xyz[:, 1] - y_min) / grid_size).astype(np.int64), 0, y_bins - 1)

    for i in range(xyz.shape[0]):
        xi = int(x_indices[i])
        yi = int(y_indices[i])
        z = float(xyz[i, 2])
        if z < ground_heights[xi, yi]:
            ground_heights[xi, yi] = z

    keep = np.zeros((xyz.shape[0],), dtype=bool)
    for i in range(xyz.shape[0]):
        xi = int(x_indices[i])
        yi = int(y_indices[i])
        z_ground = float(ground_heights[xi, yi])
        if np.isinf(z_ground):
            keep[i] = float(xyz[i, 2]) > ground_height
        else:
            keep[i] = (float(xyz[i, 2]) - z_ground) > height_threshold
    return points[keep]


def _grid_dims_from_range(point_cloud_range: List[float], voxel_size: List[float]) -> Tuple[int, int, int]:
    pc = point_cloud_range
    vs = voxel_size
    nx = int((pc[3] - pc[0]) / vs[0])
    ny = int((pc[4] - pc[1]) / vs[1])
    nz = int((pc[5] - pc[2]) / vs[2])
    return nx, ny, nz


def _voxelize_cartesian_occupancy(
    points_xyz: np.ndarray,
    point_cloud_range: List[float],
    dims_zyx: Tuple[int, int, int],
) -> np.ndarray:
    """Binary occupancy voxelization into a (1,Z,Y,X) grid."""
    nz, ny, nx = dims_zyx
    occ = np.zeros((1, nz, ny, nx), dtype=np.float32)
    if points_xyz.shape[0] == 0:
        return occ

    pc_min = np.asarray(point_cloud_range[:3], dtype=np.float32)
    pc_max = np.asarray(point_cloud_range[3:], dtype=np.float32)
    ext = pc_max - pc_min

    vx = float(ext[0] / nx)
    vy = float(ext[1] / ny)
    vz = float(ext[2] / nz)

    ix = np.floor((points_xyz[:, 0] - pc_min[0]) / vx).astype(np.int64)
    iy = np.floor((points_xyz[:, 1] - pc_min[1]) / vy).astype(np.int64)
    iz = np.floor((points_xyz[:, 2] - pc_min[2]) / vz).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    ix, iy, iz = ix[valid], iy[valid], iz[valid]
    occ[0, iz, iy, ix] = 1.0
    return occ


def _voxelize_radar_points_to_volume(
    radar_points: np.ndarray,
    point_cloud_range: List[float],
    dims_zyx: Tuple[int, int, int],
    feature: str,
) -> np.ndarray:
    """Voxelize radar points (M,5) into a (1,Z,Y,X) input volume."""
    nz, ny, nx = dims_zyx
    vol = np.zeros((1, nz, ny, nx), dtype=np.float32)
    if radar_points.shape[0] == 0:
        return vol

    pc_min = np.asarray(point_cloud_range[:3], dtype=np.float32)
    pc_max = np.asarray(point_cloud_range[3:], dtype=np.float32)
    ext = pc_max - pc_min
    vx = float(ext[0] / nx)
    vy = float(ext[1] / ny)
    vz = float(ext[2] / nz)

    xyz = radar_points[:, :3]
    ix = np.floor((xyz[:, 0] - pc_min[0]) / vx).astype(np.int64)
    iy = np.floor((xyz[:, 1] - pc_min[1]) / vy).astype(np.int64)
    iz = np.floor((xyz[:, 2] - pc_min[2]) / vz).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    ix, iy, iz = ix[valid], iy[valid], iz[valid]

    if feature == "occupancy":
        vol[0, iz, iy, ix] = 1.0
        return vol

    if feature == "count":
        # count points per voxel
        for xi, yi, zi in zip(ix.tolist(), iy.tolist(), iz.tolist()):
            vol[0, zi, yi, xi] += 1.0
        return vol

    if feature == "max_intensity":
        intens = radar_points[:, 3].astype(np.float32)[valid]
        for xi, yi, zi, s in zip(ix.tolist(), iy.tolist(), iz.tolist(), intens.tolist()):
            if s > vol[0, zi, yi, xi]:
                vol[0, zi, yi, xi] = s
        return vol

    raise ValueError(f"Unknown radar_input_feature={feature!r}; use 'occupancy', 'count', or 'max_intensity'")


@dataclass(frozen=True)
class VoDSample:
    frame_id: str
    radar_points: np.ndarray  # (M,5)
    lidar_points: np.ndarray  # (N,4)


class VoDDataset(Dataset):
    """Loads VoD `.bin` pairs and generates cartesian occupancy GT on-the-fly."""

    def __init__(self, cfg: dict, split: str):
        super().__init__()
        vod_cfg = cfg.get("vod", {})
        self.root = str(vod_cfg.get("root", "data/vod"))
        self.split = str(split)

        self.verify_files = bool(vod_cfg.get("verify_files", True))
        self.vod_sequence_filter = vod_cfg.get("vod_sequence_filter", None)

        self.radar_fov_deg = float(vod_cfg.get("radar_fov_deg", 120.0))
        self.ground_height = float(vod_cfg.get("ground_height", -1.5))
        self.elev_grid_size_m = float(vod_cfg.get("elevation_grid_size_m", 0.5))
        self.elev_height_threshold_m = float(vod_cfg.get("elevation_height_threshold_m", 0.3))

        self.point_cloud_range = list(vod_cfg.get("point_cloud_range", [0.0, -16.0, -2.0, 32.0, 16.0, 4.0]))
        self.voxel_size = list(vod_cfg.get("voxel_size", [0.1, 0.1, 0.15]))
        self.radar_input_feature = str(vod_cfg.get("radar_input_feature", "max_intensity"))

        split_file = os.path.join(self.root, "split", f"{self.split}.txt")
        with open(split_file, "r", encoding="utf-8") as f:
            frame_ids = [line.strip() for line in f if line.strip()]

        if self.vod_sequence_filter == "4drvo_net":
            n0 = len(frame_ids)
            frame_ids = _filter_frame_ids_4drvo_net(frame_ids, self.split)
            print(f"VoD filter 4drvo_net split={self.split!r}: {n0} -> {len(frame_ids)} frames")
        elif self.vod_sequence_filter not in (None, ""):
            raise ValueError(f"Unknown vod_sequence_filter={self.vod_sequence_filter!r}; use None or '4drvo_net'")

        if self.verify_files:
            ok: List[str] = []
            missing = 0
            for fid in frame_ids:
                lp = os.path.join(self.root, "lidar", f"{fid}.bin")
                rp = os.path.join(self.root, "radar", f"{fid}.bin")
                if os.path.exists(lp) and os.path.exists(rp):
                    ok.append(fid)
                else:
                    missing += 1
            if missing:
                print(f"VoD: {len(ok)} valid pairs ({missing} missing files)")
            self.frame_ids = ok
        else:
            self.frame_ids = frame_ids

        # Precompute base grid dims (ds1)
        nx, ny, nz = _grid_dims_from_range(self.point_cloud_range, self.voxel_size)
        self.dims_ds1_zyx = (nz, ny, nx)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_pair(self, fid: str) -> VoDSample:
        lidar_path = os.path.join(self.root, "lidar", f"{fid}.bin")
        radar_path = os.path.join(self.root, "radar", f"{fid}.bin")

        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        radar = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 5)
        return VoDSample(frame_id=fid, radar_points=radar, lidar_points=lidar)

    def _crop_to_fov(self, pc: np.ndarray) -> np.ndarray:
        half = np.deg2rad(self.radar_fov_deg / 2.0)
        angles = np.arctan2(pc[:, 1], pc[:, 0])
        return pc[np.abs(angles) <= half]

    def _crop_to_range(self, pc: np.ndarray) -> np.ndarray:
        pc_range = self.point_cloud_range
        mask = (
            (pc[:, 0] >= pc_range[0]) & (pc[:, 0] < pc_range[3]) &
            (pc[:, 1] >= pc_range[1]) & (pc[:, 1] < pc_range[4]) &
            (pc[:, 2] >= pc_range[2]) & (pc[:, 2] < pc_range[5])
        )
        return pc[mask]

    def _preprocess_lidar(self, lidar: np.ndarray) -> np.ndarray:
        if lidar.shape[0] == 0:
            return lidar
        lidar = remove_ground_elevation_map(
            lidar,
            ground_height=self.ground_height,
            grid_size=self.elev_grid_size_m,
            height_threshold=self.elev_height_threshold_m,
        )
        lidar = self._crop_to_fov(lidar)
        lidar = self._crop_to_range(lidar)
        return lidar

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fid = self.frame_ids[idx]
        sample = self._load_pair(fid)

        lidar = self._preprocess_lidar(sample.lidar_points)
        radar = sample.radar_points.astype(np.float32, copy=False)

        # Radar input volume at ds1 resolution
        radar_volume = _voxelize_radar_points_to_volume(
            radar,
            point_cloud_range=self.point_cloud_range,
            dims_zyx=self.dims_ds1_zyx,
            feature=self.radar_input_feature,
        )

        # Multi-scale occupancy targets from LiDAR points
        nx = self.dims_ds1_zyx[2]
        ny = self.dims_ds1_zyx[1]
        nz = self.dims_ds1_zyx[0]
        dims_hr = (nz * 2, ny * 2, nx * 2)
        dims_ds1 = (nz, ny, nx)
        dims_ds2 = (max(1, nz // 2), max(1, ny // 2), max(1, nx // 2))
        dims_ds4 = (max(1, nz // 4), max(1, ny // 4), max(1, nx // 4))

        lidar_xyz = lidar[:, :3].astype(np.float32, copy=False)
        gt_occ_hr = _voxelize_cartesian_occupancy(lidar_xyz, self.point_cloud_range, dims_hr)
        gt_occ_ds1 = _voxelize_cartesian_occupancy(lidar_xyz, self.point_cloud_range, dims_ds1)
        gt_occ_ds2 = _voxelize_cartesian_occupancy(lidar_xyz, self.point_cloud_range, dims_ds2)
        gt_occ_ds4 = _voxelize_cartesian_occupancy(lidar_xyz, self.point_cloud_range, dims_ds4)

        return {
            "frame_id": fid,
            "radar_points": torch.from_numpy(radar),                 # (M,5)
            "radar_volume": torch.from_numpy(radar_volume),          # (1,Z,Y,X)
            "gt_occ_hr": torch.from_numpy(gt_occ_hr),                # (1,2Z,2Y,2X)
            "gt_occ_ds1": torch.from_numpy(gt_occ_ds1),              # (1,Z,Y,X)
            "gt_occ_ds2": torch.from_numpy(gt_occ_ds2),
            "gt_occ_ds4": torch.from_numpy(gt_occ_ds4),
            "gt_points_radar": torch.from_numpy(lidar_xyz),          # cartesian GT points (name kept for trainer)
        }


def vod_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stacks fixed-size tensors, keeps variable-length point clouds as lists."""
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            if key in ("radar_points", "gt_points_radar"):
                collated[key] = values
            else:
                collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated

