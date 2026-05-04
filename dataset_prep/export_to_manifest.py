#!/usr/bin/env python3
"""
Esporta un dataset (layout RADIal / custom) verso il formato DenserRadar:
  <output.root>/<sequence_id>/{radar,lidar,boxes}/*.npy|json
  <output.root>/<manifest_filename>

Non modifica la pipeline in src/ — produce solo file consumabili da
  scripts/precompute_ground_truth.py  e  scripts/train.py
con un config YAML i cui radar.*_bins coincidano con target_shape (es. configs/radial_grid.yaml).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml
from scipy.ndimage import zoom
from tqdm import tqdm


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_matrix(cfg: dict, key: str, npy_key: str) -> np.ndarray:
    p = cfg.get(npy_key)
    if p:
        m = np.load(str(Path(p).expanduser()), allow_pickle=False)
        if m.shape != (4, 4):
            raise ValueError(f"{npy_key} must be (4,4), got {m.shape}")
        return m.astype(np.float32)
    if cfg.get(key) == "identity":
        return np.eye(4, dtype=np.float32)
    raise ValueError(f"Set {key} to 'identity' or provide {npy_key}")


def _discover_frame_ids(source_root: Path, pattern: str, id_regex: str) -> List[int]:
    paths = sorted(source_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {source_root}/{pattern}")
    cre = re.compile(id_regex)
    ids: List[int] = []
    for p in paths:
        m = cre.search(p.name)
        if not m:
            raise ValueError(f"Could not parse frame id from {p.name} with regex {id_regex!r}")
        ids.append(int(m.group(1)))
    return sorted(set(ids))


def _load_frame_list(path: Path) -> List[int]:
    out: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(int(line))
    return out


def _layout_to_drae(arr: np.ndarray, layout: str) -> np.ndarray:
    """Return array shaped (D, R, E, A).

    Layouts supportati:
      DRAE  — già (D, R, E, A), nessuna permuta.
      RDEA  — (R, D, E, A) da build_radial_raed_dataset.py → transpose (1,0,2,3).
      RAEC  — (R, A, E, C) stile Radar-Mamba         → transpose (3,0,2,1).
      RAED  — (R, A, E, D) alias di RAEC             → transpose (3,0,2,1).
      RAE   — (R, A, E) 3D                           → newaxis → (1,R,A,E).
    """
    layout = layout.upper()
    if layout == "DRAE":
        if arr.ndim != 4:
            raise ValueError(f"DRAE expects 4D, got {arr.shape}")
        return arr
    if layout == "RDEA":
        # (R, D, E, A) → (D, R, E, A)
        if arr.ndim != 4:
            raise ValueError(f"RDEA expects 4D (R,D,E,A), got {arr.shape}")
        return np.transpose(arr, (1, 0, 2, 3))
    if layout in ("RAEC", "RAED"):
        # (R, A, E, C) → (C, R, E, A)
        if arr.ndim != 4:
            raise ValueError(f"{layout} expects 4D (R,A,E,C), got {arr.shape}")
        return np.transpose(arr, (3, 0, 2, 1))
    if layout == "RAE":
        if arr.ndim != 3:
            raise ValueError(f"RAE expects 3D, got {arr.shape}")
        return arr[np.newaxis, ...]
    raise ValueError(f"Unknown source_layout: {layout!r}. Valid: DRAE, RDEA, RAEC, RAED, RAE")


def _resize_drae(arr: np.ndarray, target: Tuple[int, int, int, int], mode: str) -> np.ndarray:
    if tuple(arr.shape) == target:
        return arr
    if mode == "fail":
        raise ValueError(f"Shape {arr.shape} != target {target}; set radar.resize_mode: zoom")
    if mode == "none":
        return arr
    if mode == "zoom":
        factors = tuple(t / s for t, s in zip(target, arr.shape))
        return zoom(arr, factors, order=1).astype(np.float32)
    raise ValueError(f"Unknown resize_mode: {mode}")


def _load_lidar(path: Path, col: Sequence[int]) -> np.ndarray:
    x = np.load(str(path), allow_pickle=False)
    if x.ndim == 1:
        raise ValueError(f"LiDAR at {path} unexpected 1D shape {x.shape}")
    if x.shape[1] < 3:
        raise ValueError(f"LiDAR at {path} has shape {x.shape}, need at least 3 columns")
    i, j, k = col[0], col[1], col[2]
    return np.ascontiguousarray(x[:, [i, j, k]], dtype=np.float32)


def _format_path(tpl: str, source_root: Path, frame_id: int, padding: int) -> Path:
    fid = f"{frame_id:0{padding}d}"
    rel = tpl.format(frame_id=frame_id, frame_id_padded=fid)
    return source_root / rel


def _assign_splits(
    frame_ids: List[int],
    train_ratio: float,
    shuffle_seed: int,
    train_name: str,
    val_name: str,
) -> Dict[int, str]:
    rng = np.random.default_rng(shuffle_seed)
    ids = list(frame_ids)
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    train_ids = set(ids[:n_train])
    return {fid: (train_name if fid in train_ids else val_name) for fid in frame_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dataset to DenserRadar manifest layout")
    parser.add_argument("--config", type=Path, required=True, help="YAML config (see dataset_prep/configs/)")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned actions")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    out_cfg = cfg["output"]
    paths_cfg = cfg["paths"]
    radar_cfg = cfg["radar"]
    frames_cfg = cfg["frames"]
    split_cfg = cfg["split"]
    lidar_cfg = cfg["lidar"]
    cal_cfg = cfg["calibration"]
    boxes_cfg = cfg["boxes"]

    source_root = Path(paths_cfg["source_root"]).expanduser().resolve()
    output_root = Path(out_cfg["root"]).expanduser().resolve()
    seq_id = str(out_cfg["sequence_id"])
    seq_dir = output_root / seq_id
    radar_dir = seq_dir / out_cfg["subdirs"]["radar"]
    lidar_dir = seq_dir / out_cfg["subdirs"]["lidar"]
    boxes_dir = seq_dir / out_cfg["subdirs"]["boxes"]

    lidar_to_radar = _load_matrix(cal_cfg, "lidar_to_radar", "lidar_to_radar_npy")
    ego_pose = _load_matrix(cal_cfg, "ego_pose", "ego_pose_npy")

    target_shape = tuple(radar_cfg["target_shape"])
    if len(target_shape) != 4:
        raise ValueError("radar.target_shape must be [D, R, E, A]")

    list_file = frames_cfg.get("list_file")
    if list_file:
        frame_ids = _load_frame_list(Path(list_file).expanduser())
    else:
        frame_ids = _discover_frame_ids(
            source_root,
            frames_cfg["discover_glob"],
            frames_cfg["frame_id_regex"],
        )

    if not frame_ids:
        print("No frames found.", file=sys.stderr)
        sys.exit(1)

    radar_tpl = paths_cfg["radar_template"]
    lidar_tpl = paths_cfg["lidar_template"]
    pad = int(paths_cfg.get("id_padding", 6))
    skip_missing = bool(lidar_cfg.get("skip_if_missing", False))

    if args.dry_run:
        print(f"Would export up to {len(frame_ids)} frames to {output_root}")
        print(f"sequence_id={seq_id}, sample ids={frame_ids[:8]}{'...' if len(frame_ids) > 8 else ''}")
        return

    radar_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    boxes_dir.mkdir(parents=True, exist_ok=True)

    exported: List[int] = []

    for fid in tqdm(frame_ids, desc="Export"):
        r_path = _format_path(radar_tpl, source_root, fid, pad)
        l_path = _format_path(lidar_tpl, source_root, fid, pad)

        if not r_path.is_file():
            raise FileNotFoundError(f"Missing radar file: {r_path}")
        if not l_path.is_file():
            if skip_missing:
                tqdm.write(f"WARN: skip frame {fid}: missing lidar {l_path}")
                continue
            raise FileNotFoundError(f"Missing lidar file: {l_path}")

        raw = np.load(str(r_path), allow_pickle=False).astype(np.float32)
        drae = _layout_to_drae(raw, radar_cfg["source_layout"])
        drae = _resize_drae(drae, target_shape, radar_cfg.get("resize_mode", "zoom"))

        if tuple(drae.shape) != target_shape:
            raise RuntimeError(f"After resize got {drae.shape}, expected {target_shape}")

        np.save(str(radar_dir / f"{fid:06d}.npy"), np.ascontiguousarray(drae.astype(np.float32)))

        lidar_xyz = _load_lidar(l_path, lidar_cfg["columns_xyz"])
        np.save(str(lidar_dir / f"{fid:06d}.npy"), lidar_xyz)

        if boxes_cfg.get("empty_list", True):
            (boxes_dir / f"{fid:06d}.json").write_text("[]\n", encoding="utf-8")
        else:
            raise NotImplementedError("Only boxes.empty_list: true is supported")

        exported.append(fid)

    if not exported:
        print("No frames exported.", file=sys.stderr)
        sys.exit(1)

    split_map = _assign_splits(
        exported,
        float(split_cfg["train_ratio"]),
        int(split_cfg.get("shuffle_seed", 42)),
        str(split_cfg["train_name"]),
        str(split_cfg["val_name"]),
    )

    records: List[Dict[str, Any]] = []
    for i, fid in enumerate(sorted(exported)):
        records.append(
            {
                "sequence_id": seq_id,
                "frame_id": int(fid),
                "timestamp": float(i) * 0.1,
                "split": split_map[fid],
                "radar_tensor_path": f"{seq_id}/radar/{fid:06d}.npy",
                "lidar_points_path": f"{seq_id}/lidar/{fid:06d}.npy",
                "boxes_path": f"{seq_id}/boxes/{fid:06d}.json",
                "lidar_to_radar": lidar_to_radar.tolist(),
                "ego_pose": ego_pose.tolist(),
                "synthetic_radar_tensor_path": None,
            }
        )

    manifest_path = output_root / out_cfg["manifest_filename"]
    output_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    print(f"Wrote manifest with {len(records)} frames → {manifest_path}")


if __name__ == "__main__":
    main()
