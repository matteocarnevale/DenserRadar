#!/usr/bin/env python
"""Precompute dense 3D occupancy ground truth for every frame in the manifest.

This is Step A of the pipeline.  Run this ONCE before training.
It reads the manifest, loads LiDAR + radar data for each frame,
stitches multi-frame point clouds, voxelizes, and saves the results
so the training loop can load them instantly.

Usage:
    python scripts/precompute_ground_truth.py \
        --config configs/default.yaml \
        --manifest data/manifest.json \
        --output-dir artifacts/precomputed_gt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm

from denserradar.data.ground_truth import GroundTruthBuilder, load_frames_grouped
from denserradar.utils.config import load_config
from denserradar.utils.io import dump_json, ensure_dir, load_json, save_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute DenserRadar ground truth")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Where to write GT files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    frames_by_sequence = load_frames_grouped(manifest, root=manifest_path.parent)

    builder = GroundTruthBuilder(cfg)
    output_root = ensure_dir(args.output_dir)

    for sequence_id, frames in frames_by_sequence.items():
        sequence_dir = ensure_dir(output_root / sequence_id)

        for index, frame in enumerate(tqdm(frames, desc=f"GT seq={sequence_id}")):
            result = builder.build(frames_by_sequence, sequence_id, index)

            frame_dir = ensure_dir(sequence_dir / f"{frame.frame_id:06d}")
            save_npy(frame_dir / "gt_occ_hr.npy", result["gt_occ_hr"])
            save_npy(frame_dir / "gt_occ_ds1.npy", result["gt_occ_ds1"])
            save_npy(frame_dir / "gt_occ_ds2.npy", result["gt_occ_ds2"])
            save_npy(frame_dir / "gt_occ_ds4.npy", result["gt_occ_ds4"])
            save_npy(frame_dir / "gt_points_radar.npy", result["gt_points_radar"])

            dump_json({
                "sequence_id": frame.sequence_id,
                "frame_id": frame.frame_id,
                "num_gt_points_radar": int(result["gt_points_radar"].shape[0]),
                "occ_hr_sum": float(result["gt_occ_hr"].sum()),
            }, frame_dir / "metadata.json")


if __name__ == "__main__":
    main()
