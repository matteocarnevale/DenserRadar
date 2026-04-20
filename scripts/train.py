#!/usr/bin/env python
"""Train DenserRadar models.

Supports:
- Manifest pipeline (original): requires precomputed GT (Step A).
- VoD pipeline: reads `.bin` pairs and generates GT on-the-fly.

Usage:
    python scripts/train.py --config configs/default.yaml --run-dir runs/baseline

    # Manifest dataset
    python scripts/train.py --config configs/default.yaml \\
        --manifest data/manifest.json \\
        --precomputed-gt artifacts/precomputed_gt \\
        --run-dir runs/baseline
"""
from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from denserradar.data.dataset import DenserRadarDataset, denserradar_collate
from denserradar.data.vod_dataset import VoDDataset, vod_collate
from denserradar.models.denser_radar import DenserRadarNet
from denserradar.models.radar_points_unet import RadarPointsUNet
from denserradar.training.trainer import Trainer
from denserradar.utils.config import load_config
from denserradar.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenserRadar")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--manifest", required=False, help="Path to manifest JSON (manifest dataset only)")
    parser.add_argument("--precomputed-gt", required=False, help="Dir with precomputed GT (manifest dataset only)")
    parser.add_argument("--run-dir", required=True, help="Output dir for checkpoints and logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dataset_kind = str(cfg.get("data", {}).get("dataset", "manifest"))
    if dataset_kind == "manifest":
        if not args.manifest or not args.precomputed_gt:
            raise ValueError("--manifest and --precomputed-gt are required when data.dataset == 'manifest'")

        train_dataset = DenserRadarDataset(
            manifest_path=args.manifest,
            split=cfg["training"].get("train_split", "train"),
            precomputed_gt_dir=args.precomputed_gt,
        )
        val_dataset = DenserRadarDataset(
            manifest_path=args.manifest,
            split=cfg["training"].get("val_split", "val"),
            precomputed_gt_dir=args.precomputed_gt,
        )
        collate_fn = denserradar_collate
        model = DenserRadarNet(cfg["radar"], cfg["model"])
    elif dataset_kind == "vod":
        train_dataset = VoDDataset(cfg, split=cfg["training"].get("train_split", "train"))
        val_dataset = VoDDataset(cfg, split=cfg["training"].get("val_split", "test"))
        collate_fn = vod_collate
        model = RadarPointsUNet(cfg["model"])
    else:
        raise ValueError(f"Unknown data.dataset={dataset_kind!r}; use 'manifest' or 'vod'")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=collate_fn,
    )

    trainer = Trainer(model, cfg, run_dir=args.run_dir, device=device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
