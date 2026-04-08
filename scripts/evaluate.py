#!/usr/bin/env python
"""Evaluate a trained DenserRadar checkpoint on the validation split.

Usage:
    python scripts/evaluate.py \
        --config configs/default.yaml \
        --manifest data/manifest.json \
        --precomputed-gt artifacts/precomputed_gt \
        --checkpoint runs/baseline/best.pt
"""
from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from denserradar.data.dataset import DenserRadarDataset, denserradar_collate
from denserradar.models.denser_radar import DenserRadarNet
from denserradar.training.trainer import Trainer
from denserradar.utils.config import load_config
from denserradar.utils.io import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DenserRadar")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--precomputed-gt", required=True, help="Dir with precomputed GT")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Validation data
    val_dataset = DenserRadarDataset(
        manifest_path=args.manifest,
        split=cfg["training"].get("val_split", "val"),
        precomputed_gt_dir=args.precomputed_gt,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=denserradar_collate,
    )

    # Load model from checkpoint
    model = DenserRadarNet(cfg["radar"], cfg["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # Run one validation epoch
    trainer = Trainer(model, cfg, run_dir="./tmp_eval", device=device)
    result = trainer._run_epoch(val_loader, train=False)
    print(f"Evaluation | loss={result.loss:.4f}  RP-CD={result.rpcd:.4f}  RP-CA={result.rpca:.4f}")


if __name__ == "__main__":
    main()
