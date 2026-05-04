#!/usr/bin/env python3
"""Controlla che manifest + file su disco siano coerenti con un config DenserRadar (sezione radar)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="YAML con chiave radar (es. configs/radial_grid.yaml)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rc = cfg["radar"]
    exp_d = int(rc["doppler_bins"])
    exp_r = int(rc["range_bins"])
    exp_e = int(rc["elevation_bins"])
    exp_a = int(rc["azimuth_bins"])

    manifest_dir = args.manifest.parent
    records = json.loads(args.manifest.read_text(encoding="utf-8"))

    errors = 0
    for rec in records:
        rp = manifest_dir / rec["radar_tensor_path"]
        lp = manifest_dir / rec["lidar_points_path"]
        bp = manifest_dir / rec["boxes_path"]
        if not rp.is_file():
            print(f"ERR missing radar: {rp}", file=sys.stderr)
            errors += 1
            continue
        if not lp.is_file():
            print(f"ERR missing lidar: {lp}", file=sys.stderr)
            errors += 1
            continue
        if not bp.is_file():
            print(f"ERR missing boxes: {bp}", file=sys.stderr)
            errors += 1
            continue

        rt = np.load(str(rp), mmap_mode="r")
        if rt.shape != (exp_d, exp_r, exp_e, exp_a):
            print(
                f"ERR shape radar {rp}: got {rt.shape}, expected ({exp_d},{exp_r},{exp_e},{exp_a})",
                file=sys.stderr,
            )
            errors += 1

        lid = np.load(str(lp), mmap_mode="r")
        if lid.ndim != 2 or lid.shape[1] != 3:
            print(f"ERR lidar {lp}: expected (N,3), got {lid.shape}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"Validation failed with {errors} error(s).", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {len(records)} frames validated against radar tensor shape ({exp_d},{exp_r},{exp_e},{exp_a}).")


if __name__ == "__main__":
    main()
