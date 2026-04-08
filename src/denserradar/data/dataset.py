"""PyTorch Dataset and collate function for DenserRadar training."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

from denserradar.utils.io import load_json, load_npy
from .ground_truth import load_frame


class DenserRadarDataset(Dataset):
    """Loads radar tensors and precomputed GT from a manifest JSON.

    Parameters
    ----------
    manifest_path : path to the manifest JSON file.
    split : "train" or "val" — only records matching this split are used.
    precomputed_gt_dir : directory written by precompute_ground_truth.py.
    root : base directory for resolving relative paths in the manifest.
           Defaults to the directory containing the manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        precomputed_gt_dir: str | Path | None = None,
        root: str | Path | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent

        all_records = load_json(self.manifest_path)
        self.records = [r for r in all_records if r.get("split", "train") == split]

        self.precomputed_gt_dir = Path(precomputed_gt_dir) if precomputed_gt_dir else None

    def __len__(self) -> int:
        return len(self.records)

    def _gt_dir_for(self, record: dict) -> Path:
        """Return the directory containing precomputed GT files for this frame."""
        seq = str(record["sequence_id"])
        frame_id = int(record["frame_id"])
        return self.precomputed_gt_dir / seq / f"{frame_id:06d}"

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        frame = load_frame(record, root=self.root)

        item: Dict[str, Any] = {
            "sequence_id": frame.sequence_id,
            "frame_id": frame.frame_id,
            "radar_tensor": torch.from_numpy(frame.radar_tensor),
        }

        if frame.synthetic_radar_tensor is not None:
            item["synthetic_radar_tensor"] = torch.from_numpy(frame.synthetic_radar_tensor)

        if self.precomputed_gt_dir is not None:
            gt_dir = self._gt_dir_for(record)
            item["gt_occ_hr"] = torch.from_numpy(load_npy(gt_dir / "gt_occ_hr.npy"))
            item["gt_occ_ds1"] = torch.from_numpy(load_npy(gt_dir / "gt_occ_ds1.npy"))
            item["gt_occ_ds2"] = torch.from_numpy(load_npy(gt_dir / "gt_occ_ds2.npy"))
            item["gt_occ_ds4"] = torch.from_numpy(load_npy(gt_dir / "gt_occ_ds4.npy"))
            item["gt_points_radar"] = torch.from_numpy(load_npy(gt_dir / "gt_points_radar.npy"))

            metadata_path = gt_dir / "metadata.json"
            if metadata_path.exists():
                item["metadata"] = load_json(metadata_path)

        return item


def denserradar_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate: stack tensors, but keep gt_points_radar as a list
    (since each frame may have a different number of points)."""
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            if key == "gt_points_radar":
                collated[key] = values          # variable length → keep as list
            else:
                collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values

    return collated
