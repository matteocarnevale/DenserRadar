"""Training loop for DenserRadar.

Handles:
- AdamW optimizer with cosine annealing LR schedule
- Mixed precision (AMP) for memory efficiency
- Optional teacher-consistency loss on synthetic radar data
- Per-epoch logging to history.csv
- Checkpoint saving (last + best by val loss)
- RP-CD / RP-CA metric tracking
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from denserradar.losses.occupancy import hybrid_multiscale_loss
from denserradar.metrics.pointcloud import occupancy_to_points_xyz, occupancy_to_points_xyz_cartesian, rpcd_rpca
from denserradar.models.denser_radar import DenserRadarNet
from denserradar.utils.io import ensure_dir, load_checkpoint, save_checkpoint


@dataclass
class EpochResult:
    loss: float
    rpcd: float
    rpca: float


class Trainer:

    def __init__(self, model: torch.nn.Module, cfg: dict, run_dir: str | Path, device: str = "cuda"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.run_dir = ensure_dir(run_dir)

        self.training_cfg = cfg["training"]
        self.radar_cfg = cfg["radar"]
        self.loss_cfg = cfg["loss"]
        self.post_cfg = cfg["postprocess"]
        self.eval_cfg = cfg["evaluation"]
        self.data_cfg = cfg.get("data", {})
        self.vod_cfg = cfg.get("vod", {})

        # Optimizer and schedule
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(self.training_cfg.get("weight_decay", 1e-2)),
        )
        num_epochs = int(self.training_cfg.get("epochs", 30))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(num_epochs, 1))

        # Mixed precision
        use_amp = bool(self.training_cfg.get("amp", True)) and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(device, enabled=use_amp)
        self.use_amp = use_amp

        # Optional frozen teacher for synthetic consistency
        self.teacher = self._load_teacher_if_configured()
        self.best_val_loss = float("inf")

    # ── Teacher loading ────────────────────────────────────────────────

    def _load_teacher_if_configured(self) -> Optional[DenserRadarNet]:
        checkpoint_path = self.training_cfg.get("teacher_checkpoint")
        consistency_weight = float(self.training_cfg.get("synthetic_consistency_weight", 0.0))

        if not checkpoint_path or consistency_weight <= 0:
            return None

        # Teacher consistency is only defined for the raw-tensor (manifest) pipeline.
        if str(self.data_cfg.get("dataset", "manifest")) != "manifest":
            return None

        teacher = DenserRadarNet(self.cfg["radar"], self.cfg["model"]).to(self.device)
        state = load_checkpoint(checkpoint_path, map_location=self.device)
        teacher.load_state_dict(state["model"])
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    # ── Batch helpers ──────────────────────────────────────────────────

    def _move_batch(self, batch: Dict[str, object]) -> Dict[str, object]:
        """Transfer all tensors in the batch to self.device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                moved[key] = [v.to(self.device, non_blocking=True) for v in value]
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _extract_targets(batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        return {
            "gt_occ_hr": batch["gt_occ_hr"],
            "gt_occ_ds1": batch["gt_occ_ds1"],
            "gt_occ_ds2": batch["gt_occ_ds2"],
            "gt_occ_ds4": batch["gt_occ_ds4"],
        }

    def _compute_teacher_loss(self, batch: Dict[str, object], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """MSE between frozen teacher's HR prediction and the GT."""
        zero = targets["gt_occ_hr"].new_tensor(0.0)
        if self.teacher is None or "synthetic_radar_tensor" not in batch:
            return zero

        with torch.no_grad():
            teacher_out = self.teacher(batch["synthetic_radar_tensor"])
        return torch.nn.functional.mse_loss(teacher_out["pred_hr"], targets["gt_occ_hr"])

    # ── Single-batch metric evaluation ─────────────────────────────────

    def _evaluate_batch_metrics(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, object]) -> tuple[float, float]:
        """Convert the first sample's HR prediction to points, compute RP-CD / RP-CA."""
        coords = str(self.data_cfg.get("coordinates", "spherical"))
        threshold = float(self.post_cfg.get("pred_threshold", 0.5))
        max_points = self.post_cfg.get("max_points", None)
        if coords == "cartesian":
            pred_points = occupancy_to_points_xyz_cartesian(
                predictions["pred_hr"][0],
                point_cloud_range=list(self.vod_cfg.get("point_cloud_range", [0.0, -16.0, -2.0, 32.0, 16.0, 4.0])),
                threshold=threshold,
                max_points=max_points,
            )
        else:
            pred_points = occupancy_to_points_xyz(
                predictions["pred_hr"][0],
                self.radar_cfg,
                threshold=threshold,
                high_res_factor=int(self.cfg["ground_truth"].get("high_res_factor", 2)),
                max_points=max_points,
            )
        gt_points = batch["gt_points_radar"][0].detach().cpu().numpy()

        return rpcd_rpca(
            pred_points, gt_points,
            rpcd_radius=float(self.eval_cfg.get("rpcd_radius_m", 0.3)),
            rpca_radius=float(self.eval_cfg.get("rpca_radius_m", 0.5)),
        )

    # ── Epoch loop ─────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> EpochResult:
        self.model.train(train)
        if not train:
            self.model.eval()

        max_steps_key = "max_train_steps_per_epoch" if train else "max_val_steps"
        max_steps = self.training_cfg.get(max_steps_key)
        teacher_weight = float(self.training_cfg.get("synthetic_consistency_weight", 0.0))

        sum_loss, sum_rpcd, sum_rpca = 0.0, 0.0, 0.0
        num_batches = 0

        progress = tqdm(loader, desc="train" if train else "val", leave=False)
        for step, batch in enumerate(progress):
            if max_steps is not None and step >= int(max_steps):
                break

            batch = self._move_batch(batch)
            targets = self._extract_targets(batch)

            # ── Forward pass ──
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(self.device, enabled=self.use_amp):
                    if "radar_tensor" in batch:
                        predictions = self.model(batch["radar_tensor"])
                    elif "radar_volume" in batch:
                        predictions = self.model(batch["radar_volume"])
                    else:
                        raise KeyError("Batch must contain 'radar_tensor' (manifest) or 'radar_volume' (vod)")
                    main_loss, _ = hybrid_multiscale_loss(predictions, targets, self.loss_cfg)
                    teacher_loss = self._compute_teacher_loss(batch, targets)
                    total_loss = main_loss + teacher_weight * teacher_loss

                # ── Backward pass (training only) ──
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(total_loss).backward()

                    grad_clip = float(self.training_cfg.get("grad_clip_norm", 1.0))
                    if grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # ── Metrics ──
            rpcd, rpca = self._evaluate_batch_metrics(predictions, batch)

            sum_loss += float(total_loss.detach().cpu())
            sum_rpcd += rpcd
            sum_rpca += rpca
            num_batches += 1

            progress.set_postfix(
                loss=sum_loss / num_batches,
                rpcd=sum_rpcd / num_batches,
                rpca=sum_rpca / num_batches,
            )

        count = max(num_batches, 1)
        return EpochResult(loss=sum_loss / count, rpcd=sum_rpcd / count, rpca=sum_rpca / count)

    # ── Full training run ──────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        num_epochs = int(self.training_cfg.get("epochs", 30))

        history_path = self.run_dir / "history.csv"
        with open(history_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_rpcd,train_rpca,val_loss,val_rpcd,val_rpca\n")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_result = self._run_epoch(train_loader, train=True)
            val_result = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            elapsed = time.time() - epoch_start
            print(
                f"epoch={epoch:03d} | time={elapsed:.1f}s | "
                f"train loss={train_result.loss:.4f} rpcd={train_result.rpcd:.4f} rpca={train_result.rpca:.4f} | "
                f"val loss={val_result.loss:.4f} rpcd={val_result.rpcd:.4f} rpca={val_result.rpca:.4f}"
            )

            # Append to history CSV
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},"
                    f"{train_result.loss},{train_result.rpcd},{train_result.rpca},"
                    f"{val_result.loss},{val_result.rpcd},{val_result.rpca}\n"
                )

            # Save checkpoints
            checkpoint_extras = {"val_loss": val_result.loss, "val_rpcd": val_result.rpcd, "val_rpca": val_result.rpca}
            save_checkpoint(self.run_dir / "last.pt", self.model, self.optimizer, epoch, extra=checkpoint_extras)

            if val_result.loss < self.best_val_loss:
                self.best_val_loss = val_result.loss
                save_checkpoint(self.run_dir / "best.pt", self.model, self.optimizer, epoch, extra=checkpoint_extras)
