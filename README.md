# DenserRadar

Engineering reproduction of the paper **"DenserRadar: A 4D millimeter-wave radar point cloud detector based on dense LiDAR point clouds"** (Han et al., 2024 — [arXiv 2405.05131](https://arxiv.org/abs/2405.05131)).

The system learns to generate dense and accurate 4D radar point clouds from the raw mmWave radar tensor, using as supervision 3D occupancy ground truth built from multi-frame stitched LiDAR point clouds.

> **Disclaimer:** This is an independent reimplementation for educational and research purposes. It is **not** the authors' official code.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites and Installation](#3-prerequisites-and-installation)
4. [Data Preparation](#4-data-preparation)
5. [Full Step-by-Step Pipeline](#5-full-step-by-step-pipeline)
6. [Configuration Reference](#6-configuration-reference)
7. [Ablation Experiments](#7-ablation-experiments)
8. [Docker](#8-docker)
9. [Tests](#9-tests)
10. [Troubleshooting](#10-troubleshooting)
11. [Citation](#11-citation)

---

## 1. Architecture Overview

The project consists of two macro-phases:

### Phase A — Ground Truth Generation (offline)

```
LiDAR point cloud (multi-frame)
        │
        ├── static / dynamic split (via annotated bounding boxes)
        │       │                       │
        │   ground removal           dynamic stitching
        │   (RANSAC)                 (per track-id, box-to-box transform)
        │       │                       │
        │   static stitching            │
        │   (relative ego-pose)         │
        │       └───────┬───────────────┘
        │               │
        │     dense point cloud fusion
        │               │
        │     LiDAR frame → Radar frame (extrinsics)
        │               │
        │     Cartesian → Spherical → Crop radar FOV
        │               │
        │     HR voxelization (2R × 2E × 2A)
        │               │
        │     Radar intensity gating (removes voxels without signal)
        │               │
        │     Multi-scale downsample (ds1, ds2, ds4)
        │               │
        ▼               ▼
  gt_occ_hr.npy   gt_occ_ds1/ds2/ds4.npy   gt_points_radar.npy
```

### Phase B — DenserRadar Network Training (online)

```
4D Radar tensor [B, D, R, E, A]      (D = Doppler as channels)
        │
   Input 3D Conv (kernel 1×1×1)       Doppler → C feature channels
        │
   3D U-Net Encoder (3 levels)        conv 5×3×3, stride 2
        │    │    │    │
        │  skip0 skip1 skip2          3D Cross-Attention (Eq. 7)
        │    │    │    │
   3D U-Net Decoder (3 levels)        deconv + residual 5×3×3
        │    │    │
   DS heads (1×1 conv + sigmoid)       pred_ds1, pred_ds2, pred_ds4
        │
   Output deconv (stride 2)            doubles resolution
        │
   Output head (1×1 conv + sigmoid)    pred_hr  [B, 1, 2R, 2E, 2A]
        │
   Weighted Hybrid Loss (Eq. 8)        Σ 1/2^i · (Dice_i + λ_F · Focal_i)
```

### Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **RP-CD** (density) | Fraction of GT LiDAR points that have at least one predicted radar point within δ_d = 0.3 m |
| **RP-CA** (accuracy) | Fraction of predicted radar points that have at least one GT LiDAR point within δ_a = 0.5 m |

---

## 2. Repository Structure

```
DenserRadar/
├── configs/
│   └── default.yaml              # single configuration file
├── scripts/
│   ├── precompute_ground_truth.py # Phase A: offline GT generation
│   ├── train.py                   # Phase B: training
│   └── evaluate.py                # evaluation on checkpoint
├── src/denserradar/
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset, collate function
│   │   ├── geometry.py            # 3D transforms, spherical/cartesian, box mask
│   │   ├── ground_truth.py        # GroundTruthBuilder: split, stitch, voxelize
│   │   └── voxelization.py        # hard/soft spherical voxelization, intensity gating
│   ├── losses/
│   │   └── occupancy.py           # Dice loss, Focal loss, hybrid multiscale (Eq. 8)
│   ├── metrics/
│   │   └── pointcloud.py          # RP-CD, RP-CA via scipy cKDTree
│   ├── models/
│   │   ├── attention.py           # CrossAttention3D (additive attention gate, Eq. 7)
│   │   ├── blocks.py              # ConvNormAct3D, ResidualBlock3D, Down/UpsampleBlock3D
│   │   └── denser_radar.py        # DenserRadarNet: stem + U-Net + deep supervision + HR head
│   ├── training/
│   │   └── trainer.py             # AdamW, cosine LR, AMP, teacher consistency, history.csv
│   └── utils/
│       ├── config.py              # yaml loader
│       ├── io.py                  # npy / json / checkpoint I/O
│       └── seed.py                # reproducibility
├── tests/
│   ├── smoke_test.py              # basic model, loss, metrics, dimension tests
│   └── test_loss_deep_supervision.py  # detailed Eq. 8 + gradient flow tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

---

## 3. Prerequisites and Installation

### 3.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8 GB VRAM | 24+ GB VRAM (A5000, RTX 4090, A100) |
| RAM | 16 GB | 32+ GB |
| Storage | 50 GB | 200+ GB (full K-Radar dataset) |

The full K-Radar radar tensor (`64×256×107×37` float32) takes ~250 MB/frame. With `batch_size: 1` and AMP, a baseline training run fits in 8 GB VRAM.

### 3.2 Installation with conda (recommended)

```bash
# Create and activate the environment
conda create -n denserradar python=3.9 -y
conda activate denserradar

# Install PyTorch (adapt CUDA version to your GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -r requirements.txt

# Install pytest for tests
pip install pytest

# Set PYTHONPATH (add to .bashrc/.zshrc for persistence)
export PYTHONPATH=$PWD/src
```

### 3.3 Installation with venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
export PYTHONPATH=$PWD/src
```

### 3.4 Verify Installation

```bash
python -c "import denserradar; print('OK')"
PYTHONPATH=src python -m pytest tests/ -v
```

You should see **20 tests passed**.

---

## 4. Data Preparation

The repo is **dataset-agnostic**: it contains no hardcoded parser for any specific dataset. Everything goes through a **JSON manifest** that you prepare once.

### 4.1 The JSON Manifest

The manifest is a JSON list where each element describes a single frame. Create it in the `data/` directory:

```json
[
  {
    "sequence_id": "0001",
    "frame_id": 0,
    "timestamp": 0.0,
    "split": "train",
    "radar_tensor_path": "seq_0001/radar/000000.npy",
    "lidar_points_path": "seq_0001/lidar/000000.npy",
    "boxes_path": "seq_0001/boxes/000000.json",
    "lidar_to_radar": [
      [0.9998, 0.0, 0.0175, 0.5],
      [0.0, 1.0, 0.0, 0.0],
      [-0.0175, 0.0, 0.9998, -0.3],
      [0.0, 0.0, 0.0, 1.0]
    ],
    "ego_pose": [
      [1, 0, 0, 100.5],
      [0, 1, 0, 200.3],
      [0, 0, 1, 0.0],
      [0, 0, 0, 1]
    ],
    "synthetic_radar_tensor_path": null
  },
  {
    "sequence_id": "0001",
    "frame_id": 1,
    "timestamp": 0.1,
    "split": "train",
    "...": "..."
  }
]
```

> **All paths in the manifest are relative to the directory containing the manifest itself.**

### 4.2 File Formats

#### Radar tensor (`npy`)

```python
# Shape: [D, R, E, A]
# For K-Radar: [64, 256, 107, 37]
# D = Doppler bins, R = Range bins, E = Elevation bins, A = Azimuth bins
# dtype: float32
radar = np.load("seq_0001/radar/000000.npy")
assert radar.shape == (64, 256, 107, 37)
```

#### LiDAR point cloud (`npy`)

```python
# Shape: [N, 3] — (x, y, z) coordinates in the LiDAR frame
# N varies per frame (typically 60k–130k for a 64-beam LiDAR)
# dtype: float32
lidar = np.load("seq_0001/lidar/000000.npy")
assert lidar.ndim == 2 and lidar.shape[1] == 3
```

#### Bounding boxes (`json`)

List of dynamic objects in the frame. Each object must have:

```json
[
  {
    "track_id": 42,
    "class_name": "car",
    "center": [12.1, 1.2, 0.4],
    "size": [4.2, 1.8, 1.6],
    "yaw": 0.1,
    "velocity": [0.0, 0.0, 0.0]
  },
  {
    "track_id": 7,
    "class_name": "pedestrian",
    "center": [5.0, 3.2, 0.8],
    "size": [0.6, 0.6, 1.7],
    "yaw": 1.57,
    "velocity": [0.5, 0.1, 0.0]
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | int | Unique object ID within the sequence (for dynamic stitching) |
| `class_name` | str | Semantic class |
| `center` | [3] | Box center (x, y, z) in the LiDAR frame |
| `size` | [3] | Dimensions (length, width, height) in meters |
| `yaw` | float | Rotation around the Z axis (radians) |
| `velocity` | [3] | Velocity (optional, not used for GT but available) |

#### `lidar_to_radar` (4×4 matrix)

Rigid transformation matrix that maps points from the LiDAR frame to the radar frame. Derive it from the extrinsic calibration of your sensor setup:

```
P_radar = lidar_to_radar @ [P_lidar; 1]
```

#### `ego_pose` (4×4 matrix)

Vehicle pose in the world frame at the current frame's timestamp. Used for multi-frame static stitching. If you only have relative odometry, integrate the incremental poses.

### 4.3 Practical Example: Preparing K-Radar

The [K-Radar](https://github.com/kaist-avelab/K-Radar) dataset provides 4D radar tensors, LiDAR point clouds, and bounding boxes. Steps to convert it to the manifest format:

```
1. Download K-Radar (sequences 1–5 to reproduce the paper)

2. For each sequence, for each frame:
   a. Copy/link the radar tensor .npy    → data/seq_XXXX/radar/YYYYYY.npy
   b. Extract the LiDAR PC as [N,3] .npy → data/seq_XXXX/lidar/YYYYYY.npy
   c. Convert labels to boxes JSON       → data/seq_XXXX/boxes/YYYYYY.json
   d. Extract lidar_to_radar from the dataset calibration
   e. Extract ego_pose from the provided GPS/IMU odometry

3. Assign splits: 80% train, 20% val (as in the paper)

4. Generate manifest.json with a conversion script
```

Resulting on-disk structure:

```
data/
├── manifest.json
├── seq_0001/
│   ├── radar/
│   │   ├── 000000.npy    # [64, 256, 107, 37] float32
│   │   ├── 000001.npy
│   │   └── ...
│   ├── lidar/
│   │   ├── 000000.npy    # [N, 3] float32
│   │   ├── 000001.npy
│   │   └── ...
│   └── boxes/
│       ├── 000000.json
│       ├── 000001.json
│       └── ...
├── seq_0002/
│   └── ...
└── ...
```

### 4.4 Manifest Validation

Before proceeding, verify that everything is consistent:

```python
import json, numpy as np
manifest = json.load(open("data/manifest.json"))
r = manifest[0]

radar = np.load(f"data/{r['radar_tensor_path']}")
lidar = np.load(f"data/{r['lidar_points_path']}")
boxes = json.load(open(f"data/{r['boxes_path']}"))
L2R = np.array(r["lidar_to_radar"])
ego = np.array(r["ego_pose"])

print(f"Radar shape: {radar.shape}")       # expected: (64, 256, 107, 37)
print(f"LiDAR points: {lidar.shape[0]}")   # expected: ~60k–130k
print(f"Boxes: {len(boxes)}")              # expected: 0–30
print(f"L2R shape: {L2R.shape}")           # expected: (4, 4)
print(f"Ego shape: {ego.shape}")           # expected: (4, 4)
assert radar.ndim == 4
assert lidar.ndim == 2 and lidar.shape[1] == 3
assert L2R.shape == (4, 4) and ego.shape == (4, 4)
print("Manifest OK!")
```

---

## 5. Full Step-by-Step Pipeline

### Step 1 — Precompute Ground Truth

This step processes each frame offline, stitching multi-frame LiDAR points and generating occupancy volumes at all resolutions:

```bash
python scripts/precompute_ground_truth.py \
  --config configs/default.yaml \
  --manifest data/manifest.json \
  --output-dir artifacts/precomputed_gt
```

**Output per frame** (`artifacts/precomputed_gt/<seq_id>/<frame_id>/`):

| File | Shape | Description |
|------|-------|-------------|
| `gt_occ_hr.npy` | `[1, 2R, 2E, 2A]` | High-resolution occupancy (2× radar resolution) |
| `gt_occ_ds1.npy` | `[1, R, E, A]` | Downsample 1× (native radar resolution) |
| `gt_occ_ds2.npy` | `[1, ~R/2, ~E/2, ~A/2]` | Downsample 2× |
| `gt_occ_ds4.npy` | `[1, ~R/4, ~E/4, ~A/4]` | Downsample 4× |
| `gt_points_radar.npy` | `[M, 3]` | Dense point cloud in the radar frame (for metrics) |
| `metadata.json` | — | Statistics (point count, occupancy sum) |

**Estimated time**: ~1–5 sec/frame (depends on point cloud size and `temporal_window`).

> **Tip**: try with a few frames first to visually verify the GT makes sense. Check that `occ_hr_sum` in the metadata is reasonable (not zero, not fully filled).

### Step 2 — Training

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --manifest data/manifest.json \
  --precomputed-gt artifacts/precomputed_gt \
  --run-dir runs/baseline
```

**Training produces** (`runs/baseline/`):

| File | Description |
|------|-------------|
| `last.pt` | Checkpoint from the last epoch |
| `best.pt` | Checkpoint with the best val loss |
| `history.csv` | Per-epoch log: loss, RP-CD, RP-CA (train and val) |

**Terminal output** per epoch:

```
epoch=001 | time=120.3s | train loss=45.2100 rpcd=0.0312 rpca=0.1205 | val loss=42.8800 rpcd=0.0401 rpca=0.1502
epoch=002 | time=119.8s | train loss=38.6500 rpcd=0.0523 rpca=0.1890 | val loss=37.1200 rpcd=0.0612 rpca=0.2105
...
```

**Default training settings** (from the paper):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 30 | |
| Batch size | 1 | Very large tensors; increase only with GPU >24 GB |
| Optimizer | AdamW | lr=1e-4, weight_decay=1e-2 |
| Scheduler | Cosine Annealing | T_max = epochs |
| AMP | Enabled | Mixed precision to reduce VRAM |
| Gradient clipping | 1.0 | Max gradient norm |
| Loss | Dice + 700 × Focal | Deep supervision at 4 scales |

### Step 3 — Evaluation

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --manifest data/manifest.json \
  --precomputed-gt artifacts/precomputed_gt \
  --checkpoint runs/baseline/best.pt
```

Output:

```
Evaluation | loss=35.2100 rpcd=0.1400 rpca=0.3600
```

---

## 6. Configuration Reference

The file `configs/default.yaml` controls every aspect of the pipeline. Below are all parameters with their explanations.

### `radar` — Radar Sensor Specifications

```yaml
radar:
  doppler_bins: 64           # Doppler dimension of the raw tensor
  range_bins: 256            # Range bins
  elevation_bins: 107        # Elevation bins
  azimuth_bins: 37           # Azimuth bins
  range_min_m: 0.0           # Minimum radar range (m)
  range_max_m: 117.76        # Maximum range (m). For K-Radar: 256 × 0.46
  elevation_min_deg: -53.0   # Min elevation FOV (degrees)
  elevation_max_deg: 53.0    # Max elevation FOV (degrees)
  azimuth_min_deg: -18.0     # Min azimuth FOV (degrees)
  azimuth_max_deg: 18.0      # Max azimuth FOV (degrees)
  intensity_reduce: max      # How to reduce the Doppler dim for gating: max | sum | mean
  intensity_threshold_mode: percentile  # absolute | percentile
  intensity_threshold_value: 85.0       # Threshold for radar intensity gating
```

### `ground_truth` — GT Generation

```yaml
ground_truth:
  mode: multi_frame_hard       # Mode:
                                #   single_frame_hard  — current frame only, binary voxels
                                #   multi_frame_hard   — multi-frame stitching, binary voxels
                                #   multi_frame_soft   — multi-frame, Gaussian-blurred voxels
  temporal_window: 10           # Number of past and future frames to stitch (t in the paper)
  remove_ground: true           # Remove ground plane (RANSAC) from the static branch
  ground_ransac_iters: 100      # RANSAC iterations for ground fitting
  ground_distance_threshold_m: 0.15  # RANSAC inlier threshold (m)
  high_res_factor: 2            # Super-resolution factor: HR = factor × native resolution
  soft_sigma_voxels: 0.8        # Gaussian blur sigma for soft mode (in voxel units)
  crop_to_radar_fov: true       # Clip points outside the radar FOV
  dynamic_box_scale: 1.0        # Bounding box scale factor for dynamic split
  box_z_padding_m: 0.2          # Extra vertical padding on boxes (m)
```

### `model` — Network Architecture

```yaml
model:
  base_channels: 16              # Channels in the first encoder level
  channel_multipliers: [1,2,4,8] # Per-level multipliers: [16, 32, 64, 128]
  use_cross_attention: true      # Enable CrossAttention3D on skip connections (Eq. 7)
  norm_groups: 8                 # Groups for GroupNorm
  activation: silu               # Activation function: silu | relu | gelu
  dropout_p: 0.0                 # 3D Dropout (0.0 = disabled)
```

### `loss` — Loss Function

```yaml
loss:
  lambda_focal: 700.0    # λ_F in the paper: relative weight of focal vs dice
  focal_alpha: 0.25      # α of the focal loss (positive/negative balancing)
  focal_gamma: 2.0       # γ of the focal loss (focus on hard examples)
  smooth: 1.0            # Dice loss smoothing factor (numerical stability)
```

The total loss follows Eq. 8 of the paper:

```
L = Σ_{i} 1/2^i · (Dice_i + λ_F · Focal_i)
```

with i = {HR (w=1.0), ds1 (w=0.5), ds2 (w=0.25), ds4 (w=0.125)}.

### `training` — Training Parameters

```yaml
training:
  epochs: 30
  batch_size: 1
  learning_rate: 1.0e-4
  weight_decay: 1.0e-2
  grad_clip_norm: 1.0
  amp: true                        # Mixed precision (FP16/FP32)
  train_split: train               # Training split name in the manifest
  val_split: val                   # Validation split name in the manifest
  synthetic_consistency_weight: 0.0  # Teacher loss weight (0 = disabled)
  teacher_checkpoint: null           # Path to the teacher checkpoint
  max_train_steps_per_epoch: null    # Limit steps per epoch (null = all)
  max_val_steps: null                # Limit validation steps
```

### `postprocess` and `evaluation` — Inference and Metrics

```yaml
postprocess:
  pred_threshold: 0.5    # Threshold on predicted occupancy to extract points
  max_points: null       # Maximum number of points to extract (null = all)

evaluation:
  rpcd_radius_m: 0.3     # δ_d in the paper: radius for RP-CD
  rpca_radius_m: 0.5     # δ_a in the paper: radius for RP-CA
```

---

## 7. Ablation Experiments

The paper presents 4 ablation configurations. Here is how to reproduce them:

### A — Full Method (baseline)

```yaml
model:
  use_cross_attention: true
ground_truth:
  mode: multi_frame_hard
  temporal_window: 10
loss:
  lambda_focal: 700.0
```

### B — Without Cross-Attention

```yaml
model:
  use_cross_attention: false
```

### C — Vanilla Focal Loss Only (without weighted deep supervision)

Modify `configs/default.yaml` or create an override:

```yaml
loss:
  lambda_focal: 1.0    # un-amplified focal
  # Deep supervision is intrinsic to the network;
  # for "focal only at the end" you need to change
  # the loss weights to: HR=1.0, ds1=0, ds2=0, ds4=0
```

### D — Single-Frame LiDAR GT

```yaml
ground_truth:
  mode: single_frame_hard
  temporal_window: 0
```

### Soft Voxelization (extension)

```yaml
ground_truth:
  mode: multi_frame_soft
  soft_sigma_voxels: 0.8
```

### Teacher Consistency (extension)

Requires a `synthetic_radar_tensor_path` in the manifest and a checkpoint from an already-trained model:

```yaml
training:
  synthetic_consistency_weight: 0.25
  teacher_checkpoint: runs/baseline/best.pt
```

---

## 8. Docker

### 8.1 Build the Image

```bash
docker build -t denserradar:latest .
```

### 8.2 Interactive Run

```bash
docker run --rm -it \
  --gpus all \
  -v $PWD:/workspace \
  denserradar:latest bash
```

Inside the container, `PYTHONPATH` is already configured:

```bash
python scripts/train.py --config configs/default.yaml ...
```

### 8.3 With docker-compose

```bash
# Interactive shell
docker compose run --rm denserradar bash

# Or direct command
docker compose run --rm denserradar python scripts/train.py \
  --config configs/default.yaml \
  --manifest data/manifest.json \
  --precomputed-gt artifacts/precomputed_gt \
  --run-dir runs/baseline
```

---

## 9. Tests

The test suite verifies the correct functioning of all components:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

### What They Test

**`tests/smoke_test.py`** (8 tests):
- Model forward pass with even and odd dimensions (including real K-Radar sizes)
- Forward pass without cross-attention
- Isolated `CrossAttention3D`
- Base loss and with spatial mismatch
- `downsample_occupancy_max` with non-divisible dimensions
- RP-CD / RP-CA metrics

**`tests/test_loss_deep_supervision.py`** (12 tests):
- All 4 scales produce dice + focal stats
- Weights follow the geometric progression 1/2^i (Eq. 8)
- Dice loss with sparse target (class imbalance)
- Focal loss downweights easy examples
- λ_F correctly scales the focal contribution
- Good predictions yield lower loss than random predictions
- End-to-end: `model.forward()` → `loss` → `.backward()` works
- **Gradients reach all layers** (from `input_stem` to `bottleneck`): verifies that deep supervision does its job
- `ds4_head` alone produces gradients in the bottleneck and encoder
- The loss is differentiable across all outputs

---

## 10. Troubleshooting

### GT has `occ_hr_sum = 0`

- Are the `lidar_to_radar` extrinsics correct? Verify that transformed points fall within the radar FOV.
- Does the range/FOV in the config match your sensor?
- Is `intensity_threshold_value` too high? Try 50.0 for debugging.

### CUDA out of memory during training

- Verify `batch_size: 1` (default).
- Make sure `amp: true` is enabled.
- Reduce `range_bins` or tensor dimensions if experimenting.
- Use `max_train_steps_per_epoch: 10` for quick debugging.

### RP-CD and RP-CA are both 0

- `pred_threshold` might be too high. Try 0.3.
- Verify that `gt_points_radar.npy` is not empty.
- Check that the model is actually learning (is the loss decreasing?).

### Dimension mismatch error in the model

- Radar tensor dimensions must match `[doppler_bins, range_bins, elevation_bins, azimuth_bins]` in the config.
- The model automatically handles odd spatial dimensions (cropping in skip connections).

### Training is very slow

- Verify that `amp: true` is enabled and the GPU is being used (`device: cuda`).
- RP-CD/RP-CA metrics are computed at every batch. To speed up, use `max_val_steps: 50` during development.
- `num_workers: 4` in the config; increase if you have many CPU cores.

### How to Visualize Results

```python
import numpy as np
import matplotlib.pyplot as plt

gt = np.load("artifacts/precomputed_gt/0001/000012/gt_points_radar.npy")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=0.1, alpha=0.3)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.title("Dense GT point cloud (radar frame)")
plt.show()
```

### Recommended Getting-Started Order

1. **Validate the manifest** with the Python snippet from section 4.4
2. **Visualize the extrinsics**: transform 1 LiDAR frame into the radar frame and plot it
3. **Precompute 20 frames** and visually inspect the GT
4. **Smoke training** on 50 frames with `max_train_steps_per_epoch: 5` and `epochs: 2`
5. **Full training** only after verifying everything works

---

## 11. Citation

If you find this work useful, please cite the original paper:

```bibtex
@article{Han2024DenserRadar,
  title   = {DenserRadar: A 4D Millimeter-Wave Radar Point Cloud Detector Based on Dense LiDAR Point Clouds},
  author  = {Han, Zeyu and Jiang, Junkai and Ding, Xiaokang and Meng, Qingwen and Xu, Shaobing and He, Lei and Wang, Jianqiang},
  journal = {arXiv preprint arXiv:2405.05131},
  year    = {2024},
  url     = {https://arxiv.org/abs/2405.05131}
}
```
