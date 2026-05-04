# Dataset prep (solo export — pipeline DenserRadar invariata)

Questa cartella **non** modifica `src/` né gli script di training. Produce solo:

- `manifest.json` nel formato atteso da `scripts/precompute_ground_truth.py` / `scripts/train.py`
- Cartelle `radar/*.npy`, `lidar/*.npy`, `boxes/*.json` compatibili con il README principale del progetto

## Prerequisiti

Dalla root del repo (con venv attivo):

```bash
pip install -r requirements.txt
```

## 1. Configura `dataset_prep/configs/export_radial.yaml`

- **`paths.source_root`**: directory che contiene i file grezzi (es. `radar_FFT/`, `laser_PCL/` come nel repo RADIal “ready-to-use”).
- **`radar.target_shape`**: deve coincidere con **`configs/radial_grid.yaml`** (o con il tuo YAML di training) nei campi `doppler_bins`, `range_bins`, `elevation_bins`, `azimuth_bins` — ordine **\[D, R, E, A\]**.
- **`radar.source_layout`**:
  - `DRAE` se il file è già `(D, R, E, A)`
  - `RAEC` se il cubo è `(R, A, E, C)` come in Radar-Mamba / tensori polari
- **`calibration`**: default `identity` per LiDAR→radar e `ego_pose`. Per stitching multi-frame nel precompute GT, sostituisci con **pose ed estrinseche reali** (file `.npy` 4×4 oppure estensione futura).

## 2. Esporta

```bash
python dataset_prep/export_to_manifest.py --config dataset_prep/configs/export_radial.yaml
python dataset_prep/export_to_manifest.py --config dataset_prep/configs/export_radial.yaml --dry-run
```

## 3. Valida coerenza shape ↔ config

```bash
python dataset_prep/validate_manifest.py \
  --manifest exports/radial_denserradar/manifest.json \
  --config configs/radial_grid.yaml
```

## 4. Pipeline DenserRadar (invariata)

```bash
export PYTHONPATH=src

python scripts/precompute_ground_truth.py \
  --config configs/radial_grid.yaml \
  --manifest exports/radial_denserradar/manifest.json \
  --output-dir artifacts/precomputed_gt

python scripts/train.py \
  --config configs/radial_grid.yaml \
  --manifest exports/radial_denserradar/manifest.json \
  --precomputed-gt artifacts/precomputed_gt \
  --run-dir runs/radial_baseline
```

## Note FFT / shape diverse

Se il file radar ha shape diversa da `target_shape`, usa `resize_mode: zoom` (interpolazione trilineare per axis). Per cubi molto diversi (es. solo RD map 2D) serve prima un preprocessing esterno che produca un volume **4D** compatibile con `source_layout`.
