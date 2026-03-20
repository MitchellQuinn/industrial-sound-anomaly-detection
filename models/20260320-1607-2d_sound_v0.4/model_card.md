# Model Card (Brief)

## Notebook
- Category: `training`
- Source notebook: `train/2d-cnn/notebooks/2d_sound_v0.4.ipynb`
- Notebook SHA256: `859e4031f8cbc351abb7e9f4b954efb7d85662483fe47ad1d1e0398e75e53428`

## Model
- Name: `Baseline2DCNN`
- Architecture: Conv2d(2->16, k5) + ReLU + MaxPool, Conv2d(16->32, k3) + ReLU + MaxPool, Conv2d(32->64, k3) + ReLU + MaxPool, AdaptiveAvgPool2d(1,1), Linear(64->2)

## Task
- Binary machine-sound classification on pump clips
- Labels: `normal=0`, `abnormal=1`

## Inputs
- Window manifests: `preprocessing/03.training-export/output/manifests/20260319_152022_windows.parquet, preprocessing/03.training-export/output/manifests/20260319_155015_windows.parquet`
- Clip manifests: `preprocessing/03.training-export/output/manifests/20260319_152022_files.parquet, preprocessing/03.training-export/output/manifests/20260319_155015_files.parquet`
- Preprocessing configs: `preprocessing/03.training-export/output/manifests/20260319_152022_config.json, preprocessing/03.training-export/output/manifests/20260319_155015_config.json`
- Base representation: `normalized_window` shape `(96, 64)` and `active_mask` shape `(96, 64)`
- Input mode: `normalized_plus_mask`
- Final tensor shape: `(2, 96, 64)`
- Clip-level split only (no window-level random split)

## Data & Split
- Selected clips: normal=100, abnormal=100
- Train clips: normal=80, abnormal=80
- Validation clips: normal=20, abnormal=20
- Train windows: normal=1440, abnormal=1440
- Validation windows: normal=360, abnormal=360
- Exact split membership: `models/20260320-1607-2d_sound_v0.4/split_membership.csv`

## Training
- Epochs configured: 64
- Epochs completed: 64
- Batch size: 32
- Learning rate: 0.0005
- Device: cuda
- Random seed: 42
- Deterministic training flag: True

## Results (This Run)
- Final train accuracy: 0.8566
- Final validation accuracy: 0.8208
- Final validation loss: 0.3871
- Predicted distribution (val): normal=245 (0.3403), abnormal=475 (0.6597)
- Interpretation: C. Pipeline works, meaningful signal emerging
- Metrics JSON: `models/20260320-1607-2d_sound_v0.4/metrics.json`
- Run manifest: `models/20260320-1607-2d_sound_v0.4/run_manifest.json`

## Artifact Tracking
- Git-tracked artifacts:
  - `models/20260320-1607-2d_sound_v0.4/training_history.json`
  - `models/20260320-1607-2d_sound_v0.4/metrics.json`
  - `models/20260320-1607-2d_sound_v0.4/split_membership.csv`
  - `models/20260320-1607-2d_sound_v0.4/run_manifest.json`
  - `models/20260320-1607-2d_sound_v0.4/model_card.md`
- Local-only artifacts intentionally excluded from Git by `.gitignore`:
  - `models/20260320-1607-2d_sound_v0.4/final_model_state_dict.pt`
  - `models/20260320-1607-2d_sound_v0.4/best_val_model_state_dict.pt`
