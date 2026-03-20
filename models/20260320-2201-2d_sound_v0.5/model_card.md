# Model Card (Brief)

## Notebook
- Category: `training`
- Source notebook: `train/2d-cnn/notebooks/2d_sound_v0.5.ipynb`
- Notebook SHA256: `71c1b3c63576f53cc0189bf557862fd01f49487abe903b7fa7897ecfc49f0cd0`

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
- Selected clips: normal=456, abnormal=456
- Train clips: normal=365, abnormal=365
- Validation clips: normal=91, abnormal=91
- Train windows: normal=6570, abnormal=6570
- Validation windows: normal=1638, abnormal=1638
- Exact split membership: `models/20260320-2201-2d_sound_v0.5/split_membership.csv`

## Training
- Epochs configured: 256
- Epochs completed: 256
- Batch size: 32
- Learning rate: 0.0005
- Device: cuda
- Random seed: 42
- Deterministic training flag: True

## Results (This Run)
- Final train accuracy: 0.9801
- Final validation accuracy (window): 0.9020
- Final validation loss: 0.3341
- Predicted distribution (window val): normal=1923 (0.5870), abnormal=1353 (0.4130)
- Clip aggregation method: probability_mean
- Clip validation accuracy: 0.9176
- Predicted distribution (clip val): normal=104 (0.5714), abnormal=78 (0.4286)
- Interpretation: C. Pipeline works, meaningful signal emerging
- Metrics JSON: `models/20260320-2201-2d_sound_v0.5/metrics.json`
- Run manifest: `models/20260320-2201-2d_sound_v0.5/run_manifest.json`

## Artifact Tracking
- Git-tracked artifacts:
  - `models/20260320-2201-2d_sound_v0.5/training_history.json`
  - `models/20260320-2201-2d_sound_v0.5/metrics.json`
  - `models/20260320-2201-2d_sound_v0.5/split_membership.csv`
  - `models/20260320-2201-2d_sound_v0.5/clip_level_predictions.csv`
  - `models/20260320-2201-2d_sound_v0.5/clip_level_predictions.json`
  - `models/20260320-2201-2d_sound_v0.5/run_manifest.json`
  - `models/20260320-2201-2d_sound_v0.5/model_card.md`
- Local-only artifacts intentionally excluded from Git by `.gitignore`:
  - `models/20260320-2201-2d_sound_v0.5/final_model_state_dict.pt`
  - `models/20260320-2201-2d_sound_v0.5/best_val_model_state_dict.pt`
