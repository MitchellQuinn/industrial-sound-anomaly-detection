# Model Card (Brief)

## Notebook
- Category: `training`
- Source notebook: `train/2d-cnn/notebooks/2d_sound_v0.8.ipynb`
- Notebook SHA256: `4f5578d0cad5b828bad31784df2550e221235a66bd9ddebff0f72c362834f7eb`

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
- Exact split membership: `models/20260323-1653-2d_sound_v0.8/split_membership.csv`
- Split strategy: `clip_random`
- Holdout units (resolved): `[]`
- Train unit ids: `['00', '02', '04', '06']`
- Validation unit ids: `['00', '02', '04', '06']`

## Training
- Epochs configured: 4
- Epochs completed: 4
- Batch size: 32
- Learning rate: 0.0005
- Device: cuda
- Random seed: 42
- Deterministic training flag: True

## Results (This Run)
- Final train accuracy: 0.5213
- Final validation accuracy (window): 0.6044
- Final validation loss: 0.6741
- Predicted distribution (window val): normal=2762 (0.8431), abnormal=514 (0.1569)
- Clip aggregation method: probability_mean
- Clip validation accuracy: 0.6154
- Clip validation ROC AUC (mean abnormal prob): 0.6791
- Predicted distribution (clip val): normal=153 (0.8407), abnormal=29 (0.1593)
- Interpretation: B. Pipeline works, weak but nontrivial signal
- Metrics JSON: `models/20260323-1653-2d_sound_v0.8/metrics.json`
- Run manifest: `models/20260323-1653-2d_sound_v0.8/run_manifest.json`

## Artifact Tracking
- Git-tracked artifacts:
  - `models/20260323-1653-2d_sound_v0.8/training_history.json`
  - `models/20260323-1653-2d_sound_v0.8/metrics.json`
  - `models/20260323-1653-2d_sound_v0.8/split_membership.csv`
  - `models/20260323-1653-2d_sound_v0.8/clip_level_predictions.csv`
  - `models/20260323-1653-2d_sound_v0.8/clip_level_predictions.json`
  - `models/20260323-1653-2d_sound_v0.8/run_manifest.json`
  - `models/20260323-1653-2d_sound_v0.8/model_card.md`
- Local-only artifacts intentionally excluded from Git by `.gitignore`:
  - `models/20260323-1653-2d_sound_v0.8/final_model_state_dict.pt`
  - `models/20260323-1653-2d_sound_v0.8/best_val_model_state_dict.pt`
