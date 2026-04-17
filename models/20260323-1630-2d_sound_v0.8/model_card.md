# Model Card (Brief)

## Notebook
- Category: `training`
- Source notebook: `train/2d-cnn/notebooks/2d_sound_v0.8.ipynb`
- Notebook SHA256: `3bae70a931828cb6d654186849a3fa31e331d32b402f2db25a3cb9d83d62c094`

## Model
- Name: `Baseline2DCNN`
- Architecture: Conv2d(2->16, k5) + ReLU + MaxPool, Conv2d(16->32, k3) + ReLU + MaxPool, Conv2d(32->64, k3) + ReLU + MaxPool, AdaptiveAvgPool2d(1,1), Linear(64->2)

## Task
- Binary machine-sound classification on pump clips
- Labels: `normal=0`, `abnormal=1`

## Inputs
- Window manifests (local-only, intentionally excluded from Git): `preprocessing/03.training-export/output/manifests/20260319_152022_windows.parquet, preprocessing/03.training-export/output/manifests/20260319_155015_windows.parquet`
- Clip manifests (local-only, intentionally excluded from Git): `preprocessing/03.training-export/output/manifests/20260319_152022_files.parquet, preprocessing/03.training-export/output/manifests/20260319_155015_files.parquet`
- Preprocessing configs (local-only, intentionally excluded from Git): `preprocessing/03.training-export/output/manifests/20260319_152022_config.json, preprocessing/03.training-export/output/manifests/20260319_155015_config.json`
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
- Exact split membership (local-only, intentionally excluded from Git): `models/20260323-1630-2d_sound_v0.8/split_membership.csv`
- Split strategy: `clip_random`
- Holdout units (resolved): `[]`
- Train unit ids: `['00', '02', '04', '06']`
- Validation unit ids: `['00', '02', '04', '06']`

## Training
- Epochs configured: 16
- Epochs completed: 16
- Batch size: 32
- Learning rate: 0.0005
- Device: cuda
- Random seed: 42
- Deterministic training flag: True

## Results (This Run)
- Final train accuracy: 0.7639
- Final validation accuracy (window): 0.7118
- Final validation loss: 0.5726
- Predicted distribution (window val): normal=2202 (0.6722), abnormal=1074 (0.3278)
- Clip aggregation method: probability_mean
- Clip validation accuracy: 0.7308
- Clip validation ROC AUC (mean abnormal prob): 0.7854
- Predicted distribution (clip val): normal=126 (0.6923), abnormal=56 (0.3077)
- Interpretation: C. Pipeline works, meaningful signal emerging
- Metrics JSON: `models/20260323-1630-2d_sound_v0.8/metrics.json`
- Run manifest: `models/20260323-1630-2d_sound_v0.8/run_manifest.json`

## Artifact Tracking
- Git-tracked artifacts:
  - `models/20260323-1630-2d_sound_v0.8/metrics.json`
  - `models/20260323-1630-2d_sound_v0.8/model_card.md`
  - `models/20260323-1630-2d_sound_v0.8/run_manifest.json`
  - `models/20260323-1630-2d_sound_v0.8/training_history.json`
- Local-only artifacts intentionally excluded from Git by `.gitignore`:
  - `models/20260323-1630-2d_sound_v0.8/best_val_model_state_dict.pt`
  - `models/20260323-1630-2d_sound_v0.8/clip_level_predictions.csv`
  - `models/20260323-1630-2d_sound_v0.8/clip_level_predictions.json`
  - `models/20260323-1630-2d_sound_v0.8/final_model_state_dict.pt`
  - `models/20260323-1630-2d_sound_v0.8/split_membership.csv`
