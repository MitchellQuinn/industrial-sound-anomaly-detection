# Model Card (Brief)

## Notebook
- Category: `training`
- Source notebook: `train/2d-cnn/notebooks/2d_sound_v0.7.ipynb`
- Notebook SHA256: `e5ab7c9bdafcd75da56862f14b81a056a291353f3f79588ab15fa18c0ecc02fb`

## Model
- Name: `Baseline2DCNN`
- Architecture: Conv2d(2->16, k5) + ReLU + MaxPool, Conv2d(16->32, k3) + ReLU + MaxPool, Conv2d(32->64, k3) + ReLU + MaxPool, AdaptiveAvgPool2d(1,1), Linear(64->2)

## Task
- Binary machine-sound classification on pump clips
- Labels: `normal=0`, `abnormal=1`

## Inputs
- Window manifests (local-only, intentionally excluded from Git): `training-data/pump/shards/20260322_131954_pump_npz_shards_v0.1/manifests/shard_windows.parquet`
- Clip manifests: `(derived from window manifests)`
- Preprocessing configs: ``
- Base representation: `normalized_window` shape `(96, 64)` and `active_mask` shape `(96, 64)`
- Input mode: `normalized_plus_mask`
- Final tensor shape: `(2, 96, 64)`
- Clip-level split only (no window-level random split)

## Data & Split
- Selected clips: normal=1200, abnormal=1200
- Train clips: normal=960, abnormal=960
- Validation clips: normal=240, abnormal=240
- Train windows: normal=17280, abnormal=17280
- Validation windows: normal=4320, abnormal=4320
- Exact split membership (local-only, intentionally excluded from Git): `models/20260322-1335-2d_sound_v0.7/split_membership.csv`

## Training
- Epochs configured: 16
- Epochs completed: 16
- Batch size: 32
- Learning rate: 0.0005
- Device: cuda
- Random seed: 42
- Deterministic training flag: True

## Results (This Run)
- Final train accuracy: 0.9203
- Final validation accuracy (window): 0.9354
- Final validation loss: 0.1717
- Predicted distribution (window val): normal=4532 (0.5245), abnormal=4108 (0.4755)
- Clip aggregation method: probability_mean
- Clip validation accuracy: 0.9479
- Clip validation ROC AUC (mean abnormal prob): 0.9926
- Predicted distribution (clip val): normal=253 (0.5271), abnormal=227 (0.4729)
- Interpretation: C. Pipeline works, meaningful signal emerging
- Metrics JSON: `models/20260322-1335-2d_sound_v0.7/metrics.json`
- Run manifest: `models/20260322-1335-2d_sound_v0.7/run_manifest.json`

## Artifact Tracking
- Git-tracked artifacts:
  - `models/20260322-1335-2d_sound_v0.7/metrics.json`
  - `models/20260322-1335-2d_sound_v0.7/model_card.md`
  - `models/20260322-1335-2d_sound_v0.7/run_manifest.json`
  - `models/20260322-1335-2d_sound_v0.7/training_history.json`
- Local-only artifacts intentionally excluded from Git by `.gitignore`:
  - `models/20260322-1335-2d_sound_v0.7/best_val_model_state_dict.pt`
  - `models/20260322-1335-2d_sound_v0.7/clip_level_predictions.csv`
  - `models/20260322-1335-2d_sound_v0.7/clip_level_predictions.json`
  - `models/20260322-1335-2d_sound_v0.7/final_model_state_dict.pt`
  - `models/20260322-1335-2d_sound_v0.7/split_membership.csv`
