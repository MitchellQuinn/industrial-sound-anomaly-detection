# Model Card (Brief)

## Notebook
- Category: `training`
- Source notebook: `train/2d-cnn/notebooks/2d_sound_v0.8.ipynb`
- Notebook SHA256: `3a83eafbd5d42794b0ae1f72ac1d84f93dc999ce76969c0b2fae86128f070536`

## Model
- Name: `Baseline2DCNN`
- Architecture: Conv2d(2->16, k5) + ReLU + MaxPool, Conv2d(16->32, k3) + ReLU + MaxPool, Conv2d(32->64, k3) + ReLU + MaxPool, AdaptiveAvgPool2d(1,1), Linear(64->2)

## Task
- Binary machine-sound classification on pump clips
- Labels: `normal=0`, `abnormal=1`

## Inputs
- Window manifests: `training-data/pump/shards/20260322_131954_pump_npz_shards_v0.1/manifests/shard_windows.parquet`
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
- Exact split membership: `models/20260323-1656-2d_sound_v0.8/split_membership.csv`
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
- Final train accuracy: 0.8424
- Final validation accuracy (window): 0.8759
- Final validation loss: 0.3217
- Predicted distribution (window val): normal=4278 (0.4951), abnormal=4362 (0.5049)
- Clip aggregation method: probability_mean
- Clip validation accuracy: 0.8938
- Clip validation ROC AUC (mean abnormal prob): 0.9527
- Predicted distribution (clip val): normal=243 (0.5062), abnormal=237 (0.4938)
- Interpretation: C. Pipeline works, meaningful signal emerging
- Metrics JSON: `models/20260323-1656-2d_sound_v0.8/metrics.json`
- Run manifest: `models/20260323-1656-2d_sound_v0.8/run_manifest.json`

## Artifact Tracking
- Git-tracked artifacts:
  - `models/20260323-1656-2d_sound_v0.8/training_history.json`
  - `models/20260323-1656-2d_sound_v0.8/metrics.json`
  - `models/20260323-1656-2d_sound_v0.8/split_membership.csv`
  - `models/20260323-1656-2d_sound_v0.8/clip_level_predictions.csv`
  - `models/20260323-1656-2d_sound_v0.8/clip_level_predictions.json`
  - `models/20260323-1656-2d_sound_v0.8/run_manifest.json`
  - `models/20260323-1656-2d_sound_v0.8/model_card.md`
- Local-only artifacts intentionally excluded from Git by `.gitignore`:
  - `models/20260323-1656-2d_sound_v0.8/final_model_state_dict.pt`
  - `models/20260323-1656-2d_sound_v0.8/best_val_model_state_dict.pt`
