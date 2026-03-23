# Model Card (Brief)

## Notebook
- Category: `training`
- Source notebook: `train/2d-cnn/notebooks/2d_sound_v0.8.ipynb`
- Notebook SHA256: `f686b747ae9253aa3408c78275479bec7b448b5bb98c2dbf444f9338c5f1dff0`

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
- Train clips: normal=1006, abnormal=867
- Validation clips: normal=194, abnormal=333
- Train windows: normal=18108, abnormal=15606
- Validation windows: normal=3492, abnormal=5994
- Exact split membership: `models/20260322-1756-2d_sound_v0.8/split_membership.csv`
- Split strategy: `unit_holdout`
- Holdout units (resolved): `['02']`
- Train unit ids: `['00', '04', '06']`
- Validation unit ids: `['02']`

## Training
- Epochs configured: 128
- Epochs completed: 128
- Batch size: 32
- Learning rate: 0.0005
- Device: cuda
- Random seed: 42
- Deterministic training flag: True

## Results (This Run)
- Final train accuracy: 0.9918
- Final validation accuracy (window): 0.6539
- Final validation loss: 2.0890
- Predicted distribution (window val): normal=773 (0.0815), abnormal=8713 (0.9185)
- Clip aggregation method: probability_mean
- Clip validation accuracy: 0.6509
- Clip validation ROC AUC (mean abnormal prob): 0.7102
- Predicted distribution (clip val): normal=18 (0.0342), abnormal=509 (0.9658)
- Interpretation: C. Pipeline works, meaningful signal emerging
- Metrics JSON: `models/20260322-1756-2d_sound_v0.8/metrics.json`
- Run manifest: `models/20260322-1756-2d_sound_v0.8/run_manifest.json`

## Artifact Tracking
- Git-tracked artifacts:
  - `models/20260322-1756-2d_sound_v0.8/training_history.json`
  - `models/20260322-1756-2d_sound_v0.8/metrics.json`
  - `models/20260322-1756-2d_sound_v0.8/split_membership.csv`
  - `models/20260322-1756-2d_sound_v0.8/clip_level_predictions.csv`
  - `models/20260322-1756-2d_sound_v0.8/clip_level_predictions.json`
  - `models/20260322-1756-2d_sound_v0.8/run_manifest.json`
  - `models/20260322-1756-2d_sound_v0.8/model_card.md`
- Local-only artifacts intentionally excluded from Git by `.gitignore`:
  - `models/20260322-1756-2d_sound_v0.8/final_model_state_dict.pt`
  - `models/20260322-1756-2d_sound_v0.8/best_val_model_state_dict.pt`
