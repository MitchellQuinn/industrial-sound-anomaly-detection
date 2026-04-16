# Model Card (Brief)

## Notebook
- Category: `training`
- Source notebook: `train/2d-cnn/notebooks/2d_sound_v0.8.ipynb`
- Notebook SHA256: `bc485f54eab0c5d1296fd2e90b804c600e602a4e9820412f3469ec3158f83382`

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
- Exact split membership: `models/20260323-1659-2d_sound_v0.8/split_membership.csv`
- Split strategy: `unit_holdout`
- Holdout units (resolved): `['02']`
- Train unit ids: `['00', '04', '06']`
- Validation unit ids: `['02']`

## Training
- Epochs configured: 4
- Epochs completed: 4
- Batch size: 32
- Learning rate: 0.0005
- Device: cuda
- Random seed: 42
- Deterministic training flag: True

## Results (This Run)
- Final train accuracy: 0.8985
- Final validation accuracy (window): 0.6198
- Final validation loss: 0.9099
- Predicted distribution (window val): normal=3301 (0.3480), abnormal=6185 (0.6520)
- Clip aggregation method: probability_mean
- Clip validation accuracy: 0.6186
- Clip validation ROC AUC (mean abnormal prob): 0.6522
- Predicted distribution (clip val): normal=181 (0.3435), abnormal=346 (0.6565)
- Interpretation: B. Pipeline works, weak but nontrivial signal
- Metrics JSON: `models/20260323-1659-2d_sound_v0.8/metrics.json`
- Run manifest: `models/20260323-1659-2d_sound_v0.8/run_manifest.json`

## Artifact Tracking
- Git-tracked artifacts:
  - `models/20260323-1659-2d_sound_v0.8/training_history.json`
  - `models/20260323-1659-2d_sound_v0.8/metrics.json`
  - `models/20260323-1659-2d_sound_v0.8/split_membership.csv`
  - `models/20260323-1659-2d_sound_v0.8/clip_level_predictions.csv`
  - `models/20260323-1659-2d_sound_v0.8/clip_level_predictions.json`
  - `models/20260323-1659-2d_sound_v0.8/run_manifest.json`
  - `models/20260323-1659-2d_sound_v0.8/model_card.md`
- Local-only artifacts intentionally excluded from Git by `.gitignore`:
  - `models/20260323-1659-2d_sound_v0.8/final_model_state_dict.pt`
  - `models/20260323-1659-2d_sound_v0.8/best_val_model_state_dict.pt`
