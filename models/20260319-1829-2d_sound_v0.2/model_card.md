# Model Card (Brief)

## Model
- Name: `Baseline2DCNN`
- Source notebook: `2d_sound_v0.2.ipynb`
- Architecture: Conv2d(2->16, k5) + ReLU + MaxPool, Conv2d(16->32, k3) + ReLU + MaxPool, Conv2d(32->64, k3) + ReLU + MaxPool, AdaptiveAvgPool2d(1,1), Linear(64->2)

## Task
- Binary machine-sound classification on pump clips
- Labels: `normal=0`, `abnormal=1`

## Inputs
- Base representation: `normalized_window` shape `(96, 64)` and `active_mask` shape `(96, 64)`
- Input mode: `normalized_plus_mask`
- Final tensor shape: `(2, 96, 64)`

## Data & Split
- Clip-level split only (no window-level random split)
- Selected clips: normal=20, abnormal=20
- Train clips: normal=16, abnormal=16
- Validation clips: normal=4, abnormal=4
- Train windows: normal=288, abnormal=288
- Validation windows: normal=72, abnormal=72

## Training
- Epochs configured: 2
- Epochs completed: 2
- Batch size: 32
- Learning rate: 0.001
- Device: cuda
- Random seed: 42

## Results (This Run)
- Final train accuracy: 0.5000
- Final validation accuracy: 0.5000
- Final validation loss: 0.6934
- Predicted distribution (val): normal=0 (0.0000), abnormal=144 (1.0000)
- Interpretation: A. Pipeline works, model still collapses

## Artifacts
- Final model checkpoint: `final_model_state_dict.pt`
- Best-val checkpoint: `best_val_model_state_dict.pt`
- Training history JSON: `training_history.json`
