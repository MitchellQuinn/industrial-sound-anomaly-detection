# Employer-Facing Summary: Industrial Sound Anomaly Detection

This repository is meant to be readable as a portfolio artifact, not just a
dump of notebooks. The strongest story is not "I trained a classifier"; it is
"I designed a custom audio representation, built a disciplined experimental
workflow around it, and documented both the successes and the limitations."

## What This Repo Demonstrates

- Custom audio feature engineering beyond a plain spectrogram baseline
- End-to-end ML workflow design from preprocessing through evaluation
- Practical PyTorch implementation and experiment iteration
- Structured experiment metadata, model cards, and run manifests
- Honest handling of generalization limits, not just headline in-distribution
  metrics

## Strongest Publicly Verifiable Claims

### 1. I built a custom representation pipeline for industrial audio

Public evidence:

- `preprocessing/export_2d_training.py`
- `documents/binary_voxel_occupancy_inspection_v0.4_algorithm_spec.md`
- `documents/binary_voxel_occupancy_inspection_v0.4_surface_processing_spec.md`

What a reviewer can verify:

- STFT-based frontend with log-spaced band projection
- per-clip normalization plus quantized `height_bins`
- explicit `active_mask`
- voxel/surface transformation for structural inspection

### 2. I turned that representation into a usable training pipeline

Public evidence:

- `train/2d-cnn/notebooks/2d_sound_v0.7.ipynb`
- `train/2d-cnn/notebooks/2d_sound_v0.8.ipynb`
- `models/*/model_card.md`
- `models/*/run_manifest.json`

What a reviewer can verify:

- compact 2D CNN architecture
- clip-level aggregation
- random-split versus unit-holdout validation
- scaled runs backed by shard manifests

### 3. The best in-distribution run is strong, and the holdout result is honest

Public evidence:

- `models/20260322-1335-2d_sound_v0.7/metrics.json`
- `models/20260322-1335-2d_sound_v0.7/model_card.md`
- `models/20260322-1438-2d_sound_v0.8/metrics.json`
- `models/20260322-1438-2d_sound_v0.8/model_card.md`

What a reviewer can verify:

- best tracked random-split result: `94.79%` clip accuracy, `0.9926` clip ROC
  AUC
- materially weaker unseen-unit result: `64.71%` clip accuracy, `0.6757` clip
  ROC AUC
- the repo does not hide the generalization gap

### 4. I treated notebook-heavy work like a documented engineering workflow

Public evidence:

- `documents/notebook-generation-standards.md`
- `models/*/run_manifest.json`
- `models/*/training_history.json`
- `models/*/model_card.md`

What a reviewer can verify:

- explicit artifact expectations
- environment capture and entrypoint hashing in later run manifests
- tracked versus local-only artifact labeling
- enough metadata to understand what changed between major runs

## Important Public-Repo Constraint

This public repository intentionally does not redistribute MIMII audio, derived
tensor corpora, split-membership CSVs, clip-level prediction exports, or model
weights. That is why the public proof set relies on code, documentation, model
cards, run manifests, training histories, and summary metrics instead of full
development exhaust.

## Best Reading Order

1. `README.md`
2. `documents/technical-writeup-industrial-sound-anomaly-detection.md`
3. `preprocessing/export_2d_training.py`
4. `models/20260322-1335-2d_sound_v0.7/`
5. `models/20260322-1438-2d_sound_v0.8/`

## Concise Interview Summary

> I built an industrial sound anomaly detection pipeline around a custom
> heightmap-style audio representation, then validated it with a compact 2D
> CNN, clip-level aggregation, and both random-split and unseen-unit evaluation
> modes. The best tracked run reached 94.8% clip accuracy and 0.9926 ROC AUC,
> while later holdout runs showed that cross-unit generalization remained the
> main unsolved problem. The repo also shows disciplined experiment tracking and
> documentation rather than notebook-only trial and error.
