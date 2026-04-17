# Technical Writeup: Industrial Sound Anomaly Detection Repository

## Public Repo Scope

This repository is a public, employer-facing portfolio artifact rather than a
full training package. It intentionally excludes:

- MIMII audio and other third-party audio assets
- exported tensor corpora and shard files
- split-membership CSV exports
- clip-level prediction exports
- model checkpoint binaries

That means the public evidence trail lives in:

- source code
- method documentation
- model cards
- run manifests
- training histories
- summary metrics

Dataset attribution and licensing notes are documented in `README.md`,
`COPYRIGHT.md`, and `THIRD_PARTY_NOTICES.md`.

## Project Summary

This repository captures an end-to-end machine-learning workflow for industrial
sound anomaly detection focused on pump-machine audio. The core task is binary
classification of machine clips as `normal` or `abnormal`, but the most
interesting part of the work is not the classifier itself. The strongest
technical contribution is the preprocessing and representation pipeline: I
transformed raw audio into structured 2D and 3D geometric forms that behave
like spectral heightmaps, then trained and evaluated a compact 2D CNN on those
derived representations.

From an employment perspective, this project demonstrates more than model
training. It shows dataset engineering, feature design, training/evaluation
discipline, reproducibility work, experiment artifact design, and technical
documentation intended to make notebook-based ML work reviewable and traceable.

## Dataset and Task

The project was developed locally against the public MIMII dataset, using pump
machine recordings under normal and abnormal conditions. The dataset itself is
not redistributed here, but the public repo still makes the task legible:

- model cards and run manifests show clip counts, window counts, and split
  strategy
- later run manifests show the unit IDs involved in the larger `v0.8` runs:
  `00`, `02`, `04`, and `06`
- the repo documents both `clip_random` and `unit_holdout` evaluation modes

The project evolved through progressively larger tracked runs:

| Stage | Representative public artifact | Dataset scale |
| --- | --- | --- |
| Early baseline | `models/20260319-1829-2d_sound_v0.2/training_history.json` | 40 clips total, 576 train windows, 144 validation windows |
| Mid-scale tracked runs | `models/20260320-1607-2d_sound_v0.4/model_card.md` | 200 clips total, 2,880 train windows, 720 validation windows |
| Larger clip-random runs | `models/20260320-2201-2d_sound_v0.5/model_card.md` | 912 clips total, 13,140 train windows, 3,276 validation windows |
| Scaled shard-based runs | `models/20260322-1335-2d_sound_v0.7/model_card.md` | 2,400 clips total, 34,560 train windows, 8,640 validation windows |

Each clip is exported into overlapping 2D windows of shape
`(96 frequency bands x 64 frames)`. In the larger tracked runs, the clip and
window totals imply about `18` windows per clip in practice. For example, the
`v0.7` training run reports `34,560 / 1,920 = 18` train windows per clip and
`8,640 / 480 = 18` validation windows per clip.

I evaluated the task in two ways:

- `clip_random`: a standard clip-level random split for in-distribution
  validation
- `unit_holdout`: a stricter split that completely holds out one unit ID from
  training to test whether the model generalizes to unseen hardware rather than
  memorizing unit-specific signatures

That second evaluation mode is especially important because it turns this from
a simple "can the classifier separate these samples?" problem into a more
realistic "can the representation generalize across machines?" problem.

## Model and Representation Approach

### Audio frontend

The preprocessing contract is codified directly in
`preprocessing/export_2d_training.py` via the `ExportConfig` defaults:

- audio resampled to `16 kHz`
- mono conversion enabled
- STFT with `n_fft=1024`, `hop_length=256`, `win_length=1024`,
  `window='hann'`
- `96` log-spaced triangular frequency bands from `50 Hz` to `8000 Hz`
- log compression and per-clip min-max normalization
- sliding windows of `64` frames with `32`-frame stride

### Heightmap-style representation

The representation work is where this repository is most distinctive.

In `preprocessing/export_2d_training.py`, I do not stop at a conventional
spectrogram. I project STFT power into log-spaced bands, normalize it, and then
quantize it into `24` discrete `height_bins`. I also compute an `active_mask`
that indicates whether each time-frequency location is meaningfully active.

That creates a representation that behaves like a spectral heightmap:

- the x-axis is time within the window
- the y-axis is log-spaced frequency band index
- the "height" is a quantized energy level

The exporter is designed to write:

- `normalized_window`
- `height_bins`
- `active_mask`
- `frame_starts`
- config metadata and source references

Although the current best training notebooks consume the 2-channel tensor
`(normalized_window, active_mask)` rather than `height_bins` directly, the
discrete heightmap is not incidental. It is part of the feature contract, and
it underpins the representation analysis notebooks and the voxel/surface
experiments.

### Voxel and surface processing

The method documentation in:

- `documents/binary_voxel_occupancy_inspection_v0.4_algorithm_spec.md`
- `documents/binary_voxel_occupancy_inspection_v0.4_surface_processing_spec.md`

shows the next step in the same idea: converting the quantized 2D heightmap
into a 3D binary voxel volume. Two voxelization modes were explored:

- `one_hot`
- `filled_column`

From that 3D volume, I extract only the top occupied voxel in each
frequency-time column, producing a surface representation. This gave me a way
to inspect and compare clips structurally rather than only visually. Those
surface/voxel artifacts are part of the exploratory method-development story,
not the headline public evaluation evidence.

### Classifier

The classifier itself is intentionally modest. In the later training notebooks
the model is a compact baseline 2D CNN:

- Conv2d `2 -> 16`, kernel `5`
- Conv2d `16 -> 32`, kernel `3`
- Conv2d `32 -> 64`, kernel `3`
- max pooling between stages
- adaptive average pooling
- linear classifier to 2 output classes

This is important context: the main originality is not the network
architecture. It is the feature engineering and the training/evaluation
pipeline wrapped around a small, interpretable baseline model.

## What I Implemented

I implemented the project as a coherent workflow rather than a single notebook:

- A reusable export module in `preprocessing/export_2d_training.py` that
  converts WAV files into NPZ tensor bundles plus Parquet manifests.
- A preprocessing pipeline that computes normalized spectral windows, discrete
  height bins, activity masks, and metadata-rich export artifacts.
- Representation-inspection notebooks that turn the audio-derived heightmap
  into voxel occupancy volumes and surface-only views for structural
  comparison.
- A 2D CNN training pipeline that supports window-level learning and clip-level
  evaluation.
- Clip-level aggregation logic using mean abnormal probability and majority
  vote.
- Run provenance and experiment tracking through `run_manifest.json`,
  `metrics.json`, `training_history.json`, and `model_card.md`.
- A shard-packing utility in
  `utilities/pack_npz_shards_by_machine_type_v0.1.ipynb` plus the exported
  script in `notebooks/tmp/pack_npz_shards_by_machine_type_v0.1.py`, which
  enabled larger-scale training runs by repacking windows into shard NPZ files.
- RAM-preloaded dataset loading for faster scaled training.
- A stricter `unit_holdout` validation mode that tests unseen-unit
  generalization.
- An evaluation-only notebook, `evaluation/eval_mimii_baseline_comparison_v0.1.ipynb`,
  for clip-level ROC AUC comparison in a MIMII-style evaluation format.

During local development I also captured exact split membership and clip-level
prediction exports, but the public repository intentionally keeps only redacted
split summaries, counts, artifact paths, and headline metrics.

## Results and Evidence Trail

The results show a clear progression from a collapsed early baseline to strong
in-distribution performance, followed by a harder and more realistic
generalization test.

The earliest baseline predates the later `run_manifest.json` convention, so its
public evidence trail is slightly thinner than the later runs. From `v0.4`
onward, the repo becomes much more explicit about metrics and run provenance.

| Run | Window validation accuracy | Clip validation accuracy | Clip ROC AUC | Public evidence | Assessment |
| --- | ---: | ---: | ---: | --- | --- |
| `20260319-1829-2d_sound_v0.2` | `0.5000` | n/a | n/a | `training_history.json` + `model_card.md` | Collapsed to predicting the abnormal class |
| `20260320-1607-2d_sound_v0.4` | `0.8208` | n/a | n/a | `metrics.json` + `model_card.md` | Meaningful signal appears |
| `20260320-2201-2d_sound_v0.5` | `0.9020` | `0.9176` | n/a | `metrics.json` + `model_card.md` | Strong clip-level performance on random clip split |
| `20260322-1335-2d_sound_v0.7` | `0.9354` | `0.9479` | `0.9926` | `metrics.json` + `model_card.md` + `run_manifest.json` | Best tracked in-distribution result |
| `20260322-1438-2d_sound_v0.8` | `0.6388` | `0.6471` | `0.6757` | `metrics.json` + `model_card.md` + `run_manifest.json` | Much harder, exposes generalization gap |
| `20260323-1656-2d_sound_v0.8` | `0.8759` | `0.8938` | `0.9527` | `metrics.json` + `model_card.md` + `run_manifest.json` | Strong again when validation is not unit-held-out |
| `20260323-1659-2d_sound_v0.8` | `0.6198` | `0.6186` | `0.6522` | `metrics.json` + `model_card.md` + `run_manifest.json` | Confirms unseen-unit validation remains difficult |

### Best achieved result

The strongest tracked in-distribution result is
`models/20260322-1335-2d_sound_v0.7/metrics.json`:

- window validation accuracy: `93.54%`
- clip validation accuracy: `94.79%`
- clip ROC AUC: `0.9926`
- validation set size: `480` clips
- class balance: `240 normal / 240 abnormal`

The window-level recalls in that run were also strong and reasonably balanced:

- normal recall: `0.9600`
- abnormal recall: `0.9109`

This is the kind of result I would highlight in an interview: a simple CNN
achieved high performance because the representation and data pipeline were
strong enough to make the learning problem tractable.

### Harder generalization result

The most important cautionary result is the `unit_holdout` run
`models/20260322-1438-2d_sound_v0.8/metrics.json`:

- window validation accuracy: `63.88%`
- clip validation accuracy: `64.71%`
- clip ROC AUC: `0.6757`

That drop is not something I would hide. It is useful evidence that the
repository matured far enough to ask the harder question: not just whether the
model can classify clips from a familiar unit population, but whether it can
generalize across machine identity.

## What the Results Mean

My current assessment is:

- The pipeline is successful as an end-to-end anomaly detection system under
  clip-random validation.
- The representation is clearly useful and supports strong clip-level
  separation when training and validation contain the same unit population.
- Generalization to an unseen unit is not solved yet. When the repository moves
  to `unit_holdout`, performance drops materially.

My honest assessment is that the representation likely contributed
substantially to the strong random-split results, but the unit-holdout runs
show it is not yet fully unit-invariant. In other words:

- It appears strong enough to produce excellent in-distribution performance.
- It is not yet sufficient evidence of robust cross-unit generalization.

That is still a valuable engineering outcome. It demonstrates the ability to
invent and operationalize a custom representation, test it at scale, and
identify where it does and does not generalize.

## Engineering Practices Visible in the Public Tree

The repository demonstrates practical use of:

- Python
- Jupyter notebooks
- PyTorch
- TorchAudio
- NumPy
- pandas
- scikit-learn
- librosa
- matplotlib
- Plotly
- Parquet manifests
- NPZ tensor packaging
- Git-based experiment artifact tracking

It also shows concrete engineering practices that are visible from the tracked
artifacts:

- `run_manifest.json` records commit, branch, dirty-worktree status, entrypoint
  hash, environment summary, source manifest hashes, split counts, and unit
  coverage.
- `model_card.md` gives a reviewer a short summary of model architecture,
  inputs, data split, result headline, and which artifacts are tracked versus
  local-only.
- `training_history.json` preserves the learning curve and prediction-distribution
  behavior over time.
- `documents/notebook-generation-standards.md` defines notebook structure,
  artifact expectations, validation sections, and final-verdict conventions.
- The method specs in `documents/` make the representation logic legible
  without requiring a reviewer to reverse-engineer every notebook cell.

## Reviewer Guide

If a technically literate employer wants to substantiate the main claims from
the public tree, the fastest path is:

1. Read `README.md` for repository scope, public-release constraints, and MIMII
   attribution.
2. Read `documents/employment-technical-writeup.md` for the short employer-facing
   summary.
3. Read this document for the fuller technical narrative.
4. Inspect `preprocessing/export_2d_training.py` for the feature contract.
5. Inspect the method specs in `documents/` for the voxel/surface processing
   path.
6. Compare `models/20260322-1335-2d_sound_v0.7/` and
   `models/20260322-1438-2d_sound_v0.8/` to see the difference between
   in-distribution success and unseen-unit difficulty.

## Recommended Employment-Facing Summary

If I had to summarize the value of this repository in a few lines for a hiring
manager, I would describe it like this:

> I built an end-to-end industrial sound anomaly detection pipeline centered on
> custom audio representation engineering. Rather than relying only on a
> standard spectrogram baseline, I designed a heightmap-style spectral
> representation with quantized energy structure, activity masking, and
> voxel/surface inspection tooling, then trained a compact 2D CNN on the
> resulting windows. The best tracked run reached 94.8% clip accuracy and
> 0.9926 clip ROC AUC on a 480-clip validation set, while later unseen-unit
> holdout experiments exposed the remaining generalization challenge. The public
> repo also demonstrates disciplined experiment tracking through model cards,
> run manifests, training histories, metrics files, and notebook standards
> designed for traceability and employer-facing legibility.
