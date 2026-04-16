# Technical Writeup: Industrial Sound Anomaly Detection Repository

## Project Summary

This repository captures an end-to-end machine-learning workflow for industrial sound anomaly detection, focused on pump-machine audio. The core task is binary classification of machine clips as `normal` or `abnormal`, but the most interesting part of the work is not the classifier itself. The strongest technical contribution is the preprocessing and representation pipeline: I transformed raw audio into structured 2D and 3D geometric forms that behave like spectral heightmaps, then trained and evaluated a compact 2D CNN on those derived representations.

From an employment perspective, this project demonstrates more than model training. It shows dataset engineering, feature design, training/evaluation discipline, reproducibility work, experiment artifact design, and technical documentation intended to make notebook-based ML work reviewable and traceable.

## Dataset and Task

The repository is organized around industrial pump recordings under both normal and abnormal conditions. Based on the checked-in split artifacts and manifests, the data used in the larger runs spans:

- Pump clips labeled `normal` and `abnormal`
- Multiple dB conditions: `-6`, `0`, and `+6`
- Multiple unit IDs visible in the split artifacts: `00`, `02`, `04`, and `06`

The project evolved through progressively larger training subsets:

| Stage | Representative artifact | Dataset scale |
| --- | --- | --- |
| Early baseline | `models/20260319-1829-2d_sound_v0.2/training_history.json` | 40 clips total, 576 train windows, 144 validation windows |
| Mid-scale tracked runs | `models/20260320-1607-2d_sound_v0.4/model_card.md` | 200 clips total, 2,880 train windows, 720 validation windows |
| Larger clip-random runs | `models/20260320-2201-2d_sound_v0.5/model_card.md` | 912 clips total, 13,140 train windows, 3,276 validation windows |
| Scaled shard-based runs | `models/20260322-1335-2d_sound_v0.7/model_card.md` | 2,400 clips total, 34,560 train windows, 8,640 validation windows |

Each clip is exported into overlapping 2D windows of shape `(96 frequency bands x 64 frames)`. The checked-in clip prediction artifacts show that a typical clip contributed `18` windows in the exported corpus, which gave the training pipeline both window-level and clip-level evaluation paths.

I evaluated the task in two ways:

- `clip_random`: a standard clip-level random split for in-distribution validation
- `unit_holdout`: a stricter split that completely holds out one unit ID from training to test whether the model generalizes to unseen hardware rather than memorizing unit-specific signatures

That second evaluation mode is especially important because it turns this from a simple “can the classifier separate these samples?” problem into a more realistic “can the representation generalize across machines?” problem.

## Model and Representation Approach

### Audio frontend

The checked-in export configs in `preprocessing/03.training-export/output/manifests/*_config.json` show a consistent preprocessing contract:

- Audio resampled to `16 kHz`
- Mono conversion enabled
- STFT with `n_fft=1024`, `hop_length=256`, `win_length=1024`, `window='hann'`
- `96` log-spaced triangular frequency bands from `50 Hz` to `8000 Hz`
- Log compression and per-clip min-max normalization
- Sliding windows of `64` frames with `32`-frame stride

### Heightmap-style representation

The representation work is where this repository is most distinctive.

In `preprocessing/export_2d_training.py`, I do not stop at a conventional spectrogram. I project STFT power into log-spaced bands, normalize it, and then quantize it into `24` discrete `height_bins`. I also compute an `active_mask` that indicates whether each time-frequency location is meaningfully active.

That creates a representation that behaves like a spectral heightmap:

- the x-axis is time within the window
- the y-axis is log-spaced frequency band index
- the “height” is a quantized energy level

The exporter writes:

- `normalized_window`
- `height_bins`
- `active_mask`
- `frame_starts`
- config metadata and source references

Although the current best training notebooks consume the 2-channel tensor `(normalized_window, active_mask)` rather than `height_bins` directly, the discrete heightmap is not incidental. It is part of the feature contract, and it underpins the representation analysis notebooks and the voxel/surface experiments.

### Voxel and surface processing

The method documentation in:

- `documentation/binary_voxel_occupancy_inspection_v0.4_algorithm_spec.md`
- `documentation/binary_voxel_occupancy_inspection_v0.4_surface_processing_spec.md`

shows the next step in the same idea: converting the quantized 2D heightmap into a 3D binary voxel volume. Two voxelization modes were explored:

- `one_hot`
- `filled_column`

From that 3D volume, I extract only the top occupied voxel in each frequency-time column, producing a surface representation. This gave me a way to inspect and compare clips structurally rather than only visually. The checked-in preprocessing summaries show that the surface representation was dense and discriminative enough to support clip-to-clip comparison:

- `48,828` aligned surface rows in `preprocessing/02.preprocessing-display/output/20260317_111517_summary_comparison.csv`
- `43,419` changed surface rows
- mean absolute surface delta of `2.4044` bins
- max absolute delta of `15` bins

That is a useful signal that the engineered representation is capturing structured differences between sounds, not just noise.

### Classifier

The classifier itself is intentionally modest. In the later training notebooks the model is a compact baseline 2D CNN:

- Conv2d `2 -> 16`, kernel `5`
- Conv2d `16 -> 32`, kernel `3`
- Conv2d `32 -> 64`, kernel `3`
- max pooling between stages
- adaptive average pooling
- linear classifier to 2 output classes

This is important context: the main originality is not the network architecture. It is the feature engineering and the training/evaluation pipeline wrapped around a small, interpretable baseline model.

## What I Implemented

I implemented the project as a coherent workflow rather than a single notebook:

- A reusable export module in `preprocessing/export_2d_training.py` that converts WAV files into NPZ tensor bundles plus Parquet manifests.
- A preprocessing pipeline that computes normalized spectral windows, discrete height bins, activity masks, and metadata-rich export artifacts.
- Representation-inspection notebooks that turn the audio-derived heightmap into voxel occupancy volumes and surface-only views for structural comparison.
- A 2D CNN training pipeline that supports window-level learning and clip-level evaluation.
- Clip-level aggregation logic using mean abnormal probability and majority vote.
- Exact split tracking through `split_membership.csv`.
- Run provenance and experiment tracking through `run_manifest.json`, `metrics.json`, `training_history.json`, and `model_card.md`.
- A shard-packing utility in `utilities/pack_npz_shards_by_machine_type_v0.1.ipynb` plus the exported script in `notebooks/tmp/pack_npz_shards_by_machine_type_v0.1.py`, which enabled larger-scale training runs by repacking windows into shard NPZ files.
- RAM-preloaded dataset loading for faster scaled training.
- A stricter `unit_holdout` validation mode that tests unseen-unit generalization.
- An evaluation-only notebook, `evaluation/eval_mimii_baseline_comparison_v0.1.ipynb`, for clip-level ROC AUC comparison in a MIMII-style evaluation format.

## Results and Current Status

The results show a clear progression from a collapsed early baseline to strong in-distribution performance, followed by a harder and more realistic generalization test.

| Run | What changed | Window validation accuracy | Clip validation accuracy | Clip ROC AUC | Assessment |
| --- | --- | ---: | ---: | ---: | --- |
| `20260319-1829-2d_sound_v0.2` | Early small-subset baseline | `0.5000` | n/a | n/a | Collapsed to predicting the abnormal class |
| `20260320-1607-2d_sound_v0.4` | Larger subset + stronger artifact tracking | `0.8208` | n/a | n/a | Meaningful signal appears |
| `20260320-2201-2d_sound_v0.5` | Clip-level aggregation added | `0.9020` | `0.9176` | n/a | Strong clip-level performance on random clip split |
| `20260322-1335-2d_sound_v0.7` | Shard-based scale-up to 2,400 clips | `0.9354` | `0.9479` | `0.9926` | Best checked-in in-distribution result |
| `20260322-1438-2d_sound_v0.8` | Unseen-unit holdout (`unit 02`) | `0.6388` | `0.6471` | `0.6757` | Much harder, exposes generalization gap |
| `20260323-1656-2d_sound_v0.8` | Same v0.8 code path with random clip split | `0.8759` | `0.8938` | `0.9527` | Strong again when validation is not unit-held-out |
| `20260323-1659-2d_sound_v0.8` | 4-epoch unit-holdout rerun | `0.6198` | `0.6186` | `0.6522` | Confirms unseen-unit validation remains difficult |

### Best achieved result

The strongest checked-in result is `models/20260322-1335-2d_sound_v0.7/metrics.json`:

- window validation accuracy: `93.54%`
- clip validation accuracy: `94.79%`
- clip ROC AUC: `0.9926`
- validation set size: `480` clips
- class balance: `240 normal / 240 abnormal`

The window-level recalls in that run were also strong and reasonably balanced:

- normal recall: `0.9600`
- abnormal recall: `0.9109`

This is exactly the kind of result I would highlight in an interview: a simple CNN achieved high performance because the representation and data pipeline were strong enough to make the learning problem tractable.

### Current status

My current assessment is:

- The pipeline is successful as an end-to-end anomaly detection system under clip-random validation.
- The representation is clearly useful and supports strong clip-level separation when training and validation contain the same unit population.
- Generalization to an unseen unit is not solved yet. When the repository moves to `unit_holdout`, performance drops materially.

That drop is not a failure of the project. It is a useful result. It shows that the repository has matured enough to ask the harder question: not just whether the model can classify the dataset, but whether it can generalize across machine identity.

## Novelty Assessment

I would describe the novelty of this project as **practical and representation-driven rather than architectural**.

The CNN is intentionally baseline-level. The distinctive part is the preprocessing pipeline that turns sound into a structured heightmap/voxel representation and then uses that structure in multiple ways:

- continuous normalized band-energy windows for training
- discrete height-bin maps for quantized structure
- binary occupancy volumes for geometric interpretation
- surface extraction for compact comparison of spectral shape

Why this is novel enough to matter:

- It is more bespoke than a standard log-mel spectrogram pipeline.
- It explicitly separates energy shape from binary activity via `normalized_window` and `active_mask`.
- It creates a representation that can be inspected as geometry, not just plotted as an image.
- It supports both ML training and human-readable diagnostics from the same feature family.

Why it may have contributed to the results:

- Quantization into height bins regularizes the energy landscape and can reduce sensitivity to small amplitude fluctuations.
- The active mask gives the network a clean sparsity/occupancy cue in a second channel.
- Log-spaced bands preserve perceptually and mechanically meaningful frequency structure while keeping the input compact.
- The voxel/surface view likely helped refine the representation because it made local structural differences between clips explicit during preprocessing analysis.

My honest assessment is that the representation likely contributed substantially to the strong random-split results, but the unit-holdout runs show it is not yet fully unit-invariant. In other words:

- It appears strong enough to produce excellent in-distribution performance.
- It is not yet sufficient evidence of robust cross-unit generalization.

That is still a valuable engineering outcome. It demonstrates the ability to invent and operationalize a custom representation, test it at scale, and identify where it does and does not generalize.

## Tooling and Engineering Practice

This repository demonstrates practical use of:

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

It also demonstrates concrete ML/DSP engineering skills:

- STFT-based audio feature extraction
- log-spaced triangular filterbank design
- feature normalization and quantization
- sliding-window dataset generation
- binary classification with CNNs
- clip-level aggregation and ROC AUC evaluation
- deterministic seeding and reproducibility controls
- data-manifest design and schema validation
- sharded dataset packing for larger-scale training
- model cards, run manifests, and experiment metadata design

## Notebook Standards and Supporting Documentation

One of the strongest process signals in the repository is `documentation/notebook-generation-standards.md`.

That document is not decorative. It defines how notebook-based work should be done in this repo:

- explicit notebook category and purpose
- early configuration cells
- validation/diagnostics before execution
- one clear execution responsibility per notebook
- labeled artifact writeout
- final verdicts
- versioned notebook filenames
- machine-readable run artifacts where it matters

This standard shaped how the tooling was used across preprocessing, training, and utilities. You can see its influence in the repository’s later artifacts:

- `run_manifest.json`
- `metrics.json`
- `training_history.json`
- `split_membership.csv`
- `model_card.md`
- tracked preprocessing configs and manifests

For an employer, that matters because it shows I was not only building models. I was also building a disciplined, reviewer-friendly workflow for experimental ML work without depending on heavyweight external MLOps infrastructure.

## Concrete Skills Demonstrated

- Audio DSP and feature engineering
- Custom representation design for non-image data
- PyTorch model development
- Experiment design and controlled iteration
- Dataset curation and split strategy design
- Evaluation design, including clip-level ROC AUC and unseen-unit validation
- Data pipeline implementation from raw audio to train-ready tensors
- Metadata and artifact schema design
- Reproducibility and traceability engineering
- Technical documentation for ML workflows
- Turning notebook-heavy research work into a structured, reviewable codebase

## Recommended Employment-Facing Summary

If I had to summarize the value of this repository in a few lines for a hiring manager, I would describe it like this:

> I built an end-to-end industrial sound anomaly detection pipeline centered on custom audio representation engineering. Rather than relying only on a standard spectrogram baseline, I designed a heightmap-style spectral representation with quantized energy structure, activity masking, and voxel/surface inspection tooling, then trained a compact 2D CNN on the resulting windows. The best checked-in run reached 94.8% clip accuracy and 0.9926 clip ROC AUC on a 480-clip validation set, while later unseen-unit holdout experiments exposed the remaining generalization challenge. The repo also demonstrates disciplined experiment tracking through manifests, model cards, split artifacts, and notebook standards designed for traceability and employer-facing legibility.
