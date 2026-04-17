# Repository Development Timeline

## 1. Summary

The repository appears to have started with preprocessing and representation-inspection work on 2026-03-16, appears to have moved into reusable export and tensor-validation work on 2026-03-19, and likely reached first visible model training the same day. On 2026-03-20 and 2026-03-21, the project became more structured: git history begins, traceability documentation was added, run manifests and split tracking appeared, and clip-level evaluation was added. On 2026-03-22 and 2026-03-23, the work scaled up through shard-based training and stricter `unit_holdout` testing. A later phase on 2026-04-12 to 2026-04-16 focused on employer/public-facing curation rather than original model-building.

## 2. Evidence Sources Used

Sources actually used for this timeline:

- Git commit history from `git log --reverse --all` and targeted per-file history.
- Git-tracked file inventory from `git ls-files`.
- Local file modified timestamps from tracked notebooks, scripts, docs, manifests, and model artifacts.
- Export config/manifests in `preprocessing/03.training-export/output/manifests/`.
- Training artifacts in `models/*/`, especially `run_manifest.json`, `model_card.md`, `metrics.json`, and `training_history.json`.
- Existing documentation in `documents/`, especially the notebook standards and employer technical writeup.
- Notebook and script version names such as `2d_sound_v0.1` through `v0.8`, `export_2d_training_v0.3` through `v0.5`, and `pack_npz_shards_by_machine_type_v0.1`.
- Local but non-git-tracked supporting artifacts where useful:
  - `preprocessing/02.preprocessing-display/output/`
  - `preprocessing/02.preprocessing-display/.archive/`
  - `evaluation/outputs/`
  - `training-data/pump/shards/.../manifests/shard_windows.parquet`

Notes on source strength:

- Git evidence is the strongest source where available.
- File mtimes were used as fallback because the first commit already contains earlier work.
- Ignored/local artifacts were used cautiously and only as supporting evidence.

## 3. Chronological Timeline

### 2026-03-16

**Initial audio preprocessing inspection and exploratory voxel work**

- What changed: the earliest visible work appears to be a preprocessing inspection notebook, followed later the same day by archived voxel-inspection notebook versions and CSV outputs.
- Directly evidenced:
  - `preprocessing/01.preprocessing-display/audio_preprocessing_inspection.ipynb` has a local mtime of `2026-03-16 15:52 UTC`.
  - Local archived notebooks `preprocessing/02.preprocessing-display/.archive/binary_voxel_occupancy_inspection_v0.1.ipynb` through `v0.3.ipynb` have mtimes between `16:38` and `17:19 UTC`.
  - Local CSV outputs in `preprocessing/02.preprocessing-display/output/` were created later that day.
- Inferred from surrounding artifacts: this suggests the project began as hands-on representation inspection before reusable export/training tooling existed.
- Evidence used: local file timestamps, notebook names, output filenames.
- Confidence level: `medium`

### 2026-03-17 to 2026-03-18

**Voxel/surface representation stabilised and was documented**

- What changed: the voxel-inspection work appears to have consolidated into `binary_voxel_occupancy_inspection_v0.4`, with comparison outputs on 2026-03-17 and method-spec documents on 2026-03-18.
- Directly evidenced:
  - `preprocessing/02.preprocessing-display/binary_voxel_occupancy_inspection_v0.4.ipynb` has a local mtime of `2026-03-17 15:06 UTC`.
  - Local comparison outputs such as `20260317_111517_summary_comparison.csv` and `20260317_114214_summary_comparison.csv` carry dated run IDs from 2026-03-17.
  - `documents/binary_voxel_occupancy_inspection_v0.4_algorithm_spec.md` and `documents/binary_voxel_occupancy_inspection_v0.4_surface_processing_spec.md` have local mtimes of `2026-03-18 18:02 UTC` and `18:08 UTC`.
- Inferred from surrounding artifacts: this suggests the preprocessing/representation approach had reached a documented method boundary before the training pipeline was built.
- Evidence used: file timestamps, dated output filenames, method documents.
- Confidence level: `medium`

### 2026-03-19 (morning to early afternoon)

**Reusable export pipeline and tensor validation were added**

- What changed: a Python export module, an export notebook, and tensor-inspection notebooks appear by this point.
- Directly evidenced:
  - `preprocessing/03.training-export/export_2d_training_v0.3.ipynb` has a local mtime of `2026-03-19 11:11 UTC`.
  - `preprocessing/export_2d_training.py` has a local mtime of `2026-03-19 12:38 UTC`.
  - `preprocessing/04.tensor-inspection/tensor-inspection-v0.1.ipynb` and `tensor-inspection-v0.2.ipynb` have local mtimes of `12:07 UTC` and `13:02 UTC`.
- Inferred from surrounding artifacts: this likely marks the transition from exploratory inspection to reusable train-data generation and validation.
- Evidence used: file timestamps, notebook/script names.
- Confidence level: `medium`

### 2026-03-19 15:17 to 15:52 UTC

**First dated export manifests were generated across pump subsets**

- What changed: the repo contains seven dated export-config/manifests generated in a short sequence, covering pump `normal` and `abnormal` data at `-6 dB`, `0 dB`, and `+6 dB`, after an initial small sample run.
- Directly evidenced:
  - `preprocessing/03.training-export/output/manifests/20260319_151735_config.json` records `max_files = 10` for `-6db-pump-normal`.
  - Later config files `20260319_152022`, `152728`, `153211`, `155015`, `155126`, and `155233` record full export runs for `-6db`, `+0db`, and `+6db` pump normal/abnormal directories.
  - The config contents show a stable preprocessing contract: `target_sr=16000`, `n_fft=1024`, `num_bands=96`, `window_frames=64`, `stride_frames=32`, and `energy_bins=24`.
- Inferred from surrounding artifacts: the exporter was first checked on a limited subset, then run across the main pump slices the same afternoon.
- Evidence used: exact config timestamps, config contents, manifest filenames.
- Confidence level: `high`

### 2026-03-19 (late afternoon to evening)

**First training notebooks and first model run appeared**

- What changed: the first `2d_sound` training notebooks appear to have been created and the earliest checked-in model run appears to have been produced.
- Directly evidenced:
  - `train/2d-cnn/notebooks/2d_sound_v0.1.ipynb` and `2d_sound_v0.2.ipynb` have local mtimes of `17:24 UTC` and `18:28 UTC`.
  - `models/20260319-1829-2d_sound_v0.2/model_card.md` and `training_history.json` have local mtimes of `18:29 UTC`.
  - The model card records a small 40-clip experiment, 2 configured/completed epochs, and a collapsed `0.5000` validation result.
- Inferred from surrounding artifacts: this suggests the export-to-training path was functioning on the same day the export manifests were created.
- Evidence used: notebook mtimes, model card, training history.
- Confidence level: `high`

### 2026-03-20 (morning to noon)

**The first git snapshot was created and the repo gained a minimal README**

- What changed: git history starts on 2026-03-20, capturing preprocessing notebooks, manifests, early training notebooks, and early model metadata; `README.md` was added immediately afterward.
- Directly evidenced:
  - Commit `d3f7c1d` on `2026-03-20 12:20 UTC`: "Initial commit: preprocessing, training notebooks, manifests, and model metadata".
  - Commit `9cfdc4b` on `2026-03-20 12:22 UTC`: "Added README.md".
  - `train/2d-cnn/notebooks/2d_sound_v0.3.ipynb` has a local mtime of `2026-03-20 12:00 UTC`, just before those commits.
- Inferred from surrounding artifacts: git history does not capture the real start of development; it begins after several days of local work had already happened.
- Evidence used: git log, README mtime, notebook mtime.
- Confidence level: `high`

### 2026-03-20 (afternoon to evening)

**Traceability discipline and stronger tracked training runs were added**

- What changed: notebook/process documentation was added, `2d_sound_v0.4` introduced tracked run-manifest metadata, and `2d_sound_v0.5` added clip-level evaluation before expanding to a larger dataset the same day.
- Directly evidenced:
  - Commit `44e6d23` on `2026-03-20 15:26 UTC` adds `documents/notebook-generation-standards.md`, `documents/traceability-report.md`, and `train/2d-cnn/notebooks/2d_sound_v0.4.ipynb`.
  - `models/20260320-1607-2d_sound_v0.4/run_manifest.json` records `created_utc`, `git_commit`, source manifests, preprocessing configs, and exact split membership.
  - Commits `7197551`, `32c8dfe`, and `d231cd9` track the addition, validation, and later 256-epoch rerun of `2d_sound_v0.5`.
  - Model folders `20260320-1855-2d_sound_v0.5` and `20260320-2201-2d_sound_v0.5` show clip-level prediction artifacts and a jump from 200 selected clips to 912 selected clips.
- Inferred from surrounding artifacts: 2026-03-20 is the clearest shift from "working notebooks" to "tracked experiments with provenance and evaluation artifacts".
- Evidence used: git commits, run manifests, model cards, metrics, notebook mtimes.
- Confidence level: `high`

### 2026-03-21

**Evaluation notebooks and data-organisation utilities were added**

- What changed: evaluation-only notebooks were added, evaluation outputs appear to have been generated against the best `v0.5` checkpoint, and utility notebooks were added for WAV renaming and directory restructuring.
- Directly evidenced:
  - Commit `041efd8` on `2026-03-21 16:28 UTC` adds `evaluation/eval_mimii_baseline_comparison_v0.1.ipynb`, `v0.2.ipynb`, and `train/2d-cnn/notebooks/2d_sound_v0.6.ipynb`.
  - Local evaluation outputs in `evaluation/outputs/20260321-133844-mimii-eval/` and `20260321-134108-mimii-eval/` record `saved_at_utc` timestamps and point to checkpoint `models/20260320-2201-2d_sound_v0.5/best_val_model_state_dict.pt`.
  - Commits `8c9aea3` and `53bf173` add `rename_training_wavs_v0.2.ipynb` and `copy_machine_wavs_to_pump_structure_v0.1.ipynb`.
- Inferred from surrounding artifacts: between the larger `v0.5` runs and the later scale-up work, the evidence is consistent with part of the effort shifting to benchmarking and cleaning/standardising input layout.
- Evidence used: git commits, notebook mtimes, local evaluation outputs.
- Confidence level: `high` for notebook/commit additions, `medium` for the broader interpretation

### 2026-03-22

**Shard-based scale-up and stronger generalization testing**

- What changed: the project added scaling/efficiency utilities, likely created a shard manifest, ran the larger `2d_sound_v0.7` experiment, and then tested `unit_holdout` generalization in `v0.8`.
- Directly evidenced:
  - Commit `2a4257b` on `2026-03-22 13:10 UTC`: "Training data creation and efficiency improvements".
  - `utilities/compare_dataset_combo_totals_v0.1.ipynb` and `utilities/pack_npz_shards_by_machine_type_v0.1.ipynb` have local mtimes of `11:02 UTC` and `13:20 UTC`.
  - Local shard manifest `training-data/pump/shards/20260322_131954_pump_npz_shards_v0.1/manifests/shard_windows.parquet` has a local mtime of `2026-03-22 13:20 UTC`.
  - Model folders `20260322-1335-2d_sound_v0.7`, `20260322-1438-2d_sound_v0.8`, and `20260322-1756-2d_sound_v0.8` record scaled clip counts and, for `v0.8`, `unit_holdout` with holdout unit `02`.
- Inferred from surrounding artifacts: this is the clearest one-day jump from mid-scale experiments to larger-scale training and more realistic generalization testing.
- Evidence used: git commit, utility notebook mtimes, local shard manifest timestamp, model cards, run manifests.
- Confidence level: `high`

### 2026-03-23

**New export/training notebook revisions and rapid `v0.8` reruns**

- What changed: a new export notebook and a new `v0.8` training notebook were committed; later that day, four additional `v0.8` runs appear to have been saved covering both `clip_random` and `unit_holdout`, and both original-manifest and shard-backed data paths.
- Directly evidenced:
  - Commit `5d2c667` on `2026-03-23 15:56 UTC` adds `preprocessing/03.training-export/export_2d_training_v0.5.ipynb`, `train/2d-cnn/notebooks/2d_sound_v0.8.ipynb`, exported tmp scripts under `notebooks/tmp/`, and the 2026-03-22 model folders.
  - Model folders `20260323-1630-2d_sound_v0.8`, `1653`, `1656`, and `1659` have local run timestamps between `16:30 UTC` and `16:59 UTC`.
  - Their model cards/run manifests show a mixture of `clip_random` and `unit_holdout`, plus both earlier export-manifest input paths and the newer shard-manifest path.
- Inferred from surrounding artifacts: 2026-03-23 was a rapid refinement/retest day rather than a new repository phase.
- Evidence used: git commit, model cards, run manifests, notebook/script mtimes.
- Confidence level: `high`

### 2026-04-12 to 2026-04-16

**Employer-facing writeup and public-release curation**

- What changed: the repository appears to have gained an employer-oriented technical writeup and then a public-release cleanup pass that preserved lightweight run history while removing bulkier generated artifacts from git.
- Directly evidenced:
  - `documents/employment-technical-writeup.md` has a local mtime of `2026-04-12 14:39 BST`.
  - Commit `e229b5d` on `2026-04-16 10:46 BST` says "Prepare repository for public GitHub release".
  - That commit refines `.gitignore`, renames `documentation/` to `documents/`, stops tracking prediction exports and split membership CSVs, and adds the latest dated `20260323-*` run folders as lightweight metadata snapshots.
- Inferred from surrounding artifacts: this was a curation/publication pass applied after the main March build sprint, not part of the original technical build window.
- Evidence used: git commit, file mtime, `.gitignore`, documentation files.
- Confidence level: `high`

## 4. Key Development Phases

- **Initial setup and representation exploration (`2026-03-16` to `2026-03-18`)**
  - Audio preprocessing inspection, voxel/surface comparison work, and method documentation were established before the first commit.

- **Export pipeline and first training loop (`2026-03-19`)**
  - The reusable export module, dated manifests, tensor inspection, and first baseline training run all appear on the same day.

- **Traceability and evaluation maturation (`2026-03-20` to `2026-03-21`)**
  - Git history begins, notebook/process standards are added, run manifests and split tracking appear, and clip-level plus evaluation-only workflows are added.

- **Scale-up and generalization testing (`2026-03-22` to `2026-03-23`)**
  - Dataset sharding, larger training runs, and `unit_holdout` validation appear, followed by several fast reruns of `v0.8`.

- **Publication / portfolio preparation (`2026-04-12` to `2026-04-16`)**
  - Employer-facing narrative material and a public-release cleanup pass were added after the main technical build.

## 5. Gaps / Uncertainties

- The git history starts on `2026-03-20`, but several tracked files and dated artifacts clearly predate that. Early work from `2026-03-16` to `2026-03-19` therefore lacks commit-by-commit history.
- Some useful evidence is local and ignored by git, including `.archive/` notebooks, preprocessing CSV outputs, evaluation outputs, and the shard manifest under `training-data/`. These help with ordering, but they are weaker than tracked git evidence.
- File mtimes show latest local modification time, not guaranteed original creation time. They were used as fallback evidence and cross-checked against artifact names, run IDs, and later commits rather than used alone for major claims.
- Several tracked training runs from `v0.4` onward record `dirty_worktree: true` in `run_manifest.json`, which means the exact executed notebook state may differ slightly from the nearest committed notebook snapshot.
- The earliest model run, `models/20260319-1829-2d_sound_v0.2/`, does not include a `run_manifest.json`, so its provenance is weaker than later runs.
- The `20260323-*` model runs were created on 2026-03-23 but only entered git on 2026-04-16 as part of the release-prep snapshot. For those runs, artifact timestamps are more informative than commit date.
- `documents/traceability-report.md` is useful as evidence that documentation/traceability work existed by 2026-03-20, but it was not used as a precise internal date anchor for later model milestones.

## 6. Candidate “Sub-Week Build” Support

### Strong support

- From the earliest tracked preprocessing notebook (`2026-03-16 15:52 UTC`) to the scaled `v0.7` run (`2026-03-22 13:35 UTC`) is about `5.9 days`.
- From the first dated export manifest (`2026-03-19 15:17 UTC`) to the scaled `v0.7` run (`2026-03-22 13:35 UTC`) is about `2.9 days`.
- From the first model run (`2026-03-19 18:29 UTC`) to the latest March rerun (`2026-03-23 16:59 UTC`) is about `3.9 days`.
- These intervals support a claim that the core technical build, from preprocessing/export through meaningful training/evaluation, happened very quickly.

### Weak or not supported

- From the earliest visible tracked preprocessing artifact (`2026-03-16 15:52 UTC`) to the latest March run (`2026-03-23 16:59 UTC`) is about `7.05 days`, which is slightly more than one week.
- If the claim includes the employer/publication pass (`2026-04-12` to `2026-04-16`), the project clearly extends well beyond one week.

### Overall assessment

The “sub-week build” claim is well supported if it refers to the core technical build reaching scaled training/evaluation by 2026-03-22. It is moderately supported if it refers to most of the March technical work but excludes later public-release curation. It is not well supported if it is meant to cover the entire visible repository arc through the 2026-03-23 reruns and the 2026-04-16 release-prep commit.

## 7. Safe Claims / Confidence Summary

- Strongly supported by git + artifacts: by `2026-03-20`, the repository already contained preprocessing notebooks, export manifests, early training notebooks, and model metadata; from `2026-03-20` to `2026-03-23`, the addition of traceability docs, run manifests, evaluation notebooks, scale-up utilities, and later training runs is directly visible in commits plus tracked model artifacts.
- Supported mainly by local file timestamps / surrounding evidence: the pre-git sequence from `2026-03-16` through early `2026-03-19` appears consistent with notebook mtimes, dated output filenames, and adjacent artifacts, but is not commit-backed step by step.
- Strength of the “sub-week build” claim: strong if bounded to the core technical build reaching scaled training/evaluation by `2026-03-22`; moderate if extended through most March reruns; weak if stretched to the full visible repository arc or later release/publication curation.

## 8. Closing Note on Purpose

This timeline exists to improve repository legibility, keep historical claims bounded to evidence that is actually visible in git and local artifacts, and provide a clean base for later employer-facing or public-facing adaptation.
