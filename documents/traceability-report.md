# Traceability, Reproducibility, and Employer-Facing Legibility Audit

> Note: this audit reflects an internal pre-release review state. References to
> tracked export manifest snapshots, split-membership CSVs, or clip-level
> prediction exports describe local workflow artifacts that were later removed
> or redacted for the public GitHub release.

Audit scope: repository contents as currently present in Git plus the current local workspace layout where relevant.

High-level assessment: the repository already shows real rigor in preprocessing and metadata capture, but the strongest provenance is on the export side, not the training side. A technically literate employer would likely conclude that the project is serious and actively structured, but not yet fully trustworthy as a reproducible experiment record because a specific model run cannot be reconstructed from repo-visible artifacts alone.

## A. Current strengths

- The repo already separates concerns into recognizable phases: preprocessing, export, tensor inspection, training, model metadata, and documentation. The folder layout is not random.
- The preprocessing/export path is more disciplined than typical notebook-only repos. `preprocessing/export_2d_training.py` is a proper Python module, and `preprocessing/03.training-export/export_2d_training_v0.4.ipynb` is orchestration-only rather than burying the whole pipeline in notebook cells.
- Export runs produce concrete metadata artifacts:
  - checked-in config snapshots in `preprocessing/03.training-export/output/manifests/*_config.json`
  - checked-in file manifests in `preprocessing/03.training-export/output/manifests/*_files.parquet`
  - checked-in window manifests in `preprocessing/03.training-export/output/manifests/*_windows.parquet`
- The export manifests are meaningful, not toy placeholders. They contain per-file and per-window rows, timestamps, device/backend fields, and tensor paths.
- The tensor exports themselves embed config metadata. The `.npz` bundles include `config_json`, `source_file`, `relative_source_path`, and basic shape/config fields. That is a strong traceability instinct even though the tensors are intentionally excluded from Git.
- There is evidence of validation discipline around preprocessing outputs:
  - `preprocessing/04.tensor-inspection/tensor-inspection-v0.2.ipynb` is explicitly framed as validation-only
  - it ends in a clear machine verdict pattern such as `READY_TO_PROCEED` vs `PATCH_REQUIRED`
- The training notebooks are not completely opaque. They save:
  - `training_history.json`
  - a short `model_card.md`
  - checkpoints
  - confusion matrix and classification report
- The checked-in model metadata in `models/20260319-1829-2d_sound_v0.2/` is useful portfolio evidence. A reviewer can see the intended model family, task, input representation, seed, batch size, learning rate, and final metrics.
- The repo does not try to commit obviously huge raw/generated corpora. `.gitignore` excludes `.wav`, `.npz`, `training-data/`, and common run/output directories. That is reasonable and professional for a portfolio-scale ML repo.
- Method documentation exists and is better than average for a personal ML repository. The voxel-processing specs in `documents/` make the feature logic legible to a reviewer.
- Notebook versioning by filename is at least somewhat disciplined. Files like `2d_sound_v0.1.ipynb`, `2d_sound_v0.2.ipynb`, `2d_sound_v0.3.ipynb` and corresponding preprocessing notebook versions show iteration rather than silent overwrites.

## B. Current weaknesses / gaps

### Training-run traceability

- A specific training run cannot be tied to an exact code snapshot.
  - `models/20260319-1829-2d_sound_v0.2/training_history.json` says `num_epochs_config = 2`.
  - the current checked-in `train/2d-cnn/notebooks/2d_sound_v0.2.ipynb` sets `num_epochs = 64`.
  - This means the current notebook is not reliably the exact code that produced the saved run.
- Training outputs record a notebook name string, not a notebook hash or Git commit. That is too weak for provenance.
- The training run does not save the exact input manifest paths or manifest run IDs that fed the model. The notebook reads manifests from ignored `training-data/.../manifests/...`, but the saved run metadata does not preserve those references.
- The exact train/validation split membership is not saved.
  - Current metadata stores clip/window counts only.
  - It does not store the actual clip IDs or a split manifest.
- The run does not record what changed relative to prior runs.
  - There is no experiment log, run index, or diff-style notes.
  - Git history is minimal and not linked to individual experiments.
- The checkpoints themselves are only lightly described.
  - They store `model_state_dict` plus a few scalar hyperparameters.
  - They do not store manifest references, code hash, architecture config object, environment, or split membership.
- Randomness control is incomplete for GPU reproducibility.
  - The notebook sets `np.random.seed` and `torch.manual_seed`.
  - It does not capture `torch.cuda.manual_seed_all`, deterministic backend flags, or equivalent reproducibility settings.

### Data traceability

- Data origin is not clearly documented for an external reader.
  - The repo layout suggests corpus slices like `-6db-pump-normal`, `+0db-pump-abnormal`, etc.
  - There is no tracked data catalog explaining where these clips come from, how the corpus is organized, what each directory means, or what is canonical.
- The path from raw audio to training artifacts is only partially explicit.
  - A careful reader can infer it by reading notebooks and scripts.
  - A cold reviewer cannot learn it quickly from repo-facing docs.
- The export manifests do not include explicit semantic fields like `machine`, `label`, `db_level`, or a stable `clip_id`.
  - The training notebook compensates by inferring labels and machine type from filenames and path text.
  - That is fragile and not ideal scientific metadata.
- Most paths in configs/manifests are absolute local-machine paths such as `/abs/path/to/repo/...`.
  - This hurts portability.
  - It also makes the metadata feel tied to one workstation instead of repo-native.
- It is not clearly documented which artifacts are canonical metadata and which are bulky derived data.
  - The checked-in export metadata lives under `preprocessing/03.training-export/output/manifests/`.
  - The word `output` usually suggests ephemeral generated files, even though these are being treated as important tracked metadata.

### Repo legibility for employers

- `README.md` is too thin to do the job.
  - It states what the repo includes/excludes, but it does not explain the project workflow, research question, data pipeline, modeling approach, or where a reviewer should look first.
- There is no clean top-level narrative from raw audio -> preprocessing -> manifests -> training -> model artifacts -> conclusions.
- The most important artifacts are not surfaced.
  - A reviewer would have to discover the meaningful files by browsing the tree.
- The repo is disciplined enough to feel promising, but not yet curated enough to feel publication-ready or interview-ready.
- Notebook sprawl is controlled better than average, but still not fully framed.
  - Multiple versioned notebooks are present.
  - Many notebooks have tracked outputs.
  - There is no documented distinction between exploratory notebooks, validation notebooks, and presentation-quality notebooks.
- A concrete naming inconsistency exists in the current training notebooks:
  - `train/2d-cnn/notebooks/2d_sound_v0.3.ipynb` sets `notebook_filename_for_outputs = '2d_sound_baseline_v0.2.ipynb'`.
  - This shows that output naming can drift from the actual file being run.

### Documentation quality

- There is no repo overview document beyond the brief README.
- There is no data pipeline overview document.
- There is no experiment logging convention document.
- There is no run manifest schema document.
- There is no reproducibility note describing how much can be reproduced from Git alone versus what requires excluded local data.
- There is no notebook standard that says what every notebook must save, how paths should be handled, or how outputs should be curated.
- Existing method docs are good, but they are focused on one representation family rather than the whole project lifecycle.

### Artifact/versioning hygiene

- `requirements.txt` is pinned, but not sufficient to recreate the working environment.
  - The repo code uses `torch`, `torchaudio`, and parquet reading.
  - the pinned file does not include `torch`, `torchaudio`, or `pyarrow`/`fastparquet`.
  - A fresh install from `requirements.txt` cannot run the current export/training flow as-is.
- The model card lists checkpoint artifacts, but those `.pt` files are ignored by Git and are not repo-visible to an external reviewer.
  - Excluding weights is reasonable.
  - The model card should explicitly label them as local-only artifacts if they are not committed.
- The tracked manifest snapshots are appropriate to keep.
  - They are lightweight enough and materially improve traceability.
- The current directory names are mostly understandable, but `output/manifests` is semantically confusing for files that are intentionally versioned metadata.

## C. Priority fixes

### Critical

- Add a real per-training-run manifest.
  - Needed for actual scientific traceability.
  - Each run needs one canonical machine-readable record that links code, manifests, preprocessing config, split membership, hyperparameters, seeds, environment, and outputs.
- Stop relying on notebook filename strings as provenance.
  - Needed for actual scientific traceability.
  - Save Git commit, dirty-worktree flag, and a notebook/script content hash for every run.
- Save exact split membership for every run.
  - Needed for actual scientific traceability.
  - Counts are not enough.
- Make the environment recreatable from tracked files.
  - Needed for actual scientific traceability.
  - `requirements.txt` must account for the actual training/export stack, including PyTorch and parquet support.
- Add a top-level repo narrative in `README.md`.
  - Needed for employer-facing legibility.
  - Right now the repo does not quickly communicate the workflow or rigor to a cold reviewer.

### Important

- Add explicit semantic metadata to export manifests.
  - Needed for actual scientific traceability.
  - Include at least machine type, label, dB condition, and stable clip ID as columns rather than inferring them later from filenames.
- Replace absolute-path-only metadata with repo-relative canonical paths.
  - Needed for actual scientific traceability and employer-facing legibility.
- Add a lightweight experiment log / run index.
  - Needed mostly for employer-facing legibility, but also helps traceability.
  - This is the easiest way to answer “what changed and why?”
- Add a data pipeline overview and a reproducibility note.
  - Needed for employer-facing legibility and project usability.
- Document which tracked metadata is canonical and which bulky outputs remain excluded from Git.
  - Needed for both traceability and presentation.

### Nice to have

- Curate notebooks into “working” vs “published” roles.
  - Mostly employer-facing legibility.
- Add a concise results summary table at repo level.
  - Mostly employer-facing legibility.
- Consider renaming or documenting the tracked manifest location so it no longer looks ephemeral.
  - Mostly employer-facing legibility.
- Add a lightweight model/result gallery or run summary page.
  - Mostly employer-facing legibility.

## D. Recommended minimum viable traceability standard

For every training run, the minimum credible repo-native standard should be:

### Required run folder

Use `models/<run_id>/` and include:

- `run_manifest.json`
- `training_history.json`
- `metrics.json`
- `split_membership.csv` or `split_membership.parquet`
- `model_card.md`
- optional local-only weights, clearly marked as excluded from Git if not committed

### Required fields in `run_manifest.json`

- `run_id`
- `created_utc`
- `git_commit`
- `git_branch`
- `dirty_worktree`
- `entrypoint_type`
- `entrypoint_path`
- `entrypoint_sha256`
- `command_or_notebook_version`
- `source_manifest_paths`
- `source_manifest_sha256`
- `preprocessing_config_paths`
- `preprocessing_config_sha256`
- `dataset_summary`
- `split_artifact_path`
- `model_name`
- `model_architecture_summary`
- `input_representation`
- `input_shape`
- `optimizer`
- `loss_function`
- `hyperparameters`
- `random_seeds`
- `environment`
- `artifact_paths`
- `notes`

### Required fields in split membership artifact

- `clip_id`
- `label`
- `machine`
- `condition`
- `db_level`
- `split`
- `source_manifest_run_id`
- `tensor_npz_path` or canonical tensor identifier

### Required environment block

- `python_version`
- `torch_version`
- `torchaudio_version`
- `numpy_version`
- `pandas_version`
- `pyarrow_version`
- `cuda_version`
- `device`
- `requirements_hash`

### Minimum rule of thumb

If a future reader cannot answer “what exact code + what exact data split + what exact preprocessing metadata + what exact environment produced this result?” from the run folder alone, the run is not fully traceable.

## E. Recommended repo-facing artifacts for employer legibility

### README structure

Recommended `README.md` structure:

- Project purpose
- Problem statement and why industrial sound anomaly detection matters here
- Repo map
- Workflow overview
- What is tracked vs intentionally excluded
- Quick-start reproduction path
- Key results / current status
- Where to inspect one representative preprocessing run and one representative training run
- Reproducibility notes

### Repo-facing documents to add

- `documents/repo-overview.md`
  - one-page orientation for a cold reviewer
- `documents/data-pipeline.md`
  - raw audio -> preprocessing -> tensor export -> manifests -> training
- `documents/data-catalog.md`
  - corpus slices, naming scheme, labels, dB conditions, and what is excluded from Git
- `documents/run-manifest-schema.md`
  - the exact schema expected in each training run folder
- `documents/reproducibility.md`
  - what can be reproduced from Git alone, what requires local data, and how to regenerate derived corpora
- `documents/notebook-generation-standards.md`
  - standards for notebook config cells, metadata capture, path handling, output hygiene, and artifact writing
- `documents/experiment-log.md`
  - human-readable run ledger with brief “what changed / why” notes

### Employer-facing summary artifacts

- A small run index, either `documents/experiment-log.md` or `results/run-index.csv`, with one row per experiment:
  - run ID
  - date
  - notebook/entrypoint
  - data slice
  - preprocessing manifest IDs
  - model variant
  - key metrics
  - short rationale
- A representative `model_card.md` that links back to the exact run manifest and split artifact.
- One clearly marked “best current baseline” summary artifact.
  - This could be `documents/current-baseline.md`.
  - A hiring manager should not have to guess which run matters most.

### What this improves

- Needed for actual scientific traceability:
  - run manifests, split artifacts, environment notes, explicit data catalog
- Needed for employer-facing presentation:
  - top-level narrative, run index, current baseline summary, notebook standards

## F. Concrete implementation suggestions

### Files to add

- `documents/data-pipeline.md`
- `documents/data-catalog.md`
- `documents/reproducibility.md`
- `documents/run-manifest-schema.md`
- `documents/notebook-generation-standards.md`
- `documents/experiment-log.md`
- `models/README.md`
  - explain run folder conventions and what is and is not committed

### Files to modify

- `README.md`
  - expand into a real project overview and workflow guide
- `requirements.txt`
  - include the actual runtime dependencies used by export/training, or split into clearly named environment files
- `preprocessing/export_2d_training.py`
  - emit more semantic metadata into file/window manifests
  - prefer canonical repo-relative paths in addition to any absolute convenience paths
- `preprocessing/03.training-export/export_2d_training_v0.4.ipynb`
  - write or preview a concise export run summary artifact intended for Git
- `train/2d-cnn/notebooks/2d_sound_v0.2.ipynb`
  - save a proper `run_manifest.json`
  - save exact split membership
  - capture environment and Git metadata
  - stop relying on a manually typed notebook filename string
- `train/2d-cnn/notebooks/2d_sound_v0.3.ipynb`
  - fix naming drift and bring it under the same run-manifest convention

### Metadata fields to include in export manifests

- `export_run_id`
- `dataset_id`
- `machine`
- `label`
- `db_level`
- `clip_id`
- `source_relpath`
- `config_snapshot_path`
- `config_sha256`
- `num_windows`
- `duration_seconds`

### Metadata fields to include in training run manifests

- `run_id`
- `git_commit`
- `dirty_worktree`
- `entrypoint_path`
- `entrypoint_sha256`
- `input_manifest_paths`
- `input_manifest_sha256`
- `preprocessing_config_paths`
- `preprocessing_config_sha256`
- `train_clip_ids`
- `val_clip_ids`
- optional `test_clip_ids`
- `model_architecture`
- `optimizer`
- `loss`
- `hyperparameters`
- `seeds`
- `environment_versions`
- `artifact_inventory`
- `parent_run_id` or `changed_from_run_id`
- `change_note`

### Directory structure suggestions

- Keep raw audio and bulk tensors out of Git. That choice is justified.
- Keep checked-in manifests/config snapshots in Git, but either:
  - rename `preprocessing/03.training-export/output/manifests/` to a more clearly canonical metadata location, or
  - document explicitly that this path is the canonical tracked export metadata location
- Standardize each training run as a self-contained folder under `models/<run_id>/`

### Lightweight workflow recommendation

- Do not add heavy external tracking infrastructure.
- Instead, add one small helper module that every training notebook calls at the end to write:
  - `run_manifest.json`
  - `metrics.json`
  - `split_membership.csv`
  - `model_card.md`
- That will give most of the credibility benefits of experiment tracking while staying entirely repo-native and maintainable by one person.

## Bottom line

Current repo impression: promising, serious, and more structured than a typical notebook-heavy ML portfolio, especially on preprocessing and metadata capture.

Current repo risk: a future reviewer still cannot fully trust that a specific training result is reproducible from the checked-in artifacts, because code version, exact input manifests, exact split membership, and environment are not yet pinned tightly enough.

Most important next step: make each training run self-describing with one machine-readable run manifest plus a saved split artifact, then add a stronger README and experiment log so both scientific traceability and employer-facing legibility improve at the same time.
