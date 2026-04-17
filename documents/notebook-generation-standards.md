# Notebook Generation Standards

## 1. Purpose

This document defines the minimum standard for notebooks that are committed to this repository.

The goal is not to eliminate notebooks. The goal is to make notebook-based work:

- scientifically traceable
- practically reproducible
- legible to a technical reviewer or potential employer
- maintainable by one person without heavyweight MLOps infrastructure

For local development, this standard prefers exact split membership and rich
intermediate metadata capture. For the public portfolio release, some
dataset-derived artifacts may be summarized, redacted, or excluded from Git
when redistribution or attribution constraints apply. In those cases, the
public repo should still retain enough tracked metadata to explain the run and
its headline results.

These standards apply to new notebooks and to major revisions of existing notebooks in:

- `preprocessing/`
- `train/`
- `utilities/`

## 2. Design Principles

### 2.1 Scientific traceability

A future reader should be able to tell:

- what notebook or script was run
- what inputs were used
- what configuration was used
- what outputs were produced
- what the notebook concluded

### 2.2 Employer-facing legibility

A technically literate reviewer should be able to open a notebook cold and quickly understand:

- why it exists
- whether it is exploratory, validation-only, export-oriented, or training-oriented
- whether it produced canonical artifacts or only local/debug outputs
- whether the result is a successful checkpoint, a failed test, or an intermediate experiment

### 2.3 Lightweight, repo-native workflow

The repo should not depend on external tracking platforms to look disciplined.

The standard is:

- small number of explicit artifacts
- predictable naming
- explicit config cells
- machine-readable metadata where it matters
- Markdown explanations where a human reader benefits

## 3. Notebook Categories

Every committed notebook must declare its category in the first Markdown cell.

Allowed categories:

- `inspection`
  - visual or analytic review of one file, one tensor, or one intermediate representation
- `validation`
  - checks whether an export or representation is correct enough to proceed
- `export`
  - creates training-ready derived artifacts and manifests
- `training`
  - runs a model experiment and writes run artifacts
- `utility`
  - controlled repo-maintenance task such as renaming or metadata repair

The first Markdown cell should state:

- notebook title
- category
- purpose
- inputs
- expected outputs
- what decision the notebook is meant to support

## 4. Required Notebook Structure

Every committed notebook should follow this high-level structure.

### 4.1 Cell 1: Title and purpose

The first Markdown cell must include:

- notebook name and version
- one-paragraph purpose
- whether the notebook is exploratory, validation-only, export-oriented, or training-oriented
- whether it writes canonical artifacts, local-only artifacts, or no artifacts

### 4.2 Cell 2: Imports and repo-root detection

The first code cell should:

- import dependencies
- detect repo root in a stable way
- avoid assuming the notebook is launched from one exact working directory

If notebook behavior depends on repo structure, that dependency should be explicit.

### 4.3 Cell 3: Configuration

All editable run configuration must live in one clearly labeled cell near the top.

This cell should contain:

- input paths or input selectors
- output paths
- core parameters
- execution toggles
- any defaults that affect artifact generation

Do not bury important configuration later in the notebook.

### 4.4 Input selection / loading section

The notebook should make it obvious:

- what files or manifests it expects
- how they are selected
- what fallback behavior exists if widgets do not render

### 4.5 Validation / diagnostics section

Before the main operation, the notebook should check:

- required files exist
- schemas/keys/columns are present
- configuration values are valid
- the selected inputs match the notebook’s expectations

### 4.6 Main execution section

This section should do one clear job only.

Examples:

- inspect one audio transformation
- export one corpus slice
- validate one tensor export
- train one model configuration

### 4.7 Artifact writeout section

If the notebook writes files, it must do so in one clearly labeled section near the end.

### 4.8 Final verdict section

Each notebook must end with a final machine-readable or visually explicit conclusion.

Examples:

- `READY_TO_PROCEED`
- `PATCH_REQUIRED`
- `READY_FOR_LARGER_RUN`
- `READY_FOR_NEXT_DECISION`

The verdict should match the notebook’s role.

## 5. Naming and Versioning Rules

### 5.1 Notebook filenames

Notebook filenames should be:

- descriptive
- versioned
- stable once committed

Preferred style:

- `audio_preprocessing_inspection_v0.1.ipynb`
- `export_2d_training_v0.4.ipynb`
- `2d_sound_v0.3.ipynb`

### 5.2 When to increment the version

Increment the version when any of the following changes:

- algorithm behavior
- output schema
- artifact contract
- model behavior or experiment protocol
- interpretation logic

Do not silently reuse an old version number for meaningfully different behavior.

### 5.3 Output identity must come from the actual notebook

If a notebook writes run artifacts, the artifact metadata should derive notebook identity from:

- the real notebook path
- or a helper variable generated from the actual file identity

Do not manually hardcode stale notebook names into output metadata.

## 6. Path and Data Handling Rules

### 6.1 Prefer repo-relative canonical paths in saved metadata

Saved metadata should prefer repo-relative canonical references whenever possible.

Allowed:

- `training-data/-6db-pump-normal/manifests/20260319_152022_windows.parquet`
- `models/20260319-1829-2d_sound_v0.2/training_history.json`

Not preferred as the only saved reference:

- `/abs/path/to/repo/...`

Absolute paths may still be useful at runtime, but canonical saved metadata should include repo-relative paths.

### 6.2 Make tracked vs excluded artifacts explicit

Each notebook that writes files must state which outputs are:

- tracked in Git
- intentionally excluded from Git
- local-only convenience outputs

### 6.3 Use semantic metadata, not filename parsing alone

If a notebook creates canonical manifests, include semantic fields directly when possible:

- `clip_id`
- `machine`
- `label`
- `db_level`
- `condition`
- `dataset_id`

Filename parsing may still be used as a helper, but should not be the only source of meaning downstream.

## 7. Execution Standards

### 7.1 Restart-and-run-all standard

A committed notebook should be able to run top-to-bottom in a clean kernel, assuming documented dependencies and required local data are present.

### 7.2 No hidden state dependence

Do not rely on:

- variables created in deleted cells
- manually injected objects
- prior notebook sessions
- out-of-order execution history

### 7.3 Fail loudly on invalid state

When required inputs are missing or malformed, the notebook should raise a clear error rather than continue silently.

### 7.4 Preview-first behavior for utility notebooks

Utility notebooks that rename, move, or rewrite files must default to preview/dry-run behavior.

If execution is destructive or irreversible, require an explicit flag such as:

- `PROCESS = False` by default
- `APPLY_CHANGES = False` by default

## 8. Output Hygiene

### 8.1 Only commit intentional outputs

Notebook outputs should be committed only when they serve one of these purposes:

- canonical metadata
- representative evidence of a run
- polished diagnostic output that materially aids understanding

### 8.2 Avoid noisy or bulky outputs in Git

Do not commit:

- large binary outputs
- long transient debug dumps
- repeated plots that add no value
- local-only intermediate artifacts

### 8.3 Curate visible notebook outputs

If a notebook is committed with outputs, those outputs should be intentionally useful:

- one representative summary table
- a concise diagnostic result
- a final verdict
- one or two meaningful plots if needed

## 9. Required Standards by Notebook Type

## 9.1 Inspection notebooks

Examples in this repo:

- audio preprocessing inspection
- voxel occupancy inspection

Inspection notebooks must:

- state clearly that they are not training notebooks
- state whether they write export files or only visual/debug outputs
- expose key representation parameters in one config cell
- print or display selected input identity
- make any export side effects explicit

Inspection notebooks should usually end with one of:

- `INSPECTION_COMPLETE`
- `READY_TO_PROCEED`
- `PATCH_REQUIRED`

## 9.2 Validation notebooks

Examples in this repo:

- `tensor-inspection-v0.2.ipynb`

Validation notebooks must:

- say exactly what is being validated
- define pass/fail checks explicitly
- emit a final verdict
- make the validation criteria legible to a human reader

Recommended verdict vocabulary:

- `READY_TO_PROCEED`
- `PATCH_REQUIRED`

## 9.3 Export notebooks

Examples in this repo:

- `export_2d_training_v0.4.ipynb`

Export notebooks must write:

- config snapshot
- file manifest
- window manifest
- export summary

Required export metadata:

- `export_run_id`
- `run_started_utc`
- `run_finished_utc`
- `input_dir`
- `output_root`
- `backend`
- `device`
- `num_source_files_selected`
- `status_counts`
- `config_snapshot_path`
- `files_manifest_path`
- `windows_manifest_path`

Preferred manifest fields:

- `clip_id`
- `machine`
- `label`
- `db_level`
- `dataset_id`
- `relative_source_path`
- `tensor_npz_path`
- `num_windows`
- `started_utc`
- `finished_utc`

Export notebooks should also make clear:

- which manifest location is canonical
- whether derived tensors are tracked or excluded

## 9.4 Training notebooks

Examples in this repo:

- `2d_sound_v0.1.ipynb`
- `2d_sound_v0.2.ipynb`
- `2d_sound_v0.3.ipynb`

Training notebooks must write a self-contained run folder under:

- `models/<run_id>/`

Minimum required artifacts:

- `run_manifest.json`
- `training_history.json`
- `metrics.json`
- `split_membership.csv` or `split_membership.parquet` for the local canonical
  run record, or a public-safe redacted equivalent documented in
  `run_manifest.json` and `model_card.md`
- `model_card.md`

Optional local-only artifacts:

- model checkpoints
- plots
- confusion matrix images

If local-only artifacts are excluded from Git, that must be stated in `model_card.md`.

### Required `run_manifest.json` fields

- `run_id`
- `created_utc`
- `git_commit`
- `git_branch`
- `dirty_worktree`
- `entrypoint_type`
- `entrypoint_path`
- `entrypoint_sha256`
- `source_manifest_paths`
- `source_manifest_sha256`
- `preprocessing_config_paths`
- `preprocessing_config_sha256`
- `dataset_summary`
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
- `change_note`

### Required split membership fields

- `clip_id`
- `label`
- `machine`
- `condition`
- `db_level`
- `split`
- `source_manifest_path`
- `tensor_npz_path`

### Required environment fields

- `python_version`
- `torch_version`
- `torchaudio_version`
- `numpy_version`
- `pandas_version`
- `pyarrow_version`
- `cuda_version`
- `device`

### Training reproducibility rule

A training notebook is not considered fully traceable unless a future reader can recover:

- the exact data inputs
- the exact split membership
- the exact model configuration
- the exact hyperparameters
- the exact environment summary

from the run folder alone.

## 9.5 Utility notebooks

Examples in this repo:

- training WAV rename notebook

Utility notebooks must:

- explain the change they make
- show a preview table before applying changes
- default to dry-run behavior
- write a simple operation summary if they mutate data or filenames

Recommended summary fields:

- `operation_name`
- `started_utc`
- `finished_utc`
- `input_root`
- `files_considered`
- `files_changed`
- `preview_only`

## 10. Required Metadata Capture

Every notebook that writes canonical outputs must capture the following classes of metadata.

### 10.1 Notebook identity

- notebook path
- notebook version
- timestamp
- optional notebook content hash

### 10.2 Code identity

For training notebooks and export notebooks, capture:

- Git commit
- Git branch
- dirty-worktree flag

### 10.3 Input identity

- input files or manifests used
- configuration snapshot paths
- dataset slice identifiers

### 10.4 Output inventory

- every important output path
- whether it is Git-tracked, excluded, or local-only

## 11. Markdown and Presentation Standards

Committed notebooks should read like deliberate technical documents, not scratchpads.

### 11.1 First Markdown cell must answer

- what is this notebook for
- what inputs does it use
- what artifacts does it write
- what decision should the reader make from its result

### 11.2 Section headers should be explicit

Use section headers such as:

- `Configuration`
- `Manifest loading`
- `Validation checks`
- `Training`
- `Artifact writeout`
- `Final verdict`

### 11.3 Explain unusual assumptions

If the notebook makes a project-specific assumption, say so near the config cell.

Examples:

- clip-level split only
- no test split yet
- per-clip normalization
- local-only checkpoint storage

## 12. Git and Repo Hygiene Rules

### 12.1 What should usually be tracked

- notebooks
- Python modules used by notebooks
- config snapshots
- manifests
- model cards
- run manifests
- metrics files
- split membership files when redistribution allows, otherwise public-safe
  redacted summaries
- method documentation

### 12.2 What should usually not be tracked

- raw audio
- bulk tensor exports
- local checkpoints if they are large and not necessary for repo review
- transient plots or debug dumps

### 12.3 Tracked outputs must look intentional

If a file is committed, a reviewer should be able to tell why it belongs in the repository.

## 13. Recommended Minimal Training Run Schema

```json
{
  "run_id": "20260319-1829-2d_sound_v0.2",
  "created_utc": "2026-03-19T18:29:00Z",
  "git_commit": "abcdef1",
  "dirty_worktree": false,
  "entrypoint_type": "notebook",
  "entrypoint_path": "train/2d-cnn/notebooks/2d_sound_v0.2.ipynb",
  "source_manifest_paths": [
    "training-data/pump/shards/20260322_131954_pump_npz_shards_v0.1/manifests/shard_windows.parquet"
  ],
  "preprocessing_config_paths": [],
  "model_name": "Baseline2DCNN",
  "optimizer": "Adam",
  "hyperparameters": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs_config": 64
  },
  "random_seeds": {
    "numpy": 42,
    "torch": 42
  },
  "artifact_paths": {
    "training_history": "models/<run_id>/training_history.json",
    "metrics": "models/<run_id>/metrics.json",
    "split_membership_local_only": "models/<run_id>/split_membership.csv",
    "model_card": "models/<run_id>/model_card.md"
  }
}
```

## 14. Commit-Ready Checklist

Before committing a notebook or notebook-generated metadata, confirm:

- the notebook purpose is clear in the first Markdown cell
- the config is centralized near the top
- the notebook can run top-to-bottom in a clean kernel
- inputs and outputs are explicit
- canonical artifacts are written in a predictable location
- tracked metadata uses repo-relative canonical paths where possible
- any bulky local-only outputs are excluded from Git
- the notebook ends with a clear final verdict
- the notebook filename/version still matches its behavior

## 15. Immediate Adoption Rules For This Repo

These rules should be applied first to the notebooks already acting as canonical workflow entrypoints:

- `preprocessing/03.training-export/export_2d_training_v0.4.ipynb`
- `preprocessing/04.tensor-inspection/tensor-inspection-v0.2.ipynb`
- `train/2d-cnn/notebooks/2d_sound_v0.3.ipynb`

Priority implementation order:

1. Add `run_manifest.json` and local split-membership capture to training
   notebooks, with public-safe redaction when needed.
2. Add semantic metadata fields to export manifests.
3. Convert saved metadata to include repo-relative canonical paths.
4. Standardize final verdict cells and artifact writeout sections.
5. Make utility notebooks preview-first by default.

## 16. Bottom Line

Notebooks are acceptable in this project.

What is not acceptable is notebook ambiguity.

A committed notebook in this repo should behave like a small, versioned technical product:

- clear purpose
- clear configuration
- clear inputs
- clear outputs
- clear verdict
- clear evidence trail

That is the standard that will make the repository look rigorous, reproducible, and professionally engineered without forcing it into heavyweight infrastructure.
