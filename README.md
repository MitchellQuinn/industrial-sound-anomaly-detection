# Raccoon Ball / Industrial Sound Anomaly Detection

This repository contains original code, notebooks, documentation, and
lightweight experiment summaries for an industrial sound anomaly-detection
project. It is prepared as a public, employer-facing portfolio artifact rather
than a redistributable training package.

## Start Here
- [Employer-facing summary](documents/employment-technical-writeup.md)
- [Detailed technical writeup](documents/technical-writeup-industrial-sound-anomaly-detection.md)
- [Notebook/process standards](documents/notebook-generation-standards.md)
- [Best tracked in-distribution run](models/20260322-1335-2d_sound_v0.7/model_card.md)
- [Harder unseen-unit holdout run](models/20260322-1438-2d_sound_v0.8/model_card.md)

## Workflow Overview
1. Local MIMII audio is transformed into normalized spectral windows plus
   auxiliary representation features by `preprocessing/export_2d_training.py`.
2. Training notebooks under `train/2d-cnn/notebooks/` consume those exports or
   shard manifests and write per-run metadata to `models/<run_id>/`.
3. Public review is centered on `model_card.md`, `run_manifest.json`,
   `training_history.json`, `metrics.json`, and the supporting documents under
   `documents/`.

## Includes
- preprocessing scripts and notebooks
- training notebooks
- model cards, run manifests, training histories, and summary metrics
- method and process documentation

## Does Not Include
- MIMII audio or any other third-party audio assets
- derived tensor corpora, shard files, or manifest snapshots generated from
  third-party dataset audio
- split-membership exports or clip-level prediction exports
- model weight binaries or checkpoints
- notebook outputs that embed dataset-derived previews

## Dataset Attribution

This project was developed using the **MIMII Dataset**
(**Malfunctioning Industrial Machine Investigation and Inspection**) for local
training and evaluation work.

The MIMII dataset is **not redistributed** in this repository.

- Official Zenodo record: https://zenodo.org/records/3384388
- DOI: https://doi.org/10.5281/zenodo.3384388
- License at source: **CC BY-SA 4.0**
- Additional attribution details: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

If you obtain the MIMII dataset separately, you are responsible for complying
with its license terms, attribution requirements, and any downstream
obligations that apply to your own use or redistribution.

## Recommended Upstream Citations

- Purohit, H., Tanabe, R., Ichige, K., Endo, T., Nikaido, Y., Suefusa, K., and
  Kawaguchi, Y. (2019). **MIMII Dataset: Sound Dataset for Malfunctioning
  Industrial Machine Investigation and Inspection (public 1.0)** [Data set].
  Zenodo. https://doi.org/10.5281/zenodo.3384388
- Purohit, H., Tanabe, R., Ichige, K., Endo, T., Nikaido, Y., Suefusa, K., and
  Kawaguchi, Y. **MIMII Dataset: Sound Dataset for Malfunctioning Industrial
  Machine Investigation and Inspection.** arXiv:1909.09347, 2019.
- Purohit, H., Tanabe, R., Ichige, K., Endo, T., Nikaido, Y., Suefusa, K., and
  Kawaguchi, Y. **MIMII Dataset: Sound Dataset for Malfunctioning Industrial
  Machine Investigation and Inspection.** In Proceedings of the 4th Workshop on
  Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

## Repository Rights

Unless otherwise noted, the original code, notebooks, and documentation in this
repository are **not released under an open-source license**.

No permission is granted to copy, modify, redistribute, sublicense, or create
derivative works from the repository's original materials except as allowed by
applicable law or by prior written permission from the copyright holder.

See [COPYRIGHT.md](COPYRIGHT.md) for the repository rights notice and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
