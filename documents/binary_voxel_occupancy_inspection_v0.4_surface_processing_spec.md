# Binary Voxel Occupancy Inspection v0.4: Surface Processing Specification

## 1. Purpose
This specification describes only the data processing path that transforms an input audio file into its surface voxel representation.

## 2. Input and Configuration
- Input:
  - One audio file path (the same pipeline can be run independently per file).
- Key parameters:
  - Audio loading: `TARGET_SR`, `MONO`, `AUDIO_OFFSET_SECONDS`, `MAX_AUDIO_SECONDS`
  - Spectral frontend: `N_FFT`, `HOP_LENGTH`, `WIN_LENGTH`, `WINDOW`
  - Log-band projection: `NUM_BANDS`, `FMIN`, `FMAX`
  - Scaling: `LOG_POWER_REFERENCE`, `EPSILON`
  - Quantization: `NORMALIZE_BAND_ENERGY`, `NORMALIZATION_MODE`, `ENERGY_BINS`, `LOW_ENERGY_FLOOR`, `MIN_ACTIVE_BANDS_PER_FRAME`
  - Chunking: `CHUNK_START_FRAME`, `FRAMES_PER_CHUNK`
  - Voxelization: `VOXEL_MODE in {"one_hot", "filled_column"}`

## 3. Processing Pipeline (Audio -> Surface)
1. Load audio with `librosa.load` using configured sample rate and mono/stereo behavior.
2. Compute STFT magnitude:
   - `stft_mag = |STFT(y)|` with shape `(freq_bins, num_frames)`.
3. Build log-spaced triangular filterbank:
   - FFT frequencies from `rfftfreq`.
   - Log centers from `geomspace(FMIN, FMAX, NUM_BANDS)`.
   - Triangles formed from geometric-mean boundaries and normalized by each filter sum.
4. Project to band energy:
   - `power = stft_mag^2`
   - `band_energy = band_filters @ power`
   - Apply lower bound: `band_energy = max(band_energy, EPSILON)`.
5. Log-compress:
   - `log_band_energy = log((band_energy / LOG_POWER_REFERENCE) + EPSILON)`.
6. Normalize (optional):
   - If enabled and mode is `per_clip_minmax`, map to `[0,1]` using clip-level min/max.
7. Quantize and mark activity:
   - `active_mask = normalized > LOW_ENERGY_FLOOR`
   - `energy_bin_indices = floor(normalized * ENERGY_BINS)`, clipped to `[0, ENERGY_BINS - 1]`
   - Inactive positions are forced to bin `0`.
8. Optional frame-level gating:
   - If `MIN_ACTIVE_BANDS_PER_FRAME > 0`, frames below threshold are fully deactivated.
9. Extract chunk:
   - `chunk_energy_bins = energy_bin_indices[:, start:end]`
   - `chunk_active_mask = active_mask[:, start:end]`
   - where `start = CHUNK_START_FRAME`, `end = start + FRAMES_PER_CHUNK`.
10. Build binary voxel volume:
   - Output shape: `(num_bands, ENERGY_BINS, FRAMES_PER_CHUNK)`, dtype `bool`.
   - For each active `(band_idx, chunk_frame)` with bin `e`:
     - `one_hot`: set `voxel_volume[band_idx, e, chunk_frame] = True`
     - `filled_column`: set `voxel_volume[band_idx, 0:e+1, chunk_frame] = True`
11. Extract surface representation:
   - For each `(band_idx, chunk_frame)` column, find occupied bins.
   - If any are occupied, keep only `max(occupied_bins)` as the surface bin.
   - Emit one row per occupied column in `surface_voxels_df`:
     - `band_idx`, `energy_bin`, `chunk_frame`, `is_surface`, `voxel_mode`
12. Export surface table to CSV:
   - Filename pattern: `[timestamp]_[file_label]_[sanitized_audio_stem].csv`
   - Includes additional metadata used by downstream plotting.

## 4. Intermediate Data Structures and Shapes
- `band_energy`: `(NUM_BANDS, num_frames)`
- `log_band_energy`: `(NUM_BANDS, num_frames)`
- `normalized_band_energy`: `(NUM_BANDS, num_frames)`
- `energy_bin_indices`: `(NUM_BANDS, num_frames)` integer
- `active_mask`: `(NUM_BANDS, num_frames)` bool
- `chunk_energy_bins`: `(NUM_BANDS, FRAMES_PER_CHUNK)` integer
- `chunk_active_mask`: `(NUM_BANDS, FRAMES_PER_CHUNK)` bool
- `voxel_volume`: `(NUM_BANDS, ENERGY_BINS, FRAMES_PER_CHUNK)` bool
- `surface_voxels_df`: one row per occupied `(band_idx, chunk_frame)` with the top energy bin

## 5. Validation Rules
- Audio file must exist and have a supported extension.
- `fmin < fmax` after Nyquist constraints.
- `energy_bins > 0`.
- `VOXEL_MODE` must be `one_hot` or `filled_column`.
- `energy_bin_indices.shape == active_mask.shape`.
- Requested chunk range must lie within available frame range.

## 6. Behavioral Notes
- `filled_column` creates cumulative occupancy along the energy axis; this can make the surface more stable.
- `one_hot` keeps only the exact quantized bin; surface equals that bin.
- Surface extraction removes interior voxels and retains only the highest occupied point in each frequency-time column.
- The produced surface reflects only the selected chunk, not the full clip unless the chunk covers all frames.
