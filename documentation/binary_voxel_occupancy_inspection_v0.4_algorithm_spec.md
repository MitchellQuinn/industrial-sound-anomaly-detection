# Binary Voxel Occupancy Inspection v0.4: Data Processing Algorithm Specification

## 1. Purpose
This specification defines the data processing algorithm implemented in `binary_voxel_occupancy_inspection_v0.4.ipynb` for paired audio comparison. The pipeline converts two audio clips into quantized spectral representations, builds 3D binary voxel volumes, extracts surface voxels, and computes aligned surface differences.

## 2. Inputs and Primary Configuration
- Inputs:
  - Two audio files (`a`, `b`) selected through widget chooser or fallback paths.
- Core processing parameters:
  - `TARGET_SR`, `MONO`, `AUDIO_OFFSET_SECONDS`, `MAX_AUDIO_SECONDS`
  - `N_FFT`, `HOP_LENGTH`, `WIN_LENGTH`, `WINDOW`
  - `NUM_BANDS`, `FMIN`, `FMAX`
  - `LOG_POWER_REFERENCE`, `EPSILON`
  - `NORMALIZE_BAND_ENERGY`, `NORMALIZATION_MODE`
  - `ENERGY_BINS`, `LOW_ENERGY_FLOOR`, `MIN_ACTIVE_BANDS_PER_FRAME`
  - `CHUNK_START_FRAME`, `FRAMES_PER_CHUNK`
  - `VOXEL_MODE in {"one_hot", "filled_column"}`

## 3. End-to-End Processing Flow
For each file `x in {a, b}`:
1. Load waveform with `librosa.load` (resampled to `TARGET_SR`, mono if configured).
2. Compute STFT magnitude:
   - `stft_mag = |STFT(y)|`, shape `(freq_bins, num_frames)`.
3. Build triangular log-spaced band filters:
   - FFT bins from `rfftfreq`.
   - Band centers are geometric (`geomspace(FMIN, FMAX, NUM_BANDS)`).
   - Adjacent geometric means define band boundaries.
   - Each filter is triangle-shaped and normalized by filter sum.
4. Project STFT power into band energy:
   - `power = stft_mag^2`
   - `band_energy = band_filters @ power`
   - Clamp floor with `max(eps)`.
5. Log-compress energy:
   - `log_band_energy = log((band_energy / LOG_POWER_REFERENCE) + EPSILON)`.
6. Normalize (if enabled):
   - `per_clip_minmax`: `(x - min(x)) / max(max-min, eps)`, clipped to `[0,1]`.
7. Quantize to integer bins:
   - `active_mask = normalized > LOW_ENERGY_FLOOR`
   - `energy_bin = floor(normalized * ENERGY_BINS)`, clipped to `[0, ENERGY_BINS-1]`
   - Inactive cells forced to bin `0`.
8. Optional frame gating:
   - If `MIN_ACTIVE_BANDS_PER_FRAME > 0`, frames with too few active bands are fully deactivated.
9. Extract fixed temporal chunk:
   - Slice both arrays to `[CHUNK_START_FRAME, CHUNK_START_FRAME + FRAMES_PER_CHUNK)`.
10. Build binary voxel volume `(num_bands, ENERGY_BINS, FRAMES_PER_CHUNK)`:
   - Skip inactive cells.
   - `one_hot`: set only `[band, energy_bin, frame] = True`.
   - `filled_column`: set `[band, 0:energy_bin+1, frame] = True`.
11. Convert occupied voxels to dataframe (all `True` coordinates).
12. Extract surface voxels:
   - For each `(band, frame)`, keep highest occupied `energy_bin` (if any).
13. Export surface dataframe to CSV.

After both files are processed:
14. Align surface dataframes on `(band_idx, chunk_frame)` (outer merge).
15. Compute difference metrics:
   - `delta_energy_bin = energy_bin_b - energy_bin_a`
   - `abs_delta_energy_bin = |delta_energy_bin|`
   - `changed = missing_in_either OR delta != 0`
16. Optionally export aligned difference dataframe to CSV.

## 4. Data Model and Shapes
- `band_energy`: `(NUM_BANDS, num_frames)`
- `normalized_band_energy`: `(NUM_BANDS, num_frames)`
- `energy_bin_indices`: `(NUM_BANDS, num_frames)`, integer
- `active_mask`: `(NUM_BANDS, num_frames)`, bool
- `chunk_energy_bins`: `(NUM_BANDS, FRAMES_PER_CHUNK)`
- `chunk_active_mask`: `(NUM_BANDS, FRAMES_PER_CHUNK)`
- `voxel_volume`: `(NUM_BANDS, ENERGY_BINS, FRAMES_PER_CHUNK)`, bool
- `surface_voxels_df` columns:
  - `band_idx`, `energy_bin`, `chunk_frame`, `is_surface`, `voxel_mode`
  - plus metadata: `audio_path`, `file_label`, `compare_name`
- `surface_diff_df` columns:
  - `band_idx`, `chunk_frame`, `energy_bin_a`, `energy_bin_b`
  - `present_a`, `present_b`, `delta_energy_bin`, `abs_delta_energy_bin`, `changed`

## 5. Validation and Error Conditions
- File resolution and extension checks (`SUPPORTED_EXTENSIONS`).
- Chunk bounds checks:
  - non-negative start, positive length, end within available frames.
- Shape checks:
  - `energy_bin_indices.shape == active_mask.shape`
  - `chunk_energy_bins.shape == chunk_active_mask.shape`
- Parameter checks:
  - `energy_bins > 0`
  - `voxel_mode` in supported set
  - valid `fmin/fmax` after Nyquist adjustment
- Difference alignment checks:
  - required columns present
  - no duplicate `(band_idx, chunk_frame)` rows in each surface dataframe.

## 6. Output Artifacts
- Per-file surface CSV:
  - Filename pattern: `[timestamp]_[a|b]_[sanitized_audio_stem].csv`
  - Includes plotting metadata and chunk metadata.
- Optional paired difference CSV:
  - Default pattern: `[timestamp]_diff_[a_stem]_vs_[b_stem].csv`
- In-memory summary metrics:
  - Occupied voxel counts/density
  - Surface row counts
  - Aligned comparison row count
  - Changed rows, mean/max absolute delta.

## 7. Behavioral Notes
- `filled_column` encodes magnitude as cumulative occupancy and guarantees one surface point per active `(band, frame)`.
- `one_hot` encodes only the exact bin; surface extraction still returns that bin.
- The comparison is local to the selected chunk (`CHUNK_START_FRAME`, `FRAMES_PER_CHUNK`), not necessarily the full clip.
- Normalization is per clip (`per_clip_minmax`), so bin scales are clip-relative rather than globally calibrated.
