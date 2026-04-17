# %% cell 1
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_ROOT = REPO_ROOT / 'training-data'

# Set this before each run (examples: 'pump', 'fan', 'slider', 'valve')
MACHINE_TYPE = 'pump'

# Number of windows per shard file target.
SHARD_TARGET_WINDOWS = 32768

# Write location for packed shards/manifests.
OUTPUT_ROOT = TRAINING_DATA_ROOT / MACHINE_TYPE / 'shards'

# Safety controls.
OVERWRITE_EXISTING = False
DRY_RUN = False

print(f'REPO_ROOT={REPO_ROOT}')
print(f'TRAINING_DATA_ROOT={TRAINING_DATA_ROOT}')
print(f'MACHINE_TYPE={MACHINE_TYPE}')
print(f'SHARD_TARGET_WINDOWS={SHARD_TARGET_WINDOWS}')
print(f'OUTPUT_ROOT={OUTPUT_ROOT}')
print(f'OVERWRITE_EXISTING={OVERWRITE_EXISTING}')
print(f'DRY_RUN={DRY_RUN}')

# Run manifest/data sanity checks before writing shards.
RUN_SANITY_CHECK = True
print(f'RUN_SANITY_CHECK={RUN_SANITY_CHECK}')


# %% cell 2
available_machine_types = sorted([p.name for p in TRAINING_DATA_ROOT.iterdir() if p.is_dir()])
print('Available machine types under training-data:', available_machine_types)

machine_root = TRAINING_DATA_ROOT / MACHINE_TYPE
if not machine_root.exists():
    raise FileNotFoundError(
        f'MACHINE_TYPE={MACHINE_TYPE!r} not found under training-data. Available: {available_machine_types}'
    )

windows_manifest_paths = sorted(machine_root.glob('*/manifests/*_windows.parquet'))
files_manifest_paths = sorted(machine_root.glob('*/manifests/*_files.parquet'))

print(f'Found {len(windows_manifest_paths)} window manifests and {len(files_manifest_paths)} file manifests for machine={MACHINE_TYPE}.')
for p in windows_manifest_paths:
    print(' -', p)

if not windows_manifest_paths:
    raise RuntimeError(
        f'No *_windows.parquet manifests found for MACHINE_TYPE={MACHINE_TYPE!r}. '
        'Run preprocessing export first for this machine type.'
    )


# %% cell 3
def _read_parquet_many(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        if len(df) == 0:
            continue
        df = df.copy()
        df['_manifest_path'] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _resolve_tensor_npz_path(path_value) -> Path:
    raw = Path(str(path_value)).expanduser()
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
        try:
            rel_from_training_data = raw.relative_to(TRAINING_DATA_ROOT)
            # Handles legacy absolute paths missing the MACHINE_TYPE segment, e.g.
            # training-data/+0db-pump-normal/... -> training-data/pump/+0db-pump-normal/...
            candidates.append(TRAINING_DATA_ROOT / MACHINE_TYPE / rel_from_training_data)
        except ValueError:
            pass
    else:
        candidates.extend(
            [
                REPO_ROOT / raw,
                TRAINING_DATA_ROOT / raw,
                TRAINING_DATA_ROOT / MACHINE_TYPE / raw,
                machine_root / raw,
                machine_root / 'tensors' / raw,
            ]
        )

    deduped: list[Path] = []
    seen = set()
    for cand in candidates:
        resolved = cand.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(resolved)

    for cand in deduped:
        if cand.exists():
            return cand

    return deduped[0] if deduped else raw.resolve()


windows_df = _read_parquet_many(windows_manifest_paths)
files_df = _read_parquet_many(files_manifest_paths)

if windows_df.empty:
    raise RuntimeError('Combined windows manifest is empty.')

required_window_cols = {'tensor_npz_path', 'tensor_index'}
missing_window_cols = sorted(required_window_cols - set(windows_df.columns))
if missing_window_cols:
    raise ValueError(f'windows manifest missing required columns: {missing_window_cols}')

windows_df['tensor_npz_path_raw'] = windows_df['tensor_npz_path'].astype(str)
windows_df['tensor_npz_path'] = windows_df['tensor_npz_path'].map(lambda x: str(_resolve_tensor_npz_path(x)))
windows_df['tensor_index'] = windows_df['tensor_index'].astype('int64')

windows_df = windows_df.sort_values(['tensor_npz_path', 'tensor_index']).reset_index(drop=True)

if not files_df.empty and 'tensor_npz_path' in files_df.columns:
    files_df['tensor_npz_path_raw'] = files_df['tensor_npz_path'].astype(str)
    files_df['tensor_npz_path'] = files_df['tensor_npz_path'].map(lambda x: str(_resolve_tensor_npz_path(x)))

remapped_count = int((windows_df['tensor_npz_path_raw'] != windows_df['tensor_npz_path']).sum())
missing_after_remap = int((~windows_df['tensor_npz_path'].map(lambda p: Path(p).exists())).sum())

print(f'window rows: {len(windows_df):,}')
print(f'unique npz files referenced: {windows_df["tensor_npz_path"].nunique():,}')
print(f'npz paths remapped: {remapped_count:,}')
print(f'missing npz after remap: {missing_after_remap:,}')
display(windows_df.head(5))


# %% cell 4
if RUN_SANITY_CHECK:
    duplicate_rows = int(windows_df.duplicated(subset=['tensor_npz_path', 'tensor_index']).sum())
    if duplicate_rows > 0:
        raise RuntimeError(
            f'Found duplicate window keys in combined manifest: {duplicate_rows} duplicate (tensor_npz_path, tensor_index) rows.'
        )

    negative_idx = int((windows_df['tensor_index'] < 0).sum())
    if negative_idx > 0:
        raise RuntimeError(f'Found {negative_idx} rows with negative tensor_index values.')

    missing_paths = []
    missing_keys = []
    shape_errors = []
    bounds_errors = []

    for npz_path, group in windows_df.groupby('tensor_npz_path', sort=False):
        p = Path(str(npz_path))
        if not p.exists():
            missing_paths.append(str(p))
            continue

        idx = group['tensor_index'].to_numpy(dtype=np.int64, copy=True)
        if idx.size == 0:
            continue

        with np.load(p, allow_pickle=False) as npz:
            if 'normalized_window' not in npz.files or 'active_mask' not in npz.files:
                missing_keys.append(str(p))
                continue

            normalized_all = npz['normalized_window']
            active_all = npz['active_mask']

            if normalized_all.ndim != 3 or active_all.ndim != 3 or normalized_all.shape[1:] != (96, 64) or active_all.shape[1:] != (96, 64):
                shape_errors.append((str(p), tuple(normalized_all.shape), tuple(active_all.shape)))
                continue

            min_idx = int(idx.min())
            max_idx = int(idx.max())
            n = int(normalized_all.shape[0])
            if min_idx < 0 or max_idx >= n:
                bounds_errors.append((str(p), min_idx, max_idx, n))

    if missing_paths:
        preview = '; '.join(missing_paths[:5])
        raise RuntimeError(f'Missing NPZ files referenced by manifest. First examples: {preview}')
    if missing_keys:
        preview = '; '.join(missing_keys[:5])
        raise RuntimeError(
            f'NPZ files missing required arrays normalized_window/active_mask. First examples: {preview}'
        )
    if shape_errors:
        preview = '; '.join([f'{p} norm={n} mask={m}' for p, n, m in shape_errors[:5]])
        raise RuntimeError(f'Unexpected NPZ tensor shapes detected. First examples: {preview}')
    if bounds_errors:
        preview = '; '.join([f'{p} min={mn} max={mx} N={n}' for p, mn, mx, n in bounds_errors[:5]])
        raise RuntimeError(f'Manifest tensor_index out-of-bounds errors detected. First examples: {preview}')

    print('Sanity checks passed.')
    print(f' - unique npz files checked: {windows_df["tensor_npz_path"].nunique():,}')
    print(f' - total window rows checked: {len(windows_df):,}')
else:
    print('Sanity checks skipped (RUN_SANITY_CHECK=False).')


# %% cell 5
run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{MACHINE_TYPE}_npz_shards_v0.1"
run_root = OUTPUT_ROOT / run_id
shards_dir = run_root / 'shards'
manifests_dir = run_root / 'manifests'

if run_root.exists() and not OVERWRITE_EXISTING:
    raise FileExistsError(f'Output run directory already exists: {run_root}')

if not DRY_RUN:
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

index_rows = []
shard_rows = []

pending_norm = []
pending_mask = []
pending_meta = []
pending_count = 0
shard_id = 0

npz_groups = windows_df.groupby('tensor_npz_path', sort=False)


def _flush_shard() -> None:
    global pending_norm, pending_mask, pending_meta, pending_count, shard_id

    if pending_count == 0:
        return

    shard_norm = np.concatenate(pending_norm, axis=0).astype(np.float32, copy=False)
    shard_mask = np.concatenate(pending_mask, axis=0).astype(bool, copy=False)
    shard_meta = pd.concat(pending_meta, ignore_index=True)

    shard_name = f'shard_{shard_id:05d}.npz'
    shard_path = shards_dir / shard_name

    if not DRY_RUN:
        # Uncompressed zip members (np.savez) avoid per-array decompression overhead from savez_compressed.
        np.savez(
            shard_path,
            normalized_window=shard_norm,
            active_mask=shard_mask,
        )
        shard_size_bytes = int(shard_path.stat().st_size)
    else:
        shard_size_bytes = int(shard_norm.nbytes + shard_mask.nbytes)

    shard_meta = shard_meta.reset_index(drop=True)
    if 'original_tensor_npz_path' not in shard_meta.columns:
        shard_meta['original_tensor_npz_path'] = shard_meta['tensor_npz_path']
    if 'original_tensor_index' not in shard_meta.columns:
        shard_meta['original_tensor_index'] = shard_meta['tensor_index'].astype(np.int64)

    shard_tensor_index = np.arange(len(shard_meta), dtype=np.int64)
    shard_meta['shard_id'] = int(shard_id)
    shard_meta['shard_path'] = str(shard_path)
    shard_meta['shard_tensor_index'] = shard_tensor_index

    # Make output directly consumable by training notebook loaders.
    shard_meta['tensor_npz_path'] = str(shard_path)
    shard_meta['tensor_index'] = shard_tensor_index

    index_rows.extend(shard_meta.to_dict(orient='records'))

    shard_rows.append(
        {
            'shard_id': int(shard_id),
            'shard_path': str(shard_path),
            'num_windows': int(len(shard_meta)),
            'size_bytes': shard_size_bytes,
            'size_mb': float(shard_size_bytes / (1024 ** 2)),
        }
    )

    pending_norm = []
    pending_mask = []
    pending_meta = []
    pending_count = 0
    shard_id += 1


for npz_path, group in npz_groups:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f'NPZ path from manifest does not exist: {p}')

    tensor_idx = group['tensor_index'].to_numpy(dtype=np.int64, copy=True)

    with np.load(p, allow_pickle=False) as npz:
        if 'normalized_window' not in npz.files:
            raise KeyError(f"Missing key 'normalized_window' in {p}")
        if 'active_mask' not in npz.files:
            raise KeyError(f"Missing key 'active_mask' in {p}")

        normalized_all = npz['normalized_window']
        active_all = npz['active_mask']

        if normalized_all.ndim != 3 or normalized_all.shape[1:] != (96, 64):
            raise ValueError(f'Unexpected normalized_window shape in {p}: {normalized_all.shape}')
        if active_all.ndim != 3 or active_all.shape[1:] != (96, 64):
            raise ValueError(f'Unexpected active_mask shape in {p}: {active_all.shape}')

        if len(tensor_idx):
            min_idx = int(tensor_idx.min())
            max_idx = int(tensor_idx.max())
            if min_idx < 0 or max_idx >= int(normalized_all.shape[0]):
                raise IndexError(
                    f'tensor_index out of bounds in {p}: min={min_idx}, max={max_idx}, N={int(normalized_all.shape[0])}'
                )

        selected_norm = normalized_all[tensor_idx].astype(np.float32, copy=False)
        selected_mask = active_all[tensor_idx].astype(bool, copy=False)

    meta = group.copy()
    meta['machine_type'] = MACHINE_TYPE

    if pending_count > 0 and pending_count + len(meta) > int(SHARD_TARGET_WINDOWS):
        _flush_shard()

    pending_norm.append(selected_norm)
    pending_mask.append(selected_mask)
    pending_meta.append(meta)
    pending_count += len(meta)

_flush_shard()

index_df = pd.DataFrame(index_rows)
shards_df = pd.DataFrame(shard_rows)

if index_df.empty or shards_df.empty:
    raise RuntimeError('No shards were produced; check manifests and MACHINE_TYPE selection.')

manifest_windows_path = manifests_dir / 'shard_windows.parquet'
manifest_shards_path = manifests_dir / 'shard_files.parquet'
run_config_path = manifests_dir / 'run_config.json'
run_summary_path = manifests_dir / 'run_summary.json'

run_config = {
    'run_id': run_id,
    'machine_type': MACHINE_TYPE,
    'shard_target_windows': int(SHARD_TARGET_WINDOWS),
    'dry_run': bool(DRY_RUN),
    'overwrite_existing': bool(OVERWRITE_EXISTING),
    'run_sanity_check': bool(RUN_SANITY_CHECK),
    'window_manifest_paths': [str(p) for p in windows_manifest_paths],
    'file_manifest_paths': [str(p) for p in files_manifest_paths],
}

run_summary = {
    'run_id': run_id,
    'machine_type': MACHINE_TYPE,
    'num_source_window_rows': int(len(windows_df)),
    'num_source_npz_files': int(windows_df['tensor_npz_path'].nunique()),
    'num_shards_written': int(len(shards_df)),
    'num_shard_window_rows': int(len(index_df)),
    'total_shard_size_mb': float(shards_df['size_mb'].sum()),
    'manifest_windows_path': str(manifest_windows_path),
    'manifest_shards_path': str(manifest_shards_path),
}

if not DRY_RUN:
    index_df.to_parquet(manifest_windows_path, index=False)
    shards_df.to_parquet(manifest_shards_path, index=False)
    run_config_path.write_text(json.dumps(run_config, indent=2))
    run_summary_path.write_text(json.dumps(run_summary, indent=2))

print('Run summary:')
print(json.dumps(run_summary, indent=2))
print('Top shards:')
display(shards_df.head(10))
print('Window-level shard index sample:')
display(index_df.head(10))
