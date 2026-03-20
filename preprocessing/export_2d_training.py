from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd

try:
    import torch
    import torchaudio
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    torch = None
    torchaudio = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional fallback for torchaudio torchcodec issues.
    sf = None


@dataclass(slots=True)
class ExportConfig:
    """Configuration for exporting 2D training tensors from a directory tree."""

    input_dir: Path
    output_root: Path
    backend: str = "torch-cuda"  # "torch-cuda" (default) or "cpu"

    # File selection / restart behavior
    source_files: list[Path] | None = None
    max_files: int | None = None
    skip_existing: bool = True

    # Audio loading
    target_sr: int = 16000
    mono: bool = True
    audio_offset_seconds: float = 0.0
    max_audio_seconds: float | None = None

    # Spectral frontend
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    window: str = "hann"

    # Log-band projection
    num_bands: int = 96
    fmin: float = 50.0
    fmax: float = 8000.0

    # Compression / normalization
    log_power_reference: float = 1.0
    epsilon: float = 1e-8
    normalize_band_energy: bool = True
    normalization_mode: str = "per_clip_minmax"

    # Derived representations
    energy_bins: int = 24
    low_energy_floor: float = 0.0
    min_active_bands_per_frame: int = 0

    # Window extraction (2D window shape: num_bands x window_frames)
    window_frames: int = 64
    stride_frames: int = 32

    # Manifest naming
    manifest_prefix: str | None = None

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).expanduser().resolve()
        self.output_root = Path(self.output_root).expanduser().resolve()
        if self.source_files is not None:
            self.source_files = [Path(path).expanduser().resolve() for path in self.source_files]

        if self.backend not in {"torch-cuda", "cpu"}:
            raise ValueError("backend must be 'torch-cuda' or 'cpu'.")
        if self.window != "hann":
            raise ValueError("Only WINDOW='hann' is currently supported.")
        if self.target_sr <= 0:
            raise ValueError("target_sr must be positive.")
        if self.n_fft <= 0 or self.hop_length <= 0 or self.win_length <= 0:
            raise ValueError("n_fft, hop_length, and win_length must be positive.")
        if self.num_bands <= 0:
            raise ValueError("num_bands must be positive.")
        if self.fmin <= 0 or self.fmax <= 0:
            raise ValueError("fmin and fmax must be positive.")
        if self.energy_bins <= 0 or self.energy_bins > 256:
            raise ValueError("energy_bins must be in [1, 256] so height_bins can be uint8.")
        if self.window_frames <= 0 or self.stride_frames <= 0:
            raise ValueError("window_frames and stride_frames must be positive.")
        if self.audio_offset_seconds < 0:
            raise ValueError("audio_offset_seconds must be >= 0.")
        if self.max_audio_seconds is not None and self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be > 0 when provided.")
        if self.max_files is not None and self.max_files <= 0:
            raise ValueError("max_files must be > 0 when provided.")


def _require_torch_runtime() -> None:
    if torch is None or torchaudio is None:
        raise RuntimeError(
            "PyTorch and TorchAudio are required for export. "
            "Install `torch` and `torchaudio` in this environment."
        )


def _resolve_device(config: ExportConfig):
    _require_torch_runtime()
    if config.backend == "torch-cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("backend='torch-cuda' was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_tensor_output_path(config: ExportConfig, source_file: Path) -> tuple[Path, Path]:
    source_resolved = source_file.expanduser().resolve()
    try:
        relative_source = source_resolved.relative_to(config.input_dir)
    except ValueError as exc:
        raise ValueError(
            f"Source file must be under input_dir.\nsource_file={source_resolved}\ninput_dir={config.input_dir}"
        ) from exc

    tensor_path = (config.output_root / "tensors" / relative_source).with_suffix(".npz")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    return tensor_path, relative_source


def _build_log_band_filters(
    sr: int,
    n_fft: int,
    num_bands: int,
    fmin: float,
    fmax: float,
    device,
    dtype,
):
    fft_freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sr), device=device)
    valid_fmax = min(float(fmax), float(sr) / 2.0)
    valid_fmin = max(float(fmin), float(fft_freqs[1].item()) if fft_freqs.numel() > 1 else float(fmin))
    if valid_fmin >= valid_fmax:
        raise ValueError("fmin must be smaller than fmax after accounting for sample rate.")

    if hasattr(torch, "geomspace"):
        centers = torch.geomspace(
            torch.tensor(valid_fmin, dtype=dtype, device=device),
            torch.tensor(valid_fmax, dtype=dtype, device=device),
            steps=num_bands,
        )
    else:
        log_start = torch.log10(torch.tensor(valid_fmin, dtype=dtype, device=device))
        log_end = torch.log10(torch.tensor(valid_fmax, dtype=dtype, device=device))
        centers = torch.logspace(log_start, log_end, steps=num_bands, base=10.0, dtype=dtype, device=device)
    boundaries = torch.empty(num_bands + 1, dtype=dtype, device=device)
    boundaries[0] = valid_fmin
    boundaries[-1] = valid_fmax
    if num_bands > 1:
        boundaries[1:-1] = torch.sqrt(centers[:-1] * centers[1:])

    filters = torch.zeros((num_bands, fft_freqs.numel()), dtype=dtype, device=device)
    for band_idx in range(num_bands):
        left = boundaries[band_idx]
        center = centers[band_idx]
        right = boundaries[band_idx + 1]

        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        if center > left:
            filters[band_idx, left_mask] = (fft_freqs[left_mask] - left) / (center - left)
        if right > center:
            filters[band_idx, right_mask] = (right - fft_freqs[right_mask]) / (right - center)

        band_sum = filters[band_idx].sum()
        if band_sum > 0:
            filters[band_idx] = filters[band_idx] / band_sum
    return filters


def _per_clip_minmax(x, eps: float):
    x_min = torch.amin(x)
    x_max = torch.amax(x)
    denom = torch.clamp(x_max - x_min, min=eps)
    return torch.clamp((x - x_min) / denom, 0.0, 1.0)


def _extract_windows_2d(matrix_2d, window_frames: int, stride_frames: int):
    if matrix_2d.ndim != 2:
        raise ValueError(f"Expected 2D tensor of shape (bands, frames), got shape {tuple(matrix_2d.shape)}")
    num_frames = int(matrix_2d.shape[1])
    if num_frames < window_frames:
        empty_shape = (0, int(matrix_2d.shape[0]), window_frames)
        return matrix_2d.new_empty(empty_shape), matrix_2d.new_empty((0,), dtype=torch.long)
    windows = matrix_2d.unfold(dimension=1, size=window_frames, step=stride_frames).permute(1, 0, 2).contiguous()
    starts = torch.arange(0, num_frames - window_frames + 1, stride_frames, device=matrix_2d.device, dtype=torch.long)
    return windows, starts


def _read_existing_npz_metadata(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        normalized_shape = tuple(data["normalized_window"].shape) if "normalized_window" in data else (0, 0, 0)
        frame_starts = data["frame_starts"] if "frame_starts" in data else np.array([], dtype=np.int64)
    return {
        "num_windows": int(normalized_shape[0]) if len(normalized_shape) >= 1 else 0,
        "num_bands": int(normalized_shape[1]) if len(normalized_shape) >= 2 else 0,
        "window_frames": int(normalized_shape[2]) if len(normalized_shape) >= 3 else 0,
        "frame_starts": frame_starts.astype(np.int64, copy=False),
    }


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001 - we provide a focused error message.
        raise RuntimeError(
            f"Failed to write parquet manifest to {path}. "
            "Ensure a parquet engine is installed (for example `pyarrow`)."
        ) from exc


def _build_window_manifest_rows(
    source_file: Path,
    relative_source: Path,
    tensor_path: Path,
    frame_starts: np.ndarray,
    window_frames: int,
    hop_length: int,
    target_sr: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_to_sec = float(hop_length) / float(target_sr)
    for window_index, frame_start in enumerate(frame_starts.tolist()):
        frame_start_int = int(frame_start)
        frame_end_exclusive = frame_start_int + int(window_frames)
        start_sec = float(frame_start_int) * frame_to_sec
        end_sec = float(frame_end_exclusive) * frame_to_sec
        rows.append(
            {
                "source_file": str(source_file),
                "relative_source_path": relative_source.as_posix(),
                "tensor_npz_path": str(tensor_path),
                # Contract used by tensor-inspection notebook
                "tensor_index": int(window_index),
                "start_frame": frame_start_int,
                "end_frame_exclusive": frame_end_exclusive,
                "start_sec": start_sec,
                "end_sec": end_sec,
                # Backward-compatible aliases used by earlier manifests
                "window_index": int(window_index),
                "frame_start": frame_start_int,
                "frame_end_exclusive": frame_end_exclusive,
            }
        )
    return rows


def _load_waveform(source_file: Path):
    """
    Load waveform as channels-first float32 tensor.

    Primary path uses torchaudio.load (as required by pipeline contract).
    If torchaudio in this environment requires TorchCodec and it is unavailable,
    fall back to soundfile for WAV decoding.
    """
    try:
        waveform, sample_rate = torchaudio.load(str(source_file))
        return waveform.to(torch.float32), int(sample_rate), "torchaudio"
    except Exception as exc:  # noqa: BLE001 - we inspect and selectively fallback.
        message = str(exc)
        torchcodec_missing = "TorchCodec" in message and "load_with_torchcodec" in message
        if not torchcodec_missing:
            raise
        if sf is None:
            raise RuntimeError(
                "torchaudio.load requires TorchCodec in this environment, and soundfile fallback is unavailable. "
                "Install `torchcodec` or `soundfile`."
            ) from exc

        audio_np, sample_rate = sf.read(str(source_file), dtype="float32", always_2d=True)
        # soundfile returns shape (num_samples, num_channels); convert to (num_channels, num_samples)
        waveform = torch.from_numpy(audio_np.T.copy()).to(torch.float32)
        return waveform, int(sample_rate), "soundfile_fallback"


def export_single_file(config: ExportConfig, source_file: Path) -> dict[str, Any]:
    """
    Export one source WAV file into a NPZ tensor bundle.

    Returned dictionary includes a file-level summary and `window_manifest_rows`.
    """

    _require_torch_runtime()
    device = _resolve_device(config)

    source_file = Path(source_file).expanduser().resolve()
    tensor_path, relative_source = _make_tensor_output_path(config, source_file)
    started_utc = _utc_now_iso()

    result: dict[str, Any] = {
        "source_file": str(source_file),
        "relative_source_path": relative_source.as_posix(),
        "tensor_npz_path": str(tensor_path),
        "status": "unknown",
        "error": None,
        "source_sample_rate": None,
        "target_sample_rate": config.target_sr,
        "num_samples_target_sr": 0,
        "num_frames": 0,
        "num_windows": 0,
        "num_windows_total": 0,
        "audio_loader": None,
        "backend": config.backend,
        "device": str(device),
        "started_utc": started_utc,
        "finished_utc": None,
        "window_manifest_rows": [],
    }

    if config.skip_existing and tensor_path.exists():
        existing = _read_existing_npz_metadata(tensor_path)
        result.update(
            {
                "status": "skipped_existing",
                "num_windows": existing["num_windows"],
                "num_windows_total": existing["num_windows"],
                "finished_utc": _utc_now_iso(),
                "window_manifest_rows": _build_window_manifest_rows(
                    source_file=source_file,
                    relative_source=relative_source,
                    tensor_path=tensor_path,
                    frame_starts=existing["frame_starts"],
                    window_frames=existing["window_frames"] or config.window_frames,
                    hop_length=config.hop_length,
                    target_sr=config.target_sr,
                ),
            }
        )
        return result

    try:
        waveform, source_sr, audio_loader = _load_waveform(source_file)
        result["source_sample_rate"] = int(source_sr)
        result["audio_loader"] = audio_loader

        if config.audio_offset_seconds > 0.0 or config.max_audio_seconds is not None:
            start_sample = int(round(config.audio_offset_seconds * source_sr))
            if config.max_audio_seconds is None:
                end_sample = waveform.shape[1]
            else:
                end_sample = start_sample + int(round(config.max_audio_seconds * source_sr))
            waveform = waveform[:, start_sample:end_sample]

        if waveform.numel() == 0:
            result["status"] = "empty_after_crop"
            result["finished_utc"] = _utc_now_iso()
            return result

        if config.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(device)
        if source_sr != config.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=source_sr, new_freq=config.target_sr)

        waveform = waveform.squeeze(0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono waveform after preprocessing, got shape {tuple(waveform.shape)}")

        result["num_samples_target_sr"] = int(waveform.shape[0])

        if config.window == "hann":
            stft_window = torch.hann_window(config.win_length, device=device, dtype=waveform.dtype)
        else:
            raise ValueError(f"Unsupported window type: {config.window}")

        stft_complex = torch.stft(
            waveform,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=stft_window,
            return_complex=True,
        )
        stft_mag = torch.abs(stft_complex)
        power = stft_mag * stft_mag

        band_filters = _build_log_band_filters(
            sr=config.target_sr,
            n_fft=config.n_fft,
            num_bands=config.num_bands,
            fmin=config.fmin,
            fmax=config.fmax,
            device=device,
            dtype=power.dtype,
        )
        band_energy = torch.matmul(band_filters, power)
        band_energy = torch.clamp(band_energy, min=config.epsilon)

        safe_reference = max(float(config.log_power_reference), float(config.epsilon))
        log_band_energy = torch.log((band_energy / safe_reference) + config.epsilon)

        if config.normalize_band_energy:
            if config.normalization_mode != "per_clip_minmax":
                raise ValueError(f"Unsupported normalization_mode: {config.normalization_mode}")
            normalized = _per_clip_minmax(log_band_energy, eps=float(config.epsilon))
        else:
            normalized = log_band_energy

        clipped = torch.clamp(normalized, 0.0, 1.0)
        active_mask = clipped > float(config.low_energy_floor)
        height_bins = torch.floor(clipped * float(config.energy_bins)).to(torch.int64)
        height_bins = torch.clamp(height_bins, 0, int(config.energy_bins) - 1)
        height_bins = torch.where(active_mask, height_bins, torch.zeros_like(height_bins))

        if config.min_active_bands_per_frame > 0:
            active_counts_per_frame = torch.sum(active_mask, dim=0)
            valid_frames = active_counts_per_frame >= int(config.min_active_bands_per_frame)
            active_mask = active_mask & valid_frames.unsqueeze(0)
            height_bins = torch.where(active_mask, height_bins, torch.zeros_like(height_bins))

        result["num_frames"] = int(normalized.shape[1])

        normalized_windows, frame_starts = _extract_windows_2d(
            matrix_2d=normalized.to(torch.float32),
            window_frames=config.window_frames,
            stride_frames=config.stride_frames,
        )
        height_windows, _ = _extract_windows_2d(
            matrix_2d=height_bins.to(torch.uint8),
            window_frames=config.window_frames,
            stride_frames=config.stride_frames,
        )
        active_windows, _ = _extract_windows_2d(
            matrix_2d=active_mask.to(torch.bool),
            window_frames=config.window_frames,
            stride_frames=config.stride_frames,
        )

        num_windows = int(normalized_windows.shape[0])
        result["num_windows"] = num_windows
        result["num_windows_total"] = num_windows
        if num_windows == 0:
            result["status"] = "too_short"
            result["finished_utc"] = _utc_now_iso()
            return result

        normalized_np = normalized_windows.detach().to("cpu").numpy().astype(np.float32, copy=False)
        height_np = height_windows.detach().to("cpu").numpy().astype(np.uint8, copy=False)
        active_np = active_windows.detach().to("cpu").numpy().astype(bool, copy=False)
        frame_starts_np = frame_starts.detach().to("cpu").numpy().astype(np.int64, copy=False)

        np.savez_compressed(
            tensor_path,
            normalized_window=normalized_np,
            height_bins=height_np,
            active_mask=active_np,
            frame_starts=frame_starts_np,
            source_file=np.array(str(source_file)),
            relative_source_path=np.array(relative_source.as_posix()),
            target_sample_rate=np.array(config.target_sr, dtype=np.int32),
            num_bands=np.array(config.num_bands, dtype=np.int32),
            window_frames=np.array(config.window_frames, dtype=np.int32),
            stride_frames=np.array(config.stride_frames, dtype=np.int32),
            config_json=np.array(json.dumps(asdict(config), default=str)),
        )

        result["window_manifest_rows"] = _build_window_manifest_rows(
            source_file=source_file,
            relative_source=relative_source,
            tensor_path=tensor_path,
            frame_starts=frame_starts_np,
            window_frames=config.window_frames,
            hop_length=config.hop_length,
            target_sr=config.target_sr,
        )
        result["status"] = "exported"
        result["finished_utc"] = _utc_now_iso()
        return result

    except Exception as exc:  # noqa: BLE001 - surfaced in manifest as a controlled failure.
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["finished_utc"] = _utc_now_iso()
        return result


def _resolve_source_files(config: ExportConfig) -> list[Path]:
    if config.source_files is not None:
        files = [Path(p).expanduser().resolve() for p in config.source_files]
    else:
        files = sorted(p.resolve() for p in config.input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")

    if config.max_files is not None:
        files = files[: config.max_files]
    return files


def export_directory(config: ExportConfig) -> dict[str, Any]:
    """
    Export all selected WAV files and write run manifests.

    Returns a summary dictionary suitable for notebook display.
    """

    _require_torch_runtime()
    device = _resolve_device(config)
    if not config.input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {config.input_dir}")

    tensors_dir = config.output_root / "tensors"
    manifests_dir = config.output_root / "manifests"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    source_files = _resolve_source_files(config)
    if not source_files:
        raise RuntimeError(
            "No source WAV files were selected for export. "
            f"Checked input_dir={config.input_dir}. "
            "This exporter scans only '.wav' files unless `source_files` is explicitly provided."
        )

    run_started_utc = _utc_now_iso()
    run_id = config.manifest_prefix or datetime.now().strftime("%Y%m%d_%H%M%S")

    file_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for source_file in source_files:
        file_result = export_single_file(config, source_file)
        window_manifest_rows = file_result.pop("window_manifest_rows", [])
        file_rows.append(file_result)
        window_rows.extend(window_manifest_rows)

    files_df = pd.DataFrame(file_rows)
    windows_df = pd.DataFrame(window_rows)
    if files_df.empty:
        files_df = pd.DataFrame(
            columns=[
                "source_file",
                "relative_source_path",
                "tensor_npz_path",
                "status",
                "error",
                "source_sample_rate",
                "target_sample_rate",
                "num_samples_target_sr",
                "num_frames",
                "num_windows",
                "num_windows_total",
                "backend",
                "device",
                "started_utc",
                "finished_utc",
            ]
        )
    if windows_df.empty:
        windows_df = pd.DataFrame(
            columns=[
                "source_file",
                "relative_source_path",
                "tensor_npz_path",
                "tensor_index",
                "start_frame",
                "end_frame_exclusive",
                "start_sec",
                "end_sec",
                "window_index",
                "frame_start",
                "frame_end_exclusive",
            ]
        )

    files_manifest_path = manifests_dir / f"{run_id}_files.parquet"
    windows_manifest_path = manifests_dir / f"{run_id}_windows.parquet"
    config_snapshot_path = manifests_dir / f"{run_id}_config.json"

    _write_parquet(files_df, files_manifest_path)
    _write_parquet(windows_df, windows_manifest_path)
    config_snapshot_path.write_text(json.dumps(asdict(config), indent=2, default=str))

    status_counts = files_df["status"].value_counts(dropna=False).to_dict() if not files_df.empty else {}
    exported_window_count = int(windows_df.shape[0])

    summary = {
        "run_id": run_id,
        "run_started_utc": run_started_utc,
        "run_finished_utc": _utc_now_iso(),
        "input_dir": str(config.input_dir),
        "output_root": str(config.output_root),
        "backend": config.backend,
        "device": str(device),
        "num_source_files_selected": len(source_files),
        "num_file_rows": int(files_df.shape[0]),
        "num_window_rows": exported_window_count,
        "status_counts": status_counts,
        "files_manifest_path": str(files_manifest_path),
        "windows_manifest_path": str(windows_manifest_path),
        "config_snapshot_path": str(config_snapshot_path),
    }
    return summary
