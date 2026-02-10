"""
Demucs v4 stem separation wrapper.

Separates a stereo audio file into four stems:
  • vocals
  • drums
  • bass
  • other (instruments, synths, keys, etc.)

Returns numpy arrays directly instead of file paths.
Handles resampling to the project sample rate (48 kHz).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from .config import DEFAULT_DEMUCS_MODEL, SAMPLE_RATE


class StemData(NamedTuple):
    """Container for separated stems as numpy arrays."""
    vocals: np.ndarray    # (N, 2) float64
    drums: np.ndarray     # (N, 2) float64
    bass: np.ndarray      # (N, 2) float64
    other: np.ndarray     # (N, 2) float64
    sample_rate: int


def _resample_if_needed(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample audio from *orig_sr* to *target_sr* if they differ.

    Uses polyphase resampling (high quality, exact rational ratio).
    """
    if orig_sr == target_sr:
        return audio

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    if audio.ndim == 1:
        return resample_poly(audio, up, down).astype(np.float64)

    # Process each channel independently
    channels = []
    for ch in range(audio.shape[1]):
        channels.append(resample_poly(audio[:, ch], up, down))
    return np.column_stack(channels).astype(np.float64)


def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    """Ensure array is (N, 2).  Duplicate mono if needed."""
    if x.ndim == 1:
        return np.column_stack([x, x])
    if x.shape[1] == 1:
        return np.column_stack([x[:, 0], x[:, 0]])
    return x[:, :2]  # take only first two channels


def separate(
    input_path: str,
    model_name: str = DEFAULT_DEMUCS_MODEL,
    target_sr: int = SAMPLE_RATE,
    progress_callback=None,
) -> StemData:
    """Run Demucs stem separation and return audio arrays.

    Parameters
    ----------
    input_path : str
        Path to the input audio file.
    model_name : str
        Demucs model name (``"htdemucs"`` or ``"htdemucs_ft"``).
    target_sr : int
        Resample all stems to this sample rate after separation.
    progress_callback : callable, optional
        Called with ``(message: str)`` for progress updates.

    Returns
    -------
    StemData
        Named tuple with vocals, drums, bass, other arrays and sample_rate.
    """
    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    temp_dir = tempfile.mkdtemp(prefix="spatial_demucs_")

    try:
        _log(f"Separating stems with Demucs ({model_name})...")

        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model_name,
            "--out", temp_dir,
            input_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            _log(f"Demucs error output:\n{result.stderr}")
            raise RuntimeError(
                f"Demucs failed (exit code {result.returncode}):\n"
                f"{result.stderr[-1000:]}"
            )

        # Locate output stems
        filename = os.path.splitext(os.path.basename(input_path))[0]
        stem_dir = os.path.join(temp_dir, model_name, filename)

        if not os.path.isdir(stem_dir):
            raise FileNotFoundError(
                f"Demucs output not found at {stem_dir}. "
                f"Demucs stderr: {result.stderr[-500:]}"
            )

        stem_names = ["vocals", "drums", "bass", "other"]
        stems = {}

        for name in stem_names:
            path = os.path.join(stem_dir, f"{name}.wav")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing stem: {path}")

            audio, sr = sf.read(path, dtype="float64")
            audio = _ensure_stereo(audio)
            audio = _resample_if_needed(audio, sr, target_sr)
            stems[name] = audio
            _log(f"  Loaded stem: {name} ({audio.shape[0]} samples)")

        # Ensure all stems have the same length
        max_len = max(s.shape[0] for s in stems.values())
        for name in stem_names:
            if stems[name].shape[0] < max_len:
                pad = max_len - stems[name].shape[0]
                stems[name] = np.pad(
                    stems[name], ((0, pad), (0, 0)), mode="constant"
                )

        _log("Stem separation complete.")

        return StemData(
            vocals=stems["vocals"],
            drums=stems["drums"],
            bass=stems["bass"],
            other=stems["other"],
            sample_rate=target_sr,
        )

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                _log(f"Warning: could not remove temp dir: {e}")
