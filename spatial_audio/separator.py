"""
Stem separation wrapper.

Two back-ends:
  • "roformer" (v3): BS-RoFormer-SW gives vocals, drums, bass, guitar,
    piano, other; a karaoke RoFormer then splits the vocals into the lead
    vocal and the backing vocals / ad-libs.
  • Demucs v4 ("htdemucs", "htdemucs_ft"): vocals, drums, bass, other.

Returns numpy arrays directly instead of file paths.
Handles resampling to the project sample rate (48 kHz).
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from .config import (
    DEFAULT_MODEL, DEFAULT_KARAOKE_MODEL, DEMUCS_MODELS,
    ROFORMER_SEGMENT_SIZE, ROFORMER_STEM_MODEL, SAMPLE_RATE,
)

MODEL_DIR = os.path.expanduser("~/.cache/audio-separator-models")


class StemData(NamedTuple):
    """Container for separated stems as numpy arrays."""
    vocals: np.ndarray    # (N, 2) float64 — lead vocal only with RoFormer
    drums: np.ndarray     # (N, 2) float64
    bass: np.ndarray      # (N, 2) float64
    other: np.ndarray     # (N, 2) float64
    sample_rate: int
    backing: np.ndarray | None = None   # backing vocals / ad-libs (RoFormer)
    guitar: np.ndarray | None = None    # (RoFormer)
    piano: np.ndarray | None = None     # (RoFormer)


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


def _load_stems(paths: dict[str, str], target_sr: int, _log) -> dict[str, np.ndarray]:
    """Read stem WAVs, force stereo, resample and pad to a common length."""
    stems = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing stem: {path}")
        audio, sr = sf.read(path, dtype="float64")
        audio = _ensure_stereo(audio)
        audio = _resample_if_needed(audio, sr, target_sr)
        stems[name] = audio
        _log(f"  Loaded stem: {name} ({audio.shape[0]} samples)")

    max_len = max(s.shape[0] for s in stems.values())
    for name, audio in stems.items():
        if audio.shape[0] < max_len:
            stems[name] = np.pad(audio, ((0, max_len - audio.shape[0]), (0, 0)))
    return stems


def _run_roformer(model_file: str, input_path: str, out_dir: str, overlap: int) -> dict[str, str]:
    """Run one audio-separator model; return {stem name (lowercase): path}.

    Outputs already in *out_dir* (from a previous run with --stems-dir) are reused.
    """
    base = glob.escape(os.path.splitext(os.path.basename(input_path))[0])
    model_tag = os.path.splitext(model_file)[0].split(".")[0]
    done = glob.glob(os.path.join(out_dir, f"{base}_(*)_{model_tag}*.wav"))
    if len(done) >= 2:
        return _stem_paths(done)

    from audio_separator.separator import Separator  # heavy import, only when used

    sep = Separator(
        model_file_dir=MODEL_DIR,
        output_dir=out_dir,
        log_level=40,
        normalization_threshold=1.0,   # keep the stems' relative levels
        mdxc_params={
            "segment_size": ROFORMER_SEGMENT_SIZE.get(model_file, 256),
            "override_model_segment_size": model_file in ROFORMER_SEGMENT_SIZE,
            "batch_size": 1, "overlap": overlap, "pitch_shift": 0,
        },
    )
    sep.load_model(model_file)
    return _stem_paths(os.path.join(out_dir, os.path.basename(f)) for f in sep.separate(input_path))


def _stem_paths(files) -> dict[str, str]:
    """"<input>_(Stem)_<model>.wav" → {stem: path}; the stem is the last "(word)"."""
    paths = {}
    for f in files:
        tags = re.findall(r"\((\w+)\)", os.path.basename(f))
        if tags:
            paths[tags[-1].lower()] = f
    return paths


def _separate_roformer(input_path, karaoke_model, temp_dir, target_sr, _log) -> StemData:
    _log(f"Separating 6 stems with {ROFORMER_STEM_MODEL} (slow on CPU)...")
    paths = _run_roformer(ROFORMER_STEM_MODEL, input_path, temp_dir, overlap=2)

    _log(f"Splitting lead / backing vocals with {karaoke_model}...")
    kara_dir = os.path.join(temp_dir, "karaoke", os.path.splitext(karaoke_model)[0])
    kara = _run_roformer(karaoke_model, paths["vocals"], kara_dir, overlap=2)
    # Karaoke models output "vocals" (lead) and "instrumental" (= everything
    # else, which here is only the backing vocals since the input is a vocal stem).
    paths["vocals"] = kara["vocals"]
    paths["backing"] = kara["instrumental"]

    st = _load_stems(paths, target_sr, _log)
    _log("Stem separation complete.")
    return StemData(
        vocals=st["vocals"], drums=st["drums"], bass=st["bass"], other=st["other"],
        sample_rate=target_sr,
        backing=st["backing"], guitar=st["guitar"], piano=st["piano"],
    )


def _separate_demucs(input_path, model_name, temp_dir, target_sr, _log) -> StemData:
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

    names = ["vocals", "drums", "bass", "other"]
    st = _load_stems({n: os.path.join(stem_dir, f"{n}.wav") for n in names}, target_sr, _log)
    _log("Stem separation complete.")
    return StemData(
        vocals=st["vocals"], drums=st["drums"], bass=st["bass"], other=st["other"],
        sample_rate=target_sr,
    )


def separate(
    input_path: str,
    model_name: str = DEFAULT_MODEL,
    target_sr: int = SAMPLE_RATE,
    progress_callback=None,
    karaoke_model: str = DEFAULT_KARAOKE_MODEL,
    stems_dir: str | None = None,
) -> StemData:
    """Run stem separation and return audio arrays.

    Parameters
    ----------
    input_path : str
        Path to the input audio file.
    model_name : str
        ``"roformer"``, or a Demucs model (``"htdemucs"`` / ``"htdemucs_ft"``).
    target_sr : int
        Resample all stems to this sample rate after separation.
    progress_callback : callable, optional
        Called with ``(message: str)`` for progress updates.
    karaoke_model : str
        RoFormer model used to split lead / backing vocals (roformer only).
    stems_dir : str, optional
        Keep the RoFormer stems in ``stems_dir/<song>/`` and reuse them on the
        next run (to try other gains or karaoke models without re-separating).

    Returns
    -------
    StemData
    """
    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    if stems_dir and model_name not in DEMUCS_MODELS:
        keep_dir = os.path.join(stems_dir, os.path.splitext(os.path.basename(input_path))[0])
        os.makedirs(keep_dir, exist_ok=True)
        return _separate_roformer(input_path, karaoke_model, keep_dir, target_sr, _log)

    temp_dir = tempfile.mkdtemp(prefix="spatial_sep_")
    try:
        if model_name in DEMUCS_MODELS:
            return _separate_demucs(input_path, model_name, temp_dir, target_sr, _log)
        return _separate_roformer(input_path, karaoke_model, temp_dir, target_sr, _log)
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                _log(f"Warning: could not remove temp dir: {e}")
