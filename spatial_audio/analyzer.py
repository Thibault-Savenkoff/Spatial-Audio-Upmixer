"""
Local DSP-based audio analyser — replaces the Gemini AI analysis.

Analyses the input audio to auto-tune spatial mixing parameters:
  • Spectral centroid  → brightness → height channel blend
  • RMS energy by band → bass weight → LFE gain
  • Transient density  → controls decorrelation aggressiveness
  • Stereo width (L/R correlation) → surround spread
  • Dynamic range      → normalization target

All analysis is deterministic, instant, and free (no API key required).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from .config import MixPreset, SAMPLE_RATE


@dataclass
class AnalysisResult:
    """Raw analysis measurements."""
    spectral_centroid_hz: float
    bass_energy_ratio: float      # 0–1: portion of energy below 250 Hz
    transient_density: float      # 0–1: normalised transient count
    stereo_width: float           # 0 = mono, 1 = fully uncorrelated
    dynamic_range_db: float       # difference between loud and quiet
    rms_dbfs: float               # overall loudness
    description: str              # human-readable summary


def analyse(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> AnalysisResult:
    """Analyse a stereo or mono audio signal and return measurements.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(N,)`` or ``(N, 2)``.
    sample_rate : int
    """
    # Work with mono for spectral features
    if audio.ndim == 2:
        mono = (audio[:, 0] + audio[:, 1]) / 2.0
        left = audio[:, 0]
        right = audio[:, 1]
    else:
        mono = audio
        left = right = audio

    # --- Spectral centroid ---
    nperseg = min(4096, len(mono))
    f, t, Zxx = stft(mono, fs=sample_rate, nperseg=nperseg)
    magnitude = np.abs(Zxx)
    # Weighted average frequency per frame, then average across time
    mag_sum = magnitude.sum(axis=0) + 1e-12
    centroid_per_frame = (f[:, None] * magnitude).sum(axis=0) / mag_sum
    spectral_centroid = float(np.mean(centroid_per_frame))

    # --- Band energy ratios ---
    bass_mask = f < 250
    total_energy = float(np.sum(magnitude ** 2)) + 1e-12
    bass_energy = float(np.sum(magnitude[bass_mask, :] ** 2))
    bass_ratio = bass_energy / total_energy

    # --- Transient density & dynamic range ---
    # Compute short-term energy envelope and count large jumps
    frame_len = int(0.01 * sample_rate)  # 10 ms frames
    n_frames = len(mono) // frame_len
    if n_frames >= 2:
        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.sum(frames ** 2, axis=1)
        energy_db = np.asarray(10.0 * np.log10(energy + 1e-12))
        diff = np.diff(energy_db)
        # Count frames where energy jumps > 6 dB
        transients = int(np.sum(diff > 6.0))
        transient_density = float(np.clip(transients / n_frames, 0, 1))
        # Dynamic range: 95th percentile minus 10th percentile
        energy_sorted = np.sort(energy_db)
        idx_hi = int(0.95 * len(energy_sorted))
        idx_lo = int(0.10 * len(energy_sorted))
        dynamic_range = float(energy_sorted[idx_hi] - energy_sorted[idx_lo])
    else:
        transient_density = 0.5
        dynamic_range = 20.0

    # --- Stereo width (correlation coefficient) ---
    if audio.ndim == 2 and audio.shape[1] >= 2:
        corr_matrix = np.corrcoef(left, right)
        correlation = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 1.0
        # Width: 0 = perfectly correlated (mono), 1 = uncorrelated
        stereo_width = float(np.clip(1.0 - abs(correlation), 0, 1))
    else:
        stereo_width = 0.0

    # --- Overall RMS ---
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_dbfs = float(20.0 * np.log10(rms + 1e-12))

    # --- Description ---
    brightness = "bright" if spectral_centroid > 3000 else "warm" if spectral_centroid < 1500 else "balanced"
    bass_level = "bass-heavy" if bass_ratio > 0.35 else "light-bass" if bass_ratio < 0.15 else "moderate-bass"
    transient_level = "transient-rich" if transient_density > 0.15 else "smooth"
    width_label = "wide-stereo" if stereo_width > 0.4 else "narrow" if stereo_width < 0.15 else "moderate-width"

    description = f"{brightness}, {bass_level}, {transient_level}, {width_label}"

    return AnalysisResult(
        spectral_centroid_hz=spectral_centroid,
        bass_energy_ratio=bass_ratio,
        transient_density=transient_density,
        stereo_width=stereo_width,
        dynamic_range_db=dynamic_range,
        rms_dbfs=rms_dbfs,
        description=description,
    )


def adapt_preset(
    base: MixPreset,
    analysis: AnalysisResult,
) -> MixPreset:
    """Create an adapted copy of *base* preset tuned by analysis results.

    The adjustments are subtle and bounded — never wildly different from
    the preset defaults, just nudged towards what works best for the
    material.
    """
    from dataclasses import replace

    p = replace(base)  # shallow copy

    # --- Bass-heavy material → boost LFE, reduce bass in center ---
    if analysis.bass_energy_ratio > 0.30:
        p.bass_lfe_gain = min(p.bass_lfe_gain + 0.10, 1.0)
        p.bass_center_gain = max(p.bass_center_gain - 0.05, 0.50)
    elif analysis.bass_energy_ratio < 0.15:
        p.bass_lfe_gain = max(p.bass_lfe_gain - 0.10, 0.40)
        p.bass_center_gain = min(p.bass_center_gain + 0.05, 0.85)

    # --- Bright material → more height channel content ---
    if analysis.spectral_centroid_hz > 3500:
        p.other_height_gain = min(p.other_height_gain + 0.06, 0.35)
        p.drum_height_bleed = min(p.drum_height_bleed + 0.04, 0.15)
    elif analysis.spectral_centroid_hz < 1200:
        p.other_height_gain = max(p.other_height_gain - 0.05, 0.10)

    # --- Transient-rich → less decorrelation bleed (preserve punch) ---
    if analysis.transient_density > 0.20:
        p.drum_height_bleed = max(p.drum_height_bleed - 0.03, 0.03)
        p.surround_delay_ms = max(p.surround_delay_ms - 3.0, 8.0)
    elif analysis.transient_density < 0.05:
        # Smooth / ambient → more surround immersion
        p.other_side_gain = min(p.other_side_gain + 0.08, 0.80)
        p.other_rear_gain = min(p.other_rear_gain + 0.06, 0.55)
        p.surround_delay_ms = min(p.surround_delay_ms + 4.0, 25.0)

    # --- Wide stereo → more surround spread ---
    if analysis.stereo_width > 0.45:
        p.other_side_gain = min(p.other_side_gain + 0.05, 0.80)
        p.other_rear_gain = min(p.other_rear_gain + 0.04, 0.55)
        p.vocal_width_bleed = min(p.vocal_width_bleed + 0.03, 0.20)
    elif analysis.stereo_width < 0.10:
        # Very mono — conservative surround
        p.other_side_gain = max(p.other_side_gain - 0.08, 0.40)
        p.other_rear_gain = max(p.other_rear_gain - 0.05, 0.25)

    # --- Compressed / loud material → slightly lower peak target ---
    if analysis.dynamic_range_db < 12.0:
        p.target_peak_dbfs = -1.5
    elif analysis.dynamic_range_db > 30.0:
        p.target_peak_dbfs = -0.5

    return p
