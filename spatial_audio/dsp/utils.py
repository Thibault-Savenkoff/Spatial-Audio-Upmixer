"""
DSP utility functions: gain staging, normalization, mono/stereo helpers.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mono / stereo helpers
# ---------------------------------------------------------------------------

def to_mono(x: np.ndarray) -> np.ndarray:
    """Convert stereo (N, 2) to mono (N,) by averaging channels."""
    if x.ndim == 1:
        return x
    return (x[:, 0] + x[:, 1]) / 2.0


def mid_side(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose stereo (N, 2) into mid (mono sum) and side (difference).

    Returns
    -------
    mid : (N,) — centre content (vocals, bass, kick)
    side : (N,) — stereo width content (panned instruments, reverb tails)
    """
    if x.ndim == 1:
        return x, np.zeros_like(x)
    mid = (x[:, 0] + x[:, 1]) / 2.0
    side = (x[:, 0] - x[:, 1]) / 2.0
    return mid, side


def get_left(x: np.ndarray) -> np.ndarray:
    """Return left channel of a stereo signal, or the signal itself if mono."""
    if x.ndim == 1:
        return x
    return x[:, 0]


def get_right(x: np.ndarray) -> np.ndarray:
    """Return right channel, or the signal itself if mono."""
    if x.ndim == 1:
        return x
    return x[:, 1]


# ---------------------------------------------------------------------------
# Gain & normalization
# ---------------------------------------------------------------------------

def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (db / 20.0)


def linear_to_db(amp: float) -> float:
    """Convert linear amplitude to decibels."""
    if amp <= 0:
        return -np.inf
    return 20.0 * np.log10(amp)


def peak_normalize(
    x: np.ndarray, target_dbfs: float = -1.0
) -> np.ndarray:
    """Peak-normalize a multi-channel signal to *target_dbfs*.

    Parameters
    ----------
    x : np.ndarray
        Audio data, shape ``(N,)`` or ``(N, C)``.
    target_dbfs : float
        Target peak level in dBFS.  Default −1.0 dBFS.

    Returns
    -------
    Normalized copy of *x*.
    """
    peak = np.max(np.abs(x))
    if peak < 1e-10:
        return x  # silence — nothing to normalize
    target_amp = db_to_linear(target_dbfs)
    return x * (target_amp / peak)


def soft_clip(x: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Soft-knee clipper to tame inter-channel peaks without hard clipping.

    Applies a tanh-shaped saturation above *threshold*.
    """
    mask = np.abs(x) > threshold
    if not np.any(mask):
        return x

    out = x.copy()
    # Scale overshoot into tanh range and saturate
    overshoot = (np.abs(out[mask]) - threshold) / (1.0 - threshold + 1e-9)
    saturated = threshold + (1.0 - threshold) * np.tanh(overshoot)
    out[mask] = np.sign(out[mask]) * saturated
    return out


# ---------------------------------------------------------------------------
# Delay
# ---------------------------------------------------------------------------

def apply_delay(
    x: np.ndarray, delay_ms: float, sample_rate: int
) -> np.ndarray:
    """Apply a simple sample-accurate delay to a 1-D signal.

    The output length is the same as the input; late samples are clipped.
    """
    delay_samples = int(round(delay_ms * sample_rate / 1000.0))
    if delay_samples <= 0:
        return x
    return np.pad(x, (delay_samples, 0), mode="constant")[: x.shape[0]]


# ---------------------------------------------------------------------------
# Length matching
# ---------------------------------------------------------------------------

def match_lengths(*arrays: np.ndarray) -> list[np.ndarray]:
    """Pad or trim all 1-D arrays to the length of the longest one."""
    max_len = max(a.shape[0] for a in arrays)
    result = []
    for a in arrays:
        if a.shape[0] < max_len:
            pad_width = max_len - a.shape[0]
            a = np.pad(a, (0, pad_width), mode="constant")
        elif a.shape[0] > max_len:
            a = a[:max_len]
        result.append(a)
    return result
