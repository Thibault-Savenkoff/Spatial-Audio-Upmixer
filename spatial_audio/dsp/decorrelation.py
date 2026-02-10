"""
Allpass-based decorrelation engine for spatial audio.

When a mono or stereo signal is duplicated to multiple surround / height
channels, the copies are *correlated*.  Binaural renderers (AirPods Pro)
convolve each channel with a different HRTF — if the channels are
identical, the HRTFs interfere constructively / destructively at specific
frequencies, producing comb-filter artifacts and a collapsed image.

**Decorrelation** makes each channel copy perceptually similar but
phase-independent, so HRTF convolution produces a *diffuse* spatial
impression instead of comb-filter artefacts.

This module implements decorrelation via cascaded second-order allpass
filters with randomised coefficients.  Each output channel gets a
unique allpass chain (seeded by channel index), producing a unique
phase response while preserving the magnitude spectrum exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt


def _make_allpass_sos(
    num_stages: int,
    sample_rate: int,
    rng: np.random.Generator,
    min_freq: float = 300.0,
    max_freq: float = 2000.0,
) -> np.ndarray:
    """Generate cascaded 2nd-order allpass sections.

    Each section is a biquad with transfer function:

        H(z) = (a2 + a1·z⁻¹ + z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)

    The resonance frequencies are log-spaced between *min_freq* and
    *max_freq*, with slight random jitter.  The bandwidth (Q) is also
    randomised to produce a complex, non-periodic phase response.

    Returns *sos* array of shape ``(num_stages, 6)``.
    """
    freqs = np.geomspace(min_freq, max_freq, num_stages)
    # Add ±20 % random jitter to frequencies
    freqs = freqs * (1.0 + rng.uniform(-0.20, 0.20, size=num_stages))
    freqs = np.clip(freqs, 20.0, sample_rate / 2 - 1)

    qs = rng.uniform(0.3, 2.5, size=num_stages)

    sos = np.zeros((num_stages, 6))
    for i, (fc, q) in enumerate(zip(freqs, qs)):
        w0 = 2 * np.pi * fc / sample_rate
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 - alpha
        b1 = -2 * np.cos(w0)
        b2 = 1 + alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha

        sos[i] = [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]

    return sos


class Decorrelator:
    """Multi-channel allpass decorrelation engine.

    Create one instance per channel that needs decorrelation, each with
    a unique *seed*.  Call :meth:`process` to obtain the decorrelated
    output, or :meth:`process_blended` to mix it with the dry signal.
    """

    def __init__(
        self,
        sample_rate: int,
        num_stages: int = 10,
        seed: int = 42,
        min_freq: float = 300.0,
        max_freq: float = 2000.0,
    ) -> None:
        self.sample_rate = sample_rate
        rng = np.random.default_rng(seed)
        self.sos = _make_allpass_sos(
            num_stages, sample_rate, rng, min_freq, max_freq
        )

    def process(self, x: np.ndarray) -> np.ndarray:
        """Return fully decorrelated version of *x* (mono 1-D array)."""
        result = sosfilt(self.sos, x)
        return np.asarray(result)

    def process_blended(
        self, x: np.ndarray, blend: float = 0.4
    ) -> np.ndarray:
        """Return a mix of dry + decorrelated signal.

        Parameters
        ----------
        x : 1-D array
            Mono input signal.
        blend : float
            0.0 = 100 % dry (no decorrelation).
            1.0 = 100 % decorrelated.
            Typical values: 0.3–0.6 for surrounds, 0.5–0.7 for heights.
        """
        wet = self.process(x)
        return (1.0 - blend) * x + blend * wet


class DecorrelationBank:
    """Pre-built bank of decorrelators for all surround / height channels.

    Creates one :class:`Decorrelator` per channel with a unique seed so
    each channel gets a different phase response.

    Parameters
    ----------
    sample_rate : int
    num_channels : int
        How many independent decorrelators to create.
    num_stages : int
        Cascaded allpass stages per decorrelator.
    seed_base : int
        Base random seed.  Channel *i* uses ``seed_base + i``.
    """

    def __init__(
        self,
        sample_rate: int,
        num_channels: int = 8,
        num_stages: int = 10,
        seed_base: int = 42,
        min_freq: float = 300.0,
        max_freq: float = 2000.0,
    ) -> None:
        self.decorrelators = [
            Decorrelator(
                sample_rate,
                num_stages=num_stages,
                seed=seed_base + i,
                min_freq=min_freq,
                max_freq=max_freq,
            )
            for i in range(num_channels)
        ]

    def process(self, x: np.ndarray, channel: int) -> np.ndarray:
        """Fully decorrelate *x* using the decorrelator for *channel*."""
        return self.decorrelators[channel].process(x)

    def process_blended(
        self, x: np.ndarray, channel: int, blend: float = 0.4
    ) -> np.ndarray:
        """Decorrelate and blend with dry signal for *channel*."""
        return self.decorrelators[channel].process_blended(x, blend)
