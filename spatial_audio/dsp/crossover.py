"""
Linear-phase FIR crossover filters.

Replaces the IIR Butterworth filters from the original code.
FIR filters have *linear phase* (constant group delay), which means
no phase distortion at the crossover frequency.  When the lowpass and
highpass outputs are summed they reconstruct the original signal exactly
(up to the constant group delay).

Usage
-----
>>> xo = Crossover(cutoff_hz=80, sample_rate=48000, num_taps=511)
>>> low = xo.lowpass(signal)
>>> high = xo.highpass(signal)
>>> np.allclose(low + high, delay(signal, (511-1)//2))  # True
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, fftconvolve


class Crossover:
    """Two-way linear-phase FIR crossover filter."""

    def __init__(
        self,
        cutoff_hz: float,
        sample_rate: int,
        num_taps: int = 511,
        window: str = "hann",
    ) -> None:
        if num_taps % 2 == 0:
            num_taps += 1  # Must be odd for Type I FIR

        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        self.num_taps = num_taps
        self.group_delay = (num_taps - 1) // 2  # samples

        # Design lowpass FIR
        self._lp: np.ndarray = np.asarray(
            firwin(num_taps, cutoff_hz, fs=sample_rate,
                   pass_zero=True, window=window)
        )

        # Derive complementary highpass via spectral inversion
        self._hp: np.ndarray = -self._lp.copy()
        self._hp[self.group_delay] += 1.0

    # ------------------------------------------------------------------
    def lowpass(self, x: np.ndarray) -> np.ndarray:
        """Apply lowpass filter.  Works on mono or multi-channel (N, C)."""
        return self._apply(np.array(self._lp), x)

    def highpass(self, x: np.ndarray) -> np.ndarray:
        """Apply highpass filter.  Works on mono or multi-channel (N, C)."""
        return self._apply(np.array(self._hp), x)

    # ------------------------------------------------------------------
    @staticmethod
    def _apply(fir: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Convolve *fir* with signal *x* using FFT, preserving length.

        The linear-phase group delay ((taps-1)/2 samples) is removed, so
        filtered paths stay time-aligned with unfiltered ones.
        """
        delay = (len(fir) - 1) // 2
        n = x.shape[0]
        if x.ndim == 1:
            return fftconvolve(x, fir, mode="full")[delay: delay + n]
        # Multi-channel: filter each channel independently
        out = np.empty_like(x)
        for ch in range(x.shape[1]):
            out[:, ch] = fftconvolve(x[:, ch], fir, mode="full")[delay: delay + n]
        return out


def make_lfe_crossover(
    sample_rate: int, cutoff_hz: float = 80.0, num_taps: int = 511
) -> Crossover:
    """Create a crossover tuned for LFE / main-channel split."""
    return Crossover(cutoff_hz, sample_rate, num_taps)


def make_height_highpass(
    sample_rate: int, cutoff_hz: float = 500.0, num_taps: int = 511
) -> Crossover:
    """Create a crossover whose *highpass* output feeds the height channels.

    Content below *cutoff_hz* is discarded from the height bus —
    small height-channel speakers can't reproduce it, and bass in the
    heights muddies the spatial image.
    """
    return Crossover(cutoff_hz, sample_rate, num_taps)
