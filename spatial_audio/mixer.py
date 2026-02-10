"""
7.1.4 Spatial Audio Mixer — the core engine.

Takes four Demucs stems (vocals, drums, bass, other) and mixes them
into a 12-channel 7.1.4 spatial audio signal optimised for binaural
rendering on AirPods Pro 3.

Design principles
-----------------
1. **No stem doubling** — each stem has a primary channel group.
   Cross-feed to secondary channels is always <20 % and always
   decorrelated, preventing comb-filter artifacts on binaural rendering.

2. **Linear-phase crossovers** — FIR filters split frequencies between
   LFE and main channels with zero phase distortion at the crossover.

3. **Allpass decorrelation** — surround and height channels receive
   phase-decorrelated copies so binaural HRTFs produce a diffuse
   spatial impression instead of interference.

4. **Proper channel routing**:

   | Stem      | Primary channels        | Secondary (decorrelated) |
   |-----------|------------------------|--------------------------|
   | Vocals    | FC                     | FL/FR (stereo side, ≤15 %) |
   | Bass      | FC (>80 Hz) + LFE (<80) | —                        |
   | Drums     | FL/FR (>80 Hz) + LFE  | TFL/TFR (shimmer, ≤10 %) |
   | Other     | SL/SR                  | BL/BR, TFL–TBR (≤25 %)  |
"""

from __future__ import annotations

import numpy as np

from .config import (
    CHANNEL_COUNT_714,
    CH_FL, CH_FR, CH_FC, CH_LFE,
    CH_BL, CH_BR, CH_SL, CH_SR,
    CH_TFL, CH_TFR, CH_TBL, CH_TBR,
    LFE_CROSSOVER_HZ, HEIGHT_HIGHPASS_HZ,
    DECORR_BLEND_SURROUND, DECORR_BLEND_HEIGHT,
    DECORR_SEED_BASE, DECORR_MIN_FREQ, DECORR_MAX_FREQ,
    MixPreset,
)
from .dsp.crossover import Crossover
from .dsp.decorrelation import DecorrelationBank
from .dsp.utils import (
    to_mono, mid_side, get_left, get_right,
    peak_normalize, soft_clip, apply_delay, match_lengths,
)
from .separator import StemData


def mix_to_714(
    stems: StemData,
    preset: MixPreset,
    progress_callback=None,
) -> tuple[np.ndarray, int]:
    """Mix separated stems into a 7.1.4 multichannel signal.

    Parameters
    ----------
    stems : StemData
        Four separated stems + sample_rate.
    preset : MixPreset
        Mixing parameters (gains, delays, quality settings).
    progress_callback : callable, optional
        Called with ``(message: str)`` for progress updates.

    Returns
    -------
    output : np.ndarray
        Shape ``(N, 12)`` — 12-channel 7.1.4 audio.
    sample_rate : int
    """
    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    sr = stems.sample_rate
    n_samples = stems.vocals.shape[0]

    _log("Building spatial mix (7.1.4)...")

    # ------------------------------------------------------------------
    # Prepare crossover filters
    # ------------------------------------------------------------------
    _log("  Initializing crossover filters...")
    xo_lfe = Crossover(
        LFE_CROSSOVER_HZ, sr, num_taps=preset.fir_taps
    )
    xo_height = Crossover(
        HEIGHT_HIGHPASS_HZ, sr, num_taps=preset.fir_taps
    )

    # ------------------------------------------------------------------
    # Prepare decorrelation bank
    # ------------------------------------------------------------------
    # We need decorrelators for: SL, SR, BL, BR, TFL, TFR, TBL, TBR = 8
    _log("  Initializing decorrelation bank...")
    decorr = DecorrelationBank(
        sample_rate=sr,
        num_channels=8,
        num_stages=preset.decorr_stages,
        seed_base=DECORR_SEED_BASE,
        min_freq=DECORR_MIN_FREQ,
        max_freq=DECORR_MAX_FREQ,
    )
    # Map decorrelator indices to logical roles
    D_SL, D_SR, D_BL, D_BR = 0, 1, 2, 3
    D_TFL, D_TFR, D_TBL, D_TBR = 4, 5, 6, 7

    # ------------------------------------------------------------------
    # Allocate output buffer
    # ------------------------------------------------------------------
    output = np.zeros((n_samples, CHANNEL_COUNT_714), dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. VOCALS → FC (primary) + subtle FL/FR width
    # ------------------------------------------------------------------
    _log("  Routing vocals...")
    vocal_mid, vocal_side = mid_side(stems.vocals)

    # Center: mono sum of vocals, highpass to keep LFE-range out
    vocal_center = xo_lfe.highpass(vocal_mid) * preset.vocal_center_gain
    output[:, CH_FC] += vocal_center

    # Width: decorrelated side component at low level to FL/FR
    # This gives vocals natural stereo presence without being "too forward"
    vocal_side_l = vocal_side * preset.vocal_width_bleed
    vocal_side_r = -vocal_side * preset.vocal_width_bleed  # inverted for width
    output[:, CH_FL] += decorr.process_blended(
        vocal_side_l, D_SL, blend=0.3
    )
    output[:, CH_FR] += decorr.process_blended(
        vocal_side_r, D_SR, blend=0.3
    )

    # ------------------------------------------------------------------
    # 2. BASS → FC (>80Hz) + LFE (<80Hz)
    # ------------------------------------------------------------------
    _log("  Routing bass...")
    bass_mono = to_mono(stems.bass)

    # LFE: lowpass-filtered sub-bass
    bass_sub = xo_lfe.lowpass(bass_mono) * preset.bass_lfe_gain
    output[:, CH_LFE] += bass_sub

    # Center: body of the bass above crossover
    bass_body = xo_lfe.highpass(bass_mono) * preset.bass_center_gain
    output[:, CH_FC] += bass_body

    # ------------------------------------------------------------------
    # 3. DRUMS → FL/FR (>80Hz) + LFE (kick sub) + TFL/TFR (shimmer)
    # ------------------------------------------------------------------
    _log("  Routing drums...")
    drum_left = get_left(stems.drums)
    drum_right = get_right(stems.drums)
    drum_mono = to_mono(stems.drums)

    # LFE: kick sub-bass
    kick_sub = xo_lfe.lowpass(drum_mono) * preset.drum_lfe_gain
    output[:, CH_LFE] += kick_sub

    # Front L/R: stereo drum image, highpassed
    output[:, CH_FL] += xo_lfe.highpass(drum_left) * preset.drum_front_gain
    output[:, CH_FR] += xo_lfe.highpass(drum_right) * preset.drum_front_gain

    # Height shimmer: heavily decorrelated, highpassed-at-500Hz overhead "air"
    if preset.drum_height_bleed > 0.01:
        drum_hp = xo_height.highpass(drum_mono)
        output[:, CH_TFL] += decorr.process_blended(
            drum_hp * preset.drum_height_bleed, D_TFL, blend=DECORR_BLEND_HEIGHT
        )
        output[:, CH_TFR] += decorr.process_blended(
            drum_hp * preset.drum_height_bleed, D_TFR, blend=DECORR_BLEND_HEIGHT
        )

    # ------------------------------------------------------------------
    # 4. OTHER → SL/SR (primary) + BL/BR + Heights + subtle FL/FR
    # ------------------------------------------------------------------
    _log("  Routing instruments / other...")
    other_left = get_left(stems.other)
    other_right = get_right(stems.other)
    other_mono = to_mono(stems.other)

    # Highpass to remove sub-bass from surround/height routing
    other_left_hp = xo_lfe.highpass(other_left)
    other_right_hp = xo_lfe.highpass(other_right)
    other_mono_hp = xo_lfe.highpass(other_mono)

    # --- Side surrounds (SL/SR) — primary placement ---
    sl_raw = other_left_hp * preset.other_side_gain
    sr_raw = other_right_hp * preset.other_side_gain

    # Apply surround delay (Haas effect for spatial depth)
    sl_delayed = apply_delay(sl_raw, preset.surround_delay_ms, sr)
    sr_delayed = apply_delay(sr_raw, preset.surround_delay_ms, sr)

    output[:, CH_SL] += decorr.process_blended(
        sl_delayed, D_SL, blend=DECORR_BLEND_SURROUND
    )
    output[:, CH_SR] += decorr.process_blended(
        sr_delayed, D_SR, blend=DECORR_BLEND_SURROUND
    )

    # --- Back surrounds (BL/BR) — decorrelated, extra delay ---
    total_rear_delay = preset.surround_delay_ms + preset.rear_extra_delay_ms

    bl_raw = apply_delay(
        other_left_hp * preset.other_rear_gain, total_rear_delay, sr
    )
    br_raw = apply_delay(
        other_right_hp * preset.other_rear_gain, total_rear_delay, sr
    )

    output[:, CH_BL] += decorr.process_blended(
        bl_raw, D_BL, blend=DECORR_BLEND_SURROUND + 0.10
    )
    output[:, CH_BR] += decorr.process_blended(
        br_raw, D_BR, blend=DECORR_BLEND_SURROUND + 0.10
    )

    # --- Height channels (TFL/TFR/TBL/TBR) — ambient, >500Hz ---
    if preset.other_height_gain > 0.01:
        other_height_hp = xo_height.highpass(other_mono_hp)

        output[:, CH_TFL] += decorr.process_blended(
            other_height_hp * preset.other_height_gain,
            D_TFL, blend=DECORR_BLEND_HEIGHT,
        )
        output[:, CH_TFR] += decorr.process_blended(
            other_height_hp * preset.other_height_gain,
            D_TFR, blend=DECORR_BLEND_HEIGHT,
        )
        output[:, CH_TBL] += decorr.process_blended(
            other_height_hp * preset.other_height_gain * 0.8,
            D_TBL, blend=DECORR_BLEND_HEIGHT + 0.10,
        )
        output[:, CH_TBR] += decorr.process_blended(
            other_height_hp * preset.other_height_gain * 0.8,
            D_TBR, blend=DECORR_BLEND_HEIGHT + 0.10,
        )

    # --- Subtle front presence (FL/FR) ---
    if preset.other_front_bleed > 0.01:
        output[:, CH_FL] += other_left_hp * preset.other_front_bleed
        output[:, CH_FR] += other_right_hp * preset.other_front_bleed

    # ------------------------------------------------------------------
    # 5. Post-processing
    # ------------------------------------------------------------------
    _log("  Normalizing & limiting...")

    # Soft-clip to prevent harsh clipping before normalization
    output = soft_clip(output, threshold=0.95)

    # Peak normalize
    output = peak_normalize(output, target_dbfs=preset.target_peak_dbfs)

    _log(f"  Mix complete: {output.shape[0]} samples × {output.shape[1]} channels")

    return output, sr


def downmix_714_to_51(audio_714: np.ndarray) -> np.ndarray:
    """Fold down a 7.1.4 mix to 5.1 for compatibility.

    Standard downmix coefficients (ITU-R BS.775):
      FL_51 = FL + 0.707 * SL + 0.5 * BL + 0.5 * TFL + 0.35 * TBL
      FR_51 = FR + 0.707 * SR + 0.5 * BR + 0.5 * TFR + 0.35 * TBR
      FC_51 = FC
      LFE_51 = LFE
      SL_51 = SL + 0.707 * BL + 0.5 * TBL
      SR_51 = SR + 0.707 * BR + 0.5 * TBR

    Parameters
    ----------
    audio_714 : np.ndarray
        Shape ``(N, 12)``.

    Returns
    -------
    np.ndarray
        Shape ``(N, 6)``.
    """
    n = audio_714.shape[0]
    out = np.zeros((n, 6), dtype=np.float64)

    # FL
    out[:, 0] = (
        audio_714[:, CH_FL]
        + 0.707 * audio_714[:, CH_SL]
        + 0.500 * audio_714[:, CH_BL]
        + 0.500 * audio_714[:, CH_TFL]
        + 0.350 * audio_714[:, CH_TBL]
    )
    # FR
    out[:, 1] = (
        audio_714[:, CH_FR]
        + 0.707 * audio_714[:, CH_SR]
        + 0.500 * audio_714[:, CH_BR]
        + 0.500 * audio_714[:, CH_TFR]
        + 0.350 * audio_714[:, CH_TBR]
    )
    # FC
    out[:, 2] = audio_714[:, CH_FC]
    # LFE
    out[:, 3] = audio_714[:, CH_LFE]
    # SL (5.1 surround left)
    out[:, 4] = (
        audio_714[:, CH_SL]
        + 0.707 * audio_714[:, CH_BL]
        + 0.500 * audio_714[:, CH_TBL]
    )
    # SR (5.1 surround right)
    out[:, 5] = (
        audio_714[:, CH_SR]
        + 0.707 * audio_714[:, CH_BR]
        + 0.500 * audio_714[:, CH_TBR]
    )

    # Re-normalize
    out = peak_normalize(out, target_dbfs=-1.0)
    return out
