"""
Configuration constants and presets for the Spatial Audio Upmixer.

Channel layout: 7.1.4 (12 channels)
  Index 0:  FL  — Front Left
  Index 1:  FR  — Front Right
  Index 2:  FC  — Front Center
  Index 3:  LFE — Low Frequency Effects
  Index 4:  BL  — Back Left
  Index 5:  BR  — Back Right
  Index 6:  SL  — Side Left
  Index 7:  SR  — Side Right
  Index 8:  TFL — Top Front Left
  Index 9:  TFR — Top Front Right
  Index 10: TBL — Top Back Left
  Index 11: TBR — Top Back Right
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Channel constants
# ---------------------------------------------------------------------------
CHANNEL_COUNT_714 = 12
CHANNEL_COUNT_51 = 6

CHANNEL_NAMES_714 = [
    "FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR",
    "TFL", "TFR", "TBL", "TBR",
]

CHANNEL_NAMES_51 = ["FL", "FR", "FC", "LFE", "SL", "SR"]

# Channel indices (7.1.4)
CH_FL, CH_FR, CH_FC, CH_LFE = 0, 1, 2, 3
CH_BL, CH_BR, CH_SL, CH_SR = 4, 5, 6, 7
CH_TFL, CH_TFR, CH_TBL, CH_TBR = 8, 9, 10, 11

# ---------------------------------------------------------------------------
# Audio format
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000        # Required for spatial audio delivery
BIT_DEPTH = 24             # 24-bit output
SUBTYPE_WAV = "PCM_24"     # soundfile subtype string

# ---------------------------------------------------------------------------
# Crossover frequencies (Hz)
# ---------------------------------------------------------------------------
LFE_CROSSOVER_HZ = 80     # LFE ↔ main channel split
HEIGHT_HIGHPASS_HZ = 500   # Height channels carry only content above this

# ---------------------------------------------------------------------------
# FIR filter design
# ---------------------------------------------------------------------------
FIR_TAPS = 511             # Odd number → Type I symmetric FIR
FIR_WINDOW = "hann"        # Good side-lobe suppression, smooth rolloff

# Group delay in samples = (FIR_TAPS - 1) / 2 = 255 samples ≈ 5.3 ms @ 48 kHz

# ---------------------------------------------------------------------------
# Decorrelation
# ---------------------------------------------------------------------------
DECORR_ALLPASS_STAGES = 10           # Cascaded 2nd-order allpass sections per channel
DECORR_SEED_BASE = 42               # Base seed; each channel gets seed_base + ch_index
DECORR_BLEND_SURROUND = 0.40        # 40 % decorrelated / 60 % direct for SL/SR/BL/BR
DECORR_BLEND_HEIGHT = 0.65          # 65 % decorrelated / 35 % direct for TFL/TFR/TBL/TBR
DECORR_MIN_FREQ = 300.0             # Below this: minimal decorrelation
DECORR_MAX_FREQ = 2000.0            # Above this: full decorrelation strength

# ---------------------------------------------------------------------------
# Mixing gains (linear amplitude, not dB)
# ---------------------------------------------------------------------------
@dataclass
class MixPreset:
    """Spatial mix parameters. All gains are linear amplitude."""
    # Vocals
    vocal_center_gain: float = 0.90       # Vocals → FC  (slightly below unity to sit naturally)
    vocal_width_bleed: float = 0.12       # Decorrelated stereo-side component → FL/FR

    # Bass
    bass_lfe_gain: float = 0.80           # Bass sub (<80 Hz) → LFE
    bass_center_gain: float = 0.70        # Bass body (>80 Hz) → FC

    # Drums
    drum_front_gain: float = 0.85         # Drums stereo image → FL/FR
    drum_lfe_gain: float = 0.60           # Kick sub (<80 Hz) → LFE
    drum_height_bleed: float = 0.08       # Decorrelated overhead shimmer → TFL/TFR

    # Other (instruments / ambience)
    other_side_gain: float = 0.65         # "Other" stem → SL/SR (primary surround)
    other_rear_gain: float = 0.40         # Decorrelated → BL/BR
    other_height_gain: float = 0.22       # Heavily decorrelated, >500 Hz → heights
    other_front_bleed: float = 0.15       # Slight presence kept in FL/FR

    # Surround delay
    surround_delay_ms: float = 15.0       # Haas-effect delay for surrounds
    rear_extra_delay_ms: float = 8.0      # Additional delay for BL/BR vs SL/SR

    # Normalization
    target_peak_dbfs: float = -1.0        # Peak normalize to this level

    # Quality
    fir_taps: int = FIR_TAPS
    decorr_stages: int = DECORR_ALLPASS_STAGES


# Pre-built quality presets
PRESET_LOW = MixPreset(
    fir_taps=255,
    decorr_stages=6,
)

PRESET_MEDIUM = MixPreset()   # defaults

PRESET_HIGH = MixPreset(
    fir_taps=1023,
    decorr_stages=14,
    vocal_center_gain=0.88,
    other_height_gain=0.25,
)

PRESETS = {
    "low": PRESET_LOW,
    "medium": PRESET_MEDIUM,
    "high": PRESET_HIGH,
}

# ---------------------------------------------------------------------------
# Demucs models
# ---------------------------------------------------------------------------
DEMUCS_MODELS = {
    "htdemucs":    "Standard Hybrid Transformer (faster)",
    "htdemucs_ft": "Fine-tuned Hybrid Transformer (better quality)",
}
DEFAULT_DEMUCS_MODEL = "htdemucs_ft"

# ---------------------------------------------------------------------------
# FFmpeg channel layout strings
# ---------------------------------------------------------------------------
FFMPEG_LAYOUT_714 = "7.1.4"
FFMPEG_LAYOUT_51 = "5.1"
