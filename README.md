# Spatial Audio Upmixer

Convert stereo music into **7.1.4 immersive spatial audio** optimised for **AirPods Pro 3** and Apple Spatial Audio. Uses **BS-RoFormer** (or Demucs v4) for AI stem separation and advanced DSP for professional-quality spatial mixing.

## What's New (v3.0)

| Problem (v2) | Fix (v3) |
|---|---|
| Demucs artifacts ("gargling", bleed between stems) | **BS-RoFormer-SW**: 6 cleaner stems (vocals, drums, bass, guitar, piano, other) |
| Backing vocals and ad-libs stuck in the center with the lead | A **karaoke RoFormer** splits the vocals into lead + backing; ad-libs move around and above you, like in Dolby Atmos rap mixes |
| Guitar, piano and synths all in one "other" stem | Guitar and piano get their own placement |

Demucs is still available with `--model htdemucs` / `htdemucs_ft` (faster, same routing as v2).

## What Was New in v2.0

This is a complete rewrite solving the interference artifacts and unnatural vocal placement of the original 5.1 converter:

| Problem (v1) | Fix (v2) |
|---|---|
| Comb-filter interference from stem doubling | Each stem routes to **one primary channel group** — cross-feed is always <20% and decorrelated |
| Vocals "too far ahead" / disconnected | Vocals sit naturally in center with subtle stereo width bleed via mid-side decomposition |
| Phase artifacts at crossover frequencies | **Linear-phase FIR** crossover filters (was IIR Butterworth) |
| Flat, phasey surround channels | **Allpass decorrelation engine** — each surround/height channel gets a unique phase response |
| Only 5.1 output | Full **7.1.4** (12 channels) with height channels for maximum AirPods Pro immersion |
| Unreliable Gemini AI dependency | **Local DSP analysis** — deterministic, instant, free, no API key |
| Basic AAC encoding | **EAC3 encoding** with proper `7.1.4` channel layout metadata |

## Features

- **AI Stem Separation** — BS-RoFormer isolates vocals, drums, bass, guitar, piano and the rest; a second model splits lead and backing vocals
- **12-Channel Spatial Mix** — True 7.1.4 with ear-level surrounds + 4 height channels
- **Smart DSP Analysis** — Analyses spectral balance, dynamics, stereo width to auto-tune the mix
- **Linear-Phase Crossovers** — Zero phase distortion between LFE and main channels
- **Allpass Decorrelation** — Eliminates comb-filter artifacts on binaural rendering
- **3 Quality Presets** — Low (fast), Medium (balanced), High (best quality)
- **5.1 Fallback** — Automatic downmix for older devices
- **Lossless WAV Master** — Optional 24-bit/48kHz 7.1.4 WAV output
- **Modern Dark GUI** — CustomTkinter with progress tracking and batch processing
- **No API Keys** — Everything runs locally

## Channel Routing

| Stem | Primary | Secondary (decorrelated) |
|------|---------|--------------------------|
| Vocals | FC (center) | FL/FR (stereo side, 12%) |
| Bass | FC (>80Hz) + LFE (<80Hz) | — |
| Drums | FL/FR (>80Hz) + LFE (kick) | TFL/TFR (shimmer, 8%) |
| Other | SL/SR (surrounds) | BL/BR (40%) + TFL-TBR (22%) |
| Backing vocals / ad-libs | SL/SR (55%) | FL/FR (25%) + BL/BR (35%) + TFL/TFR (20%) |
| Guitar | FL/FR (55%) | SL/SR (45%, delayed) |
| Piano | FL/FR (60%) | TFL/TFR (15%) |

Backing, guitar and piano only exist with the RoFormer back-end. The backing gains above are scaled by `--backing-db` (default −1.5 dB): once moved away from the lead they are no longer masked and sound louder than in the original.

## Requirements

- **Python 3.10+**
- **FFmpeg** — must be in PATH ([download](https://www.gyan.dev/ffmpeg/builds/))
- **~2.5 GB disk** for the RoFormer models (downloaded automatically on first run into `~/.cache/audio-separator-models`)
- **GPU strongly recommended**: on a laptop CPU (i7-12650H), RoFormer (6 stems + lead/backing split) takes about **20× the track length** (a 3-minute song ≈ 1 hour). Demucs is several times faster.

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### GUI (Recommended)

```bash
python gui.py
```

Or double-click `start_gui.bat` on Windows.

- Select a file or folder
- Choose format: **7.1.4** / **5.1** / **Both**
- Choose quality: **Low** / **Medium** / **High**
- Choose the model: **roformer** (best, slow) / **htdemucs** (fast) / **htdemucs_ft**
- Optionally check **Save WAV master** for lossless output
- Click **Start Conversion**

### Command Line

```bash
# Default: 7.1.4, medium quality, roformer
python main.py song.wav

# Both formats, high quality
python main.py song.mp3 --format both --quality high

# 5.1 only, fast model
python main.py song.flac --format 5.1 --model htdemucs

# Batch process a folder + save WAV masters
python main.py ./music/ --format both --save-wav

# Try another lead/backing vocal model
python main.py song.wav -k mel_band_roformer_karaoke_becruily.ckpt

# Keep the stems, then retry other settings in seconds instead of re-separating
python main.py song.wav --stems-dir ./stems -o ./out_default
python main.py song.wav --stems-dir ./stems -o ./out_0db --backing-db 0

# Output to a specific directory
python main.py song.wav -o ./output/
```

### Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `-f, --format` | `7.1.4`, `5.1`, `both` | `7.1.4` | Output format |
| `-q, --quality` | `low`, `medium`, `high` | `medium` | Processing quality |
| `-m, --model` | `roformer`, `htdemucs`, `htdemucs_ft` | `roformer` | Separation model |
| `-k, --karaoke-model` | see `KARAOKE_MODELS` in `config.py` | aufr33/viperx Mel-RoFormer | Lead / backing vocal split (roformer only) |
| `--stems-dir` | path | off | Keep RoFormer stems and reuse them on later runs |
| `--backing-db` | dB | `-1.5` | Backing vocals level vs the original song |
| `-o, --output` | path | same as input | Output directory |
| `--save-wav` | flag | off | Also save lossless WAV master |

## How It Works

1. **Analysis** — Local DSP analyses spectral centroid, bass energy, transient density, stereo width, and dynamic range to auto-tune mixing parameters
2. **Separation** — BS-RoFormer-SW splits stereo into 6 stems, then a karaoke RoFormer splits the vocals into lead and backing (Demucs: 4 stems)
3. **Crossover** — Linear-phase FIR filters split each stem at 80 Hz (LFE) and 500 Hz (heights)
4. **Decorrelation** — Cascaded allpass filters create unique phase responses for each surround/height channel
5. **Spatial Routing** — Stems are placed in the 7.1.4 sound field with proper gain staging, Haas-effect delays, and decorrelated bleed
6. **Normalization** — Peak normalize to −1.0 dBFS with soft-knee limiting
7. **Encoding** — FFmpeg encodes to EAC3 (7.1.4) and/or AAC (5.1) with correct channel layout metadata

## Architecture

```
spatial_audio/
├── __init__.py          # Package metadata
├── config.py            # Constants, presets, channel layout
├── analyzer.py          # Local DSP analysis (replaces Gemini)
├── separator.py         # RoFormer (audio-separator) and Demucs wrappers
├── mixer.py             # 7.1.4 spatial mixer (core engine)
├── encoder.py           # FFmpeg encoding pipeline
└── dsp/
    ├── __init__.py
    ├── crossover.py     # Linear-phase FIR crossover filters
    ├── decorrelation.py # Allpass decorrelation engine
    └── utils.py         # Gain staging, normalization, helpers
gui.py                   # CustomTkinter GUI
main.py                  # CLI entry point
```

## License

MIT

The RoFormer weights are community models distributed through [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) and UVR; they are downloaded at run time, not included here, and come with their own terms.
