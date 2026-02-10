# Spatial Audio Upmixer

Convert stereo music into **7.1.4 immersive spatial audio** optimised for **AirPods Pro 3** and Apple Spatial Audio. Uses **Demucs v4** for AI stem separation and advanced DSP for professional-quality spatial mixing.

## What's New (v2.0)

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

- **AI Stem Separation** — Demucs Hybrid Transformer isolates vocals, drums, bass, and instruments
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

## Requirements

- **Python 3.10+**
- **FFmpeg** — must be in PATH ([download](https://www.gyan.dev/ffmpeg/builds/))
- **~2 GB disk** for Demucs model (downloaded automatically on first run)
- **GPU recommended** for fast Demucs separation (CUDA-capable NVIDIA GPU)

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
- Choose Demucs model: **htdemucs** (fast) / **htdemucs_ft** (best quality)
- Optionally check **Save WAV master** for lossless output
- Click **Start Conversion**

### Command Line

```bash
# Default: 7.1.4, medium quality, htdemucs_ft
python main.py song.wav

# Both formats, high quality
python main.py song.mp3 --format both --quality high

# 5.1 only, fast model
python main.py song.flac --format 5.1 --model htdemucs

# Batch process a folder + save WAV masters
python main.py ./music/ --format both --save-wav

# Output to a specific directory
python main.py song.wav -o ./output/
```

### Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `-f, --format` | `7.1.4`, `5.1`, `both` | `7.1.4` | Output format |
| `-q, --quality` | `low`, `medium`, `high` | `medium` | Processing quality |
| `-m, --model` | `htdemucs`, `htdemucs_ft` | `htdemucs_ft` | Demucs model |
| `-o, --output` | path | same as input | Output directory |
| `--save-wav` | flag | off | Also save lossless WAV master |

## How It Works

1. **Analysis** — Local DSP analyses spectral centroid, bass energy, transient density, stereo width, and dynamic range to auto-tune mixing parameters
2. **Separation** — Demucs v4 splits stereo into 4 stems (vocals, drums, bass, other)
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
├── separator.py         # Demucs wrapper
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
