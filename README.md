# Stereo to 5.1 AI Converter

An advanced audio upmixing tool that converts stereo music into true 5.1 Surround Sound. It leverages **Demucs** for AI stem separation and **Google Gemini** to analyze the track and dynamically adjust mix parameters.

## Features

- **AI Stem Separation**: Uses Meta's Demucs (Hybrid Transformer) to isolate Vocals, Drums, Bass, and Other instruments.
- **Smart Mix Analysis**: Google Gemini analyzes the audio style to optimize spatial parameters (Gain, Delay, Crossover).
- **Advanced DSP**:
  - **Center**: Isolated vocals with high-pass filtering.
  - **LFE**: Dedicated Low Frequency Effects channel combining Bass and Kick drum.
  - **Surrounds**: Ambience extraction with intelligent delay and widening.
  - **Crossover**: Clean frequency separation between LFE and main channels.
  - **Normalization**: Auto-levels output to -1.0 dB.
- **Modern GUI**: Built with CustomTkinter, featuring Dark Mode and Batch Processing.
- **Apple Compatibility**: Outputs to 5.1 AAC (`.m4a`), natively supported by iPhone and Apple TV.

## Requirements

- **Python 3.10+**
- **FFmpeg**: Must be installed and added to your system PATH.
- **Google Gemini API Key**: Required for the "Smart Analysis" feature.

## Configuration

1. **Get a Gemini API Key**:
    - Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and create a new API key.

2. **Set up Environment Variables**:
    - Create a file named `.env` in the project root directory.
    - Add your API key to the file:

        ```env
        GEMINI_API_KEY=your_api_key_here
        ```

    - You can use `.env.example` as a template.

## Installation

1. **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd Multicanaux
    ```

2. **Install Dependencies**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

    *Note: This project requires `demucs`, `google-genai`, `customtkinter`, `numpy`, `scipy`, and `soundfile`.*

3. **Install FFmpeg**:
    - **Windows**: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract, and add the `bin` folder to your Environment Variables.
    - **Mac**: `brew install ffmpeg`
    - **Linux**: `sudo apt install ffmpeg`

## Usage

### Graphical Interface (Recommended)

Launch the modern GUI:

```bash
python gui.py
```

- **Select File/Folder**: Choose a single track or a folder for batch processing.
- **Use AI Analysis**: Toggle Gemini analysis on/off.
- **Demucs Model**: Choose between `htdemucs` (Standard) and `htdemucs_ft` (Fine-tuned, better quality but slower).
- **Start Conversion**: Sit back and let the AI work.

### Command Line

You can also use the script directly in the terminal:

```bash
python stereo_to_multichannel.py "input.wav" "output.m4a" --model htdemucs_ft
```

**Options:**

- `--no-ai`: Disable Gemini analysis and use default static parameters.
- `--model`: Choose separation model (`htdemucs` or `htdemucs_ft`).

## How it Works

1. **Analysis**: Gemini listens to a snippet and determines the genre and optimal spatial settings (e.g., vocal width, surround delay).
2. **Separation**: Demucs splits the stereo track into 4 stems (Vocals, Drums, Bass, Other).
3. **Spatial Mixing**:
    - **Front L/R**: Drums + Other + Vocal bleed.
    - **Center**: Pure Vocals.
    - **LFE**: Bass + Kick Drum (Low-passed).
    - **Surrounds**: Other + Drums (High-passed & Delayed).
4. **Encoding**: Recombines channels into a 5.1 AAC file using FFmpeg.
