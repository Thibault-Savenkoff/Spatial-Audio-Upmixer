"""
Spatial Audio Upmixer — CLI entry point.

Converts stereo music to 7.1.4 immersive audio for AirPods Pro 3.

Usage:
    python main.py input.wav
    python main.py input.mp3 --format both --quality high
    python main.py ./my_music_folder/ --format 5.1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import soundfile as sf

from spatial_audio.config import (
    PRESETS,
    DEFAULT_DEMUCS_MODEL,
    DEMUCS_MODELS,
    SAMPLE_RATE,
)
from spatial_audio.analyzer import analyse, adapt_preset
from spatial_audio.separator import separate
from spatial_audio.mixer import mix_to_714
from spatial_audio.encoder import Encoder, check_ffmpeg


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".wma"}


def process_file(
    input_path: str,
    output_dir: str | None,
    output_format: str,
    quality: str,
    model: str,
    save_wav: bool,
    encoder: Encoder,
) -> None:
    """Process a single audio file through the full pipeline."""
    start_time = time.time()
    filename = os.path.basename(input_path)
    base_name = os.path.splitext(filename)[0]

    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"Quality: {quality} | Model: {model} | Format: {output_format}")
    print(f"{'='*60}")

    # --- Step 1: Read input for analysis ---
    print("\n--- Step 1: Analysing audio ---")
    audio_input, input_sr = sf.read(input_path, dtype="float64")
    analysis = analyse(audio_input, input_sr)
    print(f"  Analysis: {analysis.description}")
    print(f"  Spectral centroid: {analysis.spectral_centroid_hz:.0f} Hz")
    print(f"  Bass energy ratio: {analysis.bass_energy_ratio:.2f}")
    print(f"  Stereo width: {analysis.stereo_width:.2f}")
    print(f"  Dynamic range: {analysis.dynamic_range_db:.1f} dB")

    # --- Step 2: Adapt preset ---
    base_preset = PRESETS[quality]
    preset = adapt_preset(base_preset, analysis)

    # --- Step 3: Stem separation ---
    print("\n--- Step 2: Stem separation (Demucs) ---")
    stems = separate(input_path, model_name=model, target_sr=SAMPLE_RATE)

    # --- Step 4: Spatial mixing ---
    print("\n--- Step 3: Spatial mixing (7.1.4) ---")
    mix_714, sr = mix_to_714(stems, preset)

    # --- Step 5: Encoding ---
    print("\n--- Step 4: Encoding ---")

    outputs = []

    if output_format in ("7.1.4", "both"):
        out_path = os.path.join(output_dir, f"{base_name}_spatial_714.wav")
        result = encoder.encode_714(mix_714, sr, out_path)
        outputs.append(result)

    if output_format in ("5.1", "both"):
        out_path = os.path.join(output_dir, f"{base_name}_spatial_51.m4a")
        result = encoder.encode_51(mix_714, sr, out_path)
        outputs.append(result)

    if save_wav:
        out_path = os.path.join(output_dir, f"{base_name}_spatial_714_wav_master.wav")
        result = encoder.save_wav_master(mix_714, sr, out_path)
        outputs.append(result)

    elapsed = time.time() - start_time
    print(f"\n--- Done in {elapsed:.1f}s ---")
    for p in outputs:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  Output: {p} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial Audio Upmixer — Convert stereo to 7.1.4 immersive audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py song.wav                        # Default: 7.1.4, medium quality
  python main.py song.mp3 --format both          # Output both 7.1.4 and 5.1
  python main.py song.flac --quality high         # High quality processing
  python main.py ./music/ --format 5.1            # Batch process a folder
  python main.py song.wav --save-wav              # Also save lossless WAV master
        """,
    )

    parser.add_argument(
        "input",
        help="Path to input audio file or directory (batch mode)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: same as input)",
        default=None,
    )
    parser.add_argument(
        "-f", "--format",
        choices=["7.1.4", "5.1", "both"],
        default="7.1.4",
        help="Output format (default: 7.1.4)",
    )
    parser.add_argument(
        "-q", "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Processing quality (default: medium)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=list(DEMUCS_MODELS.keys()),
        default=DEFAULT_DEMUCS_MODEL,
        help=f"Demucs model (default: {DEFAULT_DEMUCS_MODEL})",
    )
    parser.add_argument(
        "--save-wav",
        action="store_true",
        help="Also save a lossless 7.1.4 WAV master",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: '{args.input}' not found.")
        sys.exit(1)

    # Check FFmpeg
    ffmpeg_path = check_ffmpeg()
    if ffmpeg_path is None:
        print("Error: FFmpeg not found in PATH.")
        print("Download: https://www.gyan.dev/ffmpeg/builds/")
        sys.exit(1)
    print(f"FFmpeg: {ffmpeg_path}")

    encoder = Encoder(ffmpeg_path)

    # Process
    if os.path.isdir(args.input):
        # Batch mode
        files = sorted(
            f for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"No supported audio files found in {args.input}")
            sys.exit(1)

        print(f"Found {len(files)} audio files")
        output_dir = args.output or args.input

        for i, f in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")
            try:
                process_file(
                    os.path.join(args.input, f),
                    output_dir,
                    args.format,
                    args.quality,
                    args.model,
                    args.save_wav,
                    encoder,
                )
            except Exception as e:
                print(f"Error processing {f}: {e}")
    else:
        # Single file
        process_file(
            args.input,
            args.output,
            args.format,
            args.quality,
            args.model,
            args.save_wav,
            encoder,
        )


if __name__ == "__main__":
    main()
