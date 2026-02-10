"""
FFmpeg encoding pipeline for spatial audio output.

Produces:
  1. 7.1.4 WAV (24-bit) with proper channel layout metadata
  2. 5.1 AAC in .m4a — compatibility fallback for older devices

Note: FFmpeg's compressed audio codecs (EAC3, ALAC, FLAC, AAC) all max out
at 8 channels.  True 7.1.4 (12 channels) can only be stored losslessly in
WAV with correct channel layout tags.  For Dolby Atmos delivery, import the
7.1.4 WAV into a DAW (Logic Pro, Pro Tools) and render with Dolby tools.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from .config import (
    FFMPEG_LAYOUT_714,
    FFMPEG_LAYOUT_51,
    SAMPLE_RATE,
    SUBTYPE_WAV,
)
from .mixer import downmix_714_to_51


def check_ffmpeg() -> str | None:
    """Check if FFmpeg is available and return its path, or None."""
    path = shutil.which("ffmpeg")
    if path is None:
        return None
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return path
    except Exception:
        pass
    return None


def check_ffprobe() -> str | None:
    """Check if ffprobe is available."""
    path = shutil.which("ffprobe")
    return path


class Encoder:
    """FFmpeg-based spatial audio encoder."""

    def __init__(self, ffmpeg_path: str | None = None) -> None:
        self.ffmpeg = ffmpeg_path or check_ffmpeg()
        if self.ffmpeg is None:
            raise RuntimeError(
                "FFmpeg not found. Install FFmpeg and add it to PATH.\n"
                "Download: https://www.gyan.dev/ffmpeg/builds/"
            )

    def encode_714(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        bitrate: str = "640k",
        progress_callback=None,
    ) -> str:
        """Save 7.1.4 audio as a WAV with correct channel layout metadata.

        FFmpeg's compressed codecs (EAC3, ALAC, FLAC, AAC) all cap at
        8 channels, so 12-channel 7.1.4 must be stored as lossless WAV.
        FFmpeg is used to re-mux with proper ``7.1.4`` channel layout tags
        so DAWs and players recognise the channel mapping.

        Parameters
        ----------
        audio : np.ndarray
            Shape ``(N, 12)`` — 7.1.4 audio data.
        sample_rate : int
        output_path : str
            Output file path.  Extension will be forced to ``.wav``.
        bitrate : str
            Unused (kept for API compatibility).
        progress_callback : callable, optional

        Returns
        -------
        str — path to the output file.
        """
        def _log(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Force .wav — compressed codecs can't handle 12 channels
        base, _ = os.path.splitext(output_path)
        output_path = base + ".wav"

        # Write raw WAV first
        tmp_dir = tempfile.mkdtemp(prefix="spatial_enc_")
        tmp_wav = os.path.join(tmp_dir, "mix_714_raw.wav")

        try:
            _log("Writing 7.1.4 WAV (24-bit, 48 kHz)...")
            sf.write(tmp_wav, audio, sample_rate, subtype=SUBTYPE_WAV)

            # Re-mux with FFmpeg to stamp the correct 7.1.4 channel layout
            _log("Tagging channel layout as 7.1.4...")
            cmd = [
                self.ffmpeg, "-y",
                "-i", tmp_wav,
                "-c:a", "pcm_s24le",
                "-channel_layout", "7.1.4",
                output_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                # Fallback: just copy the raw WAV (no layout tag)
                _log("FFmpeg tagging failed — saving untagged WAV...")
                shutil.copy2(tmp_wav, output_path)

            _log(f"7.1.4 output: {output_path}")
            return output_path

        finally:
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except OSError:
                    pass

    def encode_51(
        self,
        audio_714: np.ndarray,
        sample_rate: int,
        output_path: str,
        bitrate: str = "320k",
        progress_callback=None,
    ) -> str:
        """Downmix 7.1.4 to 5.1 and encode to AAC in .m4a.

        Parameters
        ----------
        audio_714 : np.ndarray
            Shape ``(N, 12)`` — 7.1.4 audio data.
        sample_rate : int
        output_path : str
            Output file path.  Extension will be forced to ``.m4a``.
        bitrate : str
            AAC bitrate (default ``"320k"``).
        progress_callback : callable, optional

        Returns
        -------
        str — path to the output file.
        """
        def _log(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Force .m4a extension
        base, _ = os.path.splitext(output_path)
        output_path = base + ".m4a"

        # Downmix
        _log("Downmixing 7.1.4 → 5.1...")
        audio_51 = downmix_714_to_51(audio_714)

        tmp_dir = tempfile.mkdtemp(prefix="spatial_enc51_")
        tmp_wav = os.path.join(tmp_dir, "mix_51.wav")

        try:
            sf.write(tmp_wav, audio_51, sample_rate, subtype=SUBTYPE_WAV)

            _log(f"Encoding to AAC 5.1 ({bitrate})...")
            cmd = [
                self.ffmpeg, "-y",
                "-i", tmp_wav,
                "-af", f"channelmap=channel_layout={FFMPEG_LAYOUT_51}",
                "-c:a", "aac",
                "-b:a", bitrate,
                output_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                # Fallback without channelmap
                cmd_fallback = [
                    self.ffmpeg, "-y",
                    "-i", tmp_wav,
                    "-c:a", "aac",
                    "-b:a", bitrate,
                    "-ac", "6",
                    output_path,
                ]
                result = subprocess.run(
                    cmd_fallback, capture_output=True, text=True, timeout=300,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"FFmpeg 5.1 encoding failed:\n{result.stderr[-500:]}"
                    )

            _log(f"5.1 output: {output_path}")
            return output_path

        finally:
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except OSError:
                    pass

    def save_wav_master(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        progress_callback=None,
    ) -> str:
        """Save a lossless 7.1.4 WAV master (24-bit, 48 kHz).

        Parameters
        ----------
        audio : np.ndarray
            Shape ``(N, 12)``.
        sample_rate : int
        output_path : str
            Output path.  Extension will be forced to ``.wav``.
        """
        def _log(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        base, _ = os.path.splitext(output_path)
        output_path = base + ".wav"

        _log(f"Saving lossless WAV master ({audio.shape[1]}ch, 24-bit)...")
        sf.write(output_path, audio, sample_rate, subtype=SUBTYPE_WAV)
        _log(f"WAV master: {output_path}")
        return output_path
