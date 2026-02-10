"""
Spatial Audio Upmixer — CustomTkinter GUI.

Modern dark-mode interface with:
  - File / folder selection
  - Format selector (7.1.4 / 5.1 / Both)
  - Quality presets (Low / Medium / High)
  - Demucs model selector
  - Progress tracking with step indicators
  - Real-time log output
"""

from __future__ import annotations

import os
import sys
import threading
import time

import customtkinter as ctk
from tkinter import filedialog

from spatial_audio.config import (
    PRESETS,
    DEMUCS_MODELS,
    DEFAULT_DEMUCS_MODEL,
    SAMPLE_RATE,
)
from spatial_audio.analyzer import analyse, adapt_preset
from spatial_audio.separator import separate
from spatial_audio.mixer import mix_to_714
from spatial_audio.encoder import Encoder, check_ffmpeg

# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TextRedirector:
    """Redirect stdout/stderr to a CTkTextbox (thread-safe)."""

    def __init__(self, widget: ctk.CTkTextbox, tag: str = "stdout") -> None:
        self.widget = widget
        self.tag = tag

    def write(self, text: str) -> None:
        self.widget.after(0, self._write_safe, text)

    def _write_safe(self, text: str) -> None:
        self.widget.configure(state="normal")
        self.widget.insert("end", text)
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Spatial Audio Upmixer")
        self.geometry("780x680")
        self.minsize(600, 500)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)  # log area expands

        self.selected_path: str | None = None
        self.is_batch = False
        self._processing = False

        self._build_header()
        self._build_input_frame()
        self._build_options_frame()
        self._build_progress_frame()
        self._build_log_area()
        self._build_action_button()

        # Redirect console output to the log area
        sys.stdout = TextRedirector(self.log_box, "stdout")
        sys.stderr = TextRedirector(self.log_box, "stderr")

        # Check FFmpeg on startup
        self.after(200, self._check_ffmpeg)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_header(self) -> None:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="ew")

        ctk.CTkLabel(
            frame,
            text="Spatial Audio Upmixer",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(side="left")

        ctk.CTkLabel(
            frame,
            text="7.1.4 Immersive \u00b7 AirPods Pro",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(side="left", padx=(12, 0), pady=(6, 0))

    def _build_input_frame(self) -> None:
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            frame, text="Select File", width=120,
            command=self._select_file,
        ).grid(row=0, column=0, padx=10, pady=10)

        self.lbl_selected = ctk.CTkLabel(
            frame, text="No file selected", text_color="gray",
        )
        self.lbl_selected.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        ctk.CTkButton(
            frame, text="Select Folder", width=120,
            command=self._select_folder,
        ).grid(row=0, column=2, padx=10, pady=10)

    def _build_options_frame(self) -> None:
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")

        # Row 0: Format
        ctk.CTkLabel(frame, text="Format:").grid(
            row=0, column=0, padx=(15, 5), pady=8, sticky="w"
        )
        self.seg_format = ctk.CTkSegmentedButton(
            frame, values=["7.1.4", "5.1", "Both"]
        )
        self.seg_format.grid(row=0, column=1, padx=5, pady=8, sticky="w")
        self.seg_format.set("7.1.4")

        # Row 0: Quality
        ctk.CTkLabel(frame, text="Quality:").grid(
            row=0, column=2, padx=(20, 5), pady=8, sticky="w"
        )
        self.seg_quality = ctk.CTkSegmentedButton(
            frame, values=["Low", "Medium", "High"]
        )
        self.seg_quality.grid(row=0, column=3, padx=5, pady=8, sticky="w")
        self.seg_quality.set("Medium")

        # Row 1: Model + Save WAV
        ctk.CTkLabel(frame, text="Model:").grid(
            row=1, column=0, padx=(15, 5), pady=(0, 8), sticky="w"
        )
        self.seg_model = ctk.CTkSegmentedButton(
            frame, values=list(DEMUCS_MODELS.keys())
        )
        self.seg_model.grid(row=1, column=1, padx=5, pady=(0, 8), sticky="w")
        self.seg_model.set(DEFAULT_DEMUCS_MODEL)

        self.chk_wav = ctk.CTkCheckBox(frame, text="Save WAV master")
        self.chk_wav.grid(
            row=1, column=2, columnspan=2, padx=(20, 15), pady=(0, 8), sticky="w"
        )

    def _build_progress_frame(self) -> None:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=3, column=0, padx=20, pady=(5, 0), sticky="ew")
        frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(frame)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5)
        self.progress_bar.set(0)

        self.lbl_progress = ctk.CTkLabel(
            frame, text="Ready", font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.lbl_progress.grid(row=1, column=0, sticky="w", padx=5, pady=(2, 0))

    def _build_log_area(self) -> None:
        self.log_box = ctk.CTkTextbox(
            self, height=250,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.log_box.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        self.log_box.configure(state="disabled")

    def _build_action_button(self) -> None:
        self.btn_convert = ctk.CTkButton(
            self,
            text="Start Conversion",
            height=44,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._start_conversion,
            state="disabled",
        )
        self.btn_convert.grid(row=5, column=0, padx=20, pady=(5, 15))

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------
    def _select_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.flac *.mp3 *.m4a *.ogg *.aac")]
        )
        if path:
            self.selected_path = path
            self.is_batch = False
            self.lbl_selected.configure(
                text=f"File: {os.path.basename(path)}",
                text_color=("black", "white"),
            )
            self.btn_convert.configure(state="normal")

    def _select_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.selected_path = path
            self.is_batch = True
            count = sum(
                1 for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            )
            self.lbl_selected.configure(
                text=f"Folder: {path} ({count} files)",
                text_color=("black", "white"),
            )
            self.btn_convert.configure(state="normal")

    # ------------------------------------------------------------------
    # FFmpeg check
    # ------------------------------------------------------------------
    def _check_ffmpeg(self) -> None:
        path = check_ffmpeg()
        if path is None:
            self.lbl_progress.configure(
                text="Warning: FFmpeg not found - install it and add to PATH",
                text_color="red",
            )
        else:
            self.lbl_progress.configure(
                text=f"FFmpeg: {path}", text_color="gray"
            )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def _start_conversion(self) -> None:
        if self._processing:
            return
        self._processing = True
        self._set_ui_enabled(False)
        self.progress_bar.set(0)

        thread = threading.Thread(target=self._run_conversion, daemon=True)
        thread.start()

    def _run_conversion(self) -> None:
        try:
            encoder = Encoder()

            fmt = self.seg_format.get().lower()
            quality = self.seg_quality.get().lower()
            model = self.seg_model.get()
            save_wav = bool(self.chk_wav.get())

            if self.is_batch:
                self._run_batch(self.selected_path, encoder, fmt, quality, model, save_wav)
            else:
                self._run_single(
                    self.selected_path, encoder, fmt, quality, model, save_wav
                )

            self._update_progress("All done!", 1.0)
            print("\nAll tasks completed successfully!")

        except Exception as e:
            print(f"\nError: {e}")
            self._update_progress(f"Error: {e}", 0)
        finally:
            self._processing = False
            self.after(0, lambda: self._set_ui_enabled(True))

    def _run_single(
        self,
        input_path: str,
        encoder: Encoder,
        fmt: str,
        quality: str,
        model: str,
        save_wav: bool,
    ) -> None:
        import soundfile as sf

        filename = os.path.basename(input_path)
        base_name = os.path.splitext(filename)[0]
        output_dir = os.path.dirname(input_path) or "."

        # Step 1 - Analysis
        self._update_progress("Analysing audio...", 0.05)
        audio_input, input_sr = sf.read(input_path, dtype="float64")
        analysis = analyse(audio_input, input_sr)
        print(f"Analysis: {analysis.description}")

        base_preset = PRESETS[quality]
        preset = adapt_preset(base_preset, analysis)

        # Step 2 - Separation
        self._update_progress("Separating stems (Demucs)...", 0.10)
        stems = separate(
            input_path, model_name=model, target_sr=SAMPLE_RATE,
            progress_callback=lambda m: print(m),
        )

        # Step 3 - Mixing
        self._update_progress("Mixing to 7.1.4...", 0.55)
        mix_714, sr = mix_to_714(
            stems, preset,
            progress_callback=lambda m: print(m),
        )

        # Step 4 - Encoding
        self._update_progress("Encoding...", 0.80)

        if fmt in ("7.1.4", "both"):
            out = os.path.join(output_dir, f"{base_name}_spatial_714.wav")
            encoder.encode_714(
                mix_714, sr, out,
                progress_callback=lambda m: print(m),
            )

        if fmt in ("5.1", "both"):
            out = os.path.join(output_dir, f"{base_name}_spatial_51.m4a")
            encoder.encode_51(
                mix_714, sr, out,
                progress_callback=lambda m: print(m),
            )

        if save_wav:
            out = os.path.join(output_dir, f"{base_name}_spatial_714_wav_master.wav")
            encoder.save_wav_master(
                mix_714, sr, out,
                progress_callback=lambda m: print(m),
            )

    def _run_batch(
        self,
        input_dir: str,
        encoder: Encoder,
        fmt: str,
        quality: str,
        model: str,
        save_wav: bool,
    ) -> None:
        files = sorted(
            f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print("No supported audio files found.")
            return

        total = len(files)
        print(f"Batch processing: {total} files\n")

        for i, f in enumerate(files):
            print(f"\n[{i+1}/{total}] {f}")
            frac = i / total
            self._update_progress(f"[{i+1}/{total}] {f}", frac)
            try:
                self._run_single(
                    os.path.join(input_dir, f),
                    encoder, fmt, quality, model, save_wav,
                )
            except Exception as e:
                print(f"  Error: {e}")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _update_progress(self, text: str, fraction: float) -> None:
        self.after(0, lambda: self.progress_bar.set(fraction))
        self.after(0, lambda: self.lbl_progress.configure(text=text))

    def _set_ui_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.btn_convert.configure(
            state=state,
            text="Start Conversion" if enabled else "Processing...",
        )
        self.seg_format.configure(state=state)
        self.seg_quality.configure(state=state)
        self.seg_model.configure(state=state)
        self.chk_wav.configure(state=state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
