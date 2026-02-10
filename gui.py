import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import sys
import os
# Import from the main script
from stereo_to_multichannel import stereo_to_5_1

# Configure appearance
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        # Schedule the update on the main thread
        self.widget.after(0, self._write_safe, str)

    def _write_safe(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
        
    def flush(self):
        pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Stereo to 5.1 Converter")
        self.geometry("700x550")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Title
        self.label_title = ctk.CTkLabel(self, text="Stereo to 5.1 AI Converter", font=ctk.CTkFont(size=24, weight="bold"))
        self.label_title.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Input Frame
        self.frame_input = ctk.CTkFrame(self)
        self.frame_input.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.frame_input.grid_columnconfigure(1, weight=1)

        self.btn_file = ctk.CTkButton(self.frame_input, text="Select File", command=self.select_file)
        self.btn_file.grid(row=0, column=0, padx=10, pady=10)

        self.btn_folder = ctk.CTkButton(self.frame_input, text="Select Folder (Batch)", command=self.select_folder)
        self.btn_folder.grid(row=0, column=2, padx=10, pady=10)

        self.label_selected = ctk.CTkLabel(self.frame_input, text="No file selected", text_color="gray")
        self.label_selected.grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 10))

        # Options Frame
        self.frame_options = ctk.CTkFrame(self)
        self.frame_options.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        self.frame_options.grid_columnconfigure(0, weight=1)
        
        self.switch_ai = ctk.CTkSwitch(self.frame_options, text="Use AI Analysis (Gemini)", onvalue=True, offvalue=False)
        self.switch_ai.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.switch_ai.select() # Default to True

        self.label_model = ctk.CTkLabel(self.frame_options, text="Demucs Model:")
        self.label_model.grid(row=0, column=1, padx=(10, 5), pady=10)
        
        self.seg_model = ctk.CTkSegmentedButton(self.frame_options, values=["htdemucs", "htdemucs_ft"])
        self.seg_model.grid(row=0, column=2, padx=(0, 20), pady=10)
        self.seg_model.set("htdemucs_ft") # Default

        # Log Area
        self.textbox_log = ctk.CTkTextbox(self, width=600, height=200)
        self.textbox_log.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        self.textbox_log.configure(state="disabled")

        # Redirect stdout/stderr
        sys.stdout = TextRedirector(self.textbox_log, "stdout")
        sys.stderr = TextRedirector(self.textbox_log, "stderr")

        # Action Button
        self.btn_convert = ctk.CTkButton(self, text="Start Conversion", command=self.start_conversion_thread, state="disabled", height=40, font=ctk.CTkFont(size=16, weight="bold"))
        self.btn_convert.grid(row=4, column=0, padx=20, pady=20)

        self.selected_path = None
        self.is_batch = False

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac *.mp3 *.m4a")])
        if file_path:
            self.selected_path = file_path
            self.is_batch = False
            self.label_selected.configure(text=f"Selected File: {os.path.basename(file_path)}", text_color=("black", "white"))
            self.btn_convert.configure(state="normal")

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.selected_path = folder_path
            self.is_batch = True
            self.label_selected.configure(text=f"Selected Folder: {folder_path}", text_color=("black", "white"))
            self.btn_convert.configure(state="normal")

    def start_conversion_thread(self):
        self.btn_convert.configure(state="disabled", text="Processing...")
        self.btn_file.configure(state="disabled")
        self.btn_folder.configure(state="disabled")
        self.switch_ai.configure(state="disabled")
        self.seg_model.configure(state="disabled")
        
        use_ai = self.switch_ai.get()
        model_name = self.seg_model.get()
        thread = threading.Thread(target=self.run_conversion, args=(use_ai, model_name))
        thread.start()

    def run_conversion(self, use_ai, model_name):
        try:
            if self.is_batch:
                self.process_batch(self.selected_path, use_ai, model_name)
            else:
                self.process_single(self.selected_path, use_ai, model_name)
            print("\nAll tasks completed successfully!")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.after(0, self.reset_ui)

    def reset_ui(self):
        self.btn_convert.configure(state="normal", text="Start Conversion")
        self.btn_file.configure(state="normal")
        self.btn_folder.configure(state="normal")
        self.switch_ai.configure(state="normal")
        self.seg_model.configure(state="normal")

    def process_single(self, input_path, use_ai, model_name):
        print(f"Starting conversion for: {input_path} (AI: {use_ai}, Model: {model_name})")
        stereo_to_5_1(input_path, use_ai=use_ai, model_name=model_name)

    def process_batch(self, input_dir, use_ai, model_name):
        supported_exts = ['.wav', '.flac', '.mp3', '.m4a', '.ogg']
        files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in supported_exts]
        
        print(f"Found {len(files)} audio files in {input_dir}")
        
        for i, f in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] Processing {f} (AI: {use_ai}, Model: {model_name})...")
            input_path = os.path.join(input_dir, f)
            try:
                stereo_to_5_1(input_path, use_ai=use_ai, model_name=model_name)
            except Exception as e:
                print(f"Failed to convert {f}: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
