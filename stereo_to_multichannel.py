import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import argparse
import os
import subprocess
import shutil
import demucs.separate
import json
from google import genai
from dotenv import load_dotenv

import tempfile

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Avertissement : Échec de l'initialisation du client Gemini : {e}")
else:
    print("Avertissement : GEMINI_API_KEY introuvable dans les variables d'environnement. L'analyse de l'IA sera désactivée.")

def get_ai_spatial_config(audio_path):
    """
    Upload le fichier sur AI Studio pour que Gemini l'analyse
    et nous renvoie les meilleurs paramètres de mixage.
    """
    if not client:
        print("Erreur : Client IA non initialisé (Clé API manquante). Utilisation des paramètres par défaut.")
        return {"center_vocals_gain": 1.0, "lfe_cutoff": 100, "surround_delay_ms": 15, "surround_gain": 0.8, "vocals_front_width": 0.2}

    print(f"--- Étape 1 : Analyse du morceau par l'IA ---")
    try:
        # Upload vers Google AI Studio
        sample_file = client.files.upload(file=audio_path)
        
        prompt = """
        Analyse ce morceau. Je vais le transformer en 5.1. 
        Propose des réglages optimaux en format JSON uniquement :
        {
          "center_vocals_gain": float (0.8 à 1.3),
          "lfe_cutoff": int (80 à 120),
          "surround_delay_ms": int (10 à 30),
          "surround_gain": float (0.5 à 1.0),
          "vocals_front_width": float (0.1 à 0.4),
          "description": "Analyse rapide du style"
        }
        """
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[sample_file, prompt]
        )
        # Extraction du JSON dans la réponse
        if response and response.text:
            json_text = response.text.replace('```json', '').replace('```', '').strip()
            config = json.loads(json_text)
        else:
            raise Exception("Empty response from AI model")
        print(f"Analyse terminée : {config['description']}")
        return config
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            print("Erreur IA : Quota dépassé (429). Utilisation des paramètres par défaut.")
        elif "404" in error_msg:
             print("Erreur IA : Modèle non trouvé (404). Utilisation des paramètres par défaut.")
        else:
            print(f"Erreur IA : {error_msg[:100]}... Utilisation des paramètres par défaut.")
        return {"center_vocals_gain": 1.0, "lfe_cutoff": 100, "surround_delay_ms": 15, "surround_gain": 0.8, "vocals_front_width": 0.2}

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def separate_stems(input_file, output_dir, model_name="htdemucs_ft"):
    print(f"Separating stems using Demucs ({model_name})...")
    # Run demucs separation
    # -n htdemucs: use the high quality hybrid transformer model
    # --two-stems=vocals: separate into vocals and other (faster) - wait, we want full separation
    # Default is 4 stems: drums, bass, other, vocals
    
    import sys
    cmd = [sys.executable, "-m", "demucs", "-n", model_name, "--out", output_dir, input_file]
    subprocess.run(cmd, check=True)
    
    # Construct paths to separated files
    filename = os.path.splitext(os.path.basename(input_file))[0]
    stem_dir = os.path.join(output_dir, model_name, filename)
    
    stems = {
        "vocals": os.path.join(stem_dir, "vocals.wav"),
        "drums": os.path.join(stem_dir, "drums.wav"),
        "bass": os.path.join(stem_dir, "bass.wav"),
        "other": os.path.join(stem_dir, "other.wav")
    }
    return stems

def stereo_to_5_1(input_file, output_file=None, use_ai=True, model_name="htdemucs_ft"):
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_5.1.m4a"

    # 1. Analyse IA
    if use_ai:
        ai_config = get_ai_spatial_config(input_file)
    else:
        print("Mode sans IA : Utilisation des paramètres par défaut.")
        ai_config = {"center_vocals_gain": 1.0, "lfe_cutoff": 100, "surround_delay_ms": 15, "surround_gain": 0.8, "vocals_front_width": 0.2}

    # 2. Séparation Demucs
    # Utilisation d'un dossier temporaire système pour la propreté
    temp_dir = tempfile.mkdtemp(prefix="demucs_sep_")
    
    try:
        stems_paths = separate_stems(input_file, temp_dir, model_name=model_name)
        
        vocals, samplerate = sf.read(stems_paths["vocals"])
        drums, _ = sf.read(stems_paths["drums"])
        bass, _ = sf.read(stems_paths["bass"])
        other, _ = sf.read(stems_paths["other"])
        
        print(f"--- Étape 3 : Mixage Spatial 5.1 (Paramètres IA) ---")

        # Fréquence de coupure pour le crossover (LFE vs Autres)
        crossover_freq = ai_config['lfe_cutoff']

        # Center: Voix principale ajustée par l'IA
        center_raw = ((vocals[:, 0] + vocals[:, 1]) / 2.0) * ai_config['center_vocals_gain']
        # On nettoie le centre des basses fréquences qui vont au LFE
        center = highpass_filter(center_raw, crossover_freq, samplerate, order=4)

        # LFE: Basse + Kick filtré (Lowpass)
        bass_mono = (bass[:, 0] + bass[:, 1]) / 2.0
        drums_mono = (drums[:, 0] + drums[:, 1]) / 2.0
        kick_lfe = lowpass_filter(drums_mono, cutoff=crossover_freq, fs=samplerate, order=4)
        # On filtre aussi la basse pour qu'elle soit propre dans le LFE
        bass_lfe = lowpass_filter(bass_mono, cutoff=crossover_freq, fs=samplerate, order=4)
        lfe = bass_lfe + kick_lfe

        # Front L/R: Batterie + Instruments + un peu de voix (largeur)
        v_width = ai_config['vocals_front_width']
        fl_raw = drums[:, 0] + other[:, 0] + (vocals[:, 0] * v_width)
        fr_raw = drums[:, 1] + other[:, 1] + (vocals[:, 1] * v_width)
        
        # Application du filtre Passe-Haut (Highpass) pour laisser la place au LFE
        fl = highpass_filter(fl_raw, crossover_freq, samplerate, order=4)
        fr = highpass_filter(fr_raw, crossover_freq, samplerate, order=4)

        # Surrounds: Instruments (ambiance) + légère percussion
        s_gain = ai_config['surround_gain']
        sl_raw = other[:, 0] * s_gain + (drums[:, 0] * 0.1)
        sr_raw = other[:, 1] * s_gain + (drums[:, 1] * 0.1)
        
        # Filtre Passe-Haut sur les surrounds aussi
        sl_filtered = highpass_filter(sl_raw, crossover_freq, samplerate, order=4)
        sr_filtered = highpass_filter(sr_raw, crossover_freq, samplerate, order=4)
        
        # Délai surround ajusté par l'IA
        delay_samples = int(ai_config['surround_delay_ms'] * samplerate / 1000)
        sl = np.pad(sl_filtered, (delay_samples, 0), mode='constant')[:sl_filtered.shape[0]]
        sr = np.pad(sr_filtered, (delay_samples, 0), mode='constant')[:sr_filtered.shape[0]]

        # Stackage des 6 canaux
        multichannel_data = np.column_stack((fl, fr, center, lfe, sl, sr))

        # Normalisation (Peak Normalization à -1.0 dB)
        print("Normalisation du mixage...")
        max_val = np.max(np.abs(multichannel_data))
        if max_val > 0:
            target_db = -1.0
            target_amp = 10 ** (target_db / 20)
            multichannel_data = multichannel_data * (target_amp / max_val)

        # Écriture temporaire et conversion FFMPEG
        temp_wav = os.path.join(temp_dir, "temp_output_51.wav")
        sf.write(temp_wav, multichannel_data, samplerate)
        
        print(f"Conversion finale en AAC 5.1 pour iPhone...")
        cmd_ffmpeg = [
            "ffmpeg", "-y", "-i", temp_wav,
            "-c:a", "aac", "-b:a", "320k", "-ac", "6",
            output_file
        ]
        subprocess.run(cmd_ffmpeg, check=True)
        # Pas besoin de remove temp_wav ici car on supprime tout le dossier temp_dir après
        print(f"Succès ! Fichier prêt : {output_file}")

    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temp dir {temp_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert stereo audio to 5.1 multichannel using AI Separation.")
    parser.add_argument("input", help="Path to input stereo file (wav, flac, etc.) or directory")
    parser.add_argument("output", nargs='?', help="Path to output 5.1 file (wav or m4a) or directory")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI analysis and use default parameters")
    parser.add_argument("--model", default="htdemucs_ft", choices=["htdemucs", "htdemucs_ft"], help="Demucs model to use (default: htdemucs_ft)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' not found.")
    else:
        if os.path.isdir(args.input):
            # Batch mode
            input_dir = args.input
            output_dir = args.output if args.output else input_dir
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            supported_exts = ['.wav', '.flac', '.mp3', '.m4a', '.ogg']
            files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in supported_exts]
            
            print(f"Found {len(files)} audio files in {input_dir}")
            
            for f in files:
                input_path = os.path.join(input_dir, f)
                # Construct output filename
                base_name = os.path.splitext(f)[0]
                output_filename = f"{base_name}_5.1.m4a"
                output_path = os.path.join(output_dir, output_filename)
                
                try:
                    stereo_to_5_1(input_path, output_path, use_ai=not args.no_ai, model_name=args.model)
                except Exception as e:
                    print(f"Failed to convert {f}: {e}")
                    
        else:
            # Single file mode
            output_file = args.output
            stereo_to_5_1(args.input, output_file, use_ai=not args.no_ai, model_name=args.model)
