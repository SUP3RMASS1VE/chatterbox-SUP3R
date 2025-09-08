import random
import numpy as np
import torch

# Try to import multilingual TTS only
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    MULTILINGUAL_AVAILABLE = True
    print("🌍 Multilingual TTS support detected")
except ImportError:
    MULTILINGUAL_AVAILABLE = False
    print("❌ Multilingual TTS not available. Please install latest chatterbox package.")
import gradio as gr
import os
import subprocess
import sys
import warnings
import re
import json
import io
from datetime import datetime
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import tempfile
import shutil
# Add new imports for advanced audio processing
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import fft, ifft, fftfreq
import librosa
import soundfile as sf
import base64
import requests
from pathlib import Path
from urllib.parse import urlparse
import threading

# Suppress the specific LoRACompatibleLinear deprecation warning
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*deprecated.*", category=FutureWarning)

# Suppress torch CUDA sdp_kernel deprecation warning
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*deprecated.*", category=FutureWarning)

# Suppress LlamaModel attention implementation warning
warnings.filterwarnings("ignore", message=".*LlamaModel is using LlamaSdpaAttention.*", category=UserWarning)

# Suppress past_key_values tuple deprecation warning
warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*deprecated.*", category=UserWarning)

# Suppress additional transformers warnings
warnings.filterwarnings("ignore", message=".*LlamaModel.*LlamaSdpaAttention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*We detected that you are passing.*past_key_values.*", category=UserWarning)

# Suppress Gradio audio conversion warning
warnings.filterwarnings("ignore", message=".*Trying to convert audio automatically.*", category=UserWarning)

# More aggressive warning suppression for transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# Suppress all warnings containing these key phrases
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*LlamaModel.*")
warnings.filterwarnings("ignore", message=".*LlamaSdpaAttention.*")

# Suppress torch/contextlib warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*contextlib.*")

# Suppress torch.load warnings related to TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*weights_only.*argument.*not explicitly passed.*")
warnings.filterwarnings("ignore", message=".*forcing weights_only=False.*")

# Suppress checkpoint manager warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*checkpoint_manager.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*perth.*")

# Suppress chatterbox TTS model warnings
warnings.filterwarnings("ignore", message=".*Detected.*repetition of token.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*forcing EOS token.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=".*chatterbox.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*alignment_stream_analyzer.*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MULTILINGUAL_MODEL = None

# Supported languages for multilingual model
SUPPORTED_LANGUAGES = {
    'ar': 'Arabic',
    'da': 'Danish', 
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'fi': 'Finnish',
    'fr': 'French',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ms': 'Malay',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'tr': 'Turkish',
    'zh': 'Chinese'
}

# Multilingual model download configuration
MULTILINGUAL_MODEL_FILES = {
    'Cangjie5_TC': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/Cangjie5_TC.json',
    'conds': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt',
    'mtl_tokenizer': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/mtl_tokenizer.json',
    's3gen': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.pt',
    't3_23lang': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_23lang.safetensors',
    've': 'https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.pt'
}

# Model download directory
MODEL_DOWNLOAD_DIR = "models/multilingual"
download_status = {"status": "ready", "progress": 0, "current_file": "", "total_files": 0}

def ensure_model_download_dir():
    """Ensure the model download directory exists."""
    if not os.path.exists(MODEL_DOWNLOAD_DIR):
        os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)
        print(f"📁 Created model download directory: {os.path.abspath(MODEL_DOWNLOAD_DIR)}")

def check_multilingual_models_exist():
    """Check if multilingual model files already exist."""
    ensure_model_download_dir()
    missing_files = []
    existing_files = []
    
    for model_name, url in MULTILINGUAL_MODEL_FILES.items():
        # Extract the correct filename with extension from the URL
        filename_with_ext = url.split('/')[-1]
        model_path = os.path.join(MODEL_DOWNLOAD_DIR, filename_with_ext)
        if os.path.exists(model_path):
            existing_files.append(model_name)
        else:
            missing_files.append(model_name)
    
    return existing_files, missing_files

def download_file_with_progress(url, filepath, filename):
    """Download a file with progress tracking."""
    try:
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        download_status["progress"] = progress
        
        print(f"✅ Downloaded {filename} successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def download_multilingual_models():
    """Download all multilingual model files."""
    global download_status
    
    try:
        ensure_model_download_dir()
        
        # Check what needs to be downloaded
        existing_files, missing_files = check_multilingual_models_exist()
        
        if not missing_files:
            download_status["status"] = "complete"
            download_status["progress"] = 100
            return "✅ All multilingual model files already exist!"
        
        download_status["status"] = "downloading"
        download_status["total_files"] = len(missing_files)
        
        print(f"🌍 Starting download of {len(missing_files)} multilingual model files...")
        
        for i, model_name in enumerate(missing_files):
            download_status["current_file"] = model_name
            download_status["progress"] = (i / len(missing_files)) * 100
            
            url = MULTILINGUAL_MODEL_FILES[model_name]
            # Extract the correct filename with extension from the URL
            filename_with_ext = url.split('/')[-1]
            filepath = os.path.join(MODEL_DOWNLOAD_DIR, filename_with_ext)
            
            success = download_file_with_progress(url, filepath, filename_with_ext)
            
            if not success:
                download_status["status"] = "error"
                return f"❌ Failed to download {model_name}"
        
        download_status["status"] = "complete"
        download_status["progress"] = 100
        download_status["current_file"] = ""
        
        print("🎉 All multilingual model files downloaded successfully!")
        return "🎉 Multilingual models downloaded successfully! You can now enable multilingual mode."
        
    except Exception as e:
        download_status["status"] = "error"
        error_msg = f"❌ Download error: {str(e)}"
        print(error_msg)
        return error_msg

def download_models_async():
    """Download models in a separate thread to avoid blocking the UI."""
    threading.Thread(target=download_multilingual_models, daemon=True).start()

def get_download_status():
    """Get current download status for UI updates."""
    status = download_status["status"]
    progress = download_status.get("progress", 0)
    current_file = download_status.get("current_file", "")
    total_files = download_status.get("total_files", 0)
    
    if status == "ready":
        return "📋 Ready to download multilingual models"
    elif status == "downloading":
        file_info = f" ({current_file})" if current_file else ""
        return f"📥 Downloading... {progress:.1f}%{file_info}"
    elif status == "complete":
        return "✅ Download complete! Multilingual models ready."
    elif status == "error":
        return "❌ Download failed. Check console for details."
    else:
        return f"Status: {status}"

def check_model_files_status():
    """Check and return the status of model files."""
    existing_files, missing_files = check_multilingual_models_exist()
    
    if not missing_files:
        status_text = f"✅ All multilingual model files present ({len(existing_files)} files)\n"
        status_text += f"📁 Location: {os.path.abspath(MODEL_DOWNLOAD_DIR)}\n"
        status_text += f"Files: {', '.join(existing_files)}"
        return status_text
    else:
        status_text = f"⚠️ Missing {len(missing_files)} model files\n"
        if existing_files:
            status_text += f"✅ Found: {', '.join(existing_files)}\n"
        status_text += f"❌ Missing: {', '.join(missing_files)}\n"
        status_text += "Click 'Download Models' to get the missing files."
        return status_text

def load_model_manually():
    """Manually load the multilingual model into memory."""
    global MULTILINGUAL_MODEL
    
    try:
        if MULTILINGUAL_MODEL is not None:
            return "✅ Model already loaded in memory!", True
        
        print("🚀 Manually loading multilingual model...")
        MULTILINGUAL_MODEL = get_or_load_model()
        
        if MULTILINGUAL_MODEL is not None:
            return "✅ Model loaded successfully! Ready for speech generation.", True
        else:
            return "❌ Failed to load model. Please check your model files.", False
            
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        return error_msg, False

def check_model_loaded_status():
    """Check if the model is loaded and return status."""
    global MULTILINGUAL_MODEL
    
    if MULTILINGUAL_MODEL is not None:
        return "✅ Model loaded in memory - Ready for generation!"
    else:
        existing_files, missing_files = check_multilingual_models_exist()
        if not missing_files:
            return "📁 Models downloaded but not loaded - Click 'Load Model' to use them"
        else:
            return "📥 Models not downloaded - Use download section to get them"

def should_show_load_button():
    """Check if the load model button should be visible."""
    global MULTILINGUAL_MODEL
    
    if MULTILINGUAL_MODEL is not None:
        return False  # Hide button if model is already loaded
    
    existing_files, missing_files = check_multilingual_models_exist()
    return len(missing_files) == 0  # Show button if all models are downloaded

# --- Voice Presets System ---
PRESETS_FILE = "voice_presets.json"
PRESETS_AUDIO_DIR = "saved_voices"

def ensure_presets_dir():
    """Ensure the presets audio directory exists."""
    if not os.path.exists(PRESETS_AUDIO_DIR):
        os.makedirs(PRESETS_AUDIO_DIR)
        print(f"📁 Created presets directory: {os.path.abspath(PRESETS_AUDIO_DIR)}")

def copy_reference_audio(ref_audio_path, preset_name):
    """Copy reference audio to presets directory."""
    if not ref_audio_path or not os.path.exists(ref_audio_path):
        return None
    
    try:
        ensure_presets_dir()
        
        # Get file extension
        _, ext = os.path.splitext(ref_audio_path)
        if not ext:
            ext = '.wav'  # Default extension
        
        # Create unique filename for this preset
        safe_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        audio_filename = f"{safe_name}_voice{ext}"
        audio_path = os.path.join(PRESETS_AUDIO_DIR, audio_filename)
        
        # Copy the file
        shutil.copy2(ref_audio_path, audio_path)
        
        print(f"🎤 Copied reference audio to: {os.path.abspath(audio_path)}")
        return audio_path
        
    except Exception as e:
        print(f"❌ Error copying reference audio: {e}")
        return None

def load_voice_presets():
    """Load voice presets from JSON file."""
    try:
        preset_path = os.path.abspath(PRESETS_FILE)
        print(f"🔍 Looking for presets file at: {preset_path}")
        
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, 'r') as f:
                presets = json.load(f)
                print(f"✅ Loaded {len(presets)} voice presets from file")
                
                # Verify audio files still exist
                for name, preset in presets.items():
                    audio_path = preset.get('ref_audio_path', '')
                    if audio_path and os.path.exists(audio_path):
                        print(f"  🎤 Preset '{name}' has valid audio file")
                    elif audio_path:
                        print(f"  ⚠️ Preset '{name}' audio file missing: {audio_path}")
                
                return presets
        else:
            print("📝 No presets file found, starting with empty presets")
    except Exception as e:
        print(f"❌ Error loading presets: {e}")
    return {}

def save_voice_presets(presets):
    """Save voice presets to JSON file."""
    try:
        preset_path = os.path.abspath(PRESETS_FILE)
        print(f"💾 Saving voice presets to: {preset_path}")
        
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
        
        print(f"✅ Successfully saved {len(presets)} voice presets")
        return True
    except Exception as e:
        print(f"❌ Error saving presets: {e}")
        return False

def save_voice_preset(name, settings):
    """Save a new voice preset with reference audio."""
    presets = load_voice_presets()
    
    # Copy the reference audio file if provided
    ref_audio_path = settings.get('ref_audio', '')
    saved_audio_path = None
    
    if ref_audio_path:
        saved_audio_path = copy_reference_audio(ref_audio_path, name)
        if not saved_audio_path:
            print(f"⚠️ Warning: Could not save reference audio for preset '{name}'")
    
    presets[name] = {
        'exaggeration': settings['exaggeration'],
        'temperature': settings['temperature'],
        'cfg_weight': settings['cfg_weight'],
        'chunk_size': settings['chunk_size'],
        'language': settings.get('language', 'en'),  # Save language setting
        'ref_audio_path': saved_audio_path or '',  # Path to saved audio file
        'original_ref_audio': ref_audio_path or '',  # Original path for reference
        'created': datetime.now().isoformat()
    }
    
    success = save_voice_presets(presets)
    if success:
        if saved_audio_path:
            print(f"🎭 Saved voice preset '{name}' with custom voice audio")
        else:
            print(f"🎭 Saved voice preset '{name}' (parameters only, no custom voice)")
    return success

def load_voice_preset(name):
    """Load a voice preset by name."""
    presets = load_voice_presets()
    preset = presets.get(name, None)
    if preset:
        audio_path = preset.get('ref_audio_path', '')
        if audio_path and os.path.exists(audio_path):
            print(f"🎭 Loaded voice preset '{name}' with custom voice: {audio_path}")
        else:
            print(f"🎭 Loaded voice preset '{name}' (parameters only)")
    return preset

def delete_voice_preset(name):
    """Delete a voice preset and its audio file."""
    presets = load_voice_presets()
    if name in presets:
        preset = presets[name]
        
        # Delete associated audio file
        audio_path = preset.get('ref_audio_path', '')
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"🗑️ Deleted audio file: {audio_path}")
            except Exception as e:
                print(f"⚠️ Could not delete audio file: {e}")
        
        del presets[name]
        success = save_voice_presets(presets)
        if success:
            print(f"🗑️ Deleted voice preset '{name}'")
        return success
    return False

def get_preset_names():
    """Get list of all preset names."""
    presets = load_voice_presets()
    names = list(presets.keys())
    print(f"📋 Available voice presets: {names}")
    return names

# --- Audio Effects ---
def apply_reverb(audio, sr, room_size=0.3, damping=0.5, wet_level=0.3):
    """Apply more noticeable reverb effect to audio."""
    try:
        # Create multiple delayed versions for richer reverb
        reverb_audio = audio.copy()
        
        # Early reflections (multiple short delays)
        delays = [0.01, 0.02, 0.03, 0.05, 0.08]  # Multiple delay times in seconds
        gains = [0.6, 0.4, 0.3, 0.2, 0.15]      # Corresponding gains
        
        for delay_time, gain in zip(delays, gains):
            delay_samples = int(sr * delay_time)
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain * (1 - damping)
                reverb_audio += delayed * wet_level
        
        # Late reverberation (longer decay)
        late_delay = int(sr * 0.1)  # 100ms
        if late_delay < len(audio):
            late_reverb = np.zeros_like(audio)
            late_reverb[late_delay:] = audio[:-late_delay] * 0.3 * (1 - damping)
            reverb_audio += late_reverb * wet_level * room_size
        
        return np.clip(reverb_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Reverb error: {e}")
        return audio

def apply_echo(audio, sr, delay=0.3, decay=0.5):
    """Apply echo effect to audio."""
    try:
        delay_samples = int(sr * delay)
        if delay_samples < len(audio):
            echo_audio = audio.copy()
            echo_audio[delay_samples:] += audio[:-delay_samples] * decay
            return np.clip(echo_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Echo error: {e}")
    return audio

def apply_pitch_shift(audio, sr, semitones):
    """Apply simple pitch shift (speed change method)."""
    try:
        if semitones == 0:
            return audio
        
        # Simple pitch shift by resampling (changes speed too)
        factor = 2 ** (semitones / 12.0)
        indices = np.arange(0, len(audio), factor)
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return audio

# --- Advanced Audio Processing ---

def test_equalizer_functionality():
    """Test function to verify equalizer is working correctly."""
    try:
        # Create test signal
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create test audio with multiple frequencies
        test_audio = (
            0.3 * np.sin(2 * np.pi * 100 * t) +    # Bass
            0.3 * np.sin(2 * np.pi * 1000 * t) +   # Mid
            0.3 * np.sin(2 * np.pi * 5000 * t)     # High
        )
        
        # Test EQ settings (boost bass, cut mid, boost high)
        eq_bands = {
            'sub_bass': 0,
            'bass': 6,
            'low_mid': 0,
            'mid': -6,
            'high_mid': 0,
            'presence': 6,
            'brilliance': 0
        }
        
        # Apply equalizer
        processed = apply_equalizer(test_audio, sr, eq_bands)
        
        if processed is not None and len(processed) == len(test_audio):
            print("✅ Equalizer test passed - processing working correctly")
            return True
        else:
            print("❌ Equalizer test failed - output issue")
            return False
            
    except Exception as e:
        print(f"❌ Equalizer test failed with error: {e}")
        return False

def apply_noise_reduction(audio, sr, noise_factor=0.02, spectral_floor=0.002):
    """Apply spectral subtraction noise reduction to clean up audio."""
    try:
        print("🧹 Applying noise reduction...")
        
        # Convert to frequency domain
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Estimate noise profile from first 0.5 seconds (assumed to be quieter)
        noise_frame_count = min(int(0.5 * sr / 512), magnitude.shape[1] // 4)
        noise_profile = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)
        
        # Apply spectral subtraction with over-subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.001  # Floor factor
        
        # Subtract noise profile
        clean_magnitude = magnitude - alpha * noise_profile
        
        # Apply spectral floor to prevent artifacts
        spectral_floor_level = beta * magnitude
        clean_magnitude = np.maximum(clean_magnitude, spectral_floor_level)
        
        # Reconstruct signal
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=512)
        
        # Ensure same length as input
        if len(clean_audio) < len(audio):
            clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))
        else:
            clean_audio = clean_audio[:len(audio)]
        
        return np.clip(clean_audio, -1.0, 1.0)
        
    except Exception as e:
        print(f"Noise reduction error: {e}")
        return audio

def apply_equalizer(audio, sr, eq_bands):
    """Apply multi-band equalizer to audio."""
    try:
        if not any(gain != 0 for gain in eq_bands.values()):
            return audio  # No EQ applied
            
        print("🎛️ Applying equalizer...")
        
        # Define frequency bands (Hz) with better ranges
        band_ranges = {
            'sub_bass': (20, 80),
            'bass': (80, 250), 
            'low_mid': (250, 800),
            'mid': (800, 2500),
            'high_mid': (2500, 5000),
            'presence': (5000, 10000),
            'brilliance': (10000, 20000)
        }
        
        # Start with original audio
        processed_audio = audio.copy().astype(np.float64)
        
        for band_name, gain_db in eq_bands.items():
            if gain_db == 0 or band_name not in band_ranges:
                continue
                
            low_freq, high_freq = band_ranges[band_name]
            
            # Ensure frequencies are within Nyquist limit
            nyquist = sr / 2
            low_freq = min(low_freq, nyquist * 0.95)
            high_freq = min(high_freq, nyquist * 0.95)
            
            if low_freq >= high_freq:
                continue
            
            # Design bandpass filter with lower order to reduce artifacts
            low_norm = max(low_freq / nyquist, 0.001)  # Avoid zero frequency
            high_norm = min(high_freq / nyquist, 0.999)  # Avoid Nyquist frequency
            
            try:
                # Use second-order filter to reduce artifacts
                if band_name == 'sub_bass':
                    # Low-pass filter for sub bass
                    b, a = butter(2, high_norm, btype='low')
                elif band_name == 'brilliance':
                    # High-pass filter for brilliance
                    b, a = butter(2, low_norm, btype='high')
                else:
                    # Bandpass filter for mid bands
                    b, a = butter(2, [low_norm, high_norm], btype='band')
                
                # Filter the audio
                band_audio = filtfilt(b, a, audio.astype(np.float64))
                
                # Convert dB gain to linear gain
                gain_linear = 10 ** (gain_db / 20.0)
                
                # Apply gain correctly: multiply band by gain, then blend with original
                if gain_db > 0:
                    # Boost: add the boosted band energy
                    boost_amount = (gain_linear - 1.0)
                    processed_audio += band_audio * boost_amount
                else:
                    # Cut: reduce the band energy in the processed audio
                    cut_amount = (1.0 - gain_linear)
                    processed_audio -= band_audio * cut_amount
                
                print(f"  Applied {gain_db:+.1f}dB to {band_name} ({low_freq}-{high_freq}Hz)")
                
            except Exception as band_error:
                print(f"EQ band {band_name} error: {band_error}")
                continue
        
        # Normalize to prevent clipping while preserving dynamics
        max_val = np.abs(processed_audio).max()
        if max_val > 0.95:
            processed_audio = processed_audio * (0.95 / max_val)
        
        return processed_audio.astype(np.float32)
        
    except Exception as e:
        print(f"Equalizer error: {e}")
        return audio

def apply_spatial_audio(audio, sr, azimuth=0, elevation=0, distance=1.0):
    """Apply 3D spatial audio positioning using simple HRTF-like processing."""
    try:
        if azimuth == 0 and elevation == 0 and distance == 1.0:
            return audio  # No spatial processing needed
            
        print(f"🎧 Applying 3D spatial audio (az: {azimuth}°, el: {elevation}°, dist: {distance})")
        
        # Convert to stereo if mono
        if len(audio.shape) == 1:
            # Simple stereo panning based on azimuth
            azimuth_rad = np.radians(azimuth)
            
            # Calculate left/right gains using equal-power panning
            left_gain = np.cos((azimuth_rad + np.pi/2) / 2)
            right_gain = np.sin((azimuth_rad + np.pi/2) / 2)
            
            # Apply distance attenuation
            distance_gain = 1.0 / max(distance, 0.1)
            left_gain *= distance_gain
            right_gain *= distance_gain
            
            # Apply elevation effect (simple high-frequency filtering)
            processed_audio = audio.copy()
            if elevation != 0:
                # High-pass filter for upward elevation, low-pass for downward
                elevation_factor = elevation / 90.0  # Normalize to -1 to 1
                if elevation_factor > 0:
                    # Upward - enhance high frequencies
                    cutoff = 2000 + elevation_factor * 3000
                    b, a = butter(2, cutoff / (sr/2), btype='high')
                    high_freq = filtfilt(b, a, audio)
                    processed_audio = audio + high_freq * elevation_factor * 0.3
                else:
                    # Downward - enhance low frequencies  
                    cutoff = 1000 + abs(elevation_factor) * 2000
                    b, a = butter(2, cutoff / (sr/2), btype='low')
                    low_freq = filtfilt(b, a, audio)
                    processed_audio = audio + low_freq * abs(elevation_factor) * 0.3
            
            # Create stereo output
            stereo_audio = np.column_stack([
                processed_audio * left_gain,
                processed_audio * right_gain
            ])
            
            return np.clip(stereo_audio, -1.0, 1.0)
        
        return audio  # Return original if already stereo or other format
        
    except Exception as e:
        print(f"Spatial audio error: {e}")
        return audio

def mix_with_background(speech_audio, sr, background_path, bg_volume=0.3, speech_volume=1.0, fade_in=1.0, fade_out=1.0):
    """Mix generated speech with background music/ambience."""
    try:
        if not background_path or not os.path.exists(background_path):
            return speech_audio
            
        print(f"🎵 Mixing with background audio: {os.path.basename(background_path)}")
        
        # Load background audio
        bg_audio, bg_sr = librosa.load(background_path, sr=sr)
        
        # Ensure speech is 1D
        if len(speech_audio.shape) > 1:
            speech_audio = np.mean(speech_audio, axis=1)
            
        speech_length = len(speech_audio)
        bg_length = len(bg_audio)
        
        # Handle different background audio lengths
        if bg_length < speech_length:
            # Loop background audio if it's shorter
            repeat_count = int(np.ceil(speech_length / bg_length))
            bg_audio = np.tile(bg_audio, repeat_count)[:speech_length]
        else:
            # Trim background audio if it's longer
            bg_audio = bg_audio[:speech_length]
        
        # Apply volume adjustments
        speech_mixed = speech_audio * speech_volume
        bg_mixed = bg_audio * bg_volume
        
        # Apply fades to background
        fade_in_samples = int(fade_in * sr)
        fade_out_samples = int(fade_out * sr)
        
        if fade_in_samples > 0:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            bg_mixed[:fade_in_samples] *= fade_in_curve
            
        if fade_out_samples > 0:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            bg_mixed[-fade_out_samples:] *= fade_out_curve
        
        # Mix the audio
        mixed_audio = speech_mixed + bg_mixed
        
        # Normalize to prevent clipping
        max_val = np.abs(mixed_audio).max()
        if max_val > 0.95:
            mixed_audio = mixed_audio / max_val * 0.95
        
        return mixed_audio
        
    except Exception as e:
        print(f"Background mixing error: {e}")
        return speech_audio

def apply_audio_effects(audio, sr, effects_settings):
    """Apply selected audio effects to the generated audio."""
    processed_audio = audio.copy()
    
    print(f"🎵 Starting audio effects processing...")
    print(f"   Input audio: shape={audio.shape}, max={np.max(np.abs(audio)):.4f}, dtype={audio.dtype}")
    
    # Basic effects (existing)
    if effects_settings.get('enable_reverb', False):
        processed_audio = apply_reverb(
            processed_audio, sr,
            room_size=effects_settings.get('reverb_room', 0.3),
            damping=effects_settings.get('reverb_damping', 0.5),
            wet_level=effects_settings.get('reverb_wet', 0.3)
        )
        print(f"   After reverb: max={np.max(np.abs(processed_audio)):.4f}")
    
    if effects_settings.get('enable_echo', False):
        processed_audio = apply_echo(
            processed_audio, sr,
            delay=effects_settings.get('echo_delay', 0.3),
            decay=effects_settings.get('echo_decay', 0.5)
        )
        print(f"   After echo: max={np.max(np.abs(processed_audio)):.4f}")
    
    if effects_settings.get('enable_pitch', False):
        processed_audio = apply_pitch_shift(
            processed_audio, sr,
            semitones=effects_settings.get('pitch_semitones', 0)
        )
        print(f"   After pitch: max={np.max(np.abs(processed_audio)):.4f}")
    
    # Advanced effects (new)
    if effects_settings.get('enable_noise_reduction', False):
        processed_audio = apply_noise_reduction(processed_audio, sr)
        print(f"   After noise reduction: max={np.max(np.abs(processed_audio)):.4f}")
    
    if effects_settings.get('enable_equalizer', False):
        eq_bands = {
            'sub_bass': effects_settings.get('eq_sub_bass', 0),
            'bass': effects_settings.get('eq_bass', 0),
            'low_mid': effects_settings.get('eq_low_mid', 0),
            'mid': effects_settings.get('eq_mid', 0),
            'high_mid': effects_settings.get('eq_high_mid', 0),
            'presence': effects_settings.get('eq_presence', 0),
            'brilliance': effects_settings.get('eq_brilliance', 0)
        }
        print(f"   EQ settings: {eq_bands}")
        print(f"   Before EQ: max={np.max(np.abs(processed_audio)):.4f}")
        processed_audio = apply_equalizer(processed_audio, sr, eq_bands)
        print(f"   After EQ: max={np.max(np.abs(processed_audio)):.4f}")
    
    if effects_settings.get('enable_spatial', False):
        processed_audio = apply_spatial_audio(
            processed_audio, sr,
            azimuth=effects_settings.get('spatial_azimuth', 0),
            elevation=effects_settings.get('spatial_elevation', 0),
            distance=effects_settings.get('spatial_distance', 1.0)
        )
        print(f"   After spatial: max={np.max(np.abs(processed_audio)):.4f}")
    
    # Background mixing (applied last)
    if effects_settings.get('enable_background', False):
        processed_audio = mix_with_background(
            processed_audio, sr,
            background_path=effects_settings.get('background_path', ''),
            bg_volume=effects_settings.get('bg_volume', 0.3),
            speech_volume=effects_settings.get('speech_volume', 1.0),
            fade_in=effects_settings.get('bg_fade_in', 1.0),
            fade_out=effects_settings.get('bg_fade_out', 1.0)
        )
        print(f"   After background: max={np.max(np.abs(processed_audio)):.4f}")
    
    print(f"🎵 Audio effects processing complete. Final max: {np.max(np.abs(processed_audio)):.4f}")
    return processed_audio

# --- Export Functions ---
EXPORT_DIR = "exports"

def ensure_export_dir():
    """Ensure the export directory exists."""
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
        print(f"📁 Created export directory: {os.path.abspath(EXPORT_DIR)}")

def export_audio(audio_data, sr, format_type="wav", quality="high"):
    """Export audio in different formats and qualities to export folder."""
    try:
        ensure_export_dir()
        
        # Normalize audio to prevent clipping and fuzzy sound
        audio_normalized = np.copy(audio_data)
        
        # Check if audio needs normalization
        max_val = np.abs(audio_normalized).max()
        if max_val > 0:
            # Normalize to 85% of full scale to prevent clipping
            audio_normalized = audio_normalized / max_val * 0.85
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"chatterbox_export_{timestamp}"
        
        if format_type == "wav":
            filename = f"{base_filename}.wav"
            filepath = os.path.join(EXPORT_DIR, filename)
            
            # Implement real quality differences with supported data types
            if quality == "high":
                # High: 16-bit, original sample rate (best quality)
                export_sr = sr
                audio_int = (audio_normalized * 32767).astype(np.int16)
                print(f"🎵 WAV High Quality: 16-bit, {export_sr} Hz")
                
            elif quality == "medium":
                # Medium: 16-bit, half sample rate (smaller file, good quality)
                from scipy import signal
                export_sr = sr // 2
                # Resample to lower sample rate
                num_samples = int(len(audio_normalized) * export_sr / sr)
                audio_resampled = signal.resample(audio_normalized, num_samples)
                audio_int = (audio_resampled * 32767).astype(np.int16)
                print(f"🎵 WAV Medium Quality: 16-bit, {export_sr} Hz (resampled)")
                
            else:  # low
                # Low: 16-bit, quarter sample rate, reduced bit depth simulation
                from scipy import signal
                export_sr = sr // 4
                # Resample to much lower sample rate
                num_samples = int(len(audio_normalized) * export_sr / sr)
                audio_resampled = signal.resample(audio_normalized, num_samples)
                # Simulate lower bit depth by quantizing to fewer levels
                audio_quantized = np.round(audio_resampled * 4096) / 4096  # 12-bit simulation
                audio_int = (audio_quantized * 32767).astype(np.int16)
                print(f"🎵 WAV Low Quality: 16-bit (12-bit simulation), {export_sr} Hz (resampled)")
            
            wavfile.write(filepath, export_sr, audio_int)
            print(f"✅ WAV exported: {filepath}")
            return filepath
            
    except Exception as e:
        print(f"❌ Export error: {e}")
        return None

def handle_export(audio_data, export_quality):
    """Handle audio export and return status message."""
    if audio_data is None:
        return "❌ No audio to export. Generate audio first!"
    
    try:
        # Extract sample rate and audio array from the tuple
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, audio_array = audio_data
        else:
            return "❌ Invalid audio data format"
        
        print(f"🎵 Exporting audio: WAV format, {export_quality} quality")
        print(f"📊 Audio stats: {len(audio_array)} samples, {sr} Hz, max level: {np.abs(audio_array).max():.3f}")
        
        # Export the audio (only WAV now)
        export_path = export_audio(audio_array, sr, "wav", export_quality)
        
        if export_path:
            relative_path = os.path.relpath(export_path)
            file_size = os.path.getsize(export_path) / 1024 / 1024  # Size in MB
            
            # Show quality info
            if export_quality == "high":
                quality_info = " (16-bit, full sample rate - best quality)"
            elif export_quality == "medium":
                quality_info = " (16-bit, half sample rate - balanced)"
            else:
                quality_info = " (16-bit, quarter sample rate - smallest file)"
            
            return f"✅ Audio exported successfully!\n📁 Saved to: {relative_path}\n📊 File size: {file_size:.1f} MB{quality_info}"
        else:
            return "❌ Export failed"
            
    except Exception as e:
        return f"❌ Export error: {str(e)}"

def clear_hf_credentials():
    """Clear any cached Hugging Face credentials that might cause 401 errors."""
    try:
        # Clear environment variables
        os.environ.pop('HF_TOKEN', None)
        os.environ.pop('HUGGINGFACE_HUB_TOKEN', None)
        
        # Try to logout using CLI
        subprocess.run([sys.executable, '-m', 'huggingface_hub.commands.huggingface_cli', 'logout'], 
                      capture_output=True, check=False)
        print("🔧 Cleared Hugging Face credentials")
        return True
    except Exception as e:
        print(f"⚠️ Could not clear HF credentials: {e}")
        return False

def get_or_load_model():
    """Loads the multilingual ChatterBox model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MULTILINGUAL_MODEL
    
    # Check if multilingual TTS is available
    if not MULTILINGUAL_AVAILABLE:
        raise RuntimeError("❌ Multilingual TTS not available. Please install latest chatterbox package.")
    
    # Load multilingual model
    if MULTILINGUAL_MODEL is None:
        print("🌍 Multilingual model not loaded, initializing...")
        try:
            # Check if we have local downloaded models
            existing_files, missing_files = check_multilingual_models_exist()
            if not missing_files:
                print(f"📁 Using local multilingual models from: {os.path.abspath(MODEL_DOWNLOAD_DIR)}")
                # Load from local directory if all files exist
                MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_local(MODEL_DOWNLOAD_DIR, device=DEVICE)
            else:
                print("🌐 Loading multilingual model from Hugging Face...")
                print("💡 Tip: Download models locally using the download section for faster loading")
                MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
            
            if hasattr(MULTILINGUAL_MODEL, 'to') and str(MULTILINGUAL_MODEL.device) != DEVICE:
                MULTILINGUAL_MODEL.to(DEVICE)
            print(f"🌍 Multilingual model loaded successfully. Internal device: {getattr(MULTILINGUAL_MODEL, 'device', 'N/A')}")
        except Exception as e:
            error_str = str(e)
            # Check if it's a 401 authentication error
            if "401" in error_str and "Unauthorized" in error_str:
                print("🔧 Detected 401 authentication error. Clearing credentials and retrying...")
                clear_hf_credentials()
                try:
                    # Retry loading the model
                    existing_files, missing_files = check_multilingual_models_exist()
                    if not missing_files:
                        print(f"📁 Retrying with local multilingual models from: {os.path.abspath(MODEL_DOWNLOAD_DIR)}")
                        MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_local(MODEL_DOWNLOAD_DIR, device=DEVICE)
                    else:
                        print("🌐 Retrying multilingual model from Hugging Face...")
                        print("💡 Tip: Download models locally using the download section for faster loading")
                        MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
                    
                    if hasattr(MULTILINGUAL_MODEL, 'to') and str(MULTILINGUAL_MODEL.device) != DEVICE:
                        MULTILINGUAL_MODEL.to(DEVICE)
                    print(f"🌍 Multilingual model loaded successfully after clearing credentials. Internal device: {getattr(MULTILINGUAL_MODEL, 'device', 'N/A')}")
                except Exception as retry_error:
                    print(f"❌ Error loading multilingual model after retry: {retry_error}")
                    raise
            else:
                print(f"❌ Error loading multilingual model: {e}")
                raise
    
    return MULTILINGUAL_MODEL

# Skip model loading at startup - models will be loaded on-demand
print("🚀 App ready - multilingual models will be loaded when needed")
print("💡 Use the download section to get multilingual models for 23-language support")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def split_text_into_chunks(text: str, max_chunk_length: int = 300) -> list[str]:
    """
    Splits text into chunks that respect sentence boundaries and word limits.
    
    Args:
        text: The input text to split
        max_chunk_length: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_length:
        return [text]
    
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) + 2 > max_chunk_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split by commas or phrases
                if len(sentence) > max_chunk_length:
                    # Split by commas or natural breaks
                    parts = re.split(r'[,;]+', sentence)
                    for part in parts:
                        part = part.strip()
                        if len(current_chunk) + len(part) + 2 > max_chunk_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += (", " if current_chunk else "") + part
                else:
                    current_chunk = sentence
        else:
            current_chunk += (". " if current_chunk else "") + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    chunk_size_input: int,
    # Language selection
    language_id_input: str = "en",
    # Basic audio effects parameters
    enable_reverb: bool = False,
    reverb_room: float = 0.3,
    reverb_damping: float = 0.5,
    reverb_wet: float = 0.3,
    enable_echo: bool = False,
    echo_delay: float = 0.3,
    echo_decay: float = 0.5,
    enable_pitch: bool = False,
    pitch_semitones: float = 0,
    # Advanced audio effects parameters
    enable_noise_reduction: bool = False,
    enable_equalizer: bool = False,
    eq_sub_bass: float = 0,
    eq_bass: float = 0,
    eq_low_mid: float = 0,
    eq_mid: float = 0,
    eq_high_mid: float = 0,
    eq_presence: float = 0,
    eq_brilliance: float = 0,
    enable_spatial: bool = False,
    spatial_azimuth: float = 0,
    spatial_elevation: float = 0,
    spatial_distance: float = 1.0,
    enable_background: bool = False,
    background_path: str = "",
    bg_volume: float = 0.3,
    speech_volume: float = 1.0,
    bg_fade_in: float = 1.0,
    bg_fade_out: float = 1.0,
    # Conversation mode parameters
    conversation_mode: bool = False,
    conversation_script: str = "",
    conversation_pause: float = 0.8,
    speaker_transition_pause: float = 0.3,
    # Speaker settings (will be passed as JSON string)
    speaker_settings_json: str = "{}",
) -> tuple[tuple[int, np.ndarray], tuple[int, np.ndarray], str]:
    """
    Generates TTS audio using the multilingual ChatterBox model.
    Returns: (audio_output, waveform_data, waveform_info)
    """
    # Load the multilingual model
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("Multilingual TTS model is not loaded.")
    
    # Show which language is being used
    language_name = SUPPORTED_LANGUAGES.get(language_id_input, f"Unknown ({language_id_input})")
    print(f"🌍 Using multilingual model for {language_name} generation")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    # Prepare effects settings
    effects_settings = {
        # Basic effects
        'enable_reverb': enable_reverb,
        'reverb_room': reverb_room,
        'reverb_damping': reverb_damping,
        'reverb_wet': reverb_wet,
        'enable_echo': enable_echo,
        'echo_delay': echo_delay,
        'echo_decay': echo_decay,
        'enable_pitch': enable_pitch,
        'pitch_semitones': pitch_semitones,
        # Advanced effects
        'enable_noise_reduction': enable_noise_reduction,
        'enable_equalizer': enable_equalizer,
        'eq_sub_bass': eq_sub_bass,
        'eq_bass': eq_bass,
        'eq_low_mid': eq_low_mid,
        'eq_mid': eq_mid,
        'eq_high_mid': eq_high_mid,
        'eq_presence': eq_presence,
        'eq_brilliance': eq_brilliance,
        'enable_spatial': enable_spatial,
        'spatial_azimuth': spatial_azimuth,
        'spatial_elevation': spatial_elevation,
        'spatial_distance': spatial_distance,
        'enable_background': enable_background,
        'background_path': background_path,
        'bg_volume': bg_volume,
        'speech_volume': speech_volume,
        'bg_fade_in': bg_fade_in,
        'bg_fade_out': bg_fade_out,
    }

    # Check if conversation mode is enabled
    if conversation_mode and conversation_script.strip():
        print("🎭 Conversation mode activated")
        
        # Parse speaker settings from JSON
        try:
            import json
            speaker_settings = json.loads(speaker_settings_json) if speaker_settings_json else {}
        except:
            speaker_settings = {}
        
        # Generate conversation
        audio_result, info_or_error = generate_conversation_audio(
            conversation_script,
            speaker_settings,
            conversation_pause_duration=conversation_pause,
            speaker_transition_pause=speaker_transition_pause,
            effects_settings=effects_settings if any(effects_settings[key] for key in ['enable_reverb', 'enable_echo', 'enable_pitch', 'enable_noise_reduction', 'enable_equalizer', 'enable_spatial', 'enable_background']) else None,
            use_multilingual=True,
            language_id=language_id_input,
            current_model=current_model
        )
        
        if audio_result is None:
            # Error occurred
            waveform_info = info_or_error
            return (current_model.sr, np.zeros(1000)), (current_model.sr, np.zeros(1000)), waveform_info
        else:
            # Success
            sr, final_audio = audio_result
            waveform_info = format_conversation_info(info_or_error)
            return (sr, final_audio), (sr, final_audio), waveform_info
        
    else:
        # Original single voice mode
        # Split text into manageable chunks
        text_chunks = split_text_into_chunks(text_input, max_chunk_length=chunk_size_input)
        
        if len(text_chunks) == 1:
            print(f"Generating audio for text: '{text_input[:50]}...'")
        else:
            print(f"Generating audio in {len(text_chunks)} chunks for text: '{text_input[:50]}...'")
        
        audio_chunks = []
        
        # Temporarily suppress ALL warnings during generation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i, chunk in enumerate(text_chunks):
                if len(text_chunks) > 1:
                    print(f"Processing chunk {i+1}/{len(text_chunks)}: '{chunk[:30]}...'")
                
                # Generate audio with multilingual parameters
                wav = current_model.generate(
                    chunk,
                    audio_prompt_path=audio_prompt_path_input,
                    language_id=language_id_input,
                    exaggeration=exaggeration_input,
                    temperature=temperature_input,
                    cfg_weight=cfgw_input,
                )
                
                audio_chunks.append(wav.squeeze(0).numpy())
        
        # Concatenate all audio chunks
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            # Add small silence between chunks for natural flow
            silence_samples = int(current_model.sr * 0.05)  # 0.05 second silence
            silence = np.zeros(silence_samples)
            
            concatenated_chunks = []
            for i, chunk in enumerate(audio_chunks):
                concatenated_chunks.append(chunk)
                if i < len(audio_chunks) - 1:  # Don't add silence after the last chunk
                    concatenated_chunks.append(silence)
            
            final_audio = np.concatenate(concatenated_chunks)
        
        # Apply audio effects (both basic and advanced)
        if any(effects_settings[key] for key in ['enable_reverb', 'enable_echo', 'enable_pitch', 'enable_noise_reduction', 'enable_equalizer', 'enable_spatial', 'enable_background']):
            print("Applying audio effects...")
            final_audio = apply_audio_effects(final_audio, current_model.sr, effects_settings)
        
        # Create audio output tuple
        audio_output = (current_model.sr, final_audio)
        
        # Generate waveform analysis
        print("🔍 Performing waveform analysis...")
        _, stats = create_waveform_visualization(final_audio, current_model.sr)
        waveform_info = format_waveform_info(stats)
        
        print("Audio generation and analysis complete.")
        return audio_output, audio_output, waveform_info

# Voice preset management functions for Gradio
def save_current_preset(preset_name, exaggeration, temperature, cfg_weight, chunk_size, ref_audio):
    """Save current settings as a preset including the reference audio."""
    if not preset_name.strip():
        return "❌ Please enter a preset name", gr.update()
    
    settings = {
        'exaggeration': exaggeration,
        'temperature': temperature,
        'cfg_weight': cfg_weight,
        'chunk_size': chunk_size,
        'ref_audio': ref_audio or ''
    }
    
    if save_voice_preset(preset_name.strip(), settings):
        updated_choices = get_preset_names()
        if ref_audio:
            return f"✅ Voice preset '{preset_name}' saved successfully with custom voice!", gr.update(choices=updated_choices, value=None)
        else:
            return f"✅ Preset '{preset_name}' saved (no custom voice audio)", gr.update(choices=updated_choices, value=None)
    else:
        return "❌ Failed to save preset", gr.update()

def load_selected_preset(preset_name):
    """Load selected preset and return its settings including reference audio."""
    if not preset_name:
        return "Please select a preset", None, None, None, None, None
    
    preset = load_voice_preset(preset_name)
    if preset:
        # Use the saved audio path, not the original
        ref_audio_path = preset.get('ref_audio_path', '')
        
        return (
            f"✅ Loaded voice preset '{preset_name}'" + (" with custom voice" if ref_audio_path else ""),
            preset['exaggeration'],
            preset['temperature'], 
            preset['cfg_weight'],
            preset['chunk_size'],
            ref_audio_path if ref_audio_path and os.path.exists(ref_audio_path) else None
        )
    else:
        return "❌ Failed to load preset", None, None, None, None, None

def delete_selected_preset(preset_name):
    """Delete selected preset and its audio file."""
    if not preset_name:
        return "Please select a preset to delete", gr.update()
    
    if delete_voice_preset(preset_name):
        updated_choices = get_preset_names()
        return f"✅ Voice preset '{preset_name}' deleted (including audio file)", gr.update(choices=updated_choices, value=None)
    else:
        return "❌ Failed to delete preset", gr.update()

def refresh_preset_dropdown():
    """Refresh the preset dropdown with current presets."""
    choices = get_preset_names()
    return gr.update(choices=choices, value=None)

# Standard Gradio theme - no custom CSS needed
def get_custom_css():
    """Using Gradio's default theme styling."""
    return ""

def create_waveform_visualization(audio_data, sr=22050):
    """Create enhanced waveform visualization data."""
    if audio_data is None:
        return None, "No audio data available"
    
    try:
        # Ensure audio is 1D
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Calculate waveform statistics
        duration = len(audio_data) / sr
        max_amplitude = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data**2))
        peak_db = 20 * np.log10(max_amplitude + 1e-10)
        rms_db = 20 * np.log10(rms_level + 1e-10)
        
        # Detect zero crossings for pitch estimation
        zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
        estimated_pitch = zero_crossings / (2 * duration)
        
        # Create frequency analysis
        fft_data = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/sr)
        magnitude = np.abs(fft_data)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
        dominant_freq = abs(freqs[dominant_freq_idx])
        
        stats = {
            "duration": f"{duration:.2f}s",
            "sample_rate": f"{sr} Hz",
            "samples": f"{len(audio_data):,}",
            "max_amplitude": f"{max_amplitude:.4f}",
            "peak_level": f"{peak_db:.1f} dB",
            "rms_level": f"{rms_db:.1f} dB",
            "estimated_pitch": f"{estimated_pitch:.1f} Hz",
            "dominant_freq": f"{dominant_freq:.1f} Hz",
            "dynamic_range": f"{peak_db - rms_db:.1f} dB"
        }
        
        return audio_data, stats
        
    except Exception as e:
        return None, f"Error analyzing audio: {str(e)}"

def format_waveform_info(stats):
    """Format waveform statistics for display."""
    if isinstance(stats, str):
        return stats
    
    info_text = f"""
🎵 Audio Analysis:
• Duration: {stats['duration']} | Sample Rate: {stats['sample_rate']} | Samples: {stats['samples']}
• Peak Level: {stats['peak_level']} | RMS Level: {stats['rms_level']} | Dynamic Range: {stats['dynamic_range']}  
• Estimated Pitch: {stats['estimated_pitch']} | Dominant Frequency: {stats['dominant_freq']}
• Max Amplitude: {stats['max_amplitude']}
    """.strip()
    
    return info_text

def analyze_audio_waveform(audio_data):
    """Analyze waveform data and return formatted information."""
    if audio_data is None:
        return "No audio data available for analysis. Generate audio first."
    
    try:
        # Extract sample rate and audio array from the tuple
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, audio_array = audio_data
        else:
            return "Invalid audio data format for analysis."
        
        print(f"🔍 Analyzing audio: {len(audio_array)} samples at {sr} Hz")
        
        # Perform waveform analysis
        _, stats = create_waveform_visualization(audio_array, sr)
        
        return format_waveform_info(stats)
        
    except Exception as e:
        return f"Error analyzing waveform: {str(e)}"

# Initialize CSS (none needed for standard theme)
initial_css = get_custom_css()

# --- Voice Conversation System ---
def parse_conversation_script(script_text):
    """Parse conversation script in Speaker: Text format."""
    try:
        lines = script_text.strip().split('\n')
        conversation = []
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains speaker designation (Speaker: Text format)
            if ':' in line and not line.startswith(' '):
                # Save previous speaker's text if exists
                if current_speaker and current_text:
                    conversation.append({
                        'speaker': current_speaker,
                        'text': current_text.strip()
                    })
                
                # Parse new speaker line
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_speaker = parts[0].strip()
                    current_text = parts[1].strip()
                else:
                    # Invalid format, treat as continuation
                    current_text += " " + line
            else:
                # Continuation of previous speaker's text
                current_text += " " + line
        
        # Add the last speaker's text
        if current_speaker and current_text:
            conversation.append({
                'speaker': current_speaker,
                'text': current_text.strip()
            })
        
        return conversation, None
        
    except Exception as e:
        return [], f"Error parsing conversation: {str(e)}"

def get_speaker_names_from_script(script_text):
    """Extract unique speaker names from conversation script."""
    conversation, error = parse_conversation_script(script_text)
    if error:
        return []
    
    speakers = list(set([item['speaker'] for item in conversation]))
    return sorted(speakers)

def generate_conversation_audio(
    conversation_script,
    speaker_settings,
    conversation_pause_duration=0.8,
    speaker_transition_pause=0.3,
    effects_settings=None,
    use_multilingual=False,
    language_id="en",
    current_model=None
):
    """Generate a complete conversation with multiple voices."""
    try:
        print("🎭 Starting conversation generation...")
        
        # Parse the conversation script
        conversation, parse_error = parse_conversation_script(conversation_script)
        if parse_error:
            return None, f"❌ Script parsing error: {parse_error}"
        
        if not conversation:
            return None, "❌ No valid conversation found in script"
        
        print(f"📝 Parsed {len(conversation)} conversation lines")
        
        # Use the passed model
        if current_model is None:
            return None, "❌ TTS model not available"
        
        conversation_audio_chunks = []
        conversation_info = []
        
        # Generate audio for each conversation line
        for i, line in enumerate(conversation):
            speaker = line['speaker']
            text = line['text']
            
            print(f"🗣️ Generating line {i+1}/{len(conversation)}: {speaker}")
            
            # Get speaker settings
            if speaker not in speaker_settings:
                return None, f"❌ No settings found for speaker '{speaker}'"
            
            settings = speaker_settings[speaker]
            
            # Suppress warnings during generation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Generate audio for this line
                try:
                    # Generate with multilingual parameters using speaker's language
                    speaker_language = settings.get('language', language_id)  # Use speaker's language or fallback to global
                    wav = current_model.generate(
                        text,
                        audio_prompt_path=settings.get('ref_audio', ''),
                        language_id=speaker_language,
                        exaggeration=settings.get('exaggeration', 0.5),
                        temperature=settings.get('temperature', 0.8),
                        cfg_weight=settings.get('cfg_weight', 0.5),
                    )
                    
                    line_audio = wav.squeeze(0).numpy()
                    
                    # Apply individual speaker effects if specified
                    if effects_settings:
                        line_audio = apply_audio_effects(line_audio, current_model.sr, effects_settings)
                    
                    conversation_audio_chunks.append(line_audio)
                    conversation_info.append({
                        'speaker': speaker,
                        'text': text[:50] + ('...' if len(text) > 50 else ''),
                        'duration': len(line_audio) / current_model.sr,
                        'samples': len(line_audio),
                        'language': speaker_language
                    })
                    
                except Exception as gen_error:
                    return None, f"❌ Error generating audio for {speaker}: {str(gen_error)}"
        
        # Combine all audio with proper timing
        print("🎵 Combining conversation audio with proper timing...")
        
        # Calculate pause durations
        conversation_pause_samples = int(current_model.sr * conversation_pause_duration)
        transition_pause_samples = int(current_model.sr * speaker_transition_pause)
        
        final_audio_parts = []
        previous_speaker = None
        
        for i, (audio_chunk, info) in enumerate(zip(conversation_audio_chunks, conversation_info)):
            current_speaker = info['speaker']
            
            # Add audio chunk
            final_audio_parts.append(audio_chunk)
            
            # Add pause after each line (except the last one)
            if i < len(conversation_audio_chunks) - 1:
                next_speaker = conversation_info[i + 1]['speaker']
                
                # Different pause duration based on speaker change
                if current_speaker != next_speaker:
                    # Speaker transition - longer pause
                    pause_samples = conversation_pause_samples
                else:
                    # Same speaker continuing - shorter pause
                    pause_samples = transition_pause_samples
                
                pause_audio = np.zeros(pause_samples)
                final_audio_parts.append(pause_audio)
        
        # Concatenate all parts
        final_conversation_audio = np.concatenate(final_audio_parts)
        
        # Create conversation summary
        total_duration = len(final_conversation_audio) / current_model.sr
        unique_speakers = len(set([info['speaker'] for info in conversation_info]))
        
        # Collect language information
        languages_used = list(set([info['language'] for info in conversation_info]))
        
        summary = {
            'total_lines': len(conversation),
            'unique_speakers': unique_speakers,
            'total_duration': total_duration,
            'speakers': list(set([info['speaker'] for info in conversation_info])),
            'languages_used': languages_used,
            'conversation_info': conversation_info
        }
        
        print(f"✅ Conversation generated: {len(conversation)} lines, {unique_speakers} speakers, {total_duration:.1f}s")
        
        return (current_model.sr, final_conversation_audio), summary
        
    except Exception as e:
        return None, f"❌ Conversation generation error: {str(e)}"

def format_conversation_info(summary):
    """Format conversation summary for display."""
    if isinstance(summary, str):
        return summary
    
    try:
        # Format languages used
        languages_used = summary.get('languages_used', [])
        language_names = [SUPPORTED_LANGUAGES.get(lang, lang) for lang in languages_used]
        
        info_text = f"""
🎭 Conversation Summary:
• Total Lines: {summary['total_lines']} | Speakers: {summary['unique_speakers']} | Duration: {summary['total_duration']:.1f}s
• Speakers: {', '.join(summary['speakers'])}
• Languages: {', '.join(language_names)} ({len(languages_used)} language{'s' if len(languages_used) != 1 else ''})

📝 Line Breakdown:"""
        
        for i, line_info in enumerate(summary['conversation_info'], 1):
            speaker = line_info['speaker']
            text_preview = line_info['text']
            duration = line_info['duration']
            language = line_info.get('language', 'en')
            language_name = SUPPORTED_LANGUAGES.get(language, language)
            info_text += f"\n{i:2d}. {speaker} ({language_name}): \"{text_preview}\" ({duration:.1f}s)"
        
        return info_text.strip()
        
    except Exception as e:
        return f"Error formatting conversation info: {str(e)}"

def update_speaker_settings_from_presets(speakers_text, current_settings_json):
    """Update speaker settings by loading from available presets."""
    try:
        current_settings = json.loads(current_settings_json) if current_settings_json else {}
        available_presets = load_voice_presets()
        
        # Get speaker names from detected speakers text
        speakers = []
        if "Found" in speakers_text and "speakers:" in speakers_text:
            lines = speakers_text.split('\n')[1:]  # Skip first line
            speakers = [line.replace('• ', '').strip() for line in lines if line.strip()]
        
        # Try to match speakers with available presets
        for speaker in speakers:
            if speaker in available_presets:
                preset = available_presets[speaker]
                current_settings[speaker] = {
                    'ref_audio': preset.get('ref_audio_path', ''),
                    'exaggeration': preset.get('exaggeration', 0.5),
                    'temperature': preset.get('temperature', 0.8),
                    'cfg_weight': preset.get('cfg_weight', 0.5)
                }
                print(f"🎭 Auto-loaded preset for speaker '{speaker}'")
        
        return json.dumps(current_settings)
        
    except Exception as e:
        print(f"Error updating speaker settings: {e}")
        return current_settings_json

def setup_speaker_audio_components(script_text):
    """Set up audio components for detected speakers and return visibility updates."""
    speakers = get_speaker_names_from_script(script_text)
    
    # Maximum 5 speakers supported in UI
    audio_components_visibility = [False] * 5
    audio_labels = ["🎤 Speaker Voice"] * 5
    lang_labels = ["🌍 Speaker Language"] * 5
    speaker_controls_visible = False
    
    if speakers:
        speaker_controls_visible = True
        for i, speaker in enumerate(speakers[:5]):  # Limit to 5 speakers
            audio_components_visibility[i] = True
            audio_labels[i] = f"🎤 {speaker}'s Voice"
            lang_labels[i] = f"🌍 {speaker}'s Language"
    
    # Create updates for all audio components
    updates = []
    for i in range(5):
        updates.append(gr.update(
            visible=audio_components_visibility[i],
            label=audio_labels[i],
            value=None  # Clear previous uploads
        ))
    
    # Create updates for all language dropdowns
    for i in range(5):
        updates.append(gr.update(
            visible=audio_components_visibility[i],
            label=lang_labels[i],
            value="en"  # Default to English
        ))
    
    # Add visibility update for the speaker controls row
    updates.append(gr.update(visible=speaker_controls_visible))
    
    # Return speakers list and settings JSON
    updates.append(speakers)
    
    # Initialize speaker settings
    speaker_settings = {}
    available_presets = load_voice_presets()
    
    for speaker in speakers:
        if speaker in available_presets:
            preset = available_presets[speaker]
            speaker_settings[speaker] = {
                'ref_audio': preset.get('ref_audio_path', ''),
                'exaggeration': preset.get('exaggeration', 0.5),
                'temperature': preset.get('temperature', 0.8),
                'cfg_weight': preset.get('cfg_weight', 0.5),
                'language': preset.get('language', 'en')
            }
        else:
            speaker_settings[speaker] = {
                'ref_audio': '',
                'exaggeration': 0.5,
                'temperature': 0.8,
                'cfg_weight': 0.5,
                'language': 'en'
            }
    
    updates.append(json.dumps(speaker_settings))
    
    return updates

def update_speaker_audio_settings(speakers_list, audio1, audio2, audio3, audio4, audio5, current_settings_json):
    """Update speaker settings JSON with uploaded audio files."""
    try:
        current_settings = json.loads(current_settings_json) if current_settings_json else {}
        audio_files = [audio1, audio2, audio3, audio4, audio5]
        
        # Update settings with uploaded audio
        for i, speaker in enumerate(speakers_list[:5]):
            if speaker in current_settings:
                if i < len(audio_files) and audio_files[i]:
                    current_settings[speaker]['ref_audio'] = audio_files[i]
                    print(f"🎤 Updated audio for {speaker}: {audio_files[i]}")
        
        return json.dumps(current_settings)
        
    except Exception as e:
        print(f"Error updating speaker audio settings: {e}")
        return current_settings_json

def update_speaker_language_settings(speakers_list, lang1, lang2, lang3, lang4, lang5, current_settings_json):
    """Update speaker settings JSON with language selections."""
    try:
        current_settings = json.loads(current_settings_json) if current_settings_json else {}
        languages = [lang1, lang2, lang3, lang4, lang5]
        
        # Update settings with selected languages
        for i, speaker in enumerate(speakers_list[:5]):
            if speaker in current_settings:
                if i < len(languages) and languages[i]:
                    current_settings[speaker]['language'] = languages[i]
                    print(f"🌍 Updated language for {speaker}: {languages[i]}")
        
        return json.dumps(current_settings)
        
    except Exception as e:
        print(f"Error updating speaker language settings: {e}")
        return current_settings_json

with gr.Blocks(title="🌍 Chatterbox TTS Pro - Multilingual") as demo:
    # Header
    if MULTILINGUAL_AVAILABLE:
        multilingual_status = "🌍 Multilingual Ready"
        status_message = "**Supports 23 languages**: Arabic, Chinese, French, German, Spanish, and many more! Download models below to get started."
    else:
        multilingual_status = "⚠️ No Models Available"
        status_message = "**Please install chatterbox-tts**: `pip install chatterbox-tts` or download models below."
    
    gr.Markdown(
        f"""
        # 🎭 Chatterbox TTS Pro {multilingual_status}
        **Advanced Multilingual Text-to-Speech with Voice Presets, Audio Effects & Export Options**
        
        Generate high-quality speech from text with reference audio styling, save your favorite voice presets, apply professional audio effects, and export in multiple formats!
        {status_message}
        
        **🚀 Getting Started**: Models are loaded on-demand. Use the download section below to get multilingual models for 23-language support.
        """
    )
    
    # Initialize presets on app startup
    initial_presets = get_preset_names()
    print(f"🚀 App starting with {len(initial_presets)} presets available")
    
    # Model Download Section - closed by default to reduce UI clutter
    with gr.Accordion("📥 Download Multilingual Models", open=False):
        gr.Markdown("""
        ### 🌍 Multilingual Model Download Manager
        **Manual Download Required**: Download the required model files for 23-language support. These models enable text-to-speech in Arabic, Chinese, French, German, Spanish, and many more languages.
        
        **No Auto-Download**: Models are only downloaded when you click the download button below.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                model_status_display = gr.Textbox(
                    label="📋 Model Files Status",
                    value=check_model_files_status(),
                    interactive=False,
                    lines=4
                )
                
                download_progress_display = gr.Textbox(
                    label="📊 Download Progress",
                    value=get_download_status(),
                    interactive=False,
                    lines=2
                )

                model_loading_status = gr.Textbox(
                    label="🚀 Model Loading Status",
                    value=check_model_loaded_status(),
                    interactive=False,
                    lines=2
                )
            
                with gr.Column(scale=1):
                    with gr.Group():
                        check_models_btn = gr.Button(
                            "🔍 Check Model Files",
                            variant="secondary",
                            size="sm"
                        )

                        download_models_btn = gr.Button(
                            "📥 Download Multilingual Models",
                            variant="primary",
                            size="lg"
                        )

                        load_model_btn = gr.Button(
                            "🚀 Load Model into Memory",
                            variant="secondary",
                            size="lg",
                            visible=False
                        )

                        refresh_status_btn = gr.Button(
                            "🔄 Refresh Status",
                            variant="secondary",
                            size="sm"
                        )
                
                gr.Markdown("""
                **Model Files:**
                - `Cangjie5_TC` - Chinese tokenizer
                - `conds` - Conditional embeddings
                - `mtl_tokenizer` - Multilingual tokenizer
                - `s3gen` - Speech generator
                - `t3_23lang` - Text-to-speech model
                - `ve` - Voice encoder
                
                **Total size:** ~2-4 GB
                """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Main text input
            text = gr.Textbox(
                value="Attention, humans of Earth! This is your refrigerator speaking. I’m tired of being opened every 3 minutes just so you can stare at me like I’m hiding life’s secrets. Spoiler alert: the cheese is still there, the milk is still judging you, and yes, that mysterious container in the back is now legally considered a biological weapon. Please stop poking me at 2 AM — I need my beauty sleep. Also, who put the socks in the freezer? This is not a cry for help, but it is a cry for better groceries. Thank you. Over and out.",
                label="📝 Text to synthesize (any length supported)",
                max_lines=10,
                placeholder="Enter your text here..."
            )
            
            # Language selection
            with gr.Group():
                gr.Markdown("### 🌍 Language Selection")
                with gr.Row():
                    language_dropdown = gr.Dropdown(
                        choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                        value="en",
                        label="🗣️ Target Language",
                        info="Select the language for text-to-speech generation"
                    )
                
                # Show language availability info
                if MULTILINGUAL_AVAILABLE:
                    gr.Markdown("*✅ Multilingual support available - Download models above to use 23 languages*")
                else:
                    gr.Markdown("*❌ No TTS models available - Please install chatterbox-tts or download models*")
                
                gr.Markdown("*💡 Models are loaded on-demand when you generate speech*")
            
            # Reference audio
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="🎤 Reference Audio File (Optional)",
                value="example/example.wav"
            )
            
            # Voice Conversation Mode Section
            with gr.Accordion("🎭 Voice Conversation Mode", open=False):
                gr.Markdown("### 🗣️ Multi-Voice Conversation Generator")
                gr.Markdown("*Generate conversations between multiple speakers with different voices*")
                
                conversation_mode = gr.Checkbox(
                    label="🎭 Enable Conversation Mode",
                    value=False,
                    info="Switch to conversation mode to generate multi-speaker dialogues"
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        conversation_script = gr.Textbox(
                            label="📝 Conversation Script",
                            placeholder="""Enter conversation in this format:

Alice: Hello there! How are you doing today?
Bob: I'm doing great, thanks for asking! How about you?
Alice: I'm wonderful! I just got back from vacation.
Bob: That sounds amazing! Where did you go?
Alice: I went to Japan. It was absolutely incredible!""",
                            lines=8,
                            info="Format: 'SpeakerName: Text' - Each line should start with speaker name followed by colon"
                        )
                        
                        # Conversation timing controls
                        with gr.Row():
                            conversation_pause = gr.Slider(
                                0.2, 2.0, step=0.1,
                                label="🔇 Speaker Change Pause (s)",
                                value=0.8,
                                info="Pause duration when speakers change"
                            )
                            speaker_transition_pause = gr.Slider(
                                0.1, 1.0, step=0.1,
                                label="⏸️ Same Speaker Pause (s)",
                                value=0.3,
                                info="Pause when same speaker continues"
                            )
                    
                    with gr.Column(scale=1):
                        # Speaker detection and management
                        detected_speakers = gr.Textbox(
                            label="🔍 Detected Speakers",
                            interactive=False,
                            lines=3,
                            info="Speakers found in your script will appear here"
                        )
                        
                        parse_script_btn = gr.Button(
                            "🔍 Analyze Script",
                            size="sm",
                            variant="secondary"
                        )
                        
                        conversation_help = gr.Markdown("""
                        **📋 Script Format Guide:**
                        - Each line: `SpeakerName: Dialogue text`
                        - Speaker names are case-sensitive
                        - Use consistent speaker names
                        - Multi-line dialogue will be joined
                        
                        **🎭 Example:**
                        ```
                        Alice: Hello Bob!
                        Bob: Hi Alice, how's it going?
                        Alice: Great! I wanted to tell you about my trip.
                        ```
                        """)

                # Dynamic speaker management section
                with gr.Group():
                    gr.Markdown("### 🎤 Speaker Voice Configuration")
                    gr.Markdown("*Configure voice settings for each speaker in your conversation*")
                    
                    # Speaker configuration will be dynamically generated
                    speaker_config_area = gr.HTML(
                        value="<p style='text-align: center; color: #666; padding: 20px;'>📝 Enter a conversation script above and click 'Analyze Script' to configure speaker voices</p>"
                    )
                    
                    # Dynamic speaker controls container
                    with gr.Row(visible=False) as speaker_controls_row:
                        with gr.Column():
                            # These will be dynamically created based on detected speakers
                            speaker_audio_1 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 1 Voice",
                                visible=False
                            )
                            speaker_audio_2 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 2 Voice",
                                visible=False
                            )
                            speaker_audio_3 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 3 Voice",
                                visible=False
                            )
                            speaker_audio_4 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 4 Voice",
                                visible=False
                            )
                            speaker_audio_5 = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎤 Speaker 5 Voice",
                                visible=False
                            )
                        
                        with gr.Column():
                            # Language selection for each speaker
                            speaker_lang_1 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 1 Language",
                                visible=False
                            )
                            speaker_lang_2 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 2 Language",
                                visible=False
                            )
                            speaker_lang_3 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 3 Language",
                                visible=False
                            )
                            speaker_lang_4 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 4 Language",
                                visible=False
                            )
                            speaker_lang_5 = gr.Dropdown(
                                choices=[(f"{lang_name} ({code})", code) for code, lang_name in SUPPORTED_LANGUAGES.items()],
                                value="en",
                                label="🌍 Speaker 5 Language",
                                visible=False
                            )
                    
                    # Hidden components to store speaker configurations
                    speaker_settings_json = gr.Textbox(
                        value="{}",
                        visible=False,
                        label="Speaker Settings JSON"
                    )
                    
                    # Store current speakers for reference
                    current_speakers = gr.State([])
                    
                    # Dynamic speaker controls (will be created programmatically)
                    dynamic_speaker_controls = gr.State({})
                
                # Conversation generation controls
                with gr.Row():
                    generate_conversation_btn = gr.Button(
                        "🎭 Generate Conversation",
                        variant="primary",
                        size="lg"
                    )
                    
                    clear_conversation_btn = gr.Button(
                        "🗑️ Clear Script",
                        variant="secondary",
                        size="sm"
                    )

            # Voice Presets Section
            with gr.Group():
                gr.Markdown("### 🎭 Voice Presets")
                gr.Markdown("*Save your complete voice setup including the reference audio file*")
                
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=initial_presets,
                        label="Select Voice Preset",
                        value=None,
                        interactive=True
                    )
                    preset_name_input = gr.Textbox(
                        label="New Voice Preset Name",
                        placeholder="Enter preset name...",
                        scale=1
                    )
                
                with gr.Row():
                    load_preset_btn = gr.Button("📥 Load Voice", size="sm")
                    save_preset_btn = gr.Button("💾 Save Voice", size="sm", variant="secondary")
                    delete_preset_btn = gr.Button("🗑️ Delete Voice", size="sm", variant="stop")
                    refresh_btn = gr.Button("🔄 Refresh List", size="sm", variant="secondary")
                
                preset_status = gr.Textbox(label="Status", interactive=False, visible=True)
                
                # Show current preset file locations
                with gr.Accordion("📁 File Locations", open=False):
                    preset_path_info = gr.Textbox(
                        label="Presets config saved to",
                        value=os.path.abspath(PRESETS_FILE),
                        interactive=False
                    )
                    audio_path_info = gr.Textbox(
                        label="Voice audio files saved to",
                        value=os.path.abspath(PRESETS_AUDIO_DIR),
                        interactive=False
                    )

            # Main controls
            with gr.Row():
                exaggeration = gr.Slider(
                    0.25, 2, step=.05, 
                    label="🎭 Exaggeration (Neutral = 0.5)", 
                    value=.5,
                    info="Higher values = more dramatic speech"
                )
                cfg_weight = gr.Slider(
                    0.2, 1, step=.05, 
                    label="⚡ CFG/Pace", 
                    value=0.5,
                    info="Controls generation speed vs quality"
                )

            with gr.Accordion("🔧 Advanced Settings", open=False):
                with gr.Row():
                    chunk_size = gr.Slider(
                        100, 400, step=25, 
                        label="📄 Chunk size (characters per chunk)", 
                        value=300,
                        info="Smaller = more consistent, larger = fewer seams"
                    )
                    temp = gr.Slider(
                        0.05, 5, step=.05, 
                        label="🌡️ Temperature", 
                        value=.8,
                        info="Higher = more creative/varied"
                    )
                    seed_num = gr.Number(
                        value=0, 
                        label="🎲 Random seed (0 for random)",
                        info="Use same seed for reproducible results"
                    )

            # Audio Effects Section
            with gr.Accordion("🎵 Audio Effects & Processing", open=False):
                gr.Markdown("### Professional audio effects and advanced processing")
                
                # Basic Effects Tab
                with gr.Tab("🎭 Basic Effects"):
                    with gr.Row():
                        with gr.Column():
                            enable_reverb = gr.Checkbox(label="🏛️ Enable Reverb", value=False)
                            reverb_room = gr.Slider(0.1, 1.0, step=0.1, label="Room Size", value=0.3, visible=True)
                            reverb_damping = gr.Slider(0.1, 1.0, step=0.1, label="Damping", value=0.5, visible=True)
                            reverb_wet = gr.Slider(0.1, 0.8, step=0.1, label="Reverb Amount", value=0.3, visible=True)
                        
                        with gr.Column():
                            enable_echo = gr.Checkbox(label="🔊 Enable Echo", value=False)
                            echo_delay = gr.Slider(0.1, 1.0, step=0.1, label="Echo Delay (s)", value=0.3, visible=True)
                            echo_decay = gr.Slider(0.1, 0.9, step=0.1, label="Echo Decay", value=0.5, visible=True)
                        
                        with gr.Column():
                            enable_pitch = gr.Checkbox(label="🎼 Enable Pitch Shift", value=False)
                            pitch_semitones = gr.Slider(-12, 12, step=1, label="Pitch (semitones)", value=0, visible=True)

                # Advanced Processing Tab
                with gr.Tab("🔧 Advanced Processing"):
                    with gr.Row():
                        with gr.Column():
                            # Noise Reduction
                            enable_noise_reduction = gr.Checkbox(label="🧹 Enable Noise Reduction", value=False)
                            gr.Markdown("*Automatically clean up reference audio*")
                        
                        with gr.Column():
                            # Audio Equalizer
                            enable_equalizer = gr.Checkbox(label="🎛️ Enable Equalizer", value=False)
                            gr.Markdown("*Fine-tune frequency bands*")
                    
                    # Equalizer Controls (shown when enabled)
                    with gr.Group():
                        gr.Markdown("#### 🎛️ 7-Band Equalizer (dB)")
                        with gr.Row():
                            eq_sub_bass = gr.Slider(-12, 12, step=1, label="Sub Bass\n(20-60 Hz)", value=0)
                            eq_bass = gr.Slider(-12, 12, step=1, label="Bass\n(60-200 Hz)", value=0)
                            eq_low_mid = gr.Slider(-12, 12, step=1, label="Low Mid\n(200-500 Hz)", value=0)
                            eq_mid = gr.Slider(-12, 12, step=1, label="Mid\n(500-2k Hz)", value=0)
                        with gr.Row():
                            eq_high_mid = gr.Slider(-12, 12, step=1, label="High Mid\n(2k-4k Hz)", value=0)
                            eq_presence = gr.Slider(-12, 12, step=1, label="Presence\n(4k-8k Hz)", value=0)
                            eq_brilliance = gr.Slider(-12, 12, step=1, label="Brilliance\n(8k-20k Hz)", value=0)

                # 3D Spatial Audio Tab
                with gr.Tab("🎧 3D Spatial Audio"):
                    enable_spatial = gr.Checkbox(label="🎧 Enable 3D Spatial Positioning", value=False)
                    gr.Markdown("*Position voices in 3D space for immersive experiences*")
                    
                    with gr.Row():
                        with gr.Column():
                            spatial_azimuth = gr.Slider(
                                -180, 180, step=5, 
                                label="🧭 Azimuth (degrees)", 
                                value=0,
                                info="Left-Right positioning (-180° to 180°)"
                            )
                            spatial_elevation = gr.Slider(
                                -90, 90, step=5, 
                                label="📐 Elevation (degrees)", 
                                value=0,
                                info="Up-Down positioning (-90° to 90°)"
                            )
                        with gr.Column():
                            spatial_distance = gr.Slider(
                                0.1, 5.0, step=0.1, 
                                label="📏 Distance", 
                                value=1.0,
                                info="Distance from listener (0.1 = close, 5.0 = far)"
                            )
                            gr.Markdown("""
                            **Quick Presets:**
                            - Center: Az=0°, El=0°, Dist=1.0
                            - Left: Az=-90°, El=0°, Dist=1.0  
                            - Right: Az=90°, El=0°, Dist=1.0
                            - Above: Az=0°, El=45°, Dist=1.0
                            - Distant: Az=0°, El=0°, Dist=3.0
                            """)

                # Background Music Mixer Tab
                with gr.Tab("🎵 Background Music"):
                    enable_background = gr.Checkbox(label="🎵 Enable Background Music/Ambience", value=False)
                    gr.Markdown("*Blend generated speech with background audio*")
                    
                    with gr.Row():
                        with gr.Column():
                            background_path = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="🎼 Background Audio File"
                            )
                            gr.Markdown("*Upload music, ambience, or sound effects*")
                            
                        with gr.Column():
                            bg_volume = gr.Slider(
                                0.0, 1.0, step=0.05, 
                                label="🔊 Background Volume", 
                                value=0.3,
                                info="Volume of background audio"
                            )
                            speech_volume = gr.Slider(
                                0.0, 2.0, step=0.05, 
                                label="🗣️ Speech Volume", 
                                value=1.0,
                                info="Volume of generated speech"
                            )
                    
                    with gr.Row():
                        bg_fade_in = gr.Slider(
                            0.0, 5.0, step=0.1, 
                            label="📈 Fade In (seconds)", 
                            value=1.0,
                            info="Background fade-in duration"
                        )
                        bg_fade_out = gr.Slider(
                            0.0, 5.0, step=0.1, 
                            label="📉 Fade Out (seconds)", 
                            value=1.0,
                            info="Background fade-out duration"
                        )
                    
                    gr.Markdown("""
                    **Background Audio Tips:**
                    - **Music**: Use instrumental tracks, keep volume low (0.2-0.4)
                    - **Ambience**: Nature sounds, room tone, atmospheric audio
                    - **SFX**: Sound effects that complement the speech content
                    - **Looping**: Short audio files will automatically loop to match speech length
                    """)

            # Generate button
            run_btn = gr.Button(
                "🚀 Generate Speech", 
                variant="primary", 
                size="lg"
            )

        with gr.Column(scale=1):
            # Enhanced Audio Output with Waveform Visualization
            with gr.Group():
                gr.Markdown("### 🎵 Generated Audio & Waveform Analysis")
                
                # Main audio output
                audio_output = gr.Audio(
                    label="🎵 Generated Audio",
                    show_download_button=True,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#4CAF50",
                        waveform_progress_color="#45a049",
                        show_recording_waveform=True,
                        skip_length=5,
                        sample_rate=22050
                    )
                )
                
                # Waveform analysis info
                waveform_info = gr.Textbox(
                    label="📊 Audio Analysis",
                    lines=4,
                    interactive=False,
                    placeholder="Audio analysis will appear here after generation..."
                )
                
                # Waveform controls
                with gr.Row():
                    analyze_btn = gr.Button("📊 Analyze Audio", size="sm", variant="secondary")
                    clear_analysis_btn = gr.Button("🗑️ Clear Analysis", size="sm", variant="stop")
            
            # Export Options
            with gr.Accordion("📤 Export Options", open=False):
                gr.Markdown("### 📥 Export your audio as WAV files")
                gr.Markdown("*Download your generated speech in different qualities and formats*")
                
                with gr.Row():
                    with gr.Column():
                        export_quality = gr.Radio(
                            choices=[
                                ("🎵 High Quality (16-bit, full sample rate)", "high"),
                                ("⚖️ Medium Quality (16-bit, half sample rate)", "medium"), 
                                ("💾 Low Quality (16-bit, quarter sample rate)", "low")
                            ],
                            value="high",
                            label="Export Quality",
                            info="Choose quality vs file size trade-off"
                        )
                        
                    with gr.Column():
                        gr.Markdown("""
                        **Quality Guide:**
                        - **High**: Best quality, largest file (~3-5MB/min)
                        - **Medium**: Good quality, balanced size (~1-2MB/min)
                        - **Low**: Smallest file, acceptable quality (~0.5MB/min)
                        """)
                
                with gr.Row():
                    export_btn = gr.Button(
                        "📥 Export Audio as WAV", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    gr.HTML("<div style='width: 20px;'></div>")  # Spacer
                
                export_status = gr.Textbox(
                    label="📋 Export Status", 
                    interactive=False,
                    placeholder="Export status will appear here...",
                    lines=2
                )
                
                # Show export folder location
                with gr.Accordion("📁 Export Location", open=False):
                    export_path_info = gr.Textbox(
                        label="Files exported to",
                        value=os.path.abspath(EXPORT_DIR),
                        interactive=False,
                        info="All exported files are saved to this directory"
                    )
                    gr.Markdown("**Note**: Files are automatically named with timestamp for easy organization.")

            # Tips and info
            with gr.Accordion("💡 Tips & Best Practices", open=False):
                gr.Markdown(
                    """
                💡 Pro Tips
                - **On-demand loading**: Models load only when you generate speech (no startup downloads)
                - **Long text**: Automatically chunked for best quality
                - **Voice presets**: Save your favorite combinations
                - **Model download**: Use the download section at the top to get multilingual models (~2-4GB)
                - **Multilingual mode**: Enable for 23 language support (Arabic, Chinese, French, Spanish, etc.)
                - **Language matching**: Match reference audio language to target language for best results
                - **Conversation mode**: Generate multi-speaker dialogues with different voices
                - **Basic effects**: Add reverb for space, echo for depth, pitch shift for character
                - **Noise reduction**: Automatically cleans up noisy reference audio
                - **Equalizer**: Boost presence (4-8kHz) for clarity, adjust bass for warmth
                - **3D spatial**: Create immersive positioning for podcasts/games
                - **Background music**: Keep volume low (0.2-0.4) for speech clarity
                - **Export**: Download in different qualities
                - **Waveform**: Analyze audio characteristics and quality
                
                ### 🎯 Best Practices
                - Use clear reference audio (3-10 seconds)
                - Keep exaggeration moderate (0.3-0.8)
                - Try temperature 0.6-1.0 for natural speech
                - Use smaller chunks for consistent quality
                - Apply noise reduction to poor quality reference audio
                - Use EQ to enhance specific voice characteristics
                - Position voices spatially for immersive experiences
                - Analyze waveform to understand audio quality
                
                ### 🌍 Multilingual Best Practices
                - **Language matching**: Reference audio should match target language
                - **CFG weight**: Lower (0.3) if reference has different language accent
                - **Supported languages**: 23 languages from Arabic to Chinese
                - **Quality**: Multilingual model maintains high quality across all languages
                - **Mixed conversations**: Each speaker in conversation mode can use a different language
                
                ### 🎭 Voice Conversation Mode Guide
                - **Script Format**: Use `SpeakerName: Dialogue text` format
                - **Individual Audio**: Upload different reference audio for each speaker
                - **Per-Speaker Languages**: Each speaker can use a different language (23 languages supported)
                - **Auto-Detection**: Speakers are automatically detected and audio upload slots appear
                - **Timing Control**: Adjust pauses between speakers and within speaker turns
                - **Voice Variety**: Each speaker can have completely different voice characteristics
                - **Consistent Names**: Keep speaker names exactly the same throughout
                - **Preset Integration**: Presets with matching speaker names load automatically
                - **Natural Flow**: Longer pauses for speaker changes, shorter for continuations
                - **Max Speakers**: Supports up to 5 different speakers per conversation
                
                ### 🎵 Audio Effects Guide
                - **Reverb**: Simulates room acoustics (church, hall, studio)
                - **Echo**: Adds depth and space to voice
                - **Pitch**: Change voice character (±12 semitones)
                - **Noise Reduction**: Clean background noise from reference
                - **Equalizer**: Shape frequency response for desired tone
                - **3D Spatial**: Position voice in 3D space for VR/AR
                - **Background**: Mix with music/ambience for atmosphere
                - **Waveform Analysis**: Understand audio characteristics and quality
                
                ### 📝 Conversation Examples
                ```
                Alice: Welcome to our podcast! I'm Alice.
                Bob: And I'm Bob. Today we're discussing AI.
                Alice: It's fascinating how quickly it's evolving.
                Bob: Absolutely! The possibilities are endless.
                ```
                
                ```
                Narrator: In a distant galaxy...
                Hero: I must save the princess!
                Villain: You'll never defeat me!
                Hero: We'll see about that!
                ```
                """
                )

    # Hidden components for waveform analysis
    waveform_data = gr.State(None)

    # Language dropdown is always visible now
    
    # Model download event handlers
    check_models_btn.click(
        fn=check_model_files_status,
        outputs=[model_status_display]
    )
    
    def start_download():
        """Start the download and return initial status."""
        download_models_async()
        return "📥 Starting download..."
    
    def download_complete_handler():
        """Handle download completion and update UI."""
        return (
            get_download_status(),
            check_model_loaded_status(),
            gr.update(visible=should_show_load_button())
        )
    
    download_models_btn.click(
        fn=start_download,
        outputs=[download_progress_display]
    )
    
    def load_model_and_update_status():
        """Load model and return status updates."""
        status_msg, success = load_model_manually()
        return status_msg, gr.update(visible=not success)
    
    load_model_btn.click(
        fn=load_model_and_update_status,
        outputs=[model_loading_status, load_model_btn]
    )
    
    def refresh_all_status():
        """Refresh all status displays and button visibility."""
        return (
            check_model_files_status(),
            get_download_status(),
            check_model_loaded_status(),
            gr.update(visible=should_show_load_button())
        )
    
    refresh_status_btn.click(
        fn=refresh_all_status,
        outputs=[model_status_display, download_progress_display, model_loading_status, load_model_btn]
    )
    
    # Auto-refresh download progress every 2 seconds when downloading
    def auto_refresh_download_status():
        status = download_status["status"]
        if status == "downloading":
            return get_download_status()
        return gr.update()
    
    # Initialize status displays on app load
    demo.load(
        fn=lambda: (check_model_files_status(), get_download_status(), check_model_loaded_status(), gr.update(visible=should_show_load_button())),
        outputs=[model_status_display, download_progress_display, model_loading_status, load_model_btn]
    )

    # Event handlers
    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text, ref_wav, exaggeration, temp, seed_num, cfg_weight, chunk_size,
            language_dropdown,
            enable_reverb, reverb_room, reverb_damping, reverb_wet,
            enable_echo, echo_delay, echo_decay,
            enable_pitch, pitch_semitones,
            enable_noise_reduction, enable_equalizer, eq_sub_bass, eq_bass, eq_low_mid, eq_mid, eq_high_mid, eq_presence, eq_brilliance,
            enable_spatial, spatial_azimuth, spatial_elevation, spatial_distance,
            enable_background, background_path, bg_volume, speech_volume, bg_fade_in, bg_fade_out,
            conversation_mode, conversation_script, conversation_pause, speaker_transition_pause, speaker_settings_json
        ],
        outputs=[audio_output, waveform_data, waveform_info],
    )
    
    # Conversation mode event handlers
    parse_script_btn.click(
        fn=setup_speaker_audio_components,
        inputs=[conversation_script],
        outputs=[
            speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5,
            speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5,
            speaker_controls_row, current_speakers, speaker_settings_json
        ]
    )
    
    clear_conversation_btn.click(
        fn=lambda: (
            "",  # Clear conversation script
            gr.update(visible=False, value=None),  # speaker_audio_1
            gr.update(visible=False, value=None),  # speaker_audio_2
            gr.update(visible=False, value=None),  # speaker_audio_3
            gr.update(visible=False, value=None),  # speaker_audio_4
            gr.update(visible=False, value=None),  # speaker_audio_5
            gr.update(visible=False, value="en"),  # speaker_lang_1
            gr.update(visible=False, value="en"),  # speaker_lang_2
            gr.update(visible=False, value="en"),  # speaker_lang_3
            gr.update(visible=False, value="en"),  # speaker_lang_4
            gr.update(visible=False, value="en"),  # speaker_lang_5
            gr.update(visible=False),  # speaker_controls_row
            [],  # current_speakers
            "{}"  # speaker_settings_json
        ),
        outputs=[
            conversation_script,
            speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5,
            speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5,
            speaker_controls_row, current_speakers, speaker_settings_json
        ]
    )
    
    # Auto-update speaker components when script changes
    conversation_script.change(
        fn=setup_speaker_audio_components,
        inputs=[conversation_script],
        outputs=[
            speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5,
            speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5,
            speaker_controls_row, current_speakers, speaker_settings_json
        ]
    )
    
    # Update speaker settings when audio files are uploaded
    for audio_component in [speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5]:
        audio_component.change(
            fn=update_speaker_audio_settings,
            inputs=[
                current_speakers,
                speaker_audio_1, speaker_audio_2, speaker_audio_3, speaker_audio_4, speaker_audio_5,
                speaker_settings_json
            ],
            outputs=[speaker_settings_json]
        )
    
    # Update speaker settings when languages are changed
    for lang_component in [speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5]:
        lang_component.change(
            fn=update_speaker_language_settings,
            inputs=[
                current_speakers,
                speaker_lang_1, speaker_lang_2, speaker_lang_3, speaker_lang_4, speaker_lang_5,
                speaker_settings_json
            ],
            outputs=[speaker_settings_json]
        )
    
    # Update detected speakers display
    current_speakers.change(
        fn=lambda speakers: f"Found {len(speakers)} speakers:\n" + "\n".join([f"• {speaker}" for speaker in speakers]) if speakers else "No speakers detected",
        inputs=[current_speakers],
        outputs=[detected_speakers]
    )
    
    # Auto-load presets for matching speaker names
    detected_speakers.change(
        fn=update_speaker_settings_from_presets,
        inputs=[detected_speakers, speaker_settings_json],
        outputs=[speaker_settings_json]
    )
    
    # Conversation generation (uses the same function as regular generation)
    generate_conversation_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text, ref_wav, exaggeration, temp, seed_num, cfg_weight, chunk_size,
            language_dropdown,
            enable_reverb, reverb_room, reverb_damping, reverb_wet,
            enable_echo, echo_delay, echo_decay,
            enable_pitch, pitch_semitones,
            enable_noise_reduction, enable_equalizer, eq_sub_bass, eq_bass, eq_low_mid, eq_mid, eq_high_mid, eq_presence, eq_brilliance,
            enable_spatial, spatial_azimuth, spatial_elevation, spatial_distance,
            enable_background, background_path, bg_volume, speech_volume, bg_fade_in, bg_fade_out,
            conversation_mode, conversation_script, conversation_pause, speaker_transition_pause, speaker_settings_json
        ],
        outputs=[audio_output, waveform_data, waveform_info],
    )
    
    # Waveform analysis handlers
    analyze_btn.click(
        fn=lambda audio_data: analyze_audio_waveform(audio_data),
        inputs=[waveform_data],
        outputs=[waveform_info]
    )
    
    clear_analysis_btn.click(
        fn=lambda: "Audio analysis cleared. Generate new audio to analyze.",
        outputs=[waveform_info]
    )
    
    # Preset management
    save_preset_btn.click(
        fn=save_current_preset,
        inputs=[preset_name_input, exaggeration, temp, cfg_weight, chunk_size, ref_wav],
        outputs=[preset_status, preset_dropdown]
    )
    
    load_preset_btn.click(
        fn=load_selected_preset,
        inputs=[preset_dropdown],
        outputs=[preset_status, exaggeration, temp, cfg_weight, chunk_size, ref_wav]
    )
    
    delete_preset_btn.click(
        fn=delete_selected_preset,
        inputs=[preset_dropdown],
        outputs=[preset_status, preset_dropdown]
    )
    
    refresh_btn.click(
        fn=refresh_preset_dropdown,
        inputs=[],
        outputs=[preset_dropdown]
    )

    # Export handler
    export_btn.click(
        fn=handle_export,
        inputs=[audio_output, export_quality],
        outputs=[export_status]
    )

demo.launch()