import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
from torch_audiomentations import (
    ApplyImpulseResponse,
    AddBackgroundNoise,
    Gain,
    PitchShift,
    LowPassFilter,
    HighPassFilter,
    BandStopFilter,
    Shift,
    PeakNormalization
)
import copy
from datetime import datetime

# --- CONFIG ---
INPUT_DIR = "data/for_augmentation/positive_samples" 
OUTPUT_DIR = "data/augmented/positive_augmented"                   
CSV_PATH = "data/augmented/positive_augmented.csv"                 
BACKGROUND_DIR = "data/preprocessed/background_sounds"
RIR_DIR = "data/preprocessed/rir_noises"
SAMPLE_RATE = 16000
N_AUGMENTATIONS_PER_FILE = 11
SEED = 42

# Volume detection thresholds
WHISPER_RMS_THRESHOLD = 0.01  # Adjust based on your data
QUIET_RMS_THRESHOLD = 0.05


# --- SET SEED FOR REPRODUCIBILITY ---
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Augmentations will run on: {DEVICE}")

def calculate_rms(waveform):
    """Calculate RMS (Root Mean Square) of the audio signal"""
    return torch.sqrt(torch.mean(waveform ** 2))

def is_whispered_audio(waveform, threshold=WHISPER_RMS_THRESHOLD):
    """Detect if audio is very quiet/whispered based on RMS"""
    rms = calculate_rms(waveform)
    return rms < threshold

def get_adaptive_gain_transform(waveform):
    """Create a Gain transform adapted to the audio's volume level"""
    rms = calculate_rms(waveform)
    
    if rms < WHISPER_RMS_THRESHOLD:
        # Very quiet/whispered audio - only allow positive gain or small negative
        return Gain(
            min_gain_in_db=-1.0,  # Minimal reduction
            max_gain_in_db=12.0,  # Allow significant boost
            p=0.7,
            output_type="dict"
        )
    elif rms < QUIET_RMS_THRESHOLD:
        # Somewhat quiet audio - be more conservative with negative gain
        return Gain(
            min_gain_in_db=-3.0,
            max_gain_in_db=8.0,
            p=0.7,
            output_type="dict"
        )
    else:
        # Normal volume audio - use original settings
        return Gain(
            min_gain_in_db=-6.0,
            max_gain_in_db=6.0,
            p=0.7,
            output_type="dict"
        )

def create_augmentation_pipeline(waveform):
    """Create augmentation pipeline adapted to the input audio"""
    
    # Use adaptive gain instead of fixed gain
    adaptive_gain = get_adaptive_gain_transform(waveform)
    
    pipeline = [
        ApplyImpulseResponse(
            ir_paths=[os.path.join(RIR_DIR, f) for f in os.listdir(RIR_DIR) if f.endswith('.wav')],
            p=0.7,
            sample_rate=SAMPLE_RATE,
            compensate_for_propagation_delay=True,
            output_type="dict"
        ),
        AddBackgroundNoise(
            background_paths=[os.path.join(BACKGROUND_DIR, f) for f in os.listdir(BACKGROUND_DIR) if f.endswith('.wav')],
            min_snr_in_db=15.0 if is_whispered_audio(waveform) else 10.0,  # Higher SNR for whispered audio
            max_snr_in_db=30.0 if is_whispered_audio(waveform) else 25.0,
            p=0.8,
            sample_rate=SAMPLE_RATE,
            output_type="dict"
        ),
        adaptive_gain,  # Use adaptive gain instead of fixed
        PitchShift(
            min_transpose_semitones=-1,
            max_transpose_semitones=1,
            p=0.4,
            sample_rate=SAMPLE_RATE,
            output_type="dict"
        ),
        LowPassFilter(
            min_cutoff_freq=3500,
            max_cutoff_freq=6000,
            p=0.3,
            sample_rate=SAMPLE_RATE,
            output_type="dict"
        ),
        HighPassFilter(
            min_cutoff_freq=100,
            max_cutoff_freq=500,
            p=0.3,
            sample_rate=SAMPLE_RATE,
            output_type="dict"
        ),
        BandStopFilter(
            min_center_frequency=300,
            max_center_frequency=3000,
            min_bandwidth_fraction=0.1,
            max_bandwidth_fraction=0.3,
            p=0.2,
            sample_rate=SAMPLE_RATE,
            output_type="dict"
        ),
        PeakNormalization(
            apply_to="only_too_loud_sounds",
            p=1.0,
            output_type="dict"
        )
    ]
    
    return pipeline

# --- OUTPUT SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_entries = []

# --- PROCESS CLIPS ---
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".wav"):
        continue

    file_path = os.path.join(INPUT_DIR, filename)
    waveform, sample_rate = torchaudio.load(file_path)
    assert sample_rate == SAMPLE_RATE, f"Sample rate mismatch in {filename}"

    waveform = waveform.unsqueeze(0)  # Add batch dim: (1, 1, T)
    
    # Check if this is whispered audio
    is_whisper = is_whispered_audio(waveform)
    original_rms = calculate_rms(waveform)

    for i in range(N_AUGMENTATIONS_PER_FILE):
        max_attempts = 5  # Retry failed augmentations
        attempt = 0
        
        while attempt < max_attempts:
            # Create adaptive pipeline for this specific audio
            pipeline = create_augmentation_pipeline(waveform)
            random.shuffle(pipeline)

            augmented_audio = waveform.clone()
            applied_transforms = []

            for transform in pipeline:
                result = transform(augmented_audio, sample_rate=SAMPLE_RATE)
                if isinstance(result, dict):
                    if "samples" in result:
                        if not torch.equal(result["samples"], augmented_audio):
                            applied_transforms.append(transform.__class__.__name__)
                            augmented_audio = result["samples"]
                else:
                    # If not dict, assume it's a tensor (future-proofing)
                    if not torch.equal(result, augmented_audio):
                        applied_transforms.append(transform.__class__.__name__)
                        augmented_audio = result

            # Check if augmentation is acceptable
            length_ok = augmented_audio.shape[-1] == waveform.shape[-1]
            has_changes = len(applied_transforms) > 0
            
            # Additional check: ensure whispered audio doesn't become too quiet
            if is_whisper:
                augmented_rms = calculate_rms(augmented_audio)
                volume_ok = augmented_rms >= (original_rms * 0.3)  # At least 30% of original volume
            else:
                volume_ok = True
            
            if length_ok and has_changes and volume_ok:
                break  # Success, exit retry loop
            
            # If no changes or volume too low, force at least one appropriate transform
            if (not has_changes or not volume_ok) and attempt == max_attempts - 1:
                if is_whisper:
                    # For whispered audio, apply positive gain boost
                    forced_gain = Gain(
                        min_gain_in_db=3.0,
                        max_gain_in_db=8.0,
                        p=1.0,  # Force application
                        output_type="dict"
                    )
                else:
                    # For normal audio, apply moderate gain
                    forced_gain = Gain(
                        min_gain_in_db=-3.0,
                        max_gain_in_db=3.0,
                        p=1.0,  # Force application
                        output_type="dict"
                    )
                result = forced_gain(augmented_audio, sample_rate=SAMPLE_RATE)
                augmented_audio = result["samples"]
                applied_transforms = ["Gain_forced"]
                break
                
            attempt += 1

        # Handle length mismatch by padding/trimming
        target_length = waveform.shape[-1]
        current_length = augmented_audio.shape[-1]
        
        if current_length != target_length:
            if current_length > target_length:
                # Trim excess
                augmented_audio = augmented_audio[..., :target_length]
            else:
                # Pad with zeros
                padding = target_length - current_length
                augmented_audio = torch.nn.functional.pad(augmented_audio, (0, padding))

        # Save augmented audio
        output_filename = f"{os.path.splitext(filename)[0]}_aug{i}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        torchaudio.save(output_path, augmented_audio.squeeze(0), sample_rate=SAMPLE_RATE)

        # Log info with volume statistics
        final_rms = calculate_rms(augmented_audio)
        log_entries.append({
            "filename": output_filename,
            "is_augmented": True,
            "augmentation_index": i,
            "augmentation_type": ",".join(applied_transforms),
        })

# --- SAVE LOG ---
df = pd.DataFrame(log_entries)
df.to_csv(CSV_PATH, index=False)

print(f"\n✅ Augmentation complete. {len(df)} files saved.")
print(f"📄 Log saved to: {CSV_PATH}")
print(f"📊 Expected: {len([f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]) * N_AUGMENTATIONS_PER_FILE}")
print(f"📊 Generated: {len(df)}")