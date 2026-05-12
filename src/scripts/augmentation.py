import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from torch_audiomentations import (
    AddBackgroundNoise,
    ApplyImpulseResponse,
    BandStopFilter,
    Compose,
    Gain,
    HighPassFilter,
    LowPassFilter,
    PeakNormalization,
    PitchShift,
)

if not hasattr(torchaudio, "info"):
    from dataclasses import dataclass as _dataclass

    import soundfile as _sf

    @_dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int = 16
        encoding: str = "PCM_S"

    def _torchaudio_info(filepath):
        info = _sf.info(str(filepath))
        return _AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
        )

    torchaudio.info = _torchaudio_info


# Configuration
INPUT_DIR = "src/data/raw/positive_add"
OUTPUT_DIR = "src/data/post_augmentation/positive_add_augmented"
CSV_PATH = "src/data/post_augmentation/positive_add_augmented.csv"
BACKGROUND_DIR = "src/data/pre_augmentation/background_sounds"
RIR_DIR = "src/data/pre_augmentation/rir_noises"
SAMPLE_RATE = 16000
N_AUGMENTATIONS_PER_FILE = 10
SEED = 42
WHISPER_RMS_THRESHOLD = 0.01
QUIET_RMS_THRESHOLD = 0.05
SOUND_ORIGIN = "positive"
SOUND_NAME = "personal_voice"


# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Augmentations will run on: {DEVICE}")


# Helpers
def calculate_rms(waveform: torch.Tensor) -> torch.Tensor:
    """RMS of the audio signal."""
    return torch.sqrt(torch.mean(waveform**2))


def is_whispered(waveform: torch.Tensor) -> bool:
    return calculate_rms(waveform).item() < WHISPER_RMS_THRESHOLD


def _wav_paths(directory: str) -> list[str]:
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    ]


def get_adaptive_gain(waveform: torch.Tensor) -> Gain:
    """Return a Gain transform whose range is tuned to the clip's volume."""
    rms = calculate_rms(waveform).item()
    if rms < WHISPER_RMS_THRESHOLD:
        return Gain(min_gain_in_db=-1.0, max_gain_in_db=12.0, p=0.7, output_type="dict")
    elif rms < QUIET_RMS_THRESHOLD:
        return Gain(min_gain_in_db=-3.0, max_gain_in_db=8.0, p=0.7, output_type="dict")
    else:
        return Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.7, output_type="dict")


def build_pipeline(waveform: torch.Tensor) -> Compose:

    whisper = is_whispered(waveform)

    transforms = [
        ApplyImpulseResponse(
            ir_paths=_wav_paths(RIR_DIR),
            p=0.7,
            sample_rate=SAMPLE_RATE,
            compensate_for_propagation_delay=True,
            output_type="dict",
        ),
        AddBackgroundNoise(
            background_paths=_wav_paths(BACKGROUND_DIR),
            min_snr_in_db=15.0 if whisper else 10.0,
            max_snr_in_db=30.0 if whisper else 25.0,
            p=0.8,
            sample_rate=SAMPLE_RATE,
            output_type="dict",
        ),
        get_adaptive_gain(waveform),
        PitchShift(
            min_transpose_semitones=-1,
            max_transpose_semitones=1,
            p=0.4,
            sample_rate=SAMPLE_RATE,
            output_type="dict",
        ),
        LowPassFilter(
            min_cutoff_freq=3500,
            max_cutoff_freq=6000,
            p=0.3,
            sample_rate=SAMPLE_RATE,
            output_type="dict",
        ),
        HighPassFilter(
            min_cutoff_freq=100,
            max_cutoff_freq=500,
            p=0.3,
            sample_rate=SAMPLE_RATE,
            output_type="dict",
        ),
        BandStopFilter(
            min_center_frequency=300,
            max_center_frequency=3000,
            min_bandwidth_fraction=0.1,
            max_bandwidth_fraction=0.3,
            p=0.2,
            sample_rate=SAMPLE_RATE,
            output_type="dict",
        ),
        PeakNormalization(
            apply_to="only_too_loud_sounds",
            p=1.0,
            output_type="dict",
        ),
    ]

    # Shuffle here so each augmentation gets a different ordering.
    random.shuffle(transforms)
    return Compose(transforms=transforms, output_type="dict")


def forced_gain(waveform: torch.Tensor, whisper: bool) -> torch.Tensor:
    """Apply a guaranteed gain as a last resort when stochastic passes all fail."""
    if whisper:
        g = Gain(min_gain_in_db=3.0, max_gain_in_db=8.0, p=1.0, output_type="dict")
    else:
        g = Gain(min_gain_in_db=-3.0, max_gain_in_db=3.0, p=1.0, output_type="dict")
    return g(waveform, sample_rate=SAMPLE_RATE)


def fix_length(audio: torch.Tensor, target: int) -> torch.Tensor:
    """Trim or zero-pad the last dimension to exactly `target` samples."""
    length = audio.shape[-1]
    if length > target:
        return audio[..., :target]
    if length < target:
        return torch.nn.functional.pad(audio, (0, target - length))
    return audio


# Output setup
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_entries: list[dict] = []

input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")]

# Augmentation loop
total_expected = len(input_files) * N_AUGMENTATIONS_PER_FILE
with tqdm(total=total_expected, desc="Augmenting", unit="clip") as pbar:
    for filename in input_files:
        file_path = os.path.join(INPUT_DIR, filename)
        waveform, sample_rate = torchaudio.load(file_path)
        assert sample_rate == SAMPLE_RATE, f"Sample rate mismatch: {filename}"

        # Compute duration in milliseconds from the original audio
        duration_ms = round(waveform.shape[-1] / SAMPLE_RATE * 1000, 2)

        # torch_audiomentations expects shape (batch, channels, time)
        waveform = waveform.unsqueeze(0).to(DEVICE)  # (1, C, T)

        whisper = is_whispered(waveform)
        original_rms = calculate_rms(waveform).item()
        target_length = waveform.shape[-1]

        for i in range(N_AUGMENTATIONS_PER_FILE):
            augmented = waveform.clone()
            aug_tag = "none"
            MAX_TRIES = 5

            for attempt in range(MAX_TRIES):
                pipeline = build_pipeline(waveform)
                candidate = pipeline(waveform.clone(), sample_rate=SAMPLE_RATE)

                # Compose returns a tensor in the modern API
                if isinstance(candidate, dict): 
                    candidate = candidate["samples"]

                changed = not torch.equal(candidate, waveform)
                length_ok = candidate.shape[-1] == target_length

                if whisper:
                    aug_rms = calculate_rms(candidate).item()
                    volume_ok = aug_rms >= original_rms * 0.3
                else:
                    volume_ok = True

                if changed and length_ok and volume_ok:
                    augmented = candidate
                    # Summarise which transform classes likely ran (those with p > 0)
                    aug_tag = ",".join(
                        t.__class__.__name__ for t in pipeline.transforms
                    )
                    break

                # Last attempt: force a gain so we always produce a distinct sample
                if attempt == MAX_TRIES - 1:
                    augmented = forced_gain(waveform.clone(), whisper)
                    augmented = fix_length(augmented, target_length)
                    aug_tag = "Gain_forced"

            # Final length guard (impulse responses can occasionally shift length)
            augmented = fix_length(augmented, target_length)

            # Save
            stem = os.path.splitext(filename)[0]
            output_filename = f"{stem}_aug{i}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            torchaudio.save(output_path, augmented.squeeze(0).cpu(), SAMPLE_RATE)

            log_entries.append(
                {
                    "filename": output_filename,
                    "duration_ms": duration_ms,
                    "sound_origin": SOUND_ORIGIN,
                    "sound_name": SOUND_NAME,
                    "is_augmented": True,
                    "augmentation_index": i,
                    "augmentation_type": aug_tag,
                }
            )
            pbar.update(1)

# Save log
df = pd.DataFrame(log_entries)
df.to_csv(CSV_PATH, index=False)

expected = len(input_files) * N_AUGMENTATIONS_PER_FILE
print(f"\n✅ Augmentation complete. {len(df)} files saved.")
print(f"📄 Log saved to: {CSV_PATH}")
print(f"📊 Expected: {expected} | Generated: {len(df)}")