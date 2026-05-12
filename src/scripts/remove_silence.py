"""
Removes all silent parts from the clips
Removes all clips that has no speech in them
Leaves only speech
"""

import torch
from pathlib import Path
from tqdm import tqdm
import time

# Set CUDA only (forces error if CUDA not available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise RuntimeError("CUDA device not available — please ensure you're using a GPU.")
torch.set_num_threads(1)

# Load Silero VAD model with CUDA
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
model.to(device)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Paths
input_dir = Path('src/data/raw/positive_addition')
output_dir = Path('src/data/raw/trimmed')
output_dir.mkdir(parents=True, exist_ok=True)

# Create logs directory
output_dir_logs = output_dir / 'logs'
output_dir_logs.mkdir(parents=True, exist_ok=True)

# Get all WAV files
wav_files = list(input_dir.glob('*.wav'))
total_files = len(wav_files)
print(f"Found {total_files} WAV files to process")
print(f"Processing files from: {input_dir}")
print(f"Saving results to: {output_dir}")
print("-" * 50)

# Start timing
start_time = time.time()

# Process each WAV file with progress bar
processed_count = 0
error_count = 0
no_speech_files = []

for wav_file in tqdm(wav_files, desc="Processing audio files", unit="file"):
    try:
        # Read audio
        wav = read_audio(str(wav_file), sampling_rate=16000)
        wav = wav.to(device)
        
        # Get speech timestamps in seconds
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=False, sampling_rate=16000)
        
        if not speech_timestamps:
            tqdm.write(f"⚠️ No speech found in: {wav_file.name}")
            no_speech_files.append(wav_file.name)
            continue
        
        # Collect speech chunks and concatenate
        speech_chunks = collect_chunks(speech_timestamps, wav)
        
        # Save new trimmed audio
        output_wav_path = output_dir / f"{wav_file.stem}.wav"
        save_audio(str(output_wav_path), speech_chunks.cpu(), sampling_rate=16000)
        
        processed_count += 1
        
    except Exception as e:
        error_count += 1
        tqdm.write(f"❌ Error processing {wav_file.name}: {e}")

# Save no-speech files to log
if no_speech_files:
    log_file = output_dir_logs / 'no_speech_files.txt'
    with open(log_file, 'w') as f:
        f.write(f"Files with no speech detected ({len(no_speech_files)} total):\n")
        f.write("=" * 50 + "\n")
        for filename in no_speech_files:
            f.write(f"{filename}\n")
    print(f"No-speech files logged to: {log_file}")

# End timing
end_time = time.time()
total_time = end_time - start_time

# Summary
print("\n" + "="*50)
print("PROCESSING COMPLETE")
print("="*50)
print(f"Total files found: {total_files}")
print(f"Successfully processed: {processed_count}")
print(f"Errors: {error_count}")
print(f"Skipped (no speech): {total_files - processed_count - error_count}")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average time per file: {total_time/total_files:.2f} seconds")
print(f"Processing rate: {total_files/total_time:.2f} files/second")
print("="*50)