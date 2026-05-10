import os
import random
from pydub import AudioSegment


# Output folder
output_folder = "src/data/raw/silence_clips"

# Number of silence files to generate
num_files = 56

# Silence duration range (seconds)
min_duration = 1
max_duration = 5

# =========================

os.makedirs(output_folder, exist_ok=True)

for i in range(1, num_files + 1):

    # Random duration
    duration_sec = random.uniform(min_duration, max_duration)
    duration_ms = int(duration_sec * 1000)

    # Generate silence
    silence = AudioSegment.silent(duration=duration_ms)

    # Output filename
    filename = f"silence_{i:03d}.wav"
    output_path = os.path.join(output_folder, filename)

    # Export WAV
    silence.export(output_path, format="wav")

    print(f"Created: {filename} | {duration_sec:.2f} seconds")

print("\nDone generating silence clips!")