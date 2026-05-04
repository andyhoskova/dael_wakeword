"""
Converts MP3 files to WAV format (16kHz, mono, 16-bit PCM)
and generates CSV metadata from the clips.
"""

import pandas as pd
import subprocess
from tqdm import tqdm
from pathlib import Path


def setup_directories():
    """Create necessary output directories if they don't exist."""
    csv_output_dir = Path("data")
    wav_output_dir = Path("data/converted")

    csv_output_dir.mkdir(parents=True, exist_ok=True)
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV output directory: {csv_output_dir.resolve()}")
    print(f"WAV output directory: {wav_output_dir.resolve()}")

    return csv_output_dir, wav_output_dir


def get_mp3_files():
    """Get list of all MP3 files in the input directory."""
    input_dir = Path("src/data/raw/positive")

    print(f"Looking for MP3 files in: {input_dir.resolve()}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir.resolve()}")

    mp3_files = list(input_dir.glob("*.mp3"))

    if not mp3_files:
        raise FileNotFoundError(f"No MP3 files found in {input_dir.resolve()}")

    return mp3_files


def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Convert MP3 to WAV using FFmpeg:
    - 16 kHz sample rate
    - mono channel
    - 16-bit PCM
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",                      # overwrite output
            "-i", str(mp3_path),       # input file
            "-ac", "1",                # mono
            "-ar", "16000",            # 16 kHz
            "-acodec", "pcm_s16le",    # 16-bit PCM
            str(wav_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            print(f"\n❌ FFmpeg error for {mp3_path.name}")
            print(result.stderr.decode("utf-8"))
            return False

        return True

    except Exception as e:
        print(f"\n❌ Unexpected error: {mp3_path.name}")
        print(e)
        return False


def process_files():
    """Main processing function."""
    print("Setting up directories...")
    csv_output_dir, wav_output_dir = setup_directories()

    print("Finding MP3 files...")
    mp3_files = get_mp3_files()
    print(f"Found {len(mp3_files)} MP3 files to process")

    csv_data = []
    successful_conversions = 0

    print("Converting MP3 files to WAV...")

    for mp3_file in tqdm(mp3_files, desc="Processing files"):
        # Skip empty files
        if mp3_file.stat().st_size == 0:
            print(f"⚠️ Skipping empty file: {mp3_file.name}")
            continue

        clip_name = mp3_file.stem
        wav_file = wav_output_dir / f"{clip_name}.wav"

        if convert_mp3_to_wav(mp3_file, wav_file):
            csv_data.append({
                "clip_id": clip_name,
                "label": "random-speech",
                "folder_location": "random_speech",
            })
            successful_conversions += 1
        else:
            print(f"Failed to convert: {mp3_file.name}")

    print(f"\n✅ Successfully converted {successful_conversions} files")

    # Save CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = csv_output_dir / "positive_samples_personal_voice.csv"

        df.to_csv(csv_path, index=False)

        print(f"CSV file saved: {csv_path}")
        print(f"CSV contains {len(df)} records")

        print("\nFirst 5 rows:")
        print(df.head())
    else:
        print("❌ No data to save")


def main():
    try:
        process_files()
        print("\n🎉 Dataset preparation completed successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()