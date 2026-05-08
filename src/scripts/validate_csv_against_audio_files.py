"""
Checks WAV filenames and filenames in CSV if they match
"""

import os
import pandas as pd

def find_missing_wavs(csv_path, audio_dir, output_csv='src/data/notFound.csv', filename_column='filename'):
    # Load expected filenames from CSV
    df = pd.read_csv(csv_path)
    
    if filename_column not in df.columns:
        raise ValueError(f"Column '{filename_column}' not found in the CSV.")
    
    expected_files = set(df[filename_column].astype(str))  # Ensure all are strings
    actual_files = set(f for f in os.listdir(audio_dir) if f.endswith('.wav'))

    # Find missing files
    missing_files = expected_files - actual_files

    # Save to CSV if any missing
    if missing_files:
        pd.DataFrame({filename_column: list(missing_files)}).to_csv(output_csv, index=False)
        print(f"{len(missing_files)} file(s) missing. Saved to {output_csv}.")
    else:
        print("All files found.")

if __name__ == "__main__":
    csv_path = 'src/data/processed/positive.csv'       # Your CSV file with the filenames
    audio_dir = 'src/data/post_augmentation/positive'      # Folder containing the .wav files
    find_missing_wavs(csv_path, audio_dir)