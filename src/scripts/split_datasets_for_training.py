import pandas as pd
import shutil
import os

# Configuration
csv_path = 'src/data/post_augmentation/negative_addition2.csv' 
audio_folder = 'src/data/raw/negative_addition'  
destination_folder = 'src/data/processed/test/negative' 
target_label = 'negative_main'
num_files_to_move = 2500

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path)

# Filter rows where sound_name == 'random_speech'
matching_rows = df[df['sound_name'] == target_label]

# Shuffle all filenames that match
all_matching_filenames = matching_rows['filename'].sample(frac=1, random_state=42).tolist()

# Move files until the desired count is reached
moved_count = 0
checked_count = 0

for fname in all_matching_filenames:
    if moved_count >= num_files_to_move:
        break

    src = os.path.join(audio_folder, fname)
    dst = os.path.join(destination_folder, fname)

    if os.path.isfile(src):
        shutil.move(src, dst)
        moved_count += 1
    else:
        print(f"Missing: {src}")
    
    checked_count += 1

print(f"\nChecked {checked_count} files.")
print(f"✅ Successfully moved {moved_count} files to '{destination_folder}'.")

if moved_count < num_files_to_move:
    print(f"⚠️ Warning: Only {moved_count} files moved. Not enough available files in folder.")