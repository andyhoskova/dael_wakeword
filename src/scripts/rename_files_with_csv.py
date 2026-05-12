'''
Rename files to embbed sound_origin and sound_name into the filenames.
Create a csv file with all necessary columns, with augmentation columns being optional.
'''

import os
import csv
import wave

# Set input directory
input_dir = 'src/data/raw/negative_addition'

SOUND_ORIGIN = 'negative'
SOUND_NAME = 'negative_main'
CSV_NAME = 'negative_addition2'
INCLUDE_AUGMENTATION_COLUMNS = True  # Set to False to omit augmentation columns: is_augmented', 'augmentation_index', 'augmentation_type'

# List to hold CSV rows
csv_rows = []

# Walk through the directory structure
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.wav'):
            full_path = os.path.join(root, file)
            filename_only = os.path.splitext(file)[0]

            # Create new filename
            new_filename = f"{filename_only}-{SOUND_ORIGIN}-{SOUND_NAME}.wav"
            new_full_path = os.path.join(root, new_filename)

            # Rename file on disk if necessary
            if file != new_filename:
                try:
                    os.rename(full_path, new_full_path)
                    print(f"Renamed: {file} → {new_filename}")
                except Exception as e:
                    print(f"Failed to rename {file}: {e}")
                    continue

            # Get duration in milliseconds
            try:
                with wave.open(new_full_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration_ms = int((frames / float(rate)) * 1000)
            except Exception as e:
                print(f"Failed to read duration of {new_full_path}: {e}")
                duration_ms = 0

            row = [new_filename, duration_ms, SOUND_ORIGIN, SOUND_NAME]

            if INCLUDE_AUGMENTATION_COLUMNS:
                row += [False, '', '']  # is_augmented, augmentation_index, augmentation_type

            csv_rows.append(row)


def get_headers():
    base = ['filename', 'duration_ms', 'sound_origin', 'sound_name']
    if INCLUDE_AUGMENTATION_COLUMNS:
        return base + ['is_augmented', 'augmentation_index', 'augmentation_type']
    return base


# Write to CSV
output_csv = f'src/data/post_augmentation/{CSV_NAME}.csv'
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(get_headers())
    writer.writerows(csv_rows)

print(f"\nCSV file created: {output_csv}")