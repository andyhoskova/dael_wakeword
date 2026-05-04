import os
import csv
import wave

# Set input directory
input_dir = 'data/imports'

# List to hold CSV rows
csv_rows = []

# Walk through the directory structure
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.wav'):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, input_dir)
            parts = rel_path.split(os.sep)

            # Expecting at least 3 parts: sound_origin / sound_name / file
            if len(parts) >= 3:
                sound_origin = parts[0]
                sound_name = parts[1]
                filename_only = os.path.splitext(file)[0]

                # Create new filename
                new_filename = f"{filename_only}-{sound_origin}-{sound_name}.wav"
                new_full_path = os.path.join(root, new_filename)

                # Rename file on disk if necessary
                if file != new_filename:
                    try:
                        os.rename(full_path, new_full_path)
                        print(f"Renamed: {file} → {new_filename}")
                    except Exception as e:
                        print(f"Failed to rename {file}: {e}")
                        continue  # Skip this file in case of error

                # Get duration in milliseconds
                try:
                    with wave.open(new_full_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        duration_ms = int((frames / float(rate)) * 1000)
                except Exception as e:
                    print(f"Failed to read duration of {new_full_path}: {e}")
                    duration_ms = 0

                # Save row
                csv_rows.append([new_filename, duration_ms, sound_origin, sound_name])
            else:
                print(f"Skipping file with unexpected path: {rel_path}")

# Write to CSV
output_csv = 'data/imports/negative_samples_renamed.csv'
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'duration_ms', 'sound_origin', 'sound_name'])
    writer.writerows(csv_rows)

print(f"\nCSV file created: {output_csv}")