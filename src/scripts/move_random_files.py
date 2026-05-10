"""
Randomly select and transfer audio files from source to target directory.
"""

import random
import shutil
from pathlib import Path


def transfer_random_audio_files(source_dir, target_dir, num_files=1680):

    # Convert to Path objects for easier handling
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Check if source directory exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for audio files in: {source_dir}")

    # Get all audio files from source directory
    audio_files = list(source_path.glob("*.wav"))

    print(f"Found {len(audio_files)} Audio files in source directory")

    # Check if we have enough files
    if len(audio_files) < num_files:
        raise ValueError(
            f"Not enough Audio files in source directory. Found {len(audio_files)}, need {num_files}"
        )

    # Randomly select files to transfer
    print(f"Randomly selecting {num_files} files...")
    selected_files = random.sample(audio_files, num_files)

    # Transfer files
    print(f"Transferring {num_files} files to: {target_dir}")

    transferred_count = 0
    failed_transfers = []

    for i, file_path in enumerate(selected_files, 1):
        try:
            # Move file to target directory
            target_file = target_path / file_path.name

            # Handle potential filename conflicts
            counter = 1
            original_target = target_file
            while target_file.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_file = target_path / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.move(str(file_path), str(target_file))
            transferred_count += 1

            # Progress indicator
            if i % 150 == 0:
                print(f"Progress: {i}/{num_files} files transferred")

        except Exception as e:
            failed_transfers.append((file_path.name, str(e)))
            print(f"Failed to transfer {file_path.name}: {e}")

    # Summary
    print("\nTransfer completed!")
    print(f"Successfully transferred: {transferred_count} files")
    print(f"Failed transfers: {len(failed_transfers)}")

    remaining_files = len(audio_files) - transferred_count
    print(f"Remaining files in source directory: {remaining_files}")

    if failed_transfers:
        print("\nFailed transfers:")
        for filename, error in failed_transfers:
            print(f"  - {filename}: {error}")

    return transferred_count, failed_transfers


def main():
    # Configuration
    source_directory = "src/data/post_augmentation/negative"
    target_directory = "src/data/post_augmentation/discard"
    files_to_transfer = 1680

    print("Audio File Transfer Script")
    print("=" * 40)
    print(f"Source: {source_directory}")
    print(f"Target: {target_directory}")
    print(f"Files to transfer: {files_to_transfer}")
    print()

    try:
        # Set random seed for reproducibility (optional)
        # random.seed(42)  # Uncomment if you want reproducible results

        # Perform the transfer
        transferred, failed = transfer_random_audio_files(
            source_directory, target_directory, files_to_transfer
        )

        print("\nOperation completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())