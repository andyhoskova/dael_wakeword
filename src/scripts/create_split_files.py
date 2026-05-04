import os
from pathlib import Path

def generate_split_files():
    # Define paths
    base_data_path = Path("data/processed")
    splits_output_path = Path("data/splits")
    
    # Create splits directory if it doesn't exist
    splits_output_path.mkdir(parents=True, exist_ok=True)
    
    # Define the splits and their corresponding folders
    splits = {
        "train_split.txt": "train",
        "test_split.txt": "test", 
        "val_split.txt": "validation"
    }
    
    # Common audio file extensions
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    
    for split_file, folder_name in splits.items():
        print(f"Processing {folder_name}...")
        
        split_path = base_data_path / folder_name
        output_file = splits_output_path / split_file
        
        file_list = []
        
        # Process positive files
        positive_path = split_path / "positive"
        if positive_path.exists():
            for file_path in positive_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    # Store relative path from the split folder
                    relative_path = f"positive/{file_path.name}"
                    file_list.append(relative_path)
        
        # Process negative files  
        negative_path = split_path / "negative"
        if negative_path.exists():
            for file_path in negative_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    # Store relative path from the split folder
                    relative_path = f"negative/{file_path.name}"
                    file_list.append(relative_path)
        
        # Sort the files for consistency
        file_list.sort()
        
        # Write to split file
        with open(output_file, 'w') as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")
        
        print(f"  Created {split_file} with {len(file_list)} files")
        
        # Show breakdown
        positive_count = sum(1 for f in file_list if f.startswith('positive/'))
        negative_count = sum(1 for f in file_list if f.startswith('negative/'))
        
        print(f"    Positive: {positive_count}")
        print(f"    Negative: {negative_count}")
        if positive_count > 0:
            ratio = negative_count / positive_count
            print(f"    Ratio (neg:pos): {ratio:.1f}:1")
        print()

def generate_split_files_with_labels():
    """Alternative version that includes labels in the split files"""
    # Define paths
    base_data_path = Path("data/processed")
    splits_output_path = Path("data/splits")
    
    # Create splits directory if it doesn't exist
    splits_output_path.mkdir(parents=True, exist_ok=True)
    
    # Define the splits and their corresponding folders
    splits = {
        "train_split_labeled.txt": "train",
        "test_split_labeled.txt": "test", 
        "val_split_labeled.txt": "validation"
    }
    
    # Common audio file extensions
    audio_extensions = {'.wav'}
    
    for split_file, folder_name in splits.items():
        print(f"Processing {folder_name} (with labels)...")
        
        split_path = base_data_path / folder_name
        output_file = splits_output_path / split_file
        
        file_list = []
        
        # Process positive files (label = 1)
        positive_path = split_path / "positive"
        if positive_path.exists():
            for file_path in positive_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    relative_path = f"positive/{file_path.name}"
                    file_list.append((relative_path, "1"))
        
        # Process negative files (label = 0)
        negative_path = split_path / "negative"
        if negative_path.exists():
            for file_path in negative_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    relative_path = f"negative/{file_path.name}"
                    file_list.append((relative_path, "0"))
        
        # Sort by filename for consistency
        file_list.sort(key=lambda x: x[0])
        
        # Write to split file
        with open(output_file, 'w') as f:
            for file_path, label in file_list:
                f.write(f"{file_path} {label}\n")
        
        print(f"  Created {split_file} with {len(file_list)} files")
        print()

if __name__ == "__main__":
    print("Generating dataset split files...")
    print("=" * 50)
    
    # Generate basic split files (just filenames)
    generate_split_files()
    
    print("=" * 50)
    print("Also generating labeled versions...")
    print("=" * 50)
    
    # Generate labeled split files (filename + label)
    generate_split_files_with_labels()
    
    print("Done! Check the data/splits/ folder for your files.")
    print("\nFiles created:")
    print("- train_split.txt, val_split.txt, test_split.txt (filenames only)")
    print("- train_split_labeled.txt, val_split_labeled.txt, test_split_labeled.txt (with labels)")