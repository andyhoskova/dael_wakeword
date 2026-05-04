"""
Checks WAV files for specific audio format requirements:
- Sample rate: 16 kHz
- Bit depth: 16-bit
- Format: PCM
"""

import os
import wave
from pathlib import Path
from typing import List, Dict

# Get the project root directory (where src/ is located)
PROJECT_ROOT = Path(__file__).parent.parent

class WAVValidator:
    def __init__(self, target_sample_rate: int = 16000, target_bit_depth: int = 16):
        self.target_sample_rate = target_sample_rate
        self.target_bit_depth = target_bit_depth
        self.results = {
            'valid_files': [],
            'invalid_files': [],
            'error_files': []
        }
    
    def check_wav_file(self, file_path: str) -> Dict:
       
        try:
            with wave.open(file_path, 'rb') as wav_file:
                # Get audio properties
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()  # in bytes
                bit_depth = sample_width * 8
                channels = wav_file.getnchannels()
                frames = wav_file.getnframes()
                duration = frames / sample_rate
                
                # Check if it's PCM (wave module only handles PCM)
                # If we can open it with wave module, it's PCM
                is_pcm = True
                
                # Validate criteria
                valid_sample_rate = sample_rate == self.target_sample_rate
                valid_bit_depth = bit_depth == self.target_bit_depth
                
                is_valid = valid_sample_rate and valid_bit_depth and is_pcm
                
                return {
                    'file_path': file_path,
                    'is_valid': is_valid,
                    'sample_rate': sample_rate,
                    'bit_depth': bit_depth,
                    'channels': channels,
                    'duration': duration,
                    'is_pcm': is_pcm,
                    'valid_sample_rate': valid_sample_rate,
                    'valid_bit_depth': valid_bit_depth,
                    'error': None
                }
                
        except Exception as e:
            return {
                'file_path': file_path,
                'is_valid': False,
                'error': str(e)
            }
    
    def find_wav_files(self, directory: str) -> List[str]:
       
        wav_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files
    
    def validate_directory(self, directory: str) -> Dict:
       
        print(f"Scanning directory: {directory}")
        wav_files = self.find_wav_files(directory)
        
        if not wav_files:
            print("No WAV files found in the specified directory.")
            return self.results
        
        print(f"Found {len(wav_files)} WAV files. Validating...")
        
        for i, file_path in enumerate(wav_files, 1):
            print(f"Processing {i}/{len(wav_files)}: {os.path.basename(file_path)}")
            
            result = self.check_wav_file(file_path)
            
            if result['error']:
                self.results['error_files'].append(result)
            elif result['is_valid']:
                self.results['valid_files'].append(result)
            else:
                self.results['invalid_files'].append(result)
        
        return self.results
    
    def print_summary(self):
        """Print a summary of validation results."""
        total_files = len(self.results['valid_files']) + len(self.results['invalid_files']) + len(self.results['error_files'])
        
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        
        # Check if all files meet the parameters
        if len(self.results['valid_files']) == total_files:
            print("✓ ALL FILES MEET THE PARAMETERS (16 kHz, 16-bit, PCM)")
            print(f"Total files validated: {total_files}")
        else:
            print("✗ NOT ALL FILES MEET THE PARAMETERS")
            print(f"Valid files: {len(self.results['valid_files'])}/{total_files}")
            
            # Show files that don't meet parameters
            if self.results['invalid_files']:
                print(f"\nFILES THAT DON'T MEET PARAMETERS:")
                for file_info in self.results['invalid_files']:
                    print(f"  • {file_info['file_path']}")
                    issues = []
                    if not file_info['valid_sample_rate']:
                        issues.append(f"Sample rate: {file_info['sample_rate']} Hz (should be {self.target_sample_rate} Hz)")
                    if not file_info['valid_bit_depth']:
                        issues.append(f"Bit depth: {file_info['bit_depth']} bit (should be {self.target_bit_depth} bit)")
                    print(f"    Issues: {', '.join(issues)}")
            
            if self.results['error_files']:
                print(f"\nFILES WITH ERRORS:")
                for file_info in self.results['error_files']:
                    print(f"  • {file_info['file_path']}")
                    print(f"    Error: {file_info['error']}")
    
    def save_report(self, output_file: str):
        """Save detailed validation report to a file."""
        with open(output_file, 'w') as f:
            f.write("WAV File Validation Report\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Target Sample Rate: {self.target_sample_rate} Hz\n")
            f.write(f"Target Bit Depth: {self.target_bit_depth} bit\n")
            f.write(f"Target Format: PCM\n\n")
            
            total_files = len(self.results['valid_files']) + len(self.results['invalid_files']) + len(self.results['error_files'])
            f.write(f"Total files processed: {total_files}\n")
            f.write(f"Valid files: {len(self.results['valid_files'])}\n")
            f.write(f"Invalid files: {len(self.results['invalid_files'])}\n")
            f.write(f"Error files: {len(self.results['error_files'])}\n\n")
            
            if self.results['valid_files']:
                f.write("VALID FILES:\n")
                f.write("-" * 20 + "\n")
                for file_info in self.results['valid_files']:
                    f.write(f"{file_info['file_path']}\n")
                f.write("\n")
            
            if self.results['invalid_files']:
                f.write("INVALID FILES:\n")
                f.write("-" * 20 + "\n")
                for file_info in self.results['invalid_files']:
                    f.write(f"File: {file_info['file_path']}\n")
                    f.write(f"  Sample Rate: {file_info['sample_rate']} Hz\n")
                    f.write(f"  Bit Depth: {file_info['bit_depth']} bit\n")
                    f.write(f"  Channels: {file_info['channels']}\n")
                    f.write(f"  Duration: {file_info['duration']:.2f} seconds\n\n")
            
            if self.results['error_files']:
                f.write("ERROR FILES:\n")
                f.write("-" * 20 + "\n")
                for file_info in self.results['error_files']:
                    f.write(f"File: {file_info['file_path']}\n")
                    f.write(f"  Error: {file_info['error']}\n\n")

def main():
    # Fixed directory path and parameters
    directory = str(PROJECT_ROOT / "src" / "data" / "raw")
    sample_rate = 16000  # 16 kHz
    bit_depth = 16       # 16-bit
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    validator = WAVValidator(sample_rate, bit_depth)
    validator.validate_directory(directory)
    validator.print_summary()
  
if __name__ == "__main__":
    main()