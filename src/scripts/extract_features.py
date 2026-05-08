import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import pickle
import warnings
warnings.filterwarnings("ignore")

# Directory Configuration
DATA_DIR   = Path("src/data/processed")
OUTPUT_DIR = Path("src/data/features")

class WakeWordFeatureExtractor:
    def __init__(self, 
                 sample_rate=16000,
                 n_mels=80,
                 n_mfcc=13,
                 n_fft=512,
                 win_length=400,  # 25ms at 16kHz
                 hop_length=160,  # 10ms at 16kHz
                 device='cuda'):
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.device = device
        
        print(f"Initializing feature extractor on {device}")
        print(f"Sample rate: {sample_rate}Hz, Mel bins: {n_mels}, MFCCs: {n_mfcc}")
        
        # Initialize transforms on GPU
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        ).to(device)
        
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'win_length': win_length,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'power': 2.0
            }
        ).to(device)
        
        # For computing deltas
        self.compute_deltas = T.ComputeDeltas().to(device)
        
        # Resampler in case audio isn't 16kHz
        self.resampler = None
        
    def load_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            waveform, orig_sr = torchaudio.load(file_path)
            
            # Resample if necessary
            if orig_sr != self.sample_rate:
                if self.resampler is None or self.resampler.orig_freq != orig_sr:
                    self.resampler = T.Resample(orig_sr, self.sample_rate).to(self.device)
                waveform = self.resampler(waveform.to(self.device))
            else:
                waveform = waveform.to(self.device)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            return waveform
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_features(self, waveform):
        """Extract all features from waveform"""
        if waveform is None:
            return None
            
        features = {}
        
        # 1. Log Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-8)  # Add small epsilon for numerical stability
        features['log_mel'] = log_mel_spec.squeeze(0)  # Remove batch dimension
        
        # 2. Delta Mel (temporal derivatives)
        delta_mel = self.compute_deltas(log_mel_spec)
        features['delta_mel'] = delta_mel.squeeze(0)
        
        # 3. MFCCs
        mfcc = self.mfcc_transform(waveform)
        features['mfcc'] = mfcc.squeeze(0)
        
        # 4. Delta MFCCs
        delta_mfcc = self.compute_deltas(mfcc)
        features['delta_mfcc'] = delta_mfcc.squeeze(0)
        
        # Stack all features along feature dimension
        stacked_features = torch.cat([
            features['log_mel'],      # 80 features
            features['delta_mel'],    # 80 features  
            features['mfcc'],         # 13 features
            features['delta_mfcc']    # 13 features
        ], dim=0)  # Total: 186 features
        
        return {
            'features': stacked_features,
            'individual': features,
            'shape': stacked_features.shape
        }
    
    def process_file(self, file_path, output_dir=OUTPUT_DIR):
        """Process a single audio file and save features"""
        # Load audio
        waveform = self.load_audio(file_path)
        if waveform is None:
            return False
            
        # Extract features
        feature_dict = self.extract_features(waveform)
        if feature_dict is None:
            return False
        
        # Create output path
        rel_path = Path(file_path).relative_to(DATA_DIR)
        output_path = Path(output_dir) / rel_path.with_suffix('.pt')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save features as PyTorch tensor
        torch.save({
            'features': feature_dict['features'].cpu(),  # Move back to CPU for storage
            'shape': feature_dict['shape'],
            'original_file': str(file_path)
        }, output_path)
        
        return True
    
    def process_dataset(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, batch_size=32):
        """Process entire dataset with GPU batching"""
        print(f"Processing dataset from {data_dir} to {output_dir}")
        
        # Find all audio files
        audio_files = []
        for split in ['train', 'test', 'validation']:
            for label in ['positive', 'negative']:
                pattern_dir = Path(data_dir) / split / label
                if pattern_dir.exists():
                    # Support common audio formats
                    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                        audio_files.extend(list(pattern_dir.glob(ext)))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process files with progress bar
        successful = 0
        failed = 0
        
        with tqdm(total=len(audio_files), desc="Extracting features") as pbar:
            for file_path in audio_files:
                success = self.process_file(file_path, output_dir)
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix({
                    'Success': successful, 
                    'Failed': failed,
                    'GPU_Memory': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                })
        
        print("\nFeature extraction complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Features saved to: {output_dir}")
        
        # Save extraction metadata
        metadata = {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_mfcc': self.n_mfcc,
            'feature_dim': self.n_mels * 2 + self.n_mfcc * 2,  # 186 total
            'successful_files': successful,
            'failed_files': failed
        }
        
        with open(Path(output_dir) / 'extraction_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        return successful, failed


def batch_process_with_multiprocessing(extractor, file_list, output_dir, num_workers=4):
    """Alternative processing with multiprocessing for I/O bound operations"""
    
    def process_single(file_path):
        return extractor.process_file(file_path, output_dir)
    
    successful = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(file_list), desc="Processing with multiprocessing") as pbar:
            futures = [executor.submit(process_single, fp) for fp in file_list]
            for future in futures:
                if future.result():
                    successful += 1
                pbar.update(1)
    
    return successful


def main():
    parser = argparse.ArgumentParser(description='Extract features for wake word detection')
    parser.add_argument('--data_dir', default=DATA_DIR, help='Input data directory')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output features directory')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel bins')
    parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCC coefficients')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Initialize extractor
    extractor = WakeWordFeatureExtractor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        device=args.device
    )
    
    # Process dataset
    successful, failed = extractor.process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    print("\n🚀 Feature extraction completed!")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    
    if torch.cuda.is_available():
        print(f"🔥 Peak GPU memory usage: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()