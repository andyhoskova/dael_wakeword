"""
Train, Validation and Test Datasets Loader
==========================================
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import numpy as np
import random
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import json
from collections import Counter
import warnings
import yaml


class DatasetLogger:
    
    def __init__(self, log_dir_dataset_loader: Union[str, Path], name: str = "dataset_loader"):
        """
        Initialize dataset logger with file rotation.
        
        Args:
            log_dir_dataset_loader: Directory to store log files
            name: Logger name
        """
        self.log_dir_dataset_loader = Path(log_dir_dataset_loader)
        self.log_dir_dataset_loader.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler with rotation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir_dataset_loader / f"{name}_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def exception(self, message: str):
        self.logger.exception(message)


class SpecAugment:
    """
    SpecAugment implementation for mel-spectrogram augmentation.
    Applies frequency and time masking to spectrograms.
    """
    
    def __init__(self, 
                 freq_mask_param: int = 20,
                 time_mask_param: int = 15,
                 num_freq_masks: int = 1,
                 num_time_masks: int = 1,
                 mask_value: float = 0.0):
        """
        Initialize SpecAugment parameters.
        
        Args:
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
            mask_value: Value to use for masking (0.0 for silence)
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to a spectrogram.
        
        Args:
            spectrogram: Input spectrogram of shape (freq_bins, time_frames)
        
        Returns:
            Augmented spectrogram
        """
        spec = spectrogram.clone()
        freq_bins, time_frames = spec.shape
        
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            if freq_bins <= 1:
                continue
            
            # Random mask size (at least 1, at most freq_mask_param)
            f = random.randint(1, min(self.freq_mask_param, freq_bins))
            # Random starting position
            f0 = random.randint(0, max(0, freq_bins - f))
            
            spec[f0:f0+f, :] = self.mask_value
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            if time_frames <= 1:
                continue
            
            # Random mask size (at least 1, at most time_mask_param)
            t = random.randint(1, min(self.time_mask_param, time_frames))
            # Random starting position
            t0 = random.randint(0, max(0, time_frames - t))
            
            spec[:, t0:t0+t] = self.mask_value
        
        return spec


class VariableLengthCollator:
    """
    Custom collate function to handle variable-length sequences.
    Pads or truncates sequences to a fixed length for batching.
    """
    
    def __init__(self, 
                 max_length: Optional[int] = None,
                 padding_value: float = 0.0,
                 truncation_strategy: str = 'pad'):
        """
        Initialize the collator.
        
        Args:
            max_length: Maximum sequence length. If None, uses the longest in each batch
            padding_value: Value to use for padding
            truncation_strategy: 'pad', 'truncate', or 'pad_truncate'
        """
        self.max_length = max_length
        self.padding_value = padding_value
        self.truncation_strategy = truncation_strategy
    
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a batch of variable-length sequences.
        
        Args:
            batch: List of (features, label) tuples
            
        Returns:
            Batched and padded features and labels
        """
        features, labels = zip(*batch)
        
        # Get dimensions
        freq_dim = features[0].shape[0]
        time_dims = [f.shape[1] for f in features]
        
        # Determine target length
        if self.max_length is not None:
            target_length = self.max_length
        else:
            target_length = max(time_dims)
        
        # Process each feature tensor
        processed_features = []
        
        for feature in features:
            current_length = feature.shape[1]
            
            if current_length == target_length:
                # Already correct length
                processed_features.append(feature)
            elif current_length < target_length:
                # Need padding
                if self.truncation_strategy in ['pad', 'pad_truncate']:
                    pad_amount = target_length - current_length
                    padded = F.pad(feature, (0, pad_amount), value=self.padding_value)
                    processed_features.append(padded)
                else:
                    # Truncate to current length (shouldn't happen in this case)
                    processed_features.append(feature)
            else:
                # Need truncation
                if self.truncation_strategy in ['truncate', 'pad_truncate']:
                    truncated = feature[:, :target_length]
                    processed_features.append(truncated)
                else:
                    # Pad to accommodate longer sequence
                    processed_features.append(feature)
        
        # Stack into batch
        try:
            batched_features = torch.stack(processed_features, dim=0)
            batched_labels = torch.stack(labels, dim=0)
            return batched_features, batched_labels
        except RuntimeError as e:
            # Fallback: pad to maximum length in batch
            max_len_in_batch = max(f.shape[1] for f in processed_features)
            fallback_features = []
            
            for feature in processed_features:
                if feature.shape[1] < max_len_in_batch:
                    pad_amount = max_len_in_batch - feature.shape[1]
                    padded = F.pad(feature, (0, pad_amount), value=self.padding_value)
                    fallback_features.append(padded)
                else:
                    fallback_features.append(feature)
            
            batched_features = torch.stack(fallback_features, dim=0)
            batched_labels = torch.stack(labels, dim=0)
            return batched_features, batched_labels


class WakeWordDataset(Dataset):
    """
    Production-grade dataset class for wake word detection.
    Handles loading of pre-extracted features with comprehensive error handling and logging.
    """
    
    def __init__(self,
                 split_file: Union[str, Path],
                 features_root_dir: Union[str, Path],
                 split_name: str,
                 use_specaugment: bool = False,
                 spec_aug_prob: float = 0.3,
                 spec_aug_params: Optional[Dict] = None,
                 logger: Optional[DatasetLogger] = None,
                 validate_features: bool = True,
                 expected_feature_shape: Optional[Tuple[int, int]] = None):
        """
        Initialize the wake word dataset.
        
        Args:
            split_file: Path to the split file (train_split.txt, val_split.txt, etc.)
            features_root_dir: Root directory containing feature files
            split_name: Name of the split (train, validation, test)
            use_specaugment: Whether to apply SpecAugment
            spec_aug_prob: Probability of applying augmentation to each sample
            spec_aug_params: SpecAugment parameters dict
            logger: Custom logger instance
            validate_features: Whether to validate feature file integrity
            expected_feature_shape: Expected shape of feature tensors (freq, time). If None, shapes are not validated.
        """
        self.split_file = Path(split_file)
        self.features_root_dir = Path(features_root_dir)
        self.split_name = split_name
        self.use_specaugment = use_specaugment
        self.spec_aug_prob = spec_aug_prob
        self.validate_features = validate_features
        self.expected_feature_shape = expected_feature_shape
        
        # Initialize logger
        if logger is None:
            self.logger = DatasetLogger(Path(paths['log_dir_dataset_loader']), f"dataset_{split_name}")
        else:
            self.logger = logger
        
        # Initialize SpecAugment
        if spec_aug_params is None:
            spec_aug_params = {
                'freq_mask_param': 20,
                'time_mask_param': 15,
                'num_freq_masks': 1,
                'num_time_masks': 1
            }
        
        self.spec_augment = SpecAugment(**spec_aug_params) if use_specaugment else None
        
        # Features directory for this split
        self.features_dir = self.features_root_dir / split_name
        
        # Storage for samples and statistics
        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Dict[int, int] = {0: 0, 1: 0}
        self.corrupted_files: List[Path] = []
        self.missing_files: List[Path] = []
        
        # Load and validate dataset
        self._load_split_file()
        self._validate_dataset()
        self._log_dataset_statistics()
    
    def _load_split_file(self):
        """
        Load the split file and populate sample list.
        """
        self.logger.info(f"Loading split file: {self.split_file}")
        self.logger.info(f"Features directory: {self.features_dir}")
        
        if not self.split_file.exists():
            error_msg = f"Split file not found: {self.split_file}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not self.features_dir.exists():
            error_msg = f"Features directory not found: {self.features_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        total_lines = 0
        processed_lines = 0
        
        with open(self.split_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
        
        self.logger.info(f"Processing {total_lines} entries from split file")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Determine class and directory
            if line.startswith('positive/'):
                label = 1
                class_dir = 'positive' 
            elif line.startswith('negative/'):
                label = 0
                class_dir = 'negative'
            else:
                self.logger.warning(f"Line {line_num}: Cannot determine class for '{line}', skipping...")
                continue
            
            # Extract filename and create feature file path
            try:
                filename = line.split('/', 1)[1]  # Remove class prefix
                pt_filename = filename.replace('.wav', '.pt')
                feature_file_path = self.features_dir / class_dir / pt_filename
                
                # Check if feature file exists
                if not feature_file_path.exists():
                    self.logger.warning(f"Line {line_num}: Feature file not found: {feature_file_path}")
                    self.missing_files.append(feature_file_path)
                    continue
                
                # Add to samples
                self.samples.append((feature_file_path, label))
                self.class_counts[label] += 1
                processed_lines += 1
                
            except Exception as e:
                self.logger.error(f"Line {line_num}: Error processing '{line}': {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {processed_lines}/{total_lines} entries")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found in the dataset")
    
    def _validate_dataset(self):
        """
        Validate dataset integrity by checking feature files.
        """
        if not self.validate_features:
            self.logger.info("Feature validation disabled, skipping...")
            return
        
        self.logger.info("Validating feature file integrity...")
        
        valid_samples = []
        validation_errors = 0
        
        for i, (feature_file_path, label) in enumerate(self.samples):
            try:
                # Try to load the feature file
                data = torch.load(feature_file_path, map_location='cpu')
                
                # Validate data structure
                if isinstance(data, dict):
                    if 'features' not in data:
                        self.logger.warning(f"Feature file missing 'features' key: {feature_file_path}")
                        self.corrupted_files.append(feature_file_path)
                        self.class_counts[label] -= 1
                        continue
                    features = data['features']
                else:
                    features = data
                
                # Validate shape (only check frequency dimension, time can vary)
                if self.expected_feature_shape:
                    expected_freq, expected_time = self.expected_feature_shape
                    actual_freq, actual_time = features.shape
                    
                    if actual_freq != expected_freq:
                        self.logger.warning(
                            f"Unexpected frequency dimension {actual_freq} (expected {expected_freq}): "
                            f"{feature_file_path}"
                        )
                    
                    # Only warn about time dimension if it's significantly different
                    if abs(actual_time - expected_time) > expected_time * 0.5:  # 50% tolerance
                        self.logger.debug(
                            f"Time dimension {actual_time} differs from expected {expected_time}: "
                            f"{feature_file_path}"
                        )
                
                # Validate data type
                if not isinstance(features, torch.Tensor):
                    self.logger.warning(f"Features not a torch.Tensor: {feature_file_path}")
                    self.corrupted_files.append(feature_file_path)
                    self.class_counts[label] -= 1
                    continue
                
                # Check for NaN or Inf values
                if torch.isnan(features).any() or torch.isinf(features).any():
                    self.logger.warning(f"Features contain NaN/Inf values: {feature_file_path}")
                    self.corrupted_files.append(feature_file_path)
                    self.class_counts[label] -= 1
                    continue
                
                valid_samples.append((feature_file_path, label))
                
            except Exception as e:
                self.logger.error(f"Error validating {feature_file_path}: {str(e)}")
                self.corrupted_files.append(feature_file_path)
                self.class_counts[label] -= 1
                validation_errors += 1
        
        # Update samples with only valid ones
        self.samples = valid_samples
        
        self.logger.info(f"Validation complete: {len(valid_samples)} valid samples, {validation_errors} errors")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples remaining after validation")
    
    def _detect_feature_dimensions(self):
        """
        Detect the actual feature dimensions from a sample of files.
        """
        if len(self.samples) == 0:
            return None, None
            
        # Sample first few files to detect dimensions
        sample_sizes = []
        freq_dims = []
        time_dims = []
        
        sample_files = self.samples[:min(10, len(self.samples))]
        
        for feature_file_path, _ in sample_files:
            try:
                data = torch.load(feature_file_path, map_location='cpu')
                if isinstance(data, dict) and 'features' in data:
                    features = data['features']
                else:
                    features = data
                
                if isinstance(features, torch.Tensor) and len(features.shape) == 2:
                    freq_dims.append(features.shape[0])
                    time_dims.append(features.shape[1])
                    
            except Exception:
                continue
        
        if freq_dims and time_dims:
            # Get most common dimensions
            from collections import Counter
            most_common_freq = Counter(freq_dims).most_common(1)[0][0]
            most_common_time = Counter(time_dims).most_common(1)[0][0]
            
            self.logger.info(f"Detected feature dimensions: ({most_common_freq}, {most_common_time})")
            self.logger.info(f"Time dimension range: {min(time_dims)} - {max(time_dims)}")
            
            return most_common_freq, most_common_time
        
        return None, None
    
    def _log_dataset_statistics(self):
        """
        Log comprehensive dataset statistics.
        """
        # Detect actual feature dimensions
        detected_freq, detected_time = self._detect_feature_dimensions()
        
        total_samples = len(self.samples)
        positive_samples = self.class_counts[1]
        negative_samples = self.class_counts[0]
        
        self.logger.info("=" * 60)
        self.logger.info(f"Dataset Statistics for '{self.split_name}' split:")
        self.logger.info(f"  Total samples: {total_samples}")
        self.logger.info(f"  Positive samples (wake word): {positive_samples}")
        self.logger.info(f"  Negative samples (background): {negative_samples}")
        
        if detected_freq and detected_time:
            self.logger.info(f"  Detected feature dimensions: ({detected_freq}, {detected_time})")
        
        if total_samples > 0:
            pos_ratio = positive_samples / total_samples
            neg_ratio = negative_samples / total_samples
            self.logger.info(f"  Class distribution: {pos_ratio:.2%} positive, {neg_ratio:.2%} negative")
            
            if pos_ratio > 0 and neg_ratio > 0:
                imbalance_ratio = negative_samples / positive_samples
                self.logger.info(f"  Class imbalance ratio (neg/pos): {imbalance_ratio:.2f}")
        
        # Log augmentation settings
        if self.use_specaugment:
            self.logger.info(f"  SpecAugment enabled: {self.spec_aug_prob:.1%} probability")
            expected_augmented = int(total_samples * self.spec_aug_prob)
            self.logger.info(f"  Expected augmented samples per epoch: ~{expected_augmented}")
        else:
            self.logger.info("  SpecAugment disabled")
        
        # Log issues
        if self.missing_files:
            self.logger.warning(f"  Missing feature files: {len(self.missing_files)}")
        
        if self.corrupted_files:
            self.logger.warning(f"  Corrupted feature files: {len(self.corrupted_files)}")
        
        self.logger.info("=" * 60)
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics as a dictionary.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            'split_name': self.split_name,
            'total_samples': len(self.samples),
            'positive_samples': self.class_counts[1],
            'negative_samples': self.class_counts[0],
            'class_distribution': {
                'positive_ratio': self.class_counts[1] / len(self.samples) if len(self.samples) > 0 else 0,
                'negative_ratio': self.class_counts[0] / len(self.samples) if len(self.samples) > 0 else 0
            },
            'augmentation': {
                'enabled': self.use_specaugment,
                'probability': self.spec_aug_prob if self.use_specaugment else 0
            },
            'issues': {
                'missing_files': len(self.missing_files),
                'corrupted_files': len(self.corrupted_files)
            }
        }
    
    def save_statistics(self, output_path: Union[str, Path]):
        """
        Save dataset statistics to a JSON file.
        
        Args:
            output_path: Path to save statistics JSON file
        """
        stats = self.get_statistics()
        stats['created_at'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Dataset statistics saved to: {output_path}")
    
    def get_sample_weights(self) -> List[float]:
        """
        Calculate sample weights for weighted sampling to handle class imbalance.
        
        Returns:
            List of weights for each sample, with minority class getting higher weights
        """
        # Get class counts
        total_samples = len(self.samples)
        class_counts = self.class_counts.copy()
        
        # Calculate class weights (inverse frequency)
        class_weights = {}
        for class_id, count in class_counts.items():
            if count > 0:
                class_weights[class_id] = total_samples / (2.0 * count)  # Normalized inverse frequency
            else:
                class_weights[class_id] = 0.0
        
        # Create sample weights
        sample_weights = []
        for _, label in self.samples:
            sample_weights.append(class_weights[label])
        
        return sample_weights
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (features, label) where features is a torch.Tensor
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        feature_file_path, label = self.samples[idx]
        
        try:
            # Load feature data
            data = torch.load(feature_file_path, map_location='cpu')
            
            if isinstance(data, dict):
                if 'features' in data:
                    features = data['features']
                else:
                    # Fallback for corrupted files that passed validation
                    self.logger.warning(f"Missing 'features' key in {feature_file_path}, using fallback")
                    features = torch.zeros(186, 100)  # Use reasonable default dimensions
            else:
                features = data
            
            # Apply SpecAugment if enabled and randomly selected
            if self.use_specaugment and random.random() < self.spec_aug_prob:
                features = self.spec_augment(features)
            
            # Convert label to tensor
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            return features, label_tensor
            
        except Exception as e:
            self.logger.error(f"Error loading sample {idx} from {feature_file_path}: {str(e)}")
            # Return fallback data
            fallback_features = torch.zeros(186, 100)  # Use reasonable default dimensions
            fallback_label = torch.tensor(0, dtype=torch.float32)
            return fallback_features, fallback_label


def create_dataloaders(
    features_root_dir: Union[str, Path],
    train_split_file: Union[str, Path],
    val_split_file: Union[str, Path],
    test_split_file: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    use_specaugment: bool = True,
    spec_aug_prob: float = 0.3,
    spec_aug_params: Optional[Dict] = None,
    logger: Optional[DatasetLogger] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    max_sequence_length: Optional[int] = None,
    padding_value: float = 0.0,
    truncation_strategy: str = 'pad_truncate',
    use_weighted_sampling: bool = True
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and optionally test sets.
    
    Args:
        features_root_dir: Root directory containing feature files
        train_split_file: Path to training split file
        val_split_file: Path to validation split file
        test_split_file: Path to test split file (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        use_specaugment: Whether to use SpecAugment on training data
        spec_aug_prob: Probability of applying augmentation
        spec_aug_params: SpecAugment parameters
        logger: Custom logger instance
        pin_memory: Whether to pin memory in dataloaders
        persistent_workers: Whether to keep workers persistent
        max_sequence_length: Maximum sequence length for padding/truncation
        padding_value: Value to use for padding shorter sequences
        truncation_strategy: How to handle variable lengths ('pad', 'truncate', 'pad_truncate')
        use_weighted_sampling: Whether to use weighted sampling for training data to handle class imbalance
    
    Returns:
        Dictionary containing dataloaders
    """
    if logger is None:
        logger = DatasetLogger(Path(paths['log_dir_dataset_loader']), "dataloader_factory")
    
    logger.info("Creating datasets and dataloaders...")
    
    # Create collate function for variable-length sequences
    collate_fn = VariableLengthCollator(
        max_length=max_sequence_length,
        padding_value=padding_value,
        truncation_strategy=truncation_strategy
    )
    
    dataloaders = {}
    
    # Training dataset (with augmentation)
    train_dataset = WakeWordDataset(
        split_file=train_split_file,
        features_root_dir=features_root_dir,
        split_name='train',
        use_specaugment=use_specaugment,
        spec_aug_prob=spec_aug_prob,
        spec_aug_params=spec_aug_params,
        logger=logger
    )
    
    # Create weighted sampler for training if requested
    train_sampler = None
    train_shuffle = True
    
    if use_weighted_sampling:
        logger.info("Creating weighted sampler for training data to handle class imbalance...")
        
        # Get sample weights from the dataset
        sample_weights = train_dataset.get_sample_weights()
        
        # Create WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow replacement to ensure balanced sampling
        )
        
        # When using sampler, shuffle must be False
        train_shuffle = False
        
        # Log weighted sampling information
        pos_weight = sample_weights[0] if train_dataset.samples[0][1] == 1 else next(w for i, w in enumerate(sample_weights) if train_dataset.samples[i][1] == 1)
        neg_weight = sample_weights[0] if train_dataset.samples[0][1] == 0 else next(w for i, w in enumerate(sample_weights) if train_dataset.samples[i][1] == 0)
        
        logger.info(f"  Positive class weight: {pos_weight:.4f}")
        logger.info(f"  Negative class weight: {neg_weight:.4f}")
        logger.info(f"  Weight ratio (pos/neg): {pos_weight/neg_weight:.2f}")
        logger.info(f"  Sampling with replacement enabled")
    
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,  # Ensure consistent batch sizes
        collate_fn=collate_fn
    )
    
    # Validation dataset
    val_dataset = WakeWordDataset(
        split_file=val_split_file,
        features_root_dir=features_root_dir,
        split_name='validation',
        use_specaugment=False,
        logger=logger
    )
    
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn
    )
    
    # Test dataset
    if test_split_file is not None:
        test_dataset = WakeWordDataset(
            split_file=test_split_file,
            features_root_dir=features_root_dir,
            split_name='test',
            use_specaugment=False,  
            logger=logger
        )
        
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn
        )
    
    # Log summary with sequence length info and weighted sampling
    logger.info("Dataloaders created successfully:")
    for split_name, dataloader in dataloaders.items():
        dataset = dataloader.dataset
        logger.info(f"  {split_name}: {len(dataset)} samples, {len(dataloader)} batches")
        
        # Log sequence length strategy
        if max_sequence_length:
            logger.info(f"    - Fixed sequence length: {max_sequence_length}")
        else:
            logger.info(f"    - Variable sequence length (batch-wise padding)")
        logger.info(f"    - Truncation strategy: {truncation_strategy}")
        
        # Log weighted sampling info for training
        if split_name == 'train' and use_weighted_sampling:
            logger.info(f"    - Weighted sampling: ENABLED")
            pos_samples = dataset.class_counts[1]
            neg_samples = dataset.class_counts[0]
            logger.info(f"    - Original distribution: {pos_samples} pos, {neg_samples} neg")
            logger.info(f"    - Expected balanced sampling per epoch")
        elif split_name == 'train':
            logger.info(f"    - Weighted sampling: DISABLED")
    
    return dataloaders


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the dataset loader.
    """

     # Load paths from YAML configuration
    with open('configs/training_config.yaml', 'r') as f:
        training_config = yaml.safe_load(f)
        paths = training_config['paths']

    # Configuration
    CONFIG = {
        'features_root_dir': paths['features_root_dir'],
        'train_split': paths['train_split'],
        'val_split': paths['val_split'],
        'test_split': paths['test_split'],
        'batch_size': 32,
        'num_workers': 4,
        'use_specaugment': True,
        'spec_aug_prob': 0.3,
        'max_sequence_length': 300,  # Set a reasonable max length
        'truncation_strategy': 'pad_truncate',
        'use_weighted_sampling': True,  # Enable weighted sampling for imbalanced dataset
        'spec_aug_params': {
            'freq_mask_param': 20,
            'time_mask_param': 15,
            'num_freq_masks': 1,
            'num_time_masks': 1
        }
    }
    
    try:
        # Create logger
        logger = DatasetLogger(Path(paths['log_dir_dataset_loader']), "dataset_test")
        
        # Create dataloaders
        dataloaders = create_dataloaders(
            features_root_dir=CONFIG['features_root_dir'],
            train_split_file=CONFIG['train_split'],
            val_split_file=CONFIG['val_split'],
            test_split_file=CONFIG['test_split'],
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            use_specaugment=CONFIG['use_specaugment'],
            spec_aug_prob=CONFIG['spec_aug_prob'],
            spec_aug_params=CONFIG['spec_aug_params'],
            max_sequence_length=CONFIG['max_sequence_length'],
            truncation_strategy=CONFIG['truncation_strategy'],
            use_weighted_sampling=CONFIG['use_weighted_sampling'],
            logger=logger
        )
        
        # Test loading a few batches
        logger.info("Testing data loading...")
        
        for split_name, dataloader in dataloaders.items():
            logger.info(f"Testing {split_name} dataloader...")
            
            # For training, test class distribution with weighted sampling
            if split_name == 'train' and CONFIG['use_weighted_sampling']:
                logger.info("Testing weighted sampling effectiveness...")
                class_counts = {0: 0, 1: 0}
                total_samples_tested = 0
                
                # Test several batches to see if sampling is working
                for i, (features, labels) in enumerate(dataloader):
                    logger.info(f"  Batch {i+1}: features shape {features.shape}, labels shape {labels.shape}")
                    
                    # Count classes in this batch
                    for label in labels:
                        class_counts[int(label.item())] += 1
                        total_samples_tested += 1
                    
                    batch_pos_ratio = (labels == 1).sum().item() / len(labels)
                    logger.info(f"    Batch positive ratio: {batch_pos_ratio:.2%}")
                    
                    # Test only first few batches
                    if i >= 4:
                        break
                
                # Log overall sampling distribution
                if total_samples_tested > 0:
                    overall_pos_ratio = class_counts[1] / total_samples_tested
                    logger.info(f"  Overall tested positive ratio: {overall_pos_ratio:.2%}")
                    logger.info(f"  Positive samples in tested batches: {class_counts[1]}")
                    logger.info(f"  Negative samples in tested batches: {class_counts[0]}")
            else:
                for i, (features, labels) in enumerate(dataloader):
                    logger.info(f"  Batch {i+1}: features shape {features.shape}, labels shape {labels.shape}")
                    
                    # Test only first few batches
                    if i >= 2:
                        break
            
            # Save statistics
            stats_file = Path(paths['log_dir_dataset_loader']) / f"{split_name}_dataset_stats.json"
            dataloader.dataset.save_statistics(stats_file)
        
        logger.info("Dataset testing completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise