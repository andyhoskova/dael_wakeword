"""
Train, Validation and Test Datasets Loader
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import random
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import json
import yaml


class DatasetLogger:

    def __init__(self, log_dir_dataset_loader: Union[str, Path], name: str = "dataset_loader"):
        self.log_dir_dataset_loader = Path(log_dir_dataset_loader)
        self.log_dir_dataset_loader.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir_dataset_loader / f"{name}_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)

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

    def __init__(self,
                 freq_mask_param: int = 20,
                 time_mask_param: int = 15,
                 num_freq_masks: int = 1,
                 num_time_masks: int = 1,
                 mask_value: float = 0.0):

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

        for _ in range(self.num_freq_masks):
            if freq_bins <= 1:
                continue
            f = random.randint(1, min(self.freq_mask_param, freq_bins))
            f0 = random.randint(0, max(0, freq_bins - f))
            spec[f0:f0 + f, :] = self.mask_value

        for _ in range(self.num_time_masks):
            if time_frames <= 1:
                continue
            t = random.randint(1, min(self.time_mask_param, time_frames))
            t0 = random.randint(0, max(0, time_frames - t))
            spec[:, t0:t0 + t] = self.mask_value

        return spec


class VariableLengthCollator:

    def __init__(self,
                 max_length: Optional[int] = None,
                 padding_value: float = 0.0,
                 truncation_strategy: str = 'pad'):

        self.max_length = max_length
        self.padding_value = padding_value
        self.truncation_strategy = truncation_strategy

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:

        features, labels = zip(*batch)

        time_dims = [f.shape[1] for f in features]

        if self.max_length is not None:
            target_length = self.max_length
        else:
            target_length = max(time_dims)

        processed_features = []

        for feature in features:
            current_length = feature.shape[1]

            if current_length == target_length:
                processed_features.append(feature)
            elif current_length < target_length:
                if self.truncation_strategy in ['pad', 'pad_truncate']:
                    pad_amount = target_length - current_length
                    padded = F.pad(feature, (0, pad_amount), value=self.padding_value)
                    processed_features.append(padded)
                else:
                    processed_features.append(feature)
            else:
                if self.truncation_strategy in ['truncate', 'pad_truncate']:
                    truncated = feature[:, :target_length]
                    processed_features.append(truncated)
                else:
                    processed_features.append(feature)

        try:
            batched_features = torch.stack(processed_features, dim=0)
            batched_labels = torch.stack(labels, dim=0)
            return batched_features, batched_labels
        except RuntimeError:
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

    def __init__(self,
                 split_file: Union[str, Path],
                 features_root_dir: Union[str, Path],
                 split_name: str,
                 use_specaugment: bool = False,
                 spec_aug_prob: float = 0.3,
                 spec_aug_params: Optional[Dict] = None,
                 # FIX: gaussian noise and time shift augmentations (were in config but never wired up)
                 gaussian_noise_std: float = 0.0,
                 time_shift_max_frames: int = 0,
                 logger: Optional[DatasetLogger] = None,
                 # FIX: default log_dir so logger=None doesn't crash with undefined `paths`
                 log_dir: Union[str, Path] = "logs/dataset_loader",
                 validate_features: bool = True,
                 expected_feature_shape: Optional[Tuple[int, int]] = None):

        self.split_file = Path(split_file)
        self.features_root_dir = Path(features_root_dir)
        self.split_name = split_name
        self.use_specaugment = use_specaugment
        self.spec_aug_prob = spec_aug_prob
        self.validate_features = validate_features
        self.expected_feature_shape = expected_feature_shape
        self.gaussian_noise_std = gaussian_noise_std
        self.time_shift_max_frames = time_shift_max_frames

        # FIX: use provided log_dir instead of referencing undefined `paths` variable
        if logger is None:
            self.logger = DatasetLogger(Path(log_dir), f"dataset_{split_name}")
        else:
            self.logger = logger

        if spec_aug_params is None:
            spec_aug_params = {
                'freq_mask_param': 20,
                'time_mask_param': 15,
                'num_freq_masks': 1,
                'num_time_masks': 1
            }

        self.spec_augment = SpecAugment(**spec_aug_params) if use_specaugment else None

        self.features_dir = self.features_root_dir / split_name

        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Dict[int, int] = {0: 0, 1: 0}
        self.corrupted_files: List[Path] = []
        self.missing_files: List[Path] = []

        self._load_split_file()
        self._validate_dataset()
        self._log_dataset_statistics()

    def _load_split_file(self):

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

        with open(self.split_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        processed_lines = 0
        self.logger.info(f"Processing {total_lines} entries from split file")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            if not line:
                continue

            if line.startswith('positive/'):
                label = 1
                class_dir = 'positive'
            elif line.startswith('negative/'):
                label = 0
                class_dir = 'negative'
            else:
                self.logger.warning(f"Line {line_num}: Cannot determine class for '{line}', skipping...")
                continue

            try:
                filename = line.split('/', 1)[1]
                pt_filename = filename.replace('.wav', '.pt')
                feature_file_path = self.features_dir / class_dir / pt_filename

                if not feature_file_path.exists():
                    self.logger.warning(f"Line {line_num}: Feature file not found: {feature_file_path}")
                    self.missing_files.append(feature_file_path)
                    continue

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

        if not self.validate_features:
            self.logger.info("Feature validation disabled, skipping...")
            return

        self.logger.info("Validating feature file integrity...")

        valid_samples = []
        validation_errors = 0

        for i, (feature_file_path, label) in enumerate(self.samples):
            try:
                data = torch.load(feature_file_path, map_location='cpu')

                if isinstance(data, dict):
                    if 'features' not in data:
                        self.logger.warning(f"Feature file missing 'features' key: {feature_file_path}")
                        self.corrupted_files.append(feature_file_path)
                        self.class_counts[label] -= 1
                        continue
                    features = data['features']
                else:
                    features = data

                if self.expected_feature_shape:
                    expected_freq, expected_time = self.expected_feature_shape
                    actual_freq, actual_time = features.shape

                    if actual_freq != expected_freq:
                        self.logger.warning(
                            f"Unexpected frequency dimension {actual_freq} (expected {expected_freq}): "
                            f"{feature_file_path}"
                        )

                    if abs(actual_time - expected_time) > expected_time * 0.5:
                        self.logger.debug(
                            f"Time dimension {actual_time} differs from expected {expected_time}: "
                            f"{feature_file_path}"
                        )

                if not isinstance(features, torch.Tensor):
                    self.logger.warning(f"Features not a torch.Tensor: {feature_file_path}")
                    self.corrupted_files.append(feature_file_path)
                    self.class_counts[label] -= 1
                    continue

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

        self.samples = valid_samples

        self.logger.info(f"Validation complete: {len(valid_samples)} valid samples, {validation_errors} errors")

        if len(self.samples) == 0:
            raise ValueError("No valid samples remaining after validation")

    def _detect_feature_dimensions(self):

        if len(self.samples) == 0:
            return None, None

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
            from collections import Counter
            most_common_freq = Counter(freq_dims).most_common(1)[0][0]
            most_common_time = Counter(time_dims).most_common(1)[0][0]

            self.logger.info(f"Detected feature dimensions: ({most_common_freq}, {most_common_time})")
            self.logger.info(f"Time dimension range: {min(time_dims)} - {max(time_dims)}")

            return most_common_freq, most_common_time

        return None, None

    def _log_dataset_statistics(self):

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

        if self.use_specaugment:
            self.logger.info(f"  SpecAugment enabled: {self.spec_aug_prob:.1%} probability")

        if self.gaussian_noise_std > 0.0:
            self.logger.info(f"  Gaussian noise: std={self.gaussian_noise_std:.4f}")

        if self.time_shift_max_frames > 0:
            self.logger.info(f"  Time shift: max ±{self.time_shift_max_frames} frames")

        if self.missing_files:
            self.logger.warning(f"  Missing feature files: {len(self.missing_files)}")

        if self.corrupted_files:
            self.logger.warning(f"  Corrupted feature files: {len(self.corrupted_files)}")

        self.logger.info("=" * 60)

    def get_statistics(self) -> Dict:

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
                'specaugment_enabled': self.use_specaugment,
                'probability': self.spec_aug_prob if self.use_specaugment else 0,
                'gaussian_noise_std': self.gaussian_noise_std,
                'time_shift_max_frames': self.time_shift_max_frames
            },
            'issues': {
                'missing_files': len(self.missing_files),
                'corrupted_files': len(self.corrupted_files)
            }
        }

    def save_statistics(self, output_path: Union[str, Path]):

        stats = self.get_statistics()
        stats['created_at'] = datetime.now().isoformat()

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Dataset statistics saved to: {output_path}")

    def get_sample_weights(self) -> List[float]:

        total_samples = len(self.samples)
        class_counts = self.class_counts.copy()

        class_weights = {}
        for class_id, count in class_counts.items():
            if count > 0:
                class_weights[class_id] = total_samples / (2.0 * count)
            else:
                class_weights[class_id] = 0.0

        sample_weights = []
        for _, label in self.samples:
            sample_weights.append(class_weights[label])

        return sample_weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        feature_file_path, label = self.samples[idx]

        try:
            data = torch.load(feature_file_path, map_location='cpu')

            if isinstance(data, dict):
                if 'features' in data:
                    features = data['features']
                else:
                    self.logger.warning(f"Missing 'features' key in {feature_file_path}, using fallback")
                    features = torch.zeros(186, 100)
            else:
                features = data

            # FIX: all augmentations share spec_aug_prob and only fire when use_specaugment=True
            # (use_specaugment is False for val/test, so augmentations are train-only)

            # SpecAugment
            if self.use_specaugment and random.random() < self.spec_aug_prob:
                features = self.spec_augment(features)

            # Gaussian noise augmentation (was in config but never implemented)
            if self.use_specaugment and self.gaussian_noise_std > 0.0 and random.random() < self.spec_aug_prob:
                noise = torch.randn_like(features) * self.gaussian_noise_std
                features = features + noise

            # Time shift augmentation (was in config but never implemented)
            # Uses torch.roll — wraps rather than zero-pads; acceptable for spectrograms
            if self.use_specaugment and self.time_shift_max_frames > 0 and random.random() < self.spec_aug_prob:
                shift = random.randint(-self.time_shift_max_frames, self.time_shift_max_frames)
                features = torch.roll(features, shift, dims=1)

            label_tensor = torch.tensor(label, dtype=torch.float32)

            return features, label_tensor

        except Exception as e:
            self.logger.error(f"Error loading sample {idx} from {feature_file_path}: {str(e)}")
            fallback_features = torch.zeros(186, 100)
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
    # FIX: gaussian noise and time shift params wired through (were dead code before)
    gaussian_noise_std: float = 0.0,
    time_shift_max_frames: int = 0,
    logger: Optional[DatasetLogger] = None,
    # FIX: default log_dir so logger=None doesn't crash with undefined `paths`
    log_dir: Union[str, Path] = "logs/dataset_loader",
    pin_memory: bool = True,
    persistent_workers: bool = True,
    # FIX: prefetch_factor was in config but never passed to DataLoader
    prefetch_factor: int = 2,
    max_sequence_length: Optional[int] = None,
    padding_value: float = 0.0,
    truncation_strategy: str = 'pad_truncate',
    use_weighted_sampling: bool = True
) -> Dict[str, DataLoader]:

    if logger is None:
        logger = DatasetLogger(Path(log_dir), "dataloader_factory")

    logger.info("Creating datasets and dataloaders...")

    collate_fn = VariableLengthCollator(
        max_length=max_sequence_length,
        padding_value=padding_value,
        truncation_strategy=truncation_strategy
    )

    dataloaders = {}

    train_dataset = WakeWordDataset(
        split_file=train_split_file,
        features_root_dir=features_root_dir,
        split_name='train',
        use_specaugment=use_specaugment,
        spec_aug_prob=spec_aug_prob,
        spec_aug_params=spec_aug_params,
        gaussian_noise_std=gaussian_noise_std,
        time_shift_max_frames=time_shift_max_frames,
        logger=logger
    )

    train_sampler = None
    train_shuffle = True

    if use_weighted_sampling:
        logger.info("Creating weighted sampler for training data to handle class imbalance...")

        sample_weights = train_dataset.get_sample_weights()

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_shuffle = False

        pos_weight = next(w for i, w in enumerate(sample_weights) if train_dataset.samples[i][1] == 1)
        neg_weight = next(w for i, w in enumerate(sample_weights) if train_dataset.samples[i][1] == 0)

        logger.info(f"  Positive class weight: {pos_weight:.4f}")
        logger.info(f"  Negative class weight: {neg_weight:.4f}")
        logger.info(f"  Weight ratio (pos/neg): {pos_weight / neg_weight:.2f}")

    # FIX: prefetch_factor is now actually passed to DataLoader
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
        collate_fn=collate_fn
    )

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
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn
    )

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
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=collate_fn
        )

    logger.info("Dataloaders created successfully:")
    for split_name, dataloader in dataloaders.items():
        dataset = dataloader.dataset
        logger.info(f"  {split_name}: {len(dataset)} samples, {len(dataloader)} batches")

        if max_sequence_length:
            logger.info(f"    - Fixed sequence length: {max_sequence_length}")
        else:
            logger.info("    - Variable sequence length (batch-wise padding)")
        logger.info(f"    - Truncation strategy: {truncation_strategy}")

        if split_name == 'train' and use_weighted_sampling:
            logger.info("    - Weighted sampling: ENABLED")
            pos_samples = dataset.class_counts[1]
            neg_samples = dataset.class_counts[0]
            logger.info(f"    - Original distribution: {pos_samples} pos, {neg_samples} neg")
        elif split_name == 'train':
            logger.info("    - Weighted sampling: DISABLED")

    return dataloaders


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Loader Test')
    parser.add_argument('--config', type=str, default='training_config.yaml',
                        help='Path to training configuration YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)

    paths = training_config['paths']
    data_config = training_config['data']
    data_loading_config = training_config.get('data_loading', {})

    try:
        logger = DatasetLogger(Path(paths['log_dir_dataset_loader']), "dataset_test")

        dataloaders = create_dataloaders(
            features_root_dir=paths['features_root_dir'],
            train_split_file=paths['train_split'],
            val_split_file=paths['val_split'],
            test_split_file=paths['test_split'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            use_specaugment=data_config['use_specaugment'],
            spec_aug_prob=data_config['spec_aug_prob'],
            spec_aug_params=data_config['spec_aug_params'],
            gaussian_noise_std=data_loading_config.get('gaussian_noise_std', 0.0),
            time_shift_max_frames=data_loading_config.get('time_shift_ms', 0),
            max_sequence_length=data_config.get('max_sequence_length'),
            truncation_strategy=data_config.get('truncation_strategy', 'pad_truncate'),
            use_weighted_sampling=data_config['use_weighted_sampling'],
            log_dir=paths['log_dir_dataset_loader'],
            pin_memory=data_loading_config.get('pin_memory', True),
            persistent_workers=data_loading_config.get('persistent_workers', True),
            prefetch_factor=data_loading_config.get('prefetch_factor', 2),
            logger=logger
        )

        logger.info("Testing data loading...")

        for split_name, dataloader in dataloaders.items():
            logger.info(f"Testing {split_name} dataloader...")

            if split_name == 'train' and data_config['use_weighted_sampling']:
                logger.info("Testing weighted sampling effectiveness...")
                class_counts = {0: 0, 1: 0}
                total_samples_tested = 0

                for i, (features, labels) in enumerate(dataloader):
                    logger.info(f"  Batch {i + 1}: features shape {features.shape}, labels shape {labels.shape}")

                    for label in labels:
                        class_counts[int(label.item())] += 1
                        total_samples_tested += 1

                    batch_pos_ratio = (labels == 1).sum().item() / len(labels)
                    logger.info(f"    Batch positive ratio: {batch_pos_ratio:.2%}")

                    if i >= 4:
                        break

                if total_samples_tested > 0:
                    overall_pos_ratio = class_counts[1] / total_samples_tested
                    logger.info(f"  Overall tested positive ratio: {overall_pos_ratio:.2%}")
            else:
                for i, (features, labels) in enumerate(dataloader):
                    logger.info(f"  Batch {i + 1}: features shape {features.shape}, labels shape {labels.shape}")
                    if i >= 2:
                        break

            stats_file = Path(paths['log_dir_dataset_loader']) / f"{split_name}_dataset_stats.json"
            dataloader.dataset.save_statistics(stats_file)

        logger.info("Dataset testing completed successfully!")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise