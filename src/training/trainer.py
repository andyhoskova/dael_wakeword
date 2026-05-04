"""
Wake Word Detection Model Trainer
=================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from pathlib import Path
import numpy as np
import time
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from logging.handlers import RotatingFileHandler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import warnings
from tqdm import tqdm
import os
import shutil

# Import your custom modules
from dataset_and_features_loader import create_dataloaders, DatasetLogger
from models import create_enhanced_wake_word_model, ModelLogger, EnhancedWakeWordModel


class TrainingLogger:
    """
    Comprehensive logging system for training with file rotation and TensorBoard integration.
    """
    
    def __init__(self, log_dir: Union[str, Path], experiment_name: str = "wake_word_training"):
        """
        Initialize training logger with multiple output streams.
        
        Args:
            log_dir: Directory to store log files
            experiment_name: Name of the training experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Python logger
        self.logger = logging.getLogger(f"trainer_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler with rotation
        log_file = self.experiment_dir / f"training_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
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
        
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(self.experiment_dir / "tensorboard")
        
        # Training metrics storage
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        self.logger.info(f"Training logger initialized for experiment: {experiment_name}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info(f"TensorBoard logs: {self.experiment_dir / 'tensorboard'}")
    
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
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str = "train"):
        """
        Log metrics to both file and TensorBoard.
        
        Args:
            metrics: Dictionary of metric name -> value
            epoch: Current epoch number
            phase: Training phase ("train", "val", "test")
        """
        # Log to file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:03d} [{phase.upper()}] - {metrics_str}")
        
        # Log to TensorBoard
        for metric_name, value in metrics.items():
            self.tb_writer.add_scalar(f"{metric_name}/{phase}", value, epoch)
        
        # Store in history if validation metrics
        if phase == "val":
            for metric_name, value in metrics.items():
                history_key = f"val_{metric_name.lower()}"
                if history_key in self.metrics_history:
                    self.metrics_history[history_key].append(value)
        elif phase == "train" and "loss" in metrics:
            self.metrics_history['train_loss'].append(metrics['loss'])
    
    def log_learning_rate(self, lr: float, epoch: int):
        """Log learning rate to TensorBoard and history."""
        self.tb_writer.add_scalar('Learning_Rate', lr, epoch)
        self.metrics_history['learning_rate'].append(lr)
    
    def save_metrics_history(self):
        """Save complete metrics history to JSON file."""
        history_file = self.experiment_dir / "metrics_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.logger.info(f"Metrics history saved to: {history_file}")
    
    def close(self):
        """Close TensorBoard writer and save metrics."""
        self.save_metrics_history()
        self.tb_writer.close()
        self.logger.info("Training logger closed successfully")


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_score: Current validation score (higher is better)
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
        
        return self.early_stop
    
    def restore_best_model(self, model: nn.Module):
        """Restore the best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelCheckpoint:
    """
    Model checkpointing system with automatic cleanup.
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path], keep_best: int = 3, keep_last: int = 2):
        """
        Initialize model checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best: Number of best checkpoints to keep
            keep_last: Number of latest checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_last = keep_last
        
        self.best_checkpoints = []  # List of (score, filepath) tuples
        self.last_checkpoints = []  # List of filepaths
    
    def save_checkpoint(self, 
                       model: nn.Module, 
                       optimizer: optim.Optimizer,
                       scheduler: Any,
                       epoch: int,
                       val_score: float,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """
        Save model checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            val_score: Validation score for ranking
            metrics: Additional metrics to save
            is_best: Whether this is the best model so far
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if is_best:
            filename = f"best_model_epoch_{epoch:03d}_{timestamp}.pt"
        else:
            filename = f"checkpoint_epoch_{epoch:03d}_{timestamp}.pt"
        
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_score': val_score,
            'metrics': metrics,
            'model_config': model.get_model_config(),
            'timestamp': timestamp,
            'is_best': is_best
        }
        
        torch.save(checkpoint, filepath)
        
        # Update checkpoint lists
        if is_best:
            self.best_checkpoints.append((val_score, filepath))
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)  # Sort by score descending
            
            # Keep only top N best checkpoints
            if len(self.best_checkpoints) > self.keep_best:
                _, old_filepath = self.best_checkpoints.pop()
                if old_filepath.exists():
                    old_filepath.unlink()
        
        self.last_checkpoints.append(filepath)
        
        # Keep only last N checkpoints
        if len(self.last_checkpoints) > self.keep_last:
            old_filepath = self.last_checkpoints.pop(0)
            if old_filepath.exists() and not any(filepath == old_filepath for _, filepath in self.best_checkpoints):
                old_filepath.unlink()
        
        return filepath


class WakeWordTrainer:
    """
    Complete training system for wake word detection models.
    """
    
    def __init__(self,
                 config_path: Union[str, Path],
                 experiment_name: Optional[str] = None):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to training configuration YAML file
            experiment_name: Custom experiment name
        """
        self.config_path = Path(config_path)
        self.experiment_name = experiment_name or f"wake_word_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load configuration
        self._load_config()
        
        # Initialize logging
        self.logger = TrainingLogger(
            log_dir=Path(self.config['paths']['log_dir_training']),
            experiment_name=self.experiment_name
        )
        
        # Initialize device
        self.device = self._setup_device()
        
        # Initialize components (will be set during training)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.dataloaders = None
        
        # Training state
        self.start_epoch = 0
        self.best_val_score = 0.0
        self.training_start_time = None
        
        # Initialize checkpointing and early stopping
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=Path(self.config['paths']['checkpoint_dir']),
            keep_best=self.config['training']['keep_best_checkpoints'],
            keep_last=self.config['training']['keep_last_checkpoints']
        )
        
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping_patience'],
            min_delta=self.config['training']['early_stopping_min_delta'],
            restore_best_weights=True
        )
        
        self.logger.info("WakeWordTrainer initialized successfully")
    
    def _load_config(self):
        """Load training configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate required configuration sections
        required_sections = ['model', 'training', 'data', 'paths']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _setup_device(self) -> str:
        """Setup and validate training device."""
        device_config = self.config['training'].get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = 'cpu'
                self.logger.warning("CUDA not available, using CPU")
        else:
            device = device_config
        
        self.logger.info(f"Training device: {device}")
        return device
    
    def _create_model(self) -> EnhancedWakeWordModel:
        """Create and initialize the wake word model."""
        self.logger.info("Creating wake word detection model...")
        
        model_config = self.config['model']
        model, device_used = create_enhanced_wake_word_model(
            input_features=model_config['input_features'],
            cnn_hidden=model_config['cnn_hidden'],
            transformer_heads=model_config['transformer_heads'],
            transformer_layers=model_config['transformer_layers'],
            transformer_hidden=model_config['transformer_hidden'],
            dropout_rate=model_config['dropout_rate'],
            classifier_hidden=model_config['classifier_hidden'],
            device=self.device,
            logger=ModelLogger(Path(self.config['paths']['log_dir_models']), "trainer_model")
        )
        
        return model
    
    def _create_dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Create training, validation, and test dataloaders."""
        self.logger.info("Creating dataloaders...")
        
        data_config = self.config['data']
        paths_config = self.config['paths']
        
        dataloaders = create_dataloaders(
            features_root_dir=paths_config['features_root_dir'],
            train_split_file=paths_config['train_split'],
            val_split_file=paths_config['val_split'],
            test_split_file=paths_config.get('test_split'),
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            use_specaugment=data_config['use_specaugment'],
            spec_aug_prob=data_config['spec_aug_prob'],
            spec_aug_params=data_config['spec_aug_params'],
            max_sequence_length=data_config.get('max_sequence_length'),
            padding_value=data_config.get('padding_value', 0.0),
            truncation_strategy=data_config.get('truncation_strategy', 'pad_truncate'),
            use_weighted_sampling=data_config['use_weighted_sampling'],
            logger=DatasetLogger(Path(self.config['paths']['log_dir_dataset_loader']), "trainer_dataset")
        )
        
        return dataloaders
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.config['training']['optimizer']
        optimizer_type = optimizer_config['type'].lower()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-2),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        self.logger.info(f"Created {optimizer_type.upper()} optimizer with lr={optimizer_config['learning_rate']}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.config['training'].get('scheduler')
        if not scheduler_config or not scheduler_config.get('enabled', False):
            return None
        
        scheduler_type = scheduler_config['type'].lower()
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # We want to maximize validation score
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-7),
            )
        elif scheduler_type == 'cosine_annealing':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get('T_0', 10),
                T_mult=scheduler_config.get('T_mult', 2),
                eta_min=scheduler_config.get('eta_min', 1e-7)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        self.logger.info(f"Created {scheduler_type} learning rate scheduler")
        return scheduler
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        criterion_config = self.config['training'].get('criterion', {'type': 'bce'})
        criterion_type = criterion_config['type'].lower()
        
        if criterion_type == 'bce':
            # Binary Cross Entropy with optional class weighting
            pos_weight = criterion_config.get('pos_weight')
            if pos_weight:
                pos_weight = torch.tensor(pos_weight).to(self.device)
            criterion = nn.BCELoss(weight=pos_weight)
        elif criterion_type == 'focal':
            # Focal loss for imbalanced datasets (custom implementation needed)
            alpha = criterion_config.get('alpha', 0.25)
            gamma = criterion_config.get('gamma', 2.0)
            criterion = self._create_focal_loss(alpha, gamma)
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")
        
        self.logger.info(f"Created {criterion_type.upper()} loss function")
        return criterion
    
    def _create_focal_loss(self, alpha: float, gamma: float) -> nn.Module:
        """Create focal loss function for imbalanced datasets."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, inputs, targets):
                bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
                return focal_loss.mean()
        
        return FocalLoss()
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Convert probabilities to binary predictions
        pred_binary = (predictions > 0.5).astype(int)
        
        # Calculate basic metrics
        accuracy = accuracy_score(targets, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, pred_binary, average='binary')
        
        # Calculate AUC-ROC
        try:
            auc = roc_auc_score(targets, predictions)
        except ValueError:
            auc = 0.0  # Handle case where only one class is present
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets, pred_binary)
        
        # Calculate additional metrics if confusion matrix is 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            specificity = 0.0
            sensitivity = recall
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'sensitivity': sensitivity
        }
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        num_batches = len(self.dataloaders['train'])
        
        # Create progress bar
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch:03d} [TRAIN]")
        
        for batch_idx, (features, labels) in enumerate(pbar):
            # Move data to device
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clipping', {}).get('enabled', False):
                max_norm = self.config['training']['gradient_clipping']['max_norm']
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg_Loss': f"{running_loss/(batch_idx+1):.4f}",
                'LR': f"{current_lr:.2e}"
            })
            
            # Log to TensorBoard every N batches
            if batch_idx % 100 == 0:
                global_step = epoch * num_batches + batch_idx
                self.logger.tb_writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                self.logger.tb_writer.add_scalar('Learning_Rate_batch', current_lr, global_step)
        
        avg_loss = running_loss / num_batches
        return {'loss': avg_loss}
    
    def _validate_epoch(self, epoch: int, dataloader_name: str = 'val') -> Dict[str, float]:
        """Validate/test for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        dataloader = self.dataloaders[dataloader_name]
        
        # Create progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [{dataloader_name.upper()}]")
        
        with torch.no_grad():
            for features, labels in pbar:
                # Move data to device
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                # Store predictions and targets
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        # Calculate metrics
        avg_loss = running_loss / len(dataloader)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _save_final_model(self, model_name: str = "final_wake_word_model"):
        """Save the final trained model."""
        final_models_dir = Path(self.config['paths']['final_models_dir'])
        final_models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save complete model
        model_path = final_models_dir / f"{model_name}_{timestamp}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_config(),
            'training_config': self.config,
            'final_metrics': self.logger.metrics_history,
            'timestamp': timestamp
        }, model_path)
        
        # Save model configuration separately
        config_path = final_models_dir / f"{model_name}_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.model.get_model_config(), f, indent=2)
        
        self.logger.info(f"Final model saved: {model_path}")
        return model_path
    
    def _export_model_for_deployment(self, model_name: str = "exported_wake_word_model"):
        """Export model for deployment (TorchScript, ONNX, etc.)."""
        export_dir = Path(self.config['paths']['export_dir'])
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create example input for tracing
        example_input = torch.randn(1, self.config['model']['input_features'], 100).to(self.device)
        
        # Initialize paths to None
        torchscript_path = None
        pytorch_path = None
        
        try:
            # Try TorchScript tracing first
            self.logger.info("Attempting TorchScript export with tracing...")
            traced_model = torch.jit.trace(self.model, example_input)
            torchscript_path = export_dir / f"{model_name}_torchscript_{timestamp}.pt"
            traced_model.save(torchscript_path)
            self.logger.info(f"TorchScript model exported: {torchscript_path}")
            
            # Verify TorchScript model
            loaded_model = torch.jit.load(torchscript_path)
            with torch.no_grad():
                original_output = self.model(example_input)
                traced_output = loaded_model(example_input)
                if torch.allclose(original_output, traced_output, atol=1e-6):
                    self.logger.info("TorchScript model verification successful")
                else:
                    self.logger.warning("TorchScript model verification failed - outputs don't match")
            
        except Exception as e:
            self.logger.error(f"Failed to export TorchScript model with tracing: {str(e)}")
            
            # Try with torch.jit.script instead
            try:
                self.logger.info("Attempting TorchScript export with scripting...")
                scripted_model = torch.jit.script(self.model)
                torchscript_path = export_dir / f"{model_name}_scripted_{timestamp}.pt"
                scripted_model.save(torchscript_path)
                self.logger.info(f"Scripted model exported successfully: {torchscript_path}")
            except Exception as script_error:
                self.logger.error(f"Failed to export scripted model: {str(script_error)}")
                
                # Final fallback - save as regular PyTorch state dict
                self.logger.info("Falling back to PyTorch state dict export...")
                pytorch_path = export_dir / f"{model_name}_pytorch_{timestamp}.pt"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_config': self.model.get_model_config(),
                    'timestamp': timestamp,
                    'export_type': 'pytorch_state_dict'
                }, pytorch_path)
                self.logger.info(f"PyTorch state dict saved: {pytorch_path}")
        
        # Export metadata regardless of export method success
        metadata = {
            'model_name': model_name,
            'export_timestamp': timestamp,
            'model_config': self.model.get_model_config(),
            'input_shape': [1, self.config['model']['input_features'], -1],  # -1 for variable time dimension
            'output_shape': [1],
            'deployment_info': {
                'recommended_device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'memory_requirements_mb': (self.model.total_params * 4) / (1024 * 1024),
                'preprocessing_required': 'mel_spectrogram_features'
            },
            'export_success': torchscript_path is not None or pytorch_path is not None,
            'export_type': 'torchscript' if torchscript_path else 'pytorch_state_dict'
        }
        
        metadata_path = export_dir / f"{model_name}_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Export metadata saved: {metadata_path}")
        
        # Return the successful export path
        return torchscript_path if torchscript_path else pytorch_path
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _estimate_time_remaining(self, epoch: int, total_epochs: int, epoch_time: float) -> str:
        """Estimate remaining training time."""
        remaining_epochs = total_epochs - epoch
        remaining_seconds = remaining_epochs * epoch_time
        return self._format_time(remaining_seconds)
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop with comprehensive monitoring and checkpointing.
        
        Args:
            resume_from_checkpoint: Path to checkpoint file to resume from
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING WAKE WORD DETECTION MODEL TRAINING")
        self.logger.info("=" * 80)
        
        try:
            # Initialize training components
            self.logger.info("Initializing training components...")
            
            # Create model
            self.model = self._create_model()
            
            # Create dataloaders
            self.dataloaders = self._create_dataloaders()
            
            # Create optimizer, scheduler, and criterion
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            self.criterion = self._create_criterion()
            
            # Resume from checkpoint if specified
            if resume_from_checkpoint:
                self._load_checkpoint(resume_from_checkpoint)
            
            # Log training configuration
            self._log_training_config()
            
            # Training parameters
            num_epochs = self.config['training']['num_epochs']
            validation_frequency = self.config['training'].get('validation_frequency', 1)
            
            # Start training timer
            self.training_start_time = time.time()
            epoch_times = []
            
            self.logger.info(f"Starting training for {num_epochs} epochs...")
            self.logger.info(f"Training samples: {len(self.dataloaders['train'].dataset)}")
            self.logger.info(f"Validation samples: {len(self.dataloaders['val'].dataset)}")
            if 'test' in self.dataloaders:
                self.logger.info(f"Test samples: {len(self.dataloaders['test'].dataset)}")
            
            # Training loop
            for epoch in range(self.start_epoch, num_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self._train_epoch(epoch)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_learning_rate(current_lr, epoch)
                
                # Validation phase
                val_metrics = None
                if epoch % validation_frequency == 0:
                    val_metrics = self._validate_epoch(epoch, 'val')
                    
                    # Log metrics
                    self.logger.log_metrics(train_metrics, epoch, 'train')
                    self.logger.log_metrics(val_metrics, epoch, 'val')
                    
                    # Calculate composite validation score (F1 score for early stopping and best model selection)
                    val_score = val_metrics['f1']
                    
                    # Check for best model
                    is_best = val_score > self.best_val_score
                    if is_best:
                        self.best_val_score = val_score
                        self.logger.info(f"🎉 New best model! F1: {val_score:.4f}")
                    
                    # Save checkpoint
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        val_score=val_score,
                        metrics=val_metrics,
                        is_best=is_best
                    )
                    
                    # Update learning rate scheduler
                    if self.scheduler:
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(val_score)
                        else:
                            self.scheduler.step()
                    
                    # Early stopping check
                    if self.early_stopping(val_score, self.model):
                        self.logger.info(f"Early stopping triggered after {epoch} epochs")
                        self.early_stopping.restore_best_model(self.model)
                        break
                
                # Calculate epoch timing
                epoch_time = time.time() - epoch_start_time
                epoch_times.append(epoch_time)
                avg_epoch_time = np.mean(epoch_times[-10:])  # Average of last 10 epochs
                
                # Calculate time estimates
                elapsed_time = time.time() - self.training_start_time
                remaining_time = self._estimate_time_remaining(epoch + 1, num_epochs, avg_epoch_time)
                
                # Log timing information
                self.logger.info(f"Epoch {epoch:03d} completed in {self._format_time(epoch_time)}")
                self.logger.info(f"Elapsed: {self._format_time(elapsed_time)} | "
                               f"Remaining: {remaining_time} | "
                               f"Avg/epoch: {self._format_time(avg_epoch_time)}")
                
                # Add timing to TensorBoard
                self.logger.tb_writer.add_scalar('Timing/epoch_time', epoch_time, epoch)
                self.logger.tb_writer.add_scalar('Timing/avg_epoch_time', avg_epoch_time, epoch)
                
                self.logger.info("-" * 60)
            
            # Final evaluation on test set if available
            if 'test' in self.dataloaders:
                self.logger.info("Evaluating on test set...")
                test_metrics = self._validate_epoch(100, 'test')  # Use epoch 100 for test
                self.logger.log_metrics(test_metrics, 100, 'test')
                
                # Log final test results
                self.logger.info("=" * 60)
                self.logger.info("FINAL TEST RESULTS:")
                for metric, value in test_metrics.items():
                    self.logger.info(f"  {metric.upper()}: {value:.4f}")
                self.logger.info("=" * 60)
            
            # Save final model
            final_model_path = self._save_final_model()
            
            # Export model for deployment
            exported_model_path = self._export_model_for_deployment()
            
            # Training completion summary
            total_training_time = time.time() - self.training_start_time
            self.logger.info("=" * 80)
            self.logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info(f"Total training time: {self._format_time(total_training_time)}")
            self.logger.info(f"Best validation F1 score: {self.best_val_score:.4f}")
            self.logger.info(f"Final model saved: {final_model_path}")
            if exported_model_path:
                self.logger.info(f"Exported model saved: {exported_model_path}")
            else:
                self.logger.warning("Model export failed, but training completed successfully")
            self.logger.info("=" * 80)
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self._save_interrupt_checkpoint()
        except Exception as e:
            self.logger.exception(f"Training failed with error: {str(e)}")
            raise
        finally:
            # Cleanup
            self.logger.close()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model and training state from checkpoint."""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_score = checkpoint.get('val_score', 0.0)
        
        self.logger.info(f"Resumed from epoch {self.start_epoch}, best val score: {self.best_val_score:.4f}")
    
    def _save_interrupt_checkpoint(self):
        """Save checkpoint when training is interrupted."""
        interrupt_dir = Path(self.config['paths']['checkpoint_dir']) / "interrupts"
        interrupt_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        interrupt_path = interrupt_dir / f"interrupt_checkpoint_{timestamp}.pt"
        
        checkpoint = {
            'epoch': getattr(self, 'current_epoch', 0),
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'interrupted_at': timestamp
        }
        
        torch.save(checkpoint, interrupt_path)
        self.logger.info(f"Interrupt checkpoint saved: {interrupt_path}")
    
    def _log_training_config(self):
        """Log comprehensive training configuration."""
        self.logger.info("Training Configuration:")
        self.logger.info("=" * 50)
        
        # Model configuration
        self.logger.info("Model:")
        model_config = self.config['model']
        for key, value in model_config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Training configuration
        self.logger.info("Training:")
        training_config = self.config['training']
        for key, value in training_config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        # Data configuration
        self.logger.info("Data:")
        data_config = self.config['data']
        for key, value in data_config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 50)


# Example usage and main execution
if __name__ == "__main__":
    """
    Example usage of the wake word trainer.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Wake Word Detection Model Trainer')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration YAML file')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    try:
        # Validate configuration file exists
        if not Path(args.config).exists():
            print(f"Error: Configuration file not found: {args.config}")
            print("Please provide a valid configuration file path.")
            exit(1)
        
        # Create trainer
        trainer = WakeWordTrainer(
            config_path=args.config,
            experiment_name=args.experiment_name
        )
        
        # Start training
        trainer.train(resume_from_checkpoint=args.resume)
        
        print("\n🎉 Training completed successfully!")
        print("Check the logs directory for detailed training logs and TensorBoard files.")
        print("Your trained model is ready for deployment!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise