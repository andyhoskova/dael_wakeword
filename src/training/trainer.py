"""
Wake Word Detection Model Trainer
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
from datetime import datetime
from typing import Dict, Optional, Union, Any, Tuple
import logging
from logging.handlers import RotatingFileHandler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from tqdm import tqdm

from dataset_and_features_loader import create_dataloaders
from models import create_enhanced_wake_word_model, ModelLogger, EnhancedWakeWordModel


class TrainingLogger:

    def __init__(self, log_dir: Union[str, Path], experiment_name: str = "wake_word_training"):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"trainer_{experiment_name}")
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        log_file = self.experiment_dir / f"training_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,
            backupCount=10
        )
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.tb_writer = SummaryWriter(self.experiment_dir / "tensorboard")

        self.metrics_history: Dict[str, list] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'val_threshold': [],
            'learning_rate': []
        }

        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info(f"TensorBoard: {self.experiment_dir / 'tensorboard'}")

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
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:03d} [{phase.upper()}] - {metrics_str}")

        for metric_name, value in metrics.items():
            self.tb_writer.add_scalar(f"{metric_name}/{phase}", value, epoch)

        if phase == "val":
            for metric_name, value in metrics.items():
                history_key = f"val_{metric_name.lower()}"
                if history_key in self.metrics_history:
                    self.metrics_history[history_key].append(value)
        elif phase == "train" and "loss" in metrics:
            self.metrics_history['train_loss'].append(metrics['loss'])

    def log_learning_rate(self, lr: float, epoch: int):
        self.tb_writer.add_scalar('Learning_Rate', lr, epoch)
        self.metrics_history['learning_rate'].append(lr)

    def save_metrics_history(self):
        history_file = self.experiment_dir / "metrics_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.logger.info(f"Metrics history saved: {history_file}")

    def close(self):
        self.save_metrics_history()
        self.tb_writer.close()
        self.logger.info("Training logger closed")


class EarlyStopping:

    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()} \
                if self.restore_best_weights else None
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()} \
                if self.restore_best_weights else None

        return self.early_stop

    def restore_best_model(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelCheckpoint:

    def __init__(self, checkpoint_dir: Union[str, Path], keep_best: int = 3, keep_last: int = 2):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_last = keep_last

        self.best_checkpoints: list = []
        self.last_checkpoints: list = []

    def save_checkpoint(self,
                        model: nn.Module,
                        optimizer: optim.Optimizer,
                        scheduler: Any,
                        epoch: int,
                        val_score: float,
                        metrics: Dict[str, float],
                        is_best: bool = False):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = "best_model" if is_best else "checkpoint"
        filename = f"{prefix}_epoch_{epoch:03d}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename

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

        if is_best:
            self.best_checkpoints.append((val_score, filepath))
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

            if len(self.best_checkpoints) > self.keep_best:
                _, old_filepath = self.best_checkpoints.pop()
                if old_filepath.exists():
                    old_filepath.unlink()

        self.last_checkpoints.append(filepath)

        if len(self.last_checkpoints) > self.keep_last:
            old_filepath = self.last_checkpoints.pop(0)
            is_protected = any(fp == old_filepath for _, fp in self.best_checkpoints)
            if old_filepath.exists() and not is_protected:
                old_filepath.unlink()

        return filepath


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with binary label smoothing.

    During training, hard targets {0, 1} are replaced with soft targets:
        0  →  smoothing / 2
        1  →  1 - smoothing / 2

    For smoothing=0.1 that means  0 → 0.05  and  1 → 0.95.  This stops
    the model from driving logits towards ±∞ to satisfy hard targets,
    which is the primary cause of poorly-calibrated output probabilities.

    Label smoothing is applied ONLY when the module is in training mode
    (criterion.train()).  Validation / test losses use hard targets so
    the numbers remain interpretable and comparable across runs.
    """

    def __init__(self, smoothing: float = 0.1, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        if not 0.0 <= smoothing < 0.5:
            raise ValueError(f"smoothing must be in [0, 0.5), got {smoothing}")
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.training and self.smoothing > 0.0:
            # Soft targets: keeps gradient signal away from saturation regions
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


class WakeWordTrainer:

    def __init__(self,
                 config_path: Union[str, Path],
                 experiment_name: Optional[str] = None):

        self.config_path = Path(config_path)
        self.experiment_name = experiment_name or \
            f"wake_word_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._load_config()

        self.logger = TrainingLogger(
            log_dir=Path(self.config['paths']['log_dir_training']),
            experiment_name=self.experiment_name
        )

        self.device = self._setup_device()

        self.model: Optional[EnhancedWakeWordModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None
        self.criterion: Optional[nn.Module] = None
        self.dataloaders: Optional[Dict] = None

        # FIX: GradScaler for mixed precision — was configured but never created
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        self.start_epoch = 0
        self.best_val_score = 0.0
        self.training_start_time: Optional[float] = None

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

        self.logger.info("WakeWordTrainer initialised successfully")

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        for section in ['model', 'training', 'data', 'paths']:
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")

    def _setup_device(self) -> str:
        device_config = self.config['training'].get('device', 'auto')

        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"CUDA: {torch.cuda.get_device_name()}")
                self.logger.info(
                    f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
            else:
                device = 'cpu'
                self.logger.warning("CUDA not available — using CPU")
        else:
            device = device_config

        self.logger.info(f"Training device: {device}")
        return device

    def _create_model(self) -> EnhancedWakeWordModel:
        self.logger.info("Creating model...")

        model_config = self.config['model']
        model, _ = create_enhanced_wake_word_model(
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
        self.logger.info("Creating dataloaders...")

        data_config = self.config['data']
        paths_config = self.config['paths']
        # FIX: read data_loading section so pin_memory / persistent_workers /
        #      prefetch_factor / augmentation params are actually used
        data_loading_config = self.config.get('data_loading', {})

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
            # FIX: these were in data_loading config but never passed through
            gaussian_noise_std=data_loading_config.get('gaussian_noise_std', 0.0),
            time_shift_max_frames=data_loading_config.get('time_shift_ms', 0),
            max_sequence_length=data_config.get('max_sequence_length'),
            padding_value=data_config.get('padding_value', 0.0),
            truncation_strategy=data_config.get('truncation_strategy', 'pad_truncate'),
            use_weighted_sampling=data_config['use_weighted_sampling'],
            log_dir=paths_config['log_dir_dataset_loader'],
            pin_memory=data_loading_config.get('pin_memory', True),
            persistent_workers=data_loading_config.get('persistent_workers', True),
            prefetch_factor=data_loading_config.get('prefetch_factor', 2),
        )

        return dataloaders

    def _create_optimizer(self) -> optim.Optimizer:
        optimizer_config = self.config['training']['optimizer']
        opt_type = optimizer_config['type'].lower()

        if opt_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-2),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        self.logger.info(
            f"Optimizer: {opt_type.upper()}, lr={optimizer_config['learning_rate']}"
        )
        return optimizer

    def _create_scheduler(self):
        scheduler_config = self.config['training'].get('scheduler')
        if not scheduler_config or not scheduler_config.get('enabled', False):
            return None

        sched_type = scheduler_config['type'].lower()

        if sched_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-7),
            )
        elif sched_type == 'cosine_annealing':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get('T_0', 10),
                T_mult=scheduler_config.get('T_mult', 2),
                eta_min=scheduler_config.get('eta_min', 1e-7)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_type}")

        self.logger.info(f"LR scheduler: {sched_type}")
        return scheduler

    def _create_criterion(self) -> nn.Module:
        """
        FIX: use BCEWithLogitsLoss instead of BCELoss.

        BCELoss.weight expects a per-sample tensor of shape [batch_size].
        pos_weight (a scalar class weight) is only supported by
        BCEWithLogitsLoss.  The old code passed a scalar as BCELoss.weight,
        which was silently ignored or caused shape errors.

        The model forward() now returns raw logits; BCEWithLogitsLoss applies
        sigmoid internally and is numerically more stable.
        """
        criterion_config = self.config['training'].get('criterion', {'type': 'bce'})
        crit_type = criterion_config['type'].lower()

        if crit_type == 'bce':
            pos_weight_val = criterion_config.get('pos_weight')
            pos_weight_tensor = None
            if pos_weight_val is not None:
                pos_weight_tensor = torch.tensor([float(pos_weight_val)]).to(self.device)
                self.logger.info(f"BCEWithLogitsLoss pos_weight: {pos_weight_val}")

            smoothing = self.config.get('regularization', {}).get('label_smoothing', 0.0)
            if smoothing > 0.0:
                criterion = LabelSmoothingBCEWithLogitsLoss(
                    smoothing=smoothing,
                    pos_weight=pos_weight_tensor,
                )
                self.logger.info(
                    f"Label smoothing: {smoothing} "
                    f"(0 → {smoothing / 2:.3f}, 1 → {1.0 - smoothing / 2:.3f})"
                )
            else:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        elif crit_type == 'focal':
            alpha = criterion_config.get('alpha', 0.25)
            gamma = criterion_config.get('gamma', 2.0)
            criterion = self._create_focal_loss(alpha, gamma)

        else:
            raise ValueError(f"Unsupported criterion: {crit_type}")

        self.logger.info(f"Loss function: {crit_type.upper()}")
        return criterion

    def _create_focal_loss(self, alpha: float, gamma: float) -> nn.Module:
        """Focal loss — works with raw logits (applies sigmoid internally)."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, logits, targets):
                # sigmoid + BCE in one numerically stable step
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                pt = torch.exp(-bce)
                loss = self.alpha * (1 - pt) ** self.gamma * bce
                return loss.mean()

        return FocalLoss()

    # ------------------------------------------------------------------
    # FIX: threshold optimisation — was configured but hardcoded to 0.5
    # ------------------------------------------------------------------

    @staticmethod
    def _find_optimal_threshold(predictions: np.ndarray,
                                targets: np.ndarray) -> Tuple[float, float]:
        """
        Sweep thresholds from 0.05 to 0.95 and return the one that
        maximises F1, along with that F1 value.
        """
        best_thresh = 0.5
        best_f1 = 0.0

        for thresh in np.arange(0.05, 0.96, 0.01):
            pred_binary = (predictions >= thresh).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                targets, pred_binary, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        return best_thresh, best_f1

    def _calculate_metrics(self,
                            predictions: np.ndarray,
                            targets: np.ndarray,
                            threshold: float = 0.5) -> Dict[str, float]:
        pred_binary = (predictions >= threshold).astype(int)

        accuracy = accuracy_score(targets, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, pred_binary, average='binary', zero_division=0
        )

        try:
            auc = roc_auc_score(targets, predictions)
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(targets, pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            specificity = 0.0
            sensitivity = float(recall)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'threshold': threshold
        }

    # ------------------------------------------------------------------
    # FIX: learning rate warmup — was configured but never implemented
    # ------------------------------------------------------------------

    def _apply_warmup_lr(self, epoch: int):
        """
        Linearly ramp LR from base_lr * warmup_factor → base_lr over
        warmup_epochs.  Call this at the START of each warmup epoch,
        before training, so epoch 0 starts at warmup_factor * base_lr
        and epoch warmup_epochs-1 reaches (nearly) base_lr.
        """
        warmup_cfg = self.config.get('warmup', {})
        if not warmup_cfg.get('enabled', False):
            return

        warmup_epochs = warmup_cfg.get('warmup_epochs', 5)
        warmup_factor = warmup_cfg.get('warmup_factor', 0.1)
        base_lr = self.config['training']['optimizer']['learning_rate']

        if epoch < warmup_epochs:
            # linear ramp: epoch 0 → warmup_factor*base_lr, epoch warmup_epochs-1 → ~base_lr
            lr = base_lr * (warmup_factor + (1.0 - warmup_factor) * (epoch + 1) / warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.logger.debug(f"Warmup LR epoch {epoch}: {lr:.2e}")

    def _in_warmup(self, epoch: int) -> bool:
        warmup_cfg = self.config.get('warmup', {})
        return (
            warmup_cfg.get('enabled', False)
            and epoch < warmup_cfg.get('warmup_epochs', 5)
        )

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        FIX: implements mixed precision (autocast + GradScaler) and
        gradient accumulation, both of which were configured but
        previously not implemented.
        """
        self.model.train()
        self.criterion.train()   # enables label smoothing for training batches

        mp_cfg = self.config.get('mixed_precision', self.config.get('performance', {}).get('mixed_precision', {}))
        use_amp = mp_cfg.get('enabled', False) and self.device == 'cuda'

        ga_cfg = self.config.get('gradient_accumulation', {})
        accum_steps = ga_cfg.get('accumulation_steps', 1) if ga_cfg.get('enabled', False) else 1

        gc_cfg = self.config['training'].get('gradient_clipping', {})
        clip_enabled = gc_cfg.get('enabled', False)
        max_norm = gc_cfg.get('max_norm', 1.0)

        running_loss = 0.0
        num_batches = len(self.dataloaders['train'])

        self.optimizer.zero_grad()

        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch:03d} [TRAIN]")

        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # FIX: mixed precision forward pass
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = self.model(features)      # raw logits
                loss = self.criterion(outputs, labels)
                # scale loss for gradient accumulation so gradients are
                # averaged over the accumulation window, not summed
                loss = loss / accum_steps

            # FIX: scale + backward with GradScaler
            if use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # FIX: step only every accum_steps batches
            is_update_step = (batch_idx + 1) % accum_steps == 0 \
                             or (batch_idx + 1) == num_batches

            if is_update_step:
                if clip_enabled:
                    if use_amp and self.scaler is not None:
                        # must unscale before clipping when using GradScaler
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                if use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # running_loss tracks the raw (un-divided) loss for logging
            running_loss += loss.item() * accum_steps

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f"{loss.item() * accum_steps:.4f}",
                'Avg': f"{running_loss / (batch_idx + 1):.4f}",
                'LR': f"{current_lr:.2e}"
            })

            if batch_idx % 100 == 0:
                global_step = epoch * num_batches + batch_idx
                self.logger.tb_writer.add_scalar('Loss/train_batch', loss.item() * accum_steps, global_step)
                self.logger.tb_writer.add_scalar('LR/batch', current_lr, global_step)

        avg_loss = running_loss / num_batches
        return {'loss': avg_loss}

    def _validate_epoch(self, epoch: int, dataloader_name: str = 'val') -> Dict[str, float]:
        """
        FIX: applies sigmoid to logits before computing metrics (model now
        returns raw logits).  Also runs threshold optimisation when configured.
        """
        self.model.eval()
        self.criterion.eval()    # hard targets for val/test loss — keeps metrics interpretable

        running_loss = 0.0
        all_predictions: list = []
        all_targets: list = []

        dataloader = self.dataloaders[dataloader_name]
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [{dataloader_name.upper()}]")

        with torch.no_grad():
            for features, labels in pbar:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(features)
                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                # FIX: apply sigmoid here to get probabilities for metrics
                probs = torch.sigmoid(logits)
                all_predictions.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / len(dataloader)
        preds_arr = np.array(all_predictions)
        tgts_arr = np.array(all_targets)

        # FIX: threshold optimisation (was configured but hardcoded to 0.5)
        val_cfg = self.config.get('validation', {})
        if val_cfg.get('threshold_optimization', False) and dataloader_name == 'val':
            optimal_threshold, _ = self._find_optimal_threshold(preds_arr, tgts_arr)
        else:
            optimal_threshold = 0.5

        metrics = self._calculate_metrics(preds_arr, tgts_arr, threshold=optimal_threshold)
        metrics['loss'] = avg_loss

        return metrics

    def _save_final_model(self, model_name: str = "final_wake_word_model") -> Path:
        final_dir = Path(self.config['paths']['final_models_dir'])
        final_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = final_dir / f"{model_name}_{timestamp}.pt"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_config(),
            'training_config': self.config,
            'final_metrics': self.logger.metrics_history,
            'timestamp': timestamp
        }, model_path)

        config_path = final_dir / f"{model_name}_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.model.get_model_config(), f, indent=2)

        self.logger.info(f"Final model saved: {model_path}")
        return model_path

    def _export_model_for_deployment(self, model_name: str = "exported_wake_word_model") -> Optional[Path]:
        export_dir = Path(self.config['paths']['export_dir'])
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model.eval()

        example_input = torch.randn(
            1, self.config['model']['input_features'], 100
        ).to(self.device)

        torchscript_path = None
        pytorch_path = None

        try:
            self.logger.info("Attempting TorchScript trace export...")
            traced = torch.jit.trace(self.model, example_input)
            torchscript_path = export_dir / f"{model_name}_torchscript_{timestamp}.pt"
            traced.save(torchscript_path)
            self.logger.info(f"TorchScript model exported: {torchscript_path}")

            # Verify
            loaded = torch.jit.load(torchscript_path)
            with torch.no_grad():
                orig = self.model(example_input)
                trac = loaded(example_input)
                if torch.allclose(orig, trac, atol=1e-6):
                    self.logger.info("TorchScript export verified successfully")
                else:
                    self.logger.warning("TorchScript outputs diverged from original")

        except Exception as e:
            self.logger.error(f"TorchScript trace failed: {e}")

            try:
                self.logger.info("Falling back to torch.jit.script...")
                scripted = torch.jit.script(self.model)
                torchscript_path = export_dir / f"{model_name}_scripted_{timestamp}.pt"
                scripted.save(torchscript_path)
                self.logger.info(f"Scripted model exported: {torchscript_path}")
            except Exception as se:
                self.logger.error(f"Scripting also failed: {se}")
                pytorch_path = export_dir / f"{model_name}_pytorch_{timestamp}.pt"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_config': self.model.get_model_config(),
                    'timestamp': timestamp
                }, pytorch_path)
                self.logger.info(f"Saved PyTorch state dict fallback: {pytorch_path}")

        metadata = {
            'model_name': model_name,
            'export_timestamp': timestamp,
            'model_config': self.model.get_model_config(),
            'input_shape': [1, self.config['model']['input_features'], -1],
            'output': 'raw_logits — apply torch.sigmoid() for probabilities',
            'export_type': 'torchscript' if torchscript_path else 'pytorch_state_dict'
        }
        meta_path = export_dir / f"{model_name}_metadata_{timestamp}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return torchscript_path if torchscript_path else pytorch_path

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

    def _estimate_time_remaining(self, epoch: int, total_epochs: int, epoch_time: float) -> str:
        return self._format_time((total_epochs - epoch) * epoch_time)

    def train(self, resume_from_checkpoint: Optional[str] = None):

        self.logger.info("=" * 80)
        self.logger.info("STARTING WAKE WORD DETECTION MODEL TRAINING")
        self.logger.info("=" * 80)

        try:
            self.model = self._create_model()
            self.dataloaders = self._create_dataloaders()
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            self.criterion = self._create_criterion()

            # FIX: create GradScaler for mixed precision (was configured but never created)
            mp_cfg = self.config.get('mixed_precision',
                                      self.config.get('performance', {}).get('mixed_precision', {}))
            if mp_cfg.get('enabled', False) and self.device == 'cuda':
                self.scaler = torch.amp.GradScaler('cuda')
                self.logger.info("Mixed precision training: ENABLED (GradScaler active)")
            else:
                self.scaler = None
                self.logger.info("Mixed precision training: DISABLED")

            if resume_from_checkpoint:
                self._load_checkpoint(resume_from_checkpoint)

            self._log_training_config()

            num_epochs = self.config['training']['num_epochs']
            validation_frequency = self.config['training'].get('validation_frequency', 1)
            warmup_cfg = self.config.get('warmup', {})
            warmup_epochs = warmup_cfg.get('warmup_epochs', 0) if warmup_cfg.get('enabled', False) else 0

            self.training_start_time = time.time()
            epoch_times: list = []

            self.logger.info(f"Training for up to {num_epochs} epochs "
                             f"(warmup: {warmup_epochs} epochs)")
            self.logger.info(f"Train samples: {len(self.dataloaders['train'].dataset)}")
            self.logger.info(f"Val samples:   {len(self.dataloaders['val'].dataset)}")
            if 'test' in self.dataloaders:
                self.logger.info(f"Test samples:  {len(self.dataloaders['test'].dataset)}")

            final_epoch = self.start_epoch  # track for test eval

            for epoch in range(self.start_epoch, num_epochs):
                final_epoch = epoch
                epoch_start = time.time()

                # FIX: apply warmup LR before training this epoch
                self._apply_warmup_lr(epoch)

                train_metrics = self._train_epoch(epoch)

                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_learning_rate(current_lr, epoch)

                if epoch % validation_frequency == 0:
                    val_metrics = self._validate_epoch(epoch, 'val')

                    self.logger.log_metrics(train_metrics, epoch, 'train')
                    self.logger.log_metrics(val_metrics, epoch, 'val')

                    val_score = val_metrics['f1']
                    is_best = val_score > self.best_val_score

                    if is_best:
                        self.best_val_score = val_score
                        self.logger.info(
                            f"New best — F1: {val_score:.4f}, "
                            f"threshold: {val_metrics['threshold']:.2f}"
                        )

                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        val_score=val_score,
                        metrics=val_metrics,
                        is_best=is_best
                    )

                    # FIX: only step ReduceLROnPlateau after warmup is finished
                    if self.scheduler and not self._in_warmup(epoch):
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(val_score)
                        else:
                            self.scheduler.step()

                    if self.early_stopping(val_score, self.model):
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        self.early_stopping.restore_best_model(self.model)
                        break

                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                avg_epoch_time = float(np.mean(epoch_times[-10:]))
                elapsed = time.time() - self.training_start_time
                remaining = self._estimate_time_remaining(epoch + 1, num_epochs, avg_epoch_time)

                self.logger.info(
                    f"Epoch {epoch:03d} done in {self._format_time(epoch_time)} | "
                    f"Elapsed: {self._format_time(elapsed)} | Remaining: {remaining}"
                )
                self.logger.info("-" * 60)

            # Final test evaluation
            if 'test' in self.dataloaders:
                self.logger.info("Evaluating on test set...")
                # FIX: use actual final epoch instead of hardcoded 100
                test_metrics = self._validate_epoch(final_epoch, 'test')
                self.logger.log_metrics(test_metrics, final_epoch, 'test')

                self.logger.info("=" * 60)
                self.logger.info("FINAL TEST RESULTS:")
                for metric, value in test_metrics.items():
                    self.logger.info(f"  {metric.upper()}: {value:.4f}")
                self.logger.info("=" * 60)

            final_model_path = self._save_final_model()
            exported_path = self._export_model_for_deployment()

            total_time = time.time() - self.training_start_time
            self.logger.info("=" * 80)
            self.logger.info("TRAINING COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total time: {self._format_time(total_time)}")
            self.logger.info(f"Best val F1: {self.best_val_score:.4f}")
            self.logger.info(f"Final model: {final_model_path}")
            if exported_path:
                self.logger.info(f"Exported:    {exported_path}")
            self.logger.info("=" * 80)

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self._save_interrupt_checkpoint()
        except Exception as e:
            self.logger.exception(f"Training failed: {str(e)}")
            raise
        finally:
            self.logger.close()

    def _load_checkpoint(self, checkpoint_path: str):
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_score = checkpoint.get('val_score', 0.0)
        self.logger.info(
            f"Resumed from epoch {self.start_epoch}, best val F1: {self.best_val_score:.4f}"
        )

    def _save_interrupt_checkpoint(self):
        interrupt_dir = Path(self.config['paths']['checkpoint_dir']) / "interrupts"
        interrupt_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = interrupt_dir / f"interrupt_checkpoint_{timestamp}.pt"

        checkpoint = {
            'epoch': getattr(self, 'current_epoch', 0),
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'interrupted_at': timestamp
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Interrupt checkpoint saved: {path}")

    def _log_training_config(self):
        self.logger.info("Training Configuration:")
        self.logger.info("=" * 50)

        for section in ['model', 'training', 'data']:
            self.logger.info(f"{section.capitalize()}:")
            cfg = self.config[section]
            for k, v in cfg.items():
                if isinstance(v, dict):
                    self.logger.info(f"  {k}:")
                    for sk, sv in v.items():
                        self.logger.info(f"    {sk}: {sv}")
                else:
                    self.logger.info(f"  {k}: {v}")

        self.logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Wake Word Detection Trainer')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training_config.yaml')
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}")
        exit(1)

    trainer = WakeWordTrainer(
        config_path=args.config,
        experiment_name=args.experiment_name
    )
    trainer.train(resume_from_checkpoint=args.resume)