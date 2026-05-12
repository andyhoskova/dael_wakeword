"""
Wake Word Detection Models
==========================

Key fixes vs previous version:
  1. EnhancedCNNFrontend no longer does global pooling — it preserves the time
     dimension and returns (batch, hidden_dim, T') so the transformer actually
     has a temporal sequence to attend over.
  2. EnhancedStreamingTransformer now receives (batch, hidden_dim, T'), permutes
     to (T', batch, hidden_dim), runs attention across time steps (not across
     batch items), then mean-pools back to (batch, hidden_dim).
  3. ModuleList replaces the hand-numbered attention_1 … attention_6 attributes
     and the getattr() call in _apply_transformer_layer, which blocked
     torch.jit.script and made adding/removing layers fragile.
  4. Sigmoid is removed from EnhancedWakeWordModel.forward so that the trainer
     can use BCEWithLogitsLoss (numerically more stable and the only correct
     way to use pos_weight). Apply torch.sigmoid() manually at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import json


class ModelLogger:

    def __init__(self, log_dir: Union[str, Path], name: str = "model_builder"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=20 * 1024 * 1024,
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


class EnhancedCNNFrontend(nn.Module):

    def __init__(self,
                 input_features: int = 186,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.15,
                 use_attention: bool = True,
                 logger: Optional[ModelLogger] = None):
        super(EnhancedCNNFrontend, self).__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        # First conv block: input_features → 128 channels, T → T/2
        self.conv1 = nn.Conv1d(input_features, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second conv block: 128 → 256, T/2 → T/4 (with residual)
        self.conv2a = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv2b = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm1d(256)
        self.bn2b = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.residual2 = nn.Conv1d(128, 256, kernel_size=1)

        # Third conv block: 256 → 384, T/4 → T/8 (with residual)
        self.conv3a = nn.Conv1d(256, 384, kernel_size=3, padding=1)
        self.conv3b = nn.Conv1d(384, 384, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm1d(384)
        self.bn3b = nn.BatchNorm1d(384)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.residual3 = nn.Conv1d(256, 384, kernel_size=1)

        # Fourth conv block: 384 → hidden_dim, T/8 stays (with residual)
        self.conv4a = nn.Conv1d(384, hidden_dim, kernel_size=3, padding=1)
        self.conv4b = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm1d(hidden_dim)
        self.bn4b = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.residual4 = nn.Conv1d(384, hidden_dim, kernel_size=1)

        # Channel attention: re-weights each of the hidden_dim channels globally
        # AdaptiveAvgPool1d(1) collapses time → (B, hidden_dim, 1), which is
        # broadcast back across time after the sigmoid gate.  This is channel-
        # wise attention and intentionally preserves the time dimension.
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(hidden_dim, hidden_dim // 16, 1),
                nn.ReLU(inplace=False),
                nn.Conv1d(hidden_dim // 16, hidden_dim, 1),
                nn.Sigmoid()
            )

        if logger:
            logger.info(f"CNN Frontend: {input_features} freq bins → {hidden_dim} channels, T → T/8")
            logger.info(f"  Dropout: {dropout_rate}, Channel attention: {use_attention}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_features, T)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        x = self.pool1(x)
        x = self.dropout1(x)
        # → (B, 128, T/2)

        # Block 2 + residual
        residual = self.residual2(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.relu(x, inplace=False)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = x + residual
        x = F.relu(x, inplace=False)
        x = self.pool2(x)
        x = self.dropout2(x)
        # → (B, 256, T/4)

        # Block 3 + residual
        residual = self.residual3(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = F.relu(x, inplace=False)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = x + residual
        x = F.relu(x, inplace=False)
        x = self.pool3(x)
        x = self.dropout3(x)
        # → (B, 384, T/8)

        # Block 4 + residual
        residual = self.residual4(x)
        x = self.conv4a(x)
        x = self.bn4a(x)
        x = F.relu(x, inplace=False)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = x + residual
        x = F.relu(x, inplace=False)
        x = self.dropout4(x)
        # → (B, hidden_dim, T/8)

        # Channel attention (keeps time dim intact)
        if self.use_attention:
            attention_weights = self.channel_attention(x)  # (B, hidden_dim, 1)
            x = x * attention_weights                       # (B, hidden_dim, T/8)

        # FIX: do NOT pool over time here.  Return the full temporal feature map
        # so the transformer has an actual sequence to attend over.
        return x  # (B, hidden_dim, T/8)

    def get_output_dim(self) -> int:
        return self.hidden_dim


class EnhancedStreamingTransformer(nn.Module):

    def __init__(self,
                 input_dim: int = 512,
                 num_heads: int = 16,
                 num_layers: int = 6,
                 hidden_dim: int = 1024,
                 dropout_rate: float = 0.1,
                 logger: Optional[ModelLogger] = None):
        super(EnhancedStreamingTransformer, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
            )

        # FIX: use ModuleList instead of hand-numbered attributes + getattr().
        # ModuleList is TorchScript-compatible when iterated in forward().
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=False  # expects (T, B, D)
            )
            for _ in range(num_layers)
        ])

        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            self._make_feedforward(input_dim, hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Final projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim)
        )
        self.final_norm = nn.LayerNorm(input_dim)

        if logger:
            logger.info(
                f"Transformer: {input_dim}D input, {num_heads} heads, "
                f"{num_layers} layers, {hidden_dim}D FFN"
            )

    @staticmethod
    def _make_feedforward(input_dim: int, hidden_dim: int, dropout_rate: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FIX: x arrives as (B, D, T') from the CNN (temporal features intact).
        # Permute to (T', B, D) which is what nn.MultiheadAttention expects when
        # batch_first=False.  This means attention is computed ACROSS TIME STEPS
        # within each sample, not across samples within a batch.
        x = x.permute(2, 0, 1)  # (T', B, D)

        # Apply transformer layers (ModuleList iteration is TorchScript-safe)
        for attn, norm1, norm2, ff in zip(
            self.attention_layers,
            self.norm1_layers,
            self.norm2_layers,
            self.ff_layers
        ):
            # Pre-norm self-attention
            residual = x
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = residual + attn_out

            # Pre-norm feed-forward
            residual = x
            x_norm = norm2(x)
            x = residual + ff(x_norm)

        # Final norm + projection
        x = self.final_norm(x)
        residual = x
        x = self.output_proj(x)
        x = x + residual

        # FIX: mean-pool over the time dimension to get one vector per sample.
        # x is (T', B, D) → mean over dim 0 → (B, D)
        x = x.mean(dim=0)

        return x  # (B, D)

    def get_output_dim(self) -> int:
        return self.input_dim


class EnhancedWakeWordModel(nn.Module):

    def __init__(self,
                 input_features: int = 186,
                 cnn_hidden: int = 512,
                 transformer_heads: int = 16,
                 transformer_layers: int = 6,
                 transformer_hidden: int = 1024,
                 dropout_rate: float = 0.15,
                 classifier_hidden: List[int] = [256, 128, 64],
                 use_attention: bool = True,
                 logger: Optional[ModelLogger] = None):
        super(EnhancedWakeWordModel, self).__init__()

        self.input_features = input_features
        self.cnn_hidden = cnn_hidden
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_hidden = transformer_hidden
        self.dropout_rate = dropout_rate
        self.classifier_hidden = classifier_hidden
        self.use_attention = use_attention

        self.cnn_frontend = EnhancedCNNFrontend(
            input_features=input_features,
            hidden_dim=cnn_hidden,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            logger=logger
        )

        self.transformer_backend = EnhancedStreamingTransformer(
            input_dim=cnn_hidden,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            hidden_dim=transformer_hidden,
            # Transformer uses lower dropout than CNN — standard practice
            dropout_rate=dropout_rate * 0.5,
            logger=logger
        )

        # Build classifier with progressive dropout increase
        if len(classifier_hidden) == 3 and classifier_hidden == [256, 128, 64]:
            self.classifier = nn.Sequential(
                nn.Linear(cnn_hidden, 256),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(256),

                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout_rate + 0.05),
                nn.BatchNorm1d(128),

                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout_rate + 0.1),
                nn.BatchNorm1d(64),

                nn.Linear(64, 1)
            )
        else:
            layers = []
            prev_dim = cnn_hidden
            for i, h in enumerate(classifier_hidden):
                layers.extend([
                    nn.Linear(prev_dim, h),
                    nn.GELU(),
                    nn.Dropout(dropout_rate + i * 0.05),
                    nn.BatchNorm1d(h)
                ])
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))
            self.classifier = nn.Sequential(*layers)

        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if logger:
            logger.info("=" * 70)
            logger.info("WAKE WORD DETECTION MODEL INITIALIZED")
            logger.info("=" * 70)
            logger.info(f"Total parameters:     {self.total_params:,}")
            logger.info(f"Trainable parameters: {self.trainable_params:,}")
            model_size_mb = (self.total_params * 4) / (1024 * 1024)
            logger.info(f"Estimated model size: {model_size_mb:.2f} MB")
            logger.info("=" * 70)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_features, T)

        # CNN: (B, input_features, T) → (B, cnn_hidden, T/8)
        cnn_features = self.cnn_frontend(x)

        # Transformer: (B, cnn_hidden, T/8) → (B, cnn_hidden)
        transformer_features = self.transformer_backend(cnn_features)

        # Classifier: (B, cnn_hidden) → (B, 1) → (B,)
        logits = self.classifier(transformer_features)

        # FIX: return raw logits (no sigmoid).
        # The trainer uses BCEWithLogitsLoss which applies sigmoid internally
        # and is numerically stable.  At inference time, apply torch.sigmoid()
        # to get probabilities.
        return logits.view(-1)

    def get_model_config(self) -> Dict[str, Any]:
        return {
            'model_version': '4.0.0',
            'model_type': 'CNN-Transformer Wake Word Detector (temporal attention fixed)',
            'input_features': self.input_features,
            'cnn_hidden': self.cnn_hidden,
            'transformer_heads': self.transformer_heads,
            'transformer_layers': self.transformer_layers,
            'transformer_hidden': self.transformer_hidden,
            'dropout_rate': self.dropout_rate,
            'classifier_hidden': self.classifier_hidden,
            'use_attention': self.use_attention,
            'total_parameters': self.total_params,
            'trainable_parameters': self.trainable_params,
            'torchscript_compatible': True,
            'output': 'raw_logits_apply_sigmoid_for_inference',
            'created_at': datetime.now().isoformat()
        }

    def save_config(self, filepath: Union[str, Path]):
        with open(filepath, 'w') as f:
            json.dump(self.get_model_config(), f, indent=2)


def create_enhanced_wake_word_model(
    input_features: int = 186,
    cnn_hidden: int = 512,
    transformer_heads: int = 16,
    transformer_layers: int = 6,
    transformer_hidden: int = 1024,
    dropout_rate: float = 0.15,
    classifier_hidden: Optional[List[int]] = None,
    use_attention: bool = True,
    device: str = 'auto',
    logger: Optional[ModelLogger] = None
) -> Tuple["EnhancedWakeWordModel", str]:

    if classifier_hidden is None:
        classifier_hidden = [256, 128, 64]

    if device == 'auto':
        if torch.cuda.is_available():
            device_used = 'cuda'
            if logger:
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
                if gpu_memory < 6.0:
                    logger.warning("GPU has less than 6 GB — consider reducing model size.")
        else:
            device_used = 'cpu'
            if logger:
                logger.info("CUDA not available, using CPU")
    else:
        device_used = device

    if logger:
        logger.info(f"Creating model on: {device_used}")

    try:
        model = EnhancedWakeWordModel(
            input_features=input_features,
            cnn_hidden=cnn_hidden,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_hidden=transformer_hidden,
            dropout_rate=dropout_rate,
            classifier_hidden=classifier_hidden,
            use_attention=use_attention,
            logger=logger
        )

        model = model.to(device_used)

        if logger:
            logger.info("Validating model with test inputs...")

        test_cases = [
            (1, input_features, 100),
            (2, input_features, 250),
            (8, input_features, 150),
            (16, input_features, 100),
            (48, input_features, 75),
        ]

        model.eval()
        with torch.no_grad():
            for batch_size, freq_bins, time_frames in test_cases:
                test_input = torch.randn(batch_size, freq_bins, time_frames).to(device_used)
                output = model(test_input)

                expected_shape = (batch_size,)
                if output.shape != expected_shape:
                    raise ValueError(
                        f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
                    )

                if logger:
                    logger.debug(f"  ✓ {test_input.shape} → {output.shape}")

        # TorchScript tracing test
        if logger:
            logger.info("Testing TorchScript tracing...")

        trace_input = torch.randn(1, input_features, 100).to(device_used)
        try:
            traced = torch.jit.trace(model, trace_input)
            with torch.no_grad():
                orig = model(trace_input)
                trac = traced(trace_input)
                if torch.allclose(orig, trac, atol=1e-6, rtol=1e-5):
                    if logger:
                        logger.info("  ✓ TorchScript tracing successful")
                else:
                    if logger:
                        logger.warning("  TorchScript trace outputs diverged (check model)")
        except Exception as e:
            if logger:
                logger.warning(f"  TorchScript tracing failed: {e}")

        model.train()

        if logger:
            logger.info("Model creation and validation complete.")

        return model, device_used

    except Exception as e:
        if logger:
            logger.exception(f"Model creation failed: {str(e)}")
        raise


def save_enhanced_model_architecture(model: EnhancedWakeWordModel,
                                     save_dir: Union[str, Path],
                                     model_name: str = "wake_word_model"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / f"{model_name}_config.json"
    model.save_config(config_path)

    arch_info = {
        'state_dict_keys': list(model.state_dict().keys()),
        'model_structure': str(model),
        'layer_details': {
            'cnn_frontend': {
                'type': 'EnhancedCNNFrontend',
                'input_features': model.cnn_frontend.input_features,
                'hidden_dim': model.cnn_frontend.hidden_dim,
                'attention': model.cnn_frontend.use_attention,
                'output': '(batch, hidden_dim, T/8) — temporal dim preserved'
            },
            'transformer_backend': {
                'type': 'EnhancedStreamingTransformer',
                'input_dim': model.transformer_backend.input_dim,
                'num_heads': model.transformer_backend.num_heads,
                'num_layers': model.transformer_backend.num_layers,
                'hidden_dim': model.transformer_backend.hidden_dim,
                'output': '(batch, hidden_dim) — mean-pooled over time'
            },
            'classifier': {
                'type': 'Multi-layer MLP',
                'hidden_layers': model.classifier_hidden,
                'output': 'raw logits — apply sigmoid for probabilities'
            }
        },
        'saved_at': datetime.now().isoformat()
    }

    arch_path = save_dir / f"{model_name}_architecture.json"
    with open(arch_path, 'w') as f:
        json.dump(arch_info, f, indent=2)


def load_enhanced_model_from_config(config_path: Union[str, Path],
                                    device: str = 'auto',
                                    logger: Optional[ModelLogger] = None) -> "EnhancedWakeWordModel":
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_params = {
        'input_features': config['input_features'],
        'cnn_hidden': config['cnn_hidden'],
        'transformer_heads': config['transformer_heads'],
        'transformer_layers': config['transformer_layers'],
        'transformer_hidden': config['transformer_hidden'],
        'dropout_rate': config['dropout_rate'],
        'classifier_hidden': config['classifier_hidden'],
        'use_attention': config.get('use_attention', True)
    }

    model, device_used = create_enhanced_wake_word_model(
        device=device,
        logger=logger,
        **model_params
    )

    if logger:
        logger.info(f"Model loaded from config: {config_path}")

    return model


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Wake Word Model Test')
    parser.add_argument('--config', type=str, default='training_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)

    model_config = training_config['model']
    paths_config = training_config['paths']

    try:
        logger = ModelLogger(Path(paths_config['log_dir_models']), "model_test")
        logger.info("Starting model test...")

        model, device = create_enhanced_wake_word_model(
            input_features=model_config['input_features'],
            cnn_hidden=model_config['cnn_hidden'],
            transformer_heads=model_config['transformer_heads'],
            transformer_layers=model_config['transformer_layers'],
            transformer_hidden=model_config['transformer_hidden'],
            dropout_rate=model_config['dropout_rate'],
            classifier_hidden=model_config['classifier_hidden'],
            device=training_config['training'].get('device', 'auto'),
            logger=logger
        )

        save_enhanced_model_architecture(model, Path(paths_config['models_architecture']))

        # Quick inference test
        model.eval()
        test_input = torch.randn(4, model_config['input_features'], 200).to(device)
        with torch.no_grad():
            logits = model(test_input)
            probs = torch.sigmoid(logits)  # apply sigmoid at inference time
        logger.info(f"Test output logits: {logits}")
        logger.info(f"Test output probs:  {probs}")

        logger.info("Model test completed successfully.")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise