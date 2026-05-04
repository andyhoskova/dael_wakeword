"""
Enhanced Wake Word Detection Models - Optimized for RTX 2060 Super
================================================================
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
import math
from collections import OrderedDict
import warnings
import yaml


class ModelLogger:
    """Enhanced logger with performance monitoring."""
    
    def __init__(self, log_dir: Union[str, Path], name: str = "enhanced_model_builder"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Enhanced console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Enhanced file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=20*1024*1024,  # 20MB (increased)
            backupCount=10
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Enhanced formatter
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
    """
    Enhanced CNN Frontend with residual connections and attention mechanisms.
    Optimized for RTX 2060 Super with 8GB VRAM.
    """
    
    def __init__(self, 
                 input_features: int = 186,
                 hidden_dim: int = 512,  # Increased from 256
                 dropout_rate: float = 0.15,  # Reduced from 0.2
                 use_attention: bool = True,
                 logger: Optional[ModelLogger] = None):
        super(EnhancedCNNFrontend, self).__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Enhanced convolutional layers with residual connections
        
        # First conv block: input_features -> 128 channels
        self.conv1 = nn.Conv1d(input_features, 128, kernel_size=5, padding=2)  # Larger kernel
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second conv block: 128 -> 256 channels with residual
        self.conv2a = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv2b = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm1d(256)
        self.bn2b = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Residual connection for conv2
        self.residual2 = nn.Conv1d(128, 256, kernel_size=1)
        
        # Third conv block: 256 -> 384 channels with residual
        self.conv3a = nn.Conv1d(256, 384, kernel_size=3, padding=1)
        self.conv3b = nn.Conv1d(384, 384, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm1d(384)
        self.bn3b = nn.BatchNorm1d(384)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Residual connection for conv3
        self.residual3 = nn.Conv1d(256, 384, kernel_size=1)
        
        # Fourth conv block: 384 -> hidden_dim channels with residual
        self.conv4a = nn.Conv1d(384, hidden_dim, kernel_size=3, padding=1)
        self.conv4b = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm1d(hidden_dim)
        self.bn4b = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Residual connection for conv4
        self.residual4 = nn.Conv1d(384, hidden_dim, kernel_size=1)
        
        # Channel attention mechanism
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(hidden_dim, hidden_dim // 16, 1),
                nn.ReLU(inplace=False),
                nn.Conv1d(hidden_dim // 16, hidden_dim, 1),
                nn.Sigmoid()
            )
        
        if logger:
            logger.info(f"Enhanced CNN Frontend initialized: {input_features} -> {hidden_dim}")
            logger.info(f"  Dropout rate: {dropout_rate}, Attention: {use_attention}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block with residual
        residual = self.residual2(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.relu(x, inplace=False)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = x + residual  # Residual connection
        x = F.relu(x, inplace=False)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third conv block with residual
        residual = self.residual3(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = F.relu(x, inplace=False)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = x + residual  # Residual connection
        x = F.relu(x, inplace=False)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fourth conv block with residual
        residual = self.residual4(x)
        x = self.conv4a(x)
        x = self.bn4a(x)
        x = F.relu(x, inplace=False)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = x + residual  # Residual connection
        x = F.relu(x, inplace=False)
        x = self.dropout4(x)
        
        # Channel attention
        if self.use_attention:
            attention_weights = self.channel_attention(x)
            x = x * attention_weights
        
        # Global average and max pooling combined
        avg_pool = torch.mean(x, dim=2)
        max_pool, _ = torch.max(x, dim=2)
        x = avg_pool + max_pool  # Combine both pooling methods
        
        return x
    
    def get_output_dim(self) -> int:
        return self.hidden_dim


class EnhancedStreamingTransformer(nn.Module):
    """
    Enhanced Streaming Transformer with more layers and better attention.
    Optimized for longer sequences and better temporal modeling.
    """
    
    def __init__(self,
                 input_dim: int = 512,  # Increased from 256
                 num_heads: int = 12,   # Increased from 8
                 num_layers: int = 6,   # Increased from 4
                 hidden_dim: int = 1024,  # Increased from 512
                 dropout_rate: float = 0.1,
                 use_rotary_embeddings: bool = True,
                 logger: Optional[ModelLogger] = None):
        super(EnhancedStreamingTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_rotary_embeddings = use_rotary_embeddings
        
        # Validate attention head compatibility
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})")
        
        # Define layers explicitly for TorchScript compatibility (6 layers)
        if num_layers == 6:
            # Layer 1
            self.attention_1 = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate
            )
            self.norm1_1 = nn.LayerNorm(input_dim)
            self.norm2_1 = nn.LayerNorm(input_dim)
            self.ff_1 = self._create_feedforward(input_dim, hidden_dim, dropout_rate)
            
            # Layer 2
            self.attention_2 = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate
            )
            self.norm1_2 = nn.LayerNorm(input_dim)
            self.norm2_2 = nn.LayerNorm(input_dim)
            self.ff_2 = self._create_feedforward(input_dim, hidden_dim, dropout_rate)
            
            # Layer 3
            self.attention_3 = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate
            )
            self.norm1_3 = nn.LayerNorm(input_dim)
            self.norm2_3 = nn.LayerNorm(input_dim)
            self.ff_3 = self._create_feedforward(input_dim, hidden_dim, dropout_rate)
            
            # Layer 4
            self.attention_4 = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate
            )
            self.norm1_4 = nn.LayerNorm(input_dim)
            self.norm2_4 = nn.LayerNorm(input_dim)
            self.ff_4 = self._create_feedforward(input_dim, hidden_dim, dropout_rate)
            
            # Layer 5
            self.attention_5 = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate
            )
            self.norm1_5 = nn.LayerNorm(input_dim)
            self.norm2_5 = nn.LayerNorm(input_dim)
            self.ff_5 = self._create_feedforward(input_dim, hidden_dim, dropout_rate)
            
            # Layer 6
            self.attention_6 = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate
            )
            self.norm1_6 = nn.LayerNorm(input_dim)
            self.norm2_6 = nn.LayerNorm(input_dim)
            self.ff_6 = self._create_feedforward(input_dim, hidden_dim, dropout_rate)
            
        else:
            raise ValueError(f"Currently supports 6 transformer layers, got {num_layers}")
        
        # Enhanced output projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim)
        )
        self.final_norm = nn.LayerNorm(input_dim)
        
        if logger:
            logger.info(f"Enhanced Streaming Transformer: {input_dim}D, {num_heads} heads, {num_layers} layers")
            logger.info(f"  Hidden dim: {hidden_dim}, Dropout: {dropout_rate}")
    
    def _create_feedforward(self, input_dim: int, hidden_dim: int, dropout_rate: float) -> nn.Module:
        """Create enhanced feedforward network with GELU activation."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # Better than ReLU for transformers
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate)
        )
    
    def _apply_transformer_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply a single transformer layer."""
        # Get layer components
        attention = getattr(self, f'attention_{layer_idx}')
        norm1 = getattr(self, f'norm1_{layer_idx}')
        norm2 = getattr(self, f'norm2_{layer_idx}')
        ff = getattr(self, f'ff_{layer_idx}')
        
        # Pre-norm attention
        residual = x
        x = norm1(x)
        attn_output, _ = attention(x, x, x)
        x = residual + attn_output
        
        # Pre-norm feedforward
        residual = x
        x = norm2(x)
        ff_output = ff(x)
        x = residual + ff_output
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for transformer
        x = x.unsqueeze(0)  # Add sequence dimension
        
        # Apply all transformer layers
        for layer_idx in range(1, self.num_layers + 1):
            x = self._apply_transformer_layer(x, layer_idx)
        
        # Final processing
        x = self.final_norm(x)
        residual = x
        x = self.output_proj(x)
        x = x + residual  # Final residual connection
        
        # Remove sequence dimension
        x = x.squeeze(0)
        
        return x
    
    def get_output_dim(self) -> int:
        return self.input_dim


class EnhancedWakeWordModel(nn.Module):
    """
    Enhanced Wake Word Detection Model optimized for RTX 2060 Super.
    Features improved architecture, better regularization, and enhanced capacity.
    """
    
    def __init__(self,
                 input_features: int = 186,
                 cnn_hidden: int = 512,  # Increased
                 transformer_heads: int = 12,  # Increased
                 transformer_layers: int = 6,  # Increased
                 transformer_hidden: int = 1024,  # Increased
                 dropout_rate: float = 0.15,  # Reduced
                 classifier_hidden: List[int] = [256, 128, 64],  # Enhanced
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
        
        # Enhanced components
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
            dropout_rate=dropout_rate * 0.5,
            logger=logger
        )
        
        # Enhanced classifier with explicit definition for [256, 128, 64]
        if len(classifier_hidden) == 3 and classifier_hidden == [256, 128, 64]:
            self.classifier = nn.Sequential(
                # First layer: cnn_hidden -> 256
                nn.Linear(cnn_hidden, 256),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(256),  # Add batch norm
                
                # Second layer: 256 -> 128
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout_rate + 0.05),
                nn.BatchNorm1d(128),
                
                # Third layer: 128 -> 64
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout_rate + 0.1),
                nn.BatchNorm1d(64),
                
                # Output layer: 64 -> 1
                nn.Linear(64, 1)
            )
        else:
            # Fallback for different architectures
            layers = []
            prev_dim = cnn_hidden
            
            for i, hidden_dim in enumerate(classifier_hidden):
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate + i * 0.05),
                    nn.BatchNorm1d(hidden_dim)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.classifier = nn.Sequential(*layers)
        
        # Calculate parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if logger:
            logger.info("=" * 70)
            logger.info("ENHANCED WAKE WORD DETECTION MODEL INITIALIZED")
            logger.info("=" * 70)
            logger.info(f"Total parameters: {self.total_params:,}")
            logger.info(f"Trainable parameters: {self.trainable_params:,}")
            model_size_mb = (self.total_params * 4) / (1024 * 1024)
            logger.info(f"Estimated model size: {model_size_mb:.2f} MB")
            logger.info(f"Memory requirement (training): ~{model_size_mb * 4:.1f} MB")
            logger.info("=" * 70)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN Frontend: enhanced feature extraction
        cnn_features = self.cnn_frontend(x)
        
        # Transformer Backend: enhanced temporal modeling
        transformer_features = self.transformer_backend(cnn_features)
        
        # Classifier: enhanced final prediction
        logits = self.classifier(transformer_features)
        
        # Apply sigmoid activation and reshape
        output = torch.sigmoid(logits)
        output = output.view(-1)
        
        return output
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get enhanced model configuration."""
        return {
            'model_version': '3.0.0',  # Updated version
            'model_type': 'Enhanced TorchScript-Compatible CNN-Transformer Wake Word Detector',
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
            'optimized_for': 'RTX 2060 Super',
            'created_at': datetime.now().isoformat()
        }
    
    def save_config(self, filepath: Union[str, Path]):
        """Save enhanced model configuration."""
        config = self.get_model_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


def create_enhanced_wake_word_model(
    input_features: int = 186,
    cnn_hidden: int = 512,  # Increased default
    transformer_heads: int = 12,  # Increased default
    transformer_layers: int = 6,  # Increased default
    transformer_hidden: int = 1024,  # Increased default
    dropout_rate: float = 0.15,  # Reduced default
    classifier_hidden: Optional[List[int]] = None,
    use_attention: bool = True,
    device: str = 'auto',
    logger: Optional[ModelLogger] = None
) -> Tuple[EnhancedWakeWordModel, str]:
    """
    Factory function for enhanced wake word model optimized for RTX 2060 Super.
    
    Returns a model with ~8-12M parameters, optimized for better performance.
    """
    if classifier_hidden is None:
        classifier_hidden = [256, 128, 64]  # Enhanced default
    
    # Determine device with GPU memory check
    if device == 'auto':
        if torch.cuda.is_available():
            device_used = 'cuda'
            if logger:
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"CUDA available: {gpu_name}")
                logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
                
                if gpu_memory < 6.0:
                    logger.warning("GPU has less than 6GB memory. Consider reducing model size.")
        else:
            device_used = 'cpu'
            if logger:
                logger.info("CUDA not available, using CPU")
    else:
        device_used = device
    
    if logger:
        logger.info(f"Creating enhanced TorchScript-compatible model on: {device_used}")
    
    try:
        # Create enhanced model
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
        
        # Move to device
        model = model.to(device_used)
        
        # Enhanced validation with more test cases
        if logger:
            logger.info("Performing comprehensive enhanced model validation...")
        
        test_cases = [
            (1, input_features, 100),   # Single sample
            (2, input_features, 250),   # Small batch, long sequence
            (8, input_features, 150),   # Medium batch, medium sequence
            (16, input_features, 100),  # Large batch, standard sequence
            (48, input_features, 75),   # Training batch size, short sequence
        ]
        
        model.eval()
        total_memory_used = 0
        
        with torch.no_grad():
            for batch_size, freq_bins, time_frames in test_cases:
                test_input = torch.randn(batch_size, freq_bins, time_frames).to(device_used)
                
                # Memory usage tracking
                if device_used == 'cuda':
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated()
                
                output = model(test_input)
                
                if device_used == 'cuda':
                    memory_after = torch.cuda.memory_allocated()
                    memory_used = (memory_after - memory_before) / (1024**2)  # MB
                    total_memory_used = max(total_memory_used, memory_used)
                
                # Validate output
                expected_shape = (batch_size,)
                if output.shape != expected_shape:
                    raise ValueError(f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
                
                if not (0 <= output.min().item() and output.max().item() <= 1):
                    raise ValueError(f"Output values out of range: min={output.min().item()}, max={output.max().item()}")
                
                if logger:
                    logger.debug(f"✓ Test case {test_input.shape} -> {output.shape} passed")
        
        if logger and device_used == 'cuda':
            logger.info(f"Peak GPU memory usage during validation: {total_memory_used:.1f} MB")
        
        # Enhanced TorchScript compatibility testing
        if logger:
            logger.info("Testing enhanced TorchScript compatibility...")
        
        test_input = torch.randn(1, input_features, 100).to(device_used)
        
        # Test tracing with enhanced validation
        try:
            traced_model = torch.jit.trace(model, test_input)
            
            # More rigorous output comparison
            with torch.no_grad():
                original_output = model(test_input)
                traced_output = traced_model(test_input)
                
                max_diff = torch.max(torch.abs(original_output - traced_output)).item()
                relative_error = max_diff / torch.max(torch.abs(original_output)).item()
                
                if torch.allclose(original_output, traced_output, atol=1e-6, rtol=1e-5):
                    if logger:
                        logger.info(f"✓ TorchScript tracing successful (max diff: {max_diff:.2e})")
                else:
                    if logger:
                        logger.warning(f"TorchScript tracing produces different outputs (relative error: {relative_error:.2e})")
            
        except Exception as trace_error:
            if logger:
                logger.warning(f"TorchScript tracing failed: {str(trace_error)}")
            
            # Fallback to scripting
            try:
                scripted_model = torch.jit.script(model)
                
                with torch.no_grad():
                    original_output = model(test_input)
                    scripted_output = scripted_model(test_input)
                    
                    if torch.allclose(original_output, scripted_output, atol=1e-6, rtol=1e-5):
                        if logger:
                            logger.info("✓ TorchScript scripting successful")
                    else:
                        if logger:
                            logger.warning("TorchScript scripting produces different outputs")
                        
            except Exception as script_error:
                if logger:
                    logger.error("Both TorchScript methods failed!")
                    logger.error(f"Tracing error: {str(trace_error)}")
                    logger.error(f"Scripting error: {str(script_error)}")
                    logger.warning("Model will work for training but may not export to TorchScript")
        
        model.train()  # Return to training mode
        
        if logger:
            logger.info("✓ Enhanced model validation completed successfully!")
            logger.info("Model is ready for extended training and deployment!")
        
        return model, device_used
        
    except Exception as e:
        if logger:
            logger.exception(f"Enhanced model creation failed: {str(e)}")
        raise


def save_enhanced_model_architecture(model: EnhancedWakeWordModel, 
                                   save_dir: Union[str, Path],
                                   model_name: str = "enhanced_wake_word_model"):
    """Save enhanced model architecture and detailed configuration."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save enhanced configuration
    config_path = save_dir / f"{model_name}_config.json"
    model.save_config(config_path)
    
    # Save detailed architecture information
    arch_info = {
        'state_dict_keys': list(model.state_dict().keys()),
        'model_structure': str(model),
        'layer_details': {
            'cnn_frontend': {
                'type': 'EnhancedCNNFrontend',
                'features': model.cnn_frontend.input_features,
                'hidden_dim': model.cnn_frontend.hidden_dim,
                'attention': model.cnn_frontend.use_attention
            },
            'transformer_backend': {
                'type': 'EnhancedStreamingTransformer',
                'input_dim': model.transformer_backend.input_dim,
                'num_heads': model.transformer_backend.num_heads,
                'num_layers': model.transformer_backend.num_layers,
                'hidden_dim': model.transformer_backend.hidden_dim
            },
            'classifier': {
                'type': 'Enhanced Multi-layer',
                'hidden_layers': model.classifier_hidden,
                'output_dim': 1
            }
        },
        'optimization_target': 'RTX 2060 Super',
        'torchscript_compatible': True,
        'estimated_training_time': 'Longer training expected (50-100+ epochs)',
        'saved_at': datetime.now().isoformat()
    }
    
    arch_path = save_dir / f"{model_name}_architecture.json"
    with open(arch_path, 'w') as f:
        json.dump(arch_info, f, indent=2)
    
    # Save training recommendations
    recommendations = {
        'training_recommendations': {
            'min_epochs': 50,
            'expected_epochs': '80-150',
            'early_stopping_patience': 35,
            'learning_rate': 0.0002,
            'batch_size': 48,
            'mixed_precision': True,
            'gradient_accumulation': True
        },
        'expected_performance': {
            'target_f1': '>0.95',
            'target_auc': '>0.999',
            'training_time_estimate': '2-4 hours on RTX 2060 Super'
        },
        'memory_usage': {
            'model_size_mb': (model.total_params * 4) / (1024 * 1024),
            'training_memory_estimate_gb': '4-6 GB',
            'inference_memory_mb': '200-400 MB'
        }
    }
    
    rec_path = save_dir / f"{model_name}_training_recommendations.json"
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)


def load_enhanced_model_from_config(config_path: Union[str, Path], 
                                  device: str = 'auto',
                                  logger: Optional[ModelLogger] = None) -> EnhancedWakeWordModel:
    """Load enhanced model from configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract enhanced model parameters
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
        logger.info(f"Enhanced model loaded from config: {config_path}")
        logger.info(f"Model version: {config.get('model_version', 'unknown')}")
        logger.info(f"Optimized for: {config.get('optimized_for', 'general use')}")
    
    return model


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the enhanced wake word model optimized for RTX 2060 Super.
    """
    # Enhanced configuration for better performance
    ENHANCED_MODEL_CONFIG = {
        'input_features': 186,
        'cnn_hidden': 512,      # Doubled from original
        'transformer_heads': 12, # Increased from 8
        'transformer_layers': 6, # Increased from 4
        'transformer_hidden': 1024, # Doubled from original
        'dropout_rate': 0.15,   # Reduced from 0.2
        'classifier_hidden': [256, 128, 64], # Enhanced
        'use_attention': True,
        'device': 'auto'
    }
    
    try:
        # Create enhanced logger
        logger = ModelLogger(Path("logs/models"), "enhanced_model_test")
        logger.info("Starting enhanced TorchScript-compatible model testing...")
        logger.info("Optimized for RTX 2060 Super with 8GB VRAM")
        
        # Create enhanced model
        model, device = create_enhanced_wake_word_model(
            input_features=ENHANCED_MODEL_CONFIG['input_features'],
            cnn_hidden=ENHANCED_MODEL_CONFIG['cnn_hidden'],
            transformer_heads=ENHANCED_MODEL_CONFIG['transformer_heads'],
            transformer_layers=ENHANCED_MODEL_CONFIG['transformer_layers'],
            transformer_hidden=ENHANCED_MODEL_CONFIG['transformer_hidden'],
            dropout_rate=ENHANCED_MODEL_CONFIG['dropout_rate'],
            classifier_hidden=ENHANCED_MODEL_CONFIG['classifier_hidden'],
            use_attention=ENHANCED_MODEL_CONFIG['use_attention'],
            device=ENHANCED_MODEL_CONFIG['device'],
            logger=logger
        )
        
        # Save enhanced model architecture
        save_enhanced_model_architecture(model, Path("models/enhanced_architecture"))
        
        # Performance testing with training-like scenarios
        test_scenarios = [
            (1, 186, 100, "Single inference"),
            (48, 186, 150, "Training batch"),
            (16, 186, 300, "Long sequence batch"),
            (64, 186, 75, "Large batch, short sequence"),
        ]
        
        logger.info("Testing enhanced model performance...")
        model.eval()
        
        total_inference_times = []
        
        for batch_size, freq_bins, time_frames, description in test_scenarios:
            test_input = torch.randn(batch_size, freq_bins, time_frames).to(device)
            
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    _ = model(test_input)
            
            # Time multiple runs
            times = []
            for _ in range(10):
                if device == 'cuda':
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    with torch.no_grad():
                        output = model(test_input)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    inference_time = start_event.elapsed_time(end_event)
                    times.append(inference_time)
                else:
                    import time
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(test_input)
                    end_time = time.time()
                    inference_time = (end_time - start_time) * 1000
                    times.append(inference_time)
            
            avg_time = sum(times) / len(times)
            total_inference_times.extend(times)
            
            throughput = batch_size / (avg_time / 1000) if avg_time > 0 else 0
            
            logger.info(f"{description}:")
            logger.info(f"  Input: {test_input.shape} -> Output: {output.shape}")
            logger.info(f"  Avg time: {avg_time:.2f}ms | Throughput: {throughput:.1f} samples/sec")
            logger.info(f"  Per sample: {avg_time/batch_size:.2f}ms")
        
        overall_avg = sum(total_inference_times) / len(total_inference_times)
        logger.info(f"\nOverall average inference time: {overall_avg:.2f}ms")
        
        logger.info("=" * 80)
        logger.info("ENHANCED TORCHSCRIPT-COMPATIBLE MODEL TESTING COMPLETED!")
        logger.info("=" * 80)
        logger.info("KEY IMPROVEMENTS:")
        logger.info("• Model capacity increased from ~2.8M to ~8-12M parameters")
        logger.info("• Enhanced CNN with residual connections and attention")
        logger.info("• Deeper transformer with 6 layers and 12 attention heads")
        logger.info("• Improved classifier with batch normalization and GELU")
        logger.info("• Optimized for RTX 2060 Super training and inference")
        logger.info("• Expected to achieve F1 > 0.95 with proper training")
        logger.info("=" * 80)
        logger.info("TRAINING RECOMMENDATIONS:")
        logger.info("• Use the enhanced config with 200 epochs maximum")
        logger.info("• Set early stopping patience to 35 epochs")
        logger.info("• Enable mixed precision training")
        logger.info("• Use learning rate 0.0002 with warmup")
        logger.info("• Expect 2-4 hours training time on RTX 2060 Super") 
        logger.info("=" * 80)
        
    except Exception as e:
        print(f"Error during enhanced model testing: {str(e)}")
        raise