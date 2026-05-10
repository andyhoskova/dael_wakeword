"""
Wake Word Detection Engine - FIXED TO MATCH TRAINING PIPELINE
This version uses the EXACT same feature extraction as your training code.
"""

import torch
import numpy as np
import pyaudio
import threading
import time
import logging
from collections import deque
import signal
import sys
from typing import Optional, Callable
import warnings
import torchaudio.transforms as T
import os
import contextlib
import ctypes
import ctypes.util
from pathlib import Path

# Suppress PyAudio warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress ALSA and JACK warnings at the OS level
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'
os.environ['ALSA_QUIET'] = '1'


# Load libc
libc = ctypes.CDLL(ctypes.util.find_library("c"))

# Get the project root directory (where src/ is located)
PROJECT_ROOT = Path(__file__).parent.parent

@contextlib.contextmanager
def suppress_audio_warnings():
    """Context manager to suppress ALSA and JACK warnings at OS level"""
    # Save original stderr file descriptor
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    
    try:
        # Open /dev/null
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        # Redirect stderr to /dev/null
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
        yield
    finally:
        # Restore original stderr
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)

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
        
        # EXACT transforms from your training code
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
        
        # Resampler for audio that isn't 16kHz
        self.resampler = None
        
    def load_audio(self, waveform, orig_sr):
        """Process audio waveform (adapted for streaming)"""
        try:
            # Waveform is already on the correct device from preprocess_audio
            
            # Resample if necessary
            if orig_sr != self.sample_rate:
                if self.resampler is None or self.resampler.orig_freq != orig_sr:
                    self.resampler = T.Resample(orig_sr, self.sample_rate).to(self.device)
                waveform = self.resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            return waveform
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    
    def extract_features(self, waveform):
        """Extract EXACT same features as training"""
        if waveform is None:
            return None
            
        # Ensure waveform is on the correct device
        if waveform.device != self.device:
            waveform = waveform.to(self.device)
            
        features = {}
        
        # 1. Log Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-8)  # Same epsilon as training
        features['log_mel'] = log_mel_spec.squeeze(0)
        
        # 2. Delta Mel (temporal derivatives)
        delta_mel = self.compute_deltas(log_mel_spec)
        features['delta_mel'] = delta_mel.squeeze(0)
        
        # 3. MFCCs
        mfcc = self.mfcc_transform(waveform)
        features['mfcc'] = mfcc.squeeze(0)
        
        # 4. Delta MFCCs
        delta_mfcc = self.compute_deltas(mfcc)
        features['delta_mfcc'] = delta_mfcc.squeeze(0)
        
        # Stack all features EXACTLY like training
        # Shape: [total_features, time_frames]
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


class WakeWordEngine:
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.7,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        window_duration: float = 1.5,
        detection_cooldown: float = 2.0,
        callback: Optional[Callable] = None
    ):
        if model_path is None:
            # Default to models/callina.pt relative to project root
            model_path = str(PROJECT_ROOT / "models" / "dael_v1.1.pt")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_duration = window_duration
        self.detection_cooldown = detection_cooldown
        self.callback = callback
        
        # Calculate buffer size
        self.buffer_size = int(sample_rate * window_duration)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Threading and state
        self.is_running = False
        self.detection_thread = None
        self.last_detection_time = 0
        
        # PyAudio setup
        self.audio = None
        self.stream = None
        
        # Model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CRITICAL: Use the EXACT same feature extractor as training
        self.feature_extractor = WakeWordFeatureExtractor(
            sample_rate=16000,
            n_mels=80,           # Same as training
            n_mfcc=13,           # Same as training  
            n_fft=512,           # Same as training
            win_length=400,      # Same as training
            hop_length=160,      # Same as training
            device=self.device
        )
        
        # Statistics tracking
        self.confidence_history = deque(maxlen=100)
        self.detection_count = 0
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """Load the TorchScript model."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            # Test model with expected input shape (186 features × time_frames)
            test_input = torch.randn(1, 186, 100).to(self.device)
            with torch.no_grad():
                output = self.model(test_input)
                self.logger.info(f"✅ Model test successful!")
                self.logger.info(f"   Input: {test_input.shape} -> Output: {output.shape}")
                self.logger.info(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False
            
    def setup_audio(self) -> bool:
        """Setup PyAudio for recording."""
        try:
            # Use context manager to suppress ALSA/JACK warnings
            with suppress_audio_warnings():
                self.audio = pyaudio.PyAudio()
            
            # Find the best input device
            device_info = None
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_info = info
                    break
            
            if device_info:
                self.logger.info(f"🎤 Using audio device: {device_info['name']}")
            
            with suppress_audio_warnings():
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )
            
            self.logger.info(f"✅ Audio stream initialized")
            self.logger.info(f"   Sample rate: {self.sample_rate}Hz, Chunk size: {self.chunk_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup audio: {e}")
            return False
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous audio capture."""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
            
        # Convert to float32 [-1, 1] range
        audio_chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio_chunk)
        return (None, pyaudio.paContinue)
        
    def preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:

        try:
            # Ensure proper length
            if len(audio_data) > self.buffer_size:
                audio_data = audio_data[-self.buffer_size:]
            elif len(audio_data) < self.buffer_size:
                # Pad with zeros at the beginning
                padding = np.zeros(self.buffer_size - len(audio_data))
                audio_data = np.concatenate([padding, audio_data])
            
            # Convert to tensor with batch dimension and move to correct device
            waveform = torch.FloatTensor(audio_data).unsqueeze(0).to(self.device)  # [1, samples] on GPU
            
            # Use the EXACT same feature extraction as training
            feature_dict = self.feature_extractor.extract_features(waveform)
            
            if feature_dict is None:
                return None
                
            # Get the stacked features (186 × time_frames)
            features = feature_dict['features']  # [186, time_frames]
            
            # Add batch dimension: [1, 186, time_frames]
            features = features.unsqueeze(0)
            
            # Verify shape
            if features.shape[1] != 186:
                self.logger.error(f"❌ Feature mismatch! Got {features.shape[1]}, expected 186")
                return None
            
            # Debug logging (occasional)
            if len(self.confidence_history) % 50 == 0:
                self.logger.debug(f"🔧 Features shape: {features.shape}")
                self.logger.debug(f"   Feature stats - mean: {features.mean():.4f}, std: {features.std():.4f}")
                self.logger.debug(f"   Feature range: [{features.min():.4f}, {features.max():.4f}]")
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing error: {e}")
            return None
            
    def detect_wake_word(self, audio_tensor: torch.Tensor) -> tuple[bool, float]:
        """Run wake word detection."""
        try:
            with torch.no_grad():
                output = self.model(audio_tensor)
                
                # Handle different output shapes
                if output.dim() > 1:
                    output = output.squeeze()
                    
                if output.dim() == 0:
                    confidence = output.item()  # Raw output, no sigmoid needed if model has it
                else:
                    confidence = output[0].item()
                
                # Apply sigmoid if output is logits (check range)
                if confidence < 0 or confidence > 1:
                    confidence = torch.sigmoid(torch.tensor(confidence)).item()
                
                # Store confidence
                self.confidence_history.append(confidence)
                
                # Only show detection messages (no extra prints)
                is_detected = confidence >= self.confidence_threshold
                return is_detected, confidence
                
        except Exception as e:
            self.logger.error(f"❌ Error during inference: {e}")
            return False, 0.0
            
    def on_wake_word_detected(self, confidence: float):
        """Handle wake word detection."""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_detection_time < self.detection_cooldown:
            return
            
        self.last_detection_time = current_time
        self.detection_count += 1
        
        detection_time = time.strftime("%H:%M:%S", time.localtime(current_time))
        
        print(f"\n🎯 WAKE WORD DETECTED! #{self.detection_count}")
        print(f"   Time: {detection_time}")
        print(f"   Confidence: {confidence:.4f} ({confidence:.2%})")
        print(f"   Threshold: {self.confidence_threshold:.4f}")
        
        # Call user callback if provided
        if self.callback:
            try:
                self.callback(confidence)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
                
    def detection_worker(self):
        """Main detection loop."""
        self.logger.info("🔄 Detection worker started")
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.is_running:
            try:
                # Wait for sufficient audio data
                if len(self.audio_buffer) < self.buffer_size:
                    time.sleep(0.1)
                    continue
                
                # Get audio data
                audio_data = np.array(list(self.audio_buffer))
                
                # Preprocess audio using EXACT training pipeline
                audio_tensor = self.preprocess_audio(audio_data)
                if audio_tensor is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("❌ Too many preprocessing errors, stopping")
                        break
                    time.sleep(0.1)
                    continue
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Run detection
                is_detected, confidence = self.detect_wake_word(audio_tensor)
                
                if is_detected:
                    self.on_wake_word_detected(confidence)
                        
                # Sleep to prevent excessive CPU usage
                time.sleep(0.05)  # 20 FPS detection rate
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"❌ Error in detection worker: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error("❌ Too many consecutive errors, stopping detection")
                    break
                time.sleep(0.5)
                
        self.logger.info("🛑 Detection worker stopped")
        
    def start(self) -> bool:
        """Start the wake word engine."""
        if self.is_running:
            self.logger.warning("⚠️ Engine is already running")
            return False
            
        self.logger.info("🚀 Starting Wake Word Engine...")
        
        # Load model  
        if not self.load_model():
            return False
            
        # Setup audio
        if not self.setup_audio():
            return False
            
        # Start audio stream
        with suppress_audio_warnings():
            self.stream.start_stream()
        self.is_running = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        
        self.logger.info("✅ Wake Word Engine started successfully!")
        self.logger.info(f"🎯 Listening for wake word with confidence threshold: {self.confidence_threshold}")
        return True
        
    def stop(self):
        """Stop the wake word engine."""
        if not self.is_running:
            return
            
        self.logger.info("🛑 Stopping Wake Word Engine...")
        self.is_running = False
        
        # Stop audio stream
        if self.stream:
            with suppress_audio_warnings():
                self.stream.stop_stream()
                self.stream.close()
            
        if self.audio:
            with suppress_audio_warnings():
                self.audio.terminate()
            
        # Wait for detection thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3.0)
            
        self.logger.info("✅ Wake Word Engine stopped")
        
        # Print final statistics
        self._print_final_stats()
        
    def _print_final_stats(self):
        """Print final detection statistics."""
        if not self.confidence_history:
            return
            
        confidences = list(self.confidence_history)
        
        # Show confidence distribution
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        print("Confidence distribution:")
        for low, high in ranges:
            count = sum(1 for c in confidences if low <= c < high)
            pct = count / len(confidences) * 100 if confidences else 0
            bar = "█" * max(1, count // (len(confidences) // 20 + 1))
            print(f"     {low:.1f}-{high:.1f}: {count:4d} ({pct:5.1f}%) {bar}")
            
    def set_threshold(self, new_threshold: float):
        """Dynamically adjust confidence threshold."""
        if 0.0 <= new_threshold <= 1.0:
            old_threshold = self.confidence_threshold
            self.confidence_threshold = new_threshold
            print(f"🔧 Threshold changed: {old_threshold:.3f} → {new_threshold:.3f}")
        else:
            print(f"❌ Invalid threshold: {new_threshold}. Must be between 0.0 and 1.0")


def signal_handler(signum, frame):
    """Handle signals gracefully."""
    print("\n\n🛑 Shutting down gracefully...")
    global engine
    if 'engine' in globals():
        engine.stop()
    sys.exit(0)


def main():
    """Main function."""
    global engine
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    
    # Create engine with reasonable defaults
    engine = WakeWordEngine(
        model_path=str(PROJECT_ROOT / "models" / "dael_v1.1.pt"),
        confidence_threshold=0.5,
        window_duration=1.5,
        detection_cooldown=2.0
    )
    
    if not engine.start():
        print("❌ Failed to start wake word engine!")
        return 1
        
    try:
        # Start listening
        print(f"\n🎯 Listening... (Threshold: {engine.confidence_threshold:.3f})")
        
        # Keep running until interrupted
        while engine.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        engine.stop()
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)