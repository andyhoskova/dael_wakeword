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

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'
os.environ['ALSA_QUIET'] = '1'

libc = ctypes.CDLL(ctypes.util.find_library("c"))

PROJECT_ROOT = Path(__file__).parent.parent

_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

try:
    from models import EnhancedWakeWordModel
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def suppress_audio_warnings():
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


# ── Feature extraction ────────────────────────────────────────────────────────

class WakeWordFeatureExtractor:
   
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_mfcc: int = 13,
                 n_fft: int = 512,
                 win_length: int = 400,
                 hop_length: int = 160,
                 device: str = 'cpu',
                 normalize: bool = True,
                 norm_eps: float = 1e-6):

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.device = device
        self.normalize = normalize
        self.norm_eps = norm_eps

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(device)

        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'win_length': win_length,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'power': 2.0,
            },
        ).to(device)

        self.compute_deltas = T.ComputeDeltas().to(device)
        self.resampler = None

    def extract_features(self, waveform: torch.Tensor) -> Optional[dict]:
        if waveform is None:
            return None
        waveform = waveform.to(self.device)

        # 1. Log Mel Spectrogram [80, T]
        mel_spec = self.mel_spectrogram(waveform)
        log_mel = torch.log(mel_spec + 1e-8).squeeze(0)

        # 2. Delta Mel [80, T]
        delta_mel = self.compute_deltas(log_mel.unsqueeze(0)).squeeze(0)

        # 3. MFCCs [13, T]
        mfcc_raw = self.mfcc_transform(waveform)
        mfcc = mfcc_raw.squeeze(0)

        # 4. Delta MFCCs [13, T]
        delta_mfcc = self.compute_deltas(mfcc_raw).squeeze(0)

        stacked = torch.cat([log_mel, delta_mel, mfcc, delta_mfcc], dim=0)

        if self.normalize:
            mean = stacked.mean()
            std  = stacked.std()
            stacked = (stacked - mean) / (std + self.norm_eps)

        return {
            'features': stacked,
            'shape': stacked.shape,
        }


#  Engine
class WakeWordEngine:
  
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        window_duration: float = 1.5,           
        detection_cooldown: float = 2.0,
        callback: Optional[Callable] = None,
        smoothing_alpha: float = 0.3,            
        normalize_features: bool = True,         
    ):
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "dael_v1.5.pt")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_duration = window_duration
        self.detection_cooldown = detection_cooldown
        self.callback = callback
        self.smoothing_alpha = smoothing_alpha

        self.buffer_size = int(sample_rate * window_duration)
        self.audio_buffer: deque = deque(maxlen=self.buffer_size)

        self.is_running = False
        self.detection_thread: Optional[threading.Thread] = None
        self.last_detection_time: float = 0.0

        self.audio = None
        self.stream = None
        self.model: Optional[torch.nn.Module] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = WakeWordFeatureExtractor(
            sample_rate=16000,
            n_mels=80,
            n_mfcc=13,
            n_fft=512,
            win_length=400,
            hop_length=160,
            device=str(self.device),
            normalize=normalize_features,        # FIX 3
        )

        # Tracking
        self.raw_confidence_history: deque = deque(maxlen=200)
        self.smoothed_confidence_history: deque = deque(maxlen=200)
        self.detection_count = 0

        # EMA state
        self._ema_confidence: float = 0.0
        self._ema_initialized: bool = False

        self._setup_logging()

    # Logging

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        self.logger = logging.getLogger(__name__)

    # Model loading

    def load_model(self) -> bool:
        try:
            self.logger.info(f"Loading model from: {self.model_path}")

            try:
                raw = torch.load(self.model_path, map_location=self.device, weights_only=False)
                is_state_dict = isinstance(raw, dict) and 'model_state_dict' in raw
            except Exception:
                is_state_dict = False
                raw = None

            if is_state_dict:
                self.model = self._load_state_dict_checkpoint(raw)
            else:
                self.model = self._load_torchscript()

            if self.model is None:
                return False

            self.model.eval()

            # Smoke-test
            test_input = torch.randn(1, 186, 100).to(self.device)
            with torch.no_grad():
                logits = self.model(test_input)
                probs  = torch.sigmoid(logits)

            self.logger.info("✅ Model loaded and verified")
            self.logger.info(f"   Input  : {list(test_input.shape)}")
            self.logger.info(f"   Logits : [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            self.logger.info(f"   Sigmoid: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False

    def _load_state_dict_checkpoint(self, checkpoint: dict) -> Optional[torch.nn.Module]:
        if not _MODELS_AVAILABLE:
            self.logger.error(
                "❌ models.py not importable — make sure it is in the same directory."
            )
            return None

        cfg = checkpoint.get('model_config', {})
        kwargs = {
            'input_features'    : cfg.get('input_features', 186),
            'cnn_hidden'        : cfg.get('cnn_hidden', 512),
            'transformer_heads' : cfg.get('transformer_heads', 16),
            'transformer_layers': cfg.get('transformer_layers', 6),
            'transformer_hidden': cfg.get('transformer_hidden', 1024),
            'dropout_rate'      : cfg.get('dropout_rate', 0.15),
            'classifier_hidden' : cfg.get('classifier_hidden', [256, 128, 64]),
            'use_attention'     : cfg.get('use_attention', True),
        }

        self.logger.info("Building EnhancedWakeWordModel from checkpoint config:")
        for k, v in kwargs.items():
            self.logger.info(f"   {k}: {v}")

        model = EnhancedWakeWordModel(**kwargs).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', '?')
        score = checkpoint.get('val_score', '?')
        self.logger.info(f"✅ State-dict checkpoint loaded (epoch={epoch}, val_score={score})")
        return model

    def _load_torchscript(self) -> Optional[torch.nn.Module]:
        try:
            model = torch.jit.load(self.model_path, map_location=self.device)
            self.logger.info("✅ TorchScript model loaded")
            return model
        except Exception as e:
            self.logger.error(
                f"❌ TorchScript load failed: {e}\n"
                "Hint: checkpoints from src/models/checkpoints/ are plain state-dict "
                "files — use a file from src/models/exported/ for TorchScript loading."
            )
            return None

    # ── Audio ─────────────────────────────────────────────────────────────────

    def setup_audio(self) -> bool:
        try:
            with suppress_audio_warnings():
                self.audio = pyaudio.PyAudio()

            device_info = None
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_info = info
                    break

            if device_info:
                self.logger.info(f"🎤 Audio device: {device_info['name']}")

            with suppress_audio_warnings():
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback,
                )

            self.logger.info(
                f"✅ Audio stream ready — {self.sample_rate} Hz, "
                f"window {self.window_duration:.1f} s"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Audio setup failed: {e}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.extend(chunk)
        return (None, pyaudio.paContinue)

    # Preprocessing
    def preprocess_audio(self, audio_data: np.ndarray) -> Optional[torch.Tensor]:
        try:
            if len(audio_data) > self.buffer_size:
                audio_data = audio_data[-self.buffer_size:]
            elif len(audio_data) < self.buffer_size:
                padding = np.zeros(self.buffer_size - len(audio_data))
                audio_data = np.concatenate([padding, audio_data])

            waveform = torch.FloatTensor(audio_data).unsqueeze(0).to(self.device)
            feature_dict = self.feature_extractor.extract_features(waveform)
            if feature_dict is None:
                return None

            features = feature_dict['features'].unsqueeze(0)    # [1, 186, T]
            if features.shape[1] != 186:
                self.logger.error(
                    f"❌ Feature dimension mismatch: {features.shape[1]} != 186"
                )
                return None

            return features

        except Exception as e:
            self.logger.error(f"❌ Preprocessing error: {e}")
            return None

    # Inference
    def detect_wake_word(self, audio_tensor: torch.Tensor) -> tuple[bool, float, float]:
       
        try:
            with torch.no_grad():
                logits = self.model(audio_tensor)
                if logits.dim() > 1:
                    logits = logits.squeeze()
                logit_val = logits.item() if logits.dim() == 0 else logits[0].item()

                # FIX 1 (from round 1): always apply sigmoid unconditionally.
                raw_conf = torch.sigmoid(torch.tensor(logit_val)).item()

            # FIX 4: EMA smoothing.
            if not self._ema_initialized:
                self._ema_confidence = raw_conf
                self._ema_initialized = True
            else:
                self._ema_confidence = (
                    self.smoothing_alpha * raw_conf
                    + (1.0 - self.smoothing_alpha) * self._ema_confidence
                )

            smoothed_conf = self._ema_confidence

            self.raw_confidence_history.append(raw_conf)
            self.smoothed_confidence_history.append(smoothed_conf)

            # Detect on the smoothed signal.
            is_detected = smoothed_conf >= self.confidence_threshold
            return is_detected, raw_conf, smoothed_conf

        except Exception as e:
            self.logger.error(f"❌ Inference error: {e}")
            return False, 0.0, 0.0

    # Detection loop
    def detection_worker(self, diagnostic_mode: bool = False):
        self.logger.info("🔄 Detection worker started")
        if diagnostic_mode:
            self.logger.info(
                "🔬 DIAGNOSTIC MODE — detections suppressed, "
                "confidences printed every second"
            )

        consecutive_errors = 0
        last_diag_print = time.time()
        diag_interval = 1.0

        while self.is_running:
            try:
                if len(self.audio_buffer) < self.buffer_size:
                    time.sleep(0.05)
                    continue

                audio_data = np.array(list(self.audio_buffer))
                audio_tensor = self.preprocess_audio(audio_data)

                if audio_tensor is None:
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        self.logger.error("❌ Too many preprocessing errors, stopping")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_errors = 0

                is_detected, raw_conf, smoothed_conf = self.detect_wake_word(audio_tensor)

                # FIX 7: Diagnostic mode prints raw vs smoothed for threshold calibration.
                if diagnostic_mode:
                    now = time.time()
                    if now - last_diag_print >= diag_interval:
                        bar_raw      = "█" * int(raw_conf * 20)
                        bar_smoothed = "█" * int(smoothed_conf * 20)
                        print(
                            f"\r  raw={raw_conf:.3f} [{bar_raw:<20}]  "
                            f"smooth={smoothed_conf:.3f} [{bar_smoothed:<20}]  "
                            f"threshold={self.confidence_threshold:.2f}",
                            end='', flush=True,
                        )
                        last_diag_print = now
                elif is_detected:
                    self.on_wake_word_detected(raw_conf, smoothed_conf)

                time.sleep(0.05)

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"❌ Detection worker error: {e}")
                if consecutive_errors >= 10:
                    break
                time.sleep(0.5)

        self.logger.info("🛑 Detection worker stopped")

    # ── Detection event ───────────────────────────────────────────────────────

    def on_wake_word_detected(self, raw_conf: float, smoothed_conf: float):
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_cooldown:
            return

        self.last_detection_time = current_time
        self.detection_count += 1

        print(f"\n🎯 WAKE WORD DETECTED! #{self.detection_count}")
        print(f"   Time          : {time.strftime('%H:%M:%S')}")
        print(f"   Raw confidence: {raw_conf:.4f} ({raw_conf:.2%})")
        print(f"   EMA confidence: {smoothed_conf:.4f} ({smoothed_conf:.2%})")
        print(f"   Threshold     : {self.confidence_threshold:.4f}")

        if self.callback:
            try:
                self.callback(smoothed_conf)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

    # Lifecycle
    def start(self) -> bool:
        if self.is_running:
            self.logger.warning("⚠️ Engine is already running")
            return False

        self.logger.info("🚀 Starting Wake Word Engine...")
        if not self.load_model():
            return False
        if not self.setup_audio():
            return False

        with suppress_audio_warnings():
            self.stream.start_stream()
        self.is_running = True

        self.detection_thread = threading.Thread(
            target=self.detection_worker, daemon=True
        )
        self.detection_thread.start()

        self.logger.info("✅ Engine running")
        self.logger.info(f"🎯 Threshold : {self.confidence_threshold}")
        self.logger.info(f"🔈 Window    : {self.window_duration:.1f} s")
        self.logger.info(f"📊 Smoothing : EMA alpha={self.smoothing_alpha}")
        return True

    def run_diagnostics(self, seconds: float = 30.0):
      
        self.logger.info(f"🔬 Running diagnostics for {seconds:.0f} s...")
        if not self.load_model():
            return
        if not self.setup_audio():
            return

        with suppress_audio_warnings():
            self.stream.start_stream()
        self.is_running = True

        self.detection_thread = threading.Thread(
            target=lambda: self.detection_worker(diagnostic_mode=True),
            daemon=True,
        )
        self.detection_thread.start()

        print(
            f"\nSay the wake word several times.  "
            f"Watch the smoothed bar cross {self.confidence_threshold:.2f}.\n"
            f"Press Ctrl-C to stop early.\n"
        )

        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop(print_stats=True)

    def stop(self, print_stats: bool = True):
        if not self.is_running:
            return
        self.logger.info("🛑 Stopping Wake Word Engine...")
        self.is_running = False

        if self.stream:
            with suppress_audio_warnings():
                self.stream.stop_stream()
                self.stream.close()
        if self.audio:
            with suppress_audio_warnings():
                self.audio.terminate()

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3.0)

        self.logger.info("✅ Engine stopped")
        if print_stats:
            self._print_final_stats()

    def set_threshold(self, new_threshold: float):
        if 0.0 <= new_threshold <= 1.0:
            old = self.confidence_threshold
            self.confidence_threshold = new_threshold
            print(f"🔧 Threshold: {old:.3f} -> {new_threshold:.3f}")
        else:
            print(f"❌ Invalid threshold {new_threshold}. Must be in [0.0, 1.0]")

    def _print_final_stats(self):
        raws     = list(self.raw_confidence_history)
        smoothed = list(self.smoothed_confidence_history)
        if not raws:
            return

        print("\n── Confidence statistics ─────────────────────────────────")
        for label, values in [("Raw", raws), ("Smoothed", smoothed)]:
            a = np.array(values)
            print(
                f"  {label:8s}  mean={a.mean():.3f}  "
                f"max={a.max():.3f}  "
                f"p95={np.percentile(a, 95):.3f}"
            )

        print("\n  Raw confidence distribution:")
        a = np.array(raws)
        for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
            count = int(((a >= lo) & (a < hi)).sum())
            pct   = count / len(a) * 100
            bar   = "█" * max(1, count // max(1, len(a) // 40))
            print(f"  {lo:.1f}-{hi:.1f}: {count:4d} ({pct:5.1f}%) {bar}")
        print("──────────────────────────────────────────────────────────\n")


# Entry point
def signal_handler(signum, frame):
    print("\n\n🛑 Shutting down gracefully...")
    global engine
    if 'engine' in globals():
        engine.stop()
    sys.exit(0)


def main() -> int:
    global engine

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    engine = WakeWordEngine(
       
        model_path=str(PROJECT_ROOT / "models" / "dael_v1.5.pt"),

        # ── Tune these after running run_diagnostics() ──────────────────
        # Start here, then check logs for the trainer's reported best threshold.
        confidence_threshold=0.4,

        # If confidences don't rise at all when you say the wake word,
        # try normalize_features=False — your .pt files may be raw.
        normalize_features=False,

        window_duration=1.0,          # increase if your wake word is long
        smoothing_alpha=0.5,          # lower = smoother, slower to react
        detection_cooldown=2.0,
    )

    # Diagnotics (comment out to use the normal script)
    #engine.run_diagnostics(seconds=60)
    #return 0

    if not engine.start():
        print("❌ Failed to start wake word engine")
        return 1

    try:
        print(f"\n🎯 Listening... (threshold: {engine.confidence_threshold:.3f})")
        while engine.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        engine.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())