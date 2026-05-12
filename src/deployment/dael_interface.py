import time
import signal
import sys
import subprocess
import threading
import logging
import warnings
import os
import ctypes
import ctypes.util
import contextlib
from pathlib import Path
from collections import deque
from typing import Optional, Callable

import numpy as np
import pyaudio
import torch
import torchaudio.transforms as T
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Suppress ALSA / JACK noise on Linux
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["ALSA_PCM_CARD"] = "0"
os.environ["ALSA_PCM_DEVICE"] = "0"
os.environ["ALSA_QUIET"] = "1"

libc = ctypes.CDLL(ctypes.util.find_library("c"))


@contextlib.contextmanager
def suppress_audio_warnings():
    """Redirect stderr to /dev/null for the duration of the block."""
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)


# Paths
MODEL_PATH = "src/models/dael.onnx"
SOUND_FILE = Path("src/deployment/confirm.wav")


# Feature extractor
class WakeWordFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_mfcc: int = 13,
        n_fft: int = 512,
        win_length: int = 400,   # 25 ms at 16 kHz
        hop_length: int = 160,   # 10 ms at 16 kHz
    ):
        self.sample_rate = sample_rate
        self.n_mels      = n_mels
        self.n_mfcc      = n_mfcc
        self.device      = torch.device("cpu")

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft":       n_fft,
                "win_length":  win_length,
                "hop_length":  hop_length,
                "n_mels":      n_mels,
                "power":       2.0,
            },
        )

        self.compute_deltas = T.ComputeDeltas()
        self._resampler: Optional[T.Resample] = None

    def _to_mono_16k(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample + stereo → mono."""
        if orig_sr != self.sample_rate:
            if self._resampler is None or self._resampler.orig_freq != orig_sr:
                self._resampler = T.Resample(orig_sr, self.sample_rate)
            waveform = self._resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


    def extract_features(self, waveform: torch.Tensor) -> Optional[np.ndarray]:
    
        try:
            # Log-mel (80)
            mel_spec     = self.mel_spectrogram(waveform)
            log_mel_spec = torch.log(mel_spec + 1e-8)
            log_mel      = log_mel_spec.squeeze(0)     # [80, T]

            # Delta-mel (80)
            delta_mel    = self.compute_deltas(log_mel_spec).squeeze(0)  # [80, T]

            # MFCC (13)
            mfcc_spec    = self.mfcc_transform(waveform)
            mfcc         = mfcc_spec.squeeze(0)        # [13, T]

            # Delta-MFCC (13)
            delta_mfcc   = self.compute_deltas(mfcc_spec).squeeze(0)    # [13, T]

            # Stack → [186, T]
            stacked = torch.cat([log_mel, delta_mel, mfcc, delta_mfcc], dim=0)

            # Add batch dim → [1, 186, T]  and convert to float32 numpy
            return stacked.unsqueeze(0).numpy().astype(np.float32)

        except Exception as exc:
            print(f"[FeatureExtractor] Error: {exc}")
            return None


# ONNX-backed wake word engine
class DaelONNXEngine:
    def __init__(
        self,
        model_path: str  = str(MODEL_PATH),
        confidence_threshold: float = 0.8,
        sample_rate: int = 16000,
        chunk_size:  int = 1024,
        window_duration:    float = 1.5,
        detection_cooldown: float = 2.0,
        callback: Optional[Callable[[float], None]] = None,
        silent: bool = True,
    ):
        self.model_path           = model_path
        self.confidence_threshold = confidence_threshold
        self.sample_rate          = sample_rate
        self.chunk_size           = chunk_size
        self.window_duration      = window_duration
        self.detection_cooldown   = detection_cooldown
        self.callback             = callback
        self.silent               = silent

        # Audio ring-buffer  (holds exactly one detection window)
        self.buffer_size  = int(sample_rate * window_duration)
        self.audio_buffer = deque(maxlen=self.buffer_size)

        # Threading / state
        self.is_running          = False
        self.detection_thread: Optional[threading.Thread] = None
        self.last_detection_time = 0.0

        # PyAudio handles
        self.audio  = None
        self.stream = None

        # ONNX session (populated in load_model)
        self.session:     Optional[ort.InferenceSession] = None
        self.input_name:  Optional[str] = None
        self.output_name: Optional[str] = None

        # Feature extractor (CPU-only, no torch model needed)
        self.feature_extractor = WakeWordFeatureExtractor()

        # Stats
        self.confidence_history = deque(maxlen=100)
        self.detection_count    = 0

        if not self.silent:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s  %(levelname)s  %(message)s",
                datefmt="%H:%M:%S",
            )
            self._logger = logging.getLogger(__name__)

    # Internal helpers
    def _log(self, msg: str, level: str = "info"):
        if not self.silent:
            getattr(self._logger, level)(msg)

    # Model
    def load_model(self) -> bool:
        try:
            self._log(f"Loading ONNX model: {self.model_path}")
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            self.input_name  = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # Sanity-check with a dummy input [1, 186, 100]
            dummy = np.random.randn(1, 186, 100).astype(np.float32)
            out   = self.session.run([self.output_name], {self.input_name: dummy})[0]

            self._log("✅ ONNX model loaded and verified.")
            self._log(f"   Input : {self.input_name}  {dummy.shape}")
            self._log(f"   Output: {self.output_name}  {out.shape}  range [{out.min():.4f}, {out.max():.4f}]")
            return True

        except Exception as exc:
            print(f"[DaelONNXEngine] ❌ Failed to load model: {exc}")
            return False

    # Audio
    def setup_audio(self) -> bool:
        try:
            with suppress_audio_warnings():
                self.audio = pyaudio.PyAudio()

            # Pick the first available input device
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    self._log(f"🎤 Audio device: {info['name']}")
                    break

            with suppress_audio_warnings():
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback,
                )

            self._log(f"✅ Audio stream ready  ({self.sample_rate} Hz, chunk {self.chunk_size})")
            return True

        except Exception as exc:
            print(f"[DaelONNXEngine] ❌ Audio setup failed: {exc}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Runs in the PyAudio thread — just fill the ring-buffer."""
        if status and not self.silent:
            self._log(f"Audio callback status: {status}", "warning")
        chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.extend(chunk)
        return (None, pyaudio.paContinue)


    # Preprocessing
    def _preprocess(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        try:
            # Hard-trim / zero-pad to exactly buffer_size
            if len(audio_data) > self.buffer_size:
                audio_data = audio_data[-self.buffer_size:]
            elif len(audio_data) < self.buffer_size:
                audio_data = np.concatenate(
                    [np.zeros(self.buffer_size - len(audio_data)), audio_data]
                )

            waveform = torch.FloatTensor(audio_data).unsqueeze(0)

            features = self.feature_extractor.extract_features(waveform)
            if features is None:
                return None

            if features.shape[1] != 186:
                self._log(f"❌ Feature dim mismatch: got {features.shape[1]}, expected 186", "error")
                return None

            return features  # [1, 186, T]  float32 numpy

        except Exception as exc:
            self._log(f"❌ Preprocessing error: {exc}", "error")
            return None

    # Inference
    def _run_inference(self, features: np.ndarray) -> tuple[bool, float]:
        try:
            raw = self.session.run([self.output_name], {self.input_name: features})[0]

            # Flatten to scalar
            raw = raw.flatten()
            confidence = float(raw[0])

            # Apply sigmoid if the model outputs raw logits
            if not (0.0 <= confidence <= 1.0):
                confidence = float(1.0 / (1.0 + np.exp(-confidence)))

            self.confidence_history.append(confidence)
            return confidence >= self.confidence_threshold, confidence

        except Exception as exc:
            self._log(f"❌ Inference error: {exc}", "error")
            return False, 0.0

    # Detection handling
    def _handle_detection(self, confidence: float):
        now = time.time()
        if now - self.last_detection_time < self.detection_cooldown:
            return

        self.last_detection_time = now
        self.detection_count    += 1
        ts = time.strftime("%H:%M:%S", time.localtime(now))

        print(f"\n🎯 WAKE WORD DETECTED!  #{self.detection_count}")
        print(f"   Time       : {ts}")
        print(f"   Confidence : {confidence:.4f}  ({confidence:.2%})")
        print(f"   Threshold  : {self.confidence_threshold:.4f}\n")

        if self.callback:
            try:
                self.callback(confidence)
            except Exception as exc:
                self._log(f"Callback error: {exc}", "error")

    # Detection worker thread
    def _detection_worker(self):
        self._log("🔄 Detection worker started")
        consecutive_errors = 0

        while self.is_running:
            try:
                if len(self.audio_buffer) < self.buffer_size:
                    time.sleep(0.1)
                    continue

                audio_data = np.array(list(self.audio_buffer))
                features   = self._preprocess(audio_data)

                if features is None:
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        print("[DaelONNXEngine] ❌ Too many preprocessing errors — stopping.")
                        break
                    time.sleep(0.1)
                    continue

                consecutive_errors = 0
                detected, confidence = self._run_inference(features)

                if detected:
                    self._handle_detection(confidence)

                time.sleep(0.05)  # ~20 inferences / second

            except Exception as exc:
                consecutive_errors += 1
                self._log(f"❌ Worker error: {exc}", "error")
                if consecutive_errors >= 10:
                    print("[DaelONNXEngine] ❌ Too many consecutive errors — stopping.")
                    break
                time.sleep(0.5)

        self._log("🛑 Detection worker stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> bool:
        if self.is_running:
            self._log("⚠️  Engine already running", "warning")
            return False

        print("🚀 Starting DAEL ONNX Wake Word Engine …")

        if not self.load_model():
            return False
        if not self.setup_audio():
            return False

        with suppress_audio_warnings():
            self.stream.start_stream()

        self.is_running = True
        self.detection_thread = threading.Thread(
            target=self._detection_worker, daemon=True
        )
        self.detection_thread.start()

        print(f"✅ Engine running.  Confidence threshold: {self.confidence_threshold}")
        return True

    def stop(self):
        if not self.is_running:
            return

        print("\n🛑 Stopping engine …")
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

        self._print_stats()
        print("✅ Engine stopped.")

    def _print_stats(self):
        if not self.confidence_history:
            return
        confs = list(self.confidence_history)
        print(f"\n📊 Session stats — {self.detection_count} detection(s)")
        print(f"   avg confidence : {np.mean(confs):.4f}")
        print(f"   max confidence : {np.max(confs):.4f}")
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        print("   Distribution:")
        for lo, hi in ranges:
            n   = sum(1 for c in confs if lo <= c < hi)
            pct = n / len(confs) * 100
            bar = "█" * max(1, n // max(1, len(confs) // 20))
            print(f"     {lo:.1f}–{hi:.1f}: {n:4d} ({pct:5.1f}%) {bar}")


# Entry point
def play_sound():
    """Non-blocking playback of confirm.wav via aplay."""
    if SOUND_FILE.exists():
        subprocess.Popen(
            ["aplay", str(SOUND_FILE)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print(f"[sound] ⚠️  {SOUND_FILE} not found — skipping audio cue.")


def on_wake_word(confidence: float):
    """Callback fired on every confirmed detection."""
    play_sound()


engine: Optional[DaelONNXEngine] = None


def _signal_handler(signum, frame):
    global engine
    if engine:
        engine.stop()
    sys.exit(0)


def main() -> int:
    global engine

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    engine = DaelONNXEngine(
        model_path           = str(MODEL_PATH),
        confidence_threshold = 0.8,
        window_duration      = 0.75,
        detection_cooldown   = 2.0,
        callback             = on_wake_word,
        silent               = True,
    )

    if not engine.start():
        return 1

    print("🎙  Listening … (Ctrl-C to quit)\n")
    try:
        while engine.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())