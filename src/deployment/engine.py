#!/usr/bin/env python3
"""
Simple Wake Word Inference Engine
"""

import time
import signal
import sys
import subprocess
from pathlib import Path
from engine_logic import WakeWordEngine

# Get the project root directory (where src/ is located)
PROJECT_ROOT = Path(__file__).parent.parent

# Sound file path
SOUND_FILE = Path(__file__).parent / "answer.wav"

def play_sound():
    """Play a sound file when wake word is detected."""
    if SOUND_FILE.exists():
        subprocess.run(['aplay', str(SOUND_FILE)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def on_wake_word_detected(confidence):
    """Callback function when wake word is detected."""
    print(f"WAKE WORD DETECTED! Confidence: {confidence:.4f}")
    play_sound()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global engine
    if 'engine' in globals():
        engine.stop()
    sys.exit(0)

def main():
    """Main function."""
    global engine
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    engine = WakeWordEngine(
        model_path=str(PROJECT_ROOT / "models" / "callina.pt"),
        confidence_threshold=0.995,
        window_duration=1.5,
        detection_cooldown=2.0,
        callback=on_wake_word_detected,
        silent=True  # Back to silent mode
    )
    
    if not engine.start():
        return 1
    
    print("Start speaking...")
    try:
        while engine.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)