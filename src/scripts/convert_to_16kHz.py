import os
import json
import subprocess
import tempfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Config ───────────────────────────────────────────────────────────────────
INPUT_DIR = "src/data/processed/validation/negative"
WORKERS   = 12
# ──────────────────────────────────────────────────────────────────────────────

def needs_conversion(filepath):
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", filepath
    ], capture_output=True, text=True)

    if result.returncode != 0:
        return True

    streams = json.loads(result.stdout).get("streams", [])
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)

    if not audio:
        return False

    return not (
        int(audio.get("sample_rate", 0)) == 16000 and
        int(audio.get("channels", 0))    == 1     and
        audio.get("sample_fmt", "")      == "s16"
    )


def process_file(filepath):
    """Returns a status string: 'converted', 'skipped', or 'failed'."""
    if not needs_conversion(filepath):
        return filepath, "skipped"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=os.path.dirname(filepath))
    os.close(tmp_fd)

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", filepath,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        tmp_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode == 0:
        os.replace(tmp_path, filepath)
        return filepath, "converted"
    else:
        os.remove(tmp_path)
        return filepath, "failed"


wav_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(INPUT_DIR)
    for f in files if f.lower().endswith(".wav")
]

if not wav_files:
    print("No .wav files found.")
else:
    skipped   = 0
    converted = 0
    failed    = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_file, f): f for f in wav_files}

        with tqdm(as_completed(futures), total=len(wav_files), unit="file") as progress:
            for future in progress:
                filepath, status = future.result()
                progress.set_description(os.path.basename(filepath))

                if status == "converted":
                    converted += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    tqdm.write(f"  ✗ FAILED: {filepath}")
                    failed += 1

    print(f"\nDone! — {converted} converted, {skipped} skipped, {failed} failed.")