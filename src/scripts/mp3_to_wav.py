"""
Convert MP3 → WAV (16kHz, mono, 16-bit PCM)
"""

import subprocess
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torchaudio] Available. Device: {DEVICE.upper()}")
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    DEVICE = "cpu"
    print("[torchaudio] Not installed. Falling back to FFmpeg only.")

TARGET_SR = 16_000
MAX_WORKERS = 32


def setup_directories():
    wav_output_dir = Path("src/data/raw/negative_addition")
    wav_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"WAV output: {wav_output_dir.resolve()}")
    return wav_output_dir


def get_mp3_files():
    input_dir = Path("src/data/raw/negative_add")
    print(f"Scanning: {input_dir.resolve()}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir.resolve()}")
    mp3_files = [f for f in input_dir.glob("*.mp3") if f.stat().st_size > 0]
    if not mp3_files:
        raise FileNotFoundError(f"No MP3 files found in {input_dir.resolve()}")
    return mp3_files


# If CUDA fails
def convert_via_ffmpeg(mp3_path: Path, wav_path: Path) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp3_path),
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-acodec", "pcm_s16le",
        str(wav_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"\n❌ FFmpeg error [{mp3_path.name}]: {result.stderr.decode()[:200]}")
        return False
    return True


def convert_via_torchaudio(mp3_path: Path, wav_path: Path) -> bool:
    try:
        waveform, sr = torchaudio.load(str(mp3_path))      # CPU decode
        waveform = waveform.to(DEVICE)                      # → GPU

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # mix down to mono

        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=TARGET_SR
            ).to(DEVICE)
            waveform = resampler(waveform)

        waveform = waveform.cpu()                           # → CPU for saving
        torchaudio.save(
            str(wav_path), waveform, TARGET_SR,
            encoding="PCM_S", bits_per_sample=16,
        )
        return True

    except Exception as e:
        print(f"\n⚠️  torchaudio failed for {mp3_path.name}: {e}. Retrying with FFmpeg...")
        return convert_via_ffmpeg(mp3_path, wav_path)


def convert_single(mp3_path: Path, wav_path: Path) -> bool:
    if TORCHAUDIO_AVAILABLE:
        return convert_via_torchaudio(mp3_path, wav_path)
    return convert_via_ffmpeg(mp3_path, wav_path)


def process_files():
    wav_output_dir = setup_directories()
    mp3_files = get_mp3_files()
    total = len(mp3_files)
    print(f"Found {total} files  |  Workers: {MAX_WORKERS}\n")

    tasks = [(mp3, wav_output_dir / f"{mp3.stem}.wav") for mp3 in mp3_files]

    successful = 0
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_mp3 = {
            executor.submit(convert_single, mp3, wav): mp3
            for mp3, wav in tasks
        }
        with tqdm(total=total, desc="Converting", unit="file") as pbar:
            for future in as_completed(future_to_mp3):
                mp3_file = future_to_mp3[future]
                try:
                    ok = future.result()
                except Exception as exc:
                    ok = False
                    print(f"\n❌ Worker exception [{mp3_file.name}]: {exc}")

                if ok:
                    successful += 1
                else:
                    failed.append(mp3_file.name)

                pbar.update(1)
                pbar.set_postfix(ok=successful, fail=len(failed))

    print(f"\n✅ Converted : {successful}/{total}")
    if failed:
        print(f"❌ Failed    : {len(failed)}")
        Path("data/failed_conversions.txt").write_text("\n".join(failed))
        print("   → Failures saved to data/failed_conversions.txt")


def main():
    try:
        process_files()
        print("\n🎉 Done!")
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()