from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class PreprocessError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise PreprocessError("ffmpeg not found on PATH")
    return exe


def to_whisper_wav(src: Path, dest: Path, sample_rate: int = 16000) -> Path:
    """Decode any audio to mono PCM WAV at sample_rate for Whisper."""
    ffmpeg = require_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreprocessError(
            f"ffmpeg failed ({proc.returncode}): {proc.stderr[-2000:]}"
        )
    if not dest.is_file() or dest.stat().st_size == 0:
        raise PreprocessError(f"ffmpeg produced empty output: {dest}")
    return dest


def probe_duration_s(src: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None
