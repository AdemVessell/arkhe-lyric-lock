from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class SeparateError(RuntimeError):
    pass


# Cleanroom venv (outside product tree) — admitted tool runtime
DEFAULT_CLEANROOM = Path.home() / "ArkheCleanroom" / "lyric-lock"
DEFAULT_VENV_PYTHON = DEFAULT_CLEANROOM / ".venv" / "bin" / "python"


def cleanroom_python() -> Path:
    override = os.environ.get("ARKHE_LYRIC_LOCK_CLEANROOM_PYTHON")
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p
    if DEFAULT_VENV_PYTHON.is_file():
        return DEFAULT_VENV_PYTHON
    raise SeparateError(
        f"Cleanroom Python not found at {DEFAULT_VENV_PYTHON}. "
        "Install demucs into ~/ArkheCleanroom/lyric-lock/.venv first."
    )


def separate_vocals(
    audio: Path,
    out_dir: Path,
    *,
    model: str = "htdemucs",
    device: str = "cpu",
    two_stems: str = "vocals",
) -> Path:
    """
    Run admitted demucs in cleanroom; return path to vocals stem WAV.

    Does not vendor demucs source — only invokes cleanroom env.
    """
    audio = audio.expanduser().resolve()
    if not audio.is_file():
        raise SeparateError(f"audio missing: {audio}")

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    py = cleanroom_python()

    # demucs -n htdemucs --two-stems=vocals -o out_dir audio
    cmd = [
        str(py),
        "-m",
        "demucs",
        "-n",
        model,
        "--two-stems",
        two_stems,
        "-d",
        device,
        "-o",
        str(out_dir),
        str(audio),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SeparateError(
            f"demucs failed ({proc.returncode}):\n"
            f"{proc.stderr[-3000:] or proc.stdout[-3000:]}"
        )

    # Output layout: out_dir/<model>/<track_name>/vocals.wav
    stem_name = audio.stem
    candidates = list(out_dir.glob(f"**/{stem_name}/vocals.wav"))
    if not candidates:
        # demucs may sanitize names differently
        candidates = list(out_dir.glob("**/vocals.wav"))
    if not candidates:
        raise SeparateError(
            f"vocals.wav not found under {out_dir}. stderr tail:\n{proc.stderr[-1500:]}"
        )
    # Prefer newest
    vocals = max(candidates, key=lambda p: p.stat().st_mtime)
    return vocals.resolve()
