from __future__ import annotations

from pathlib import Path
from typing import Any


def _ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000.0))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[dict[str, Any]], path: Path) -> Path:
    """Write SubRip. Each cue: start, end, text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    n = 0
    for cue in cues:
        text = (cue.get("text") or "").strip()
        if not text:
            continue
        start = float(cue["start"])
        end = float(cue["end"])
        if end <= start:
            end = start + 0.05
        n += 1
        lines.append(str(n))
        lines.append(f"{_ts(start)} --> {_ts(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
