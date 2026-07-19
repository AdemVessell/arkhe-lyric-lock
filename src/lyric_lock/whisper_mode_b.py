from __future__ import annotations

from pathlib import Path
from typing import Any


def _pick_device(requested: str | None) -> str:
    """Prefer explicit device; avoid MPS (SparseMPS load failures with Whisper+torch)."""
    if requested:
        return requested
    # CPU is the reliable default on this workstation for openai-whisper.
    return "cpu"


def transcribe_mode_b(
    wav_path: Path,
    *,
    model_name: str = "large-v3",
    language: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Mode B: no provided lyrics. Whisper transcribes with word timestamps.

    Music-oriented decode knobs: default Whisper no_speech_threshold often
    swallows sung intros (song C medium first cue was ~31.6s while audio is
    already loud from t≈0). Lower no_speech + permissive logprob + no
    condition_on_previous_text reduces cascading skip.
    """
    import whisper

    dev = _pick_device(device)
    model = whisper.load_model(model_name, device=dev)

    # Prefer near-default Whisper decode. Aggressive no_speech / temperature
    # ladders were observed to *truncate* full-song ASR on song C stem (22 words
    # vs 57 vanilla). Stem isolation is the music lever; keep ASR stable.
    decode: dict[str, Any] = {
        "word_timestamps": True,
        "verbose": False,
        "task": "transcribe",
        "fp16": False,  # safer on CPU / mixed machines
        "condition_on_previous_text": True,
        "temperature": 0.0,
    }
    if language:
        decode["language"] = language

    result = model.transcribe(str(wav_path), **decode)

    lines: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    w_i = 0
    for li, seg in enumerate(result.get("segments") or []):
        line_id = f"L{li}"
        seg_text = (seg.get("text") or "").strip()
        lines.append(
            {
                "id": line_id,
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "text": seg_text,
                "confidence": None,
            }
        )
        for w in seg.get("words") or []:
            wt = (w.get("word") or "").strip()
            if not wt:
                continue
            words.append(
                {
                    "id": f"W{w_i}",
                    "start": float(w.get("start") or 0.0),
                    "end": float(w.get("end") or 0.0),
                    "text": wt,
                    "line_id": line_id,
                    "confidence": w.get("probability"),
                }
            )
            w_i += 1

    return {
        "language": result.get("language"),
        "text": (result.get("text") or "").strip(),
        "lines": lines,
        "words": words,
        "engine": {
            "name": "openai-whisper",
            "model": model_name,
            "word_timestamps": True,
            "task": "transcribe",
            "temperature": 0.0,
            "condition_on_previous_text": True,
        },
    }
