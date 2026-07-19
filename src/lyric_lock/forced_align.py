from __future__ import annotations

from pathlib import Path
from typing import Any


def _lyrics_plain(text: str) -> str:
    """Strip comments/section tags; keep spoken lyric lines."""
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        lines.append(line)
    return "\n".join(lines)


def forced_align_words(
    audio_wav: Path,
    lyrics_text: str,
    *,
    model_name: str = "medium",
    language: str = "en",
    device: str = "cpu",
    max_word_dur: float = 8.0,
    word_dur_factor: float = 2.5,
) -> dict[str, Any]:
    """
    True Mode A timing: known lyrics aligned to audio at word level.

    Uses stable-ts Whisper.align (cross-attention word alignment of fixed text),
    not free ASR + fuzzy paste.

    max_word_dur raised for singing holds (default stable-ts 3s clips long notes).
    """
    import stable_whisper

    plain = _lyrics_plain(lyrics_text)
    if not plain.strip():
        raise ValueError("lyrics text empty after stripping comments")

    audio_wav = audio_wav.expanduser().resolve()
    model = stable_whisper.load_model(model_name, device=device)

    # align plain text → word timestamps against this audio (stem preferred)
    align_kwargs = dict(
        language=language,
        max_word_dur=max_word_dur,
        word_dur_factor=word_dur_factor,
        remove_instant_words=False,
        original_split=False,  # avoid split bugs when trailing words fail
        ignore_compatibility=True,
        failure_threshold=None,
    )
    try:
        result = model.align(str(audio_wav), plain, **align_kwargs)
    except Exception as e1:
        # retry without optional knobs
        try:
            result = model.align(
                str(audio_wav),
                plain,
                language=language,
                ignore_compatibility=True,
            )
        except Exception as e2:
            raise RuntimeError(f"forced align failed: {e1!r} / retry: {e2!r}") from e2

    # refine timestamps against audio (improves boundaries / holds)
    try:
        result = model.refine(str(audio_wav), result, word_level=True)
    except Exception:
        pass

    words: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    w_i = 0
    segs = list(result.segments) if hasattr(result, "segments") else []

    for li, seg in enumerate(segs):
        line_id = f"L{li}"
        seg_text = (getattr(seg, "text", None) or "").strip()
        seg_start = float(getattr(seg, "start", 0.0) or 0.0)
        seg_end = float(getattr(seg, "end", 0.0) or 0.0)
        lines.append(
            {
                "id": line_id,
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "confidence": None,
            }
        )
        seg_words = getattr(seg, "words", None) or []
        for w in seg_words:
            wt = (getattr(w, "word", None) or getattr(w, "text", None) or "").strip()
            if not wt:
                continue
            ws = float(getattr(w, "start", 0.0) or 0.0)
            we = float(getattr(w, "end", 0.0) or 0.0)
            if we <= ws:
                we = ws + 0.08
            prob = getattr(w, "probability", None)
            words.append(
                {
                    "id": f"W{w_i}",
                    "start": ws,
                    "end": we,
                    "text": wt,
                    "line_id": line_id,
                    "confidence": float(prob) if prob is not None else None,
                    "source": "forced_align",
                    "duration_s": we - ws,
                }
            )
            w_i += 1

    # if segments empty, try flat words API
    if not words and hasattr(result, "all_words"):
        for w in result.all_words():
            wt = (getattr(w, "word", None) or "").strip()
            if not wt:
                continue
            ws = float(w.start)
            we = float(w.end)
            words.append(
                {
                    "id": f"W{w_i}",
                    "start": ws,
                    "end": we,
                    "text": wt,
                    "line_id": "L0",
                    "confidence": getattr(w, "probability", None),
                    "source": "forced_align",
                    "duration_s": we - ws,
                }
            )
            w_i += 1
        if words:
            lines = [
                {
                    "id": "L0",
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "text": " ".join(w["text"] for w in words),
                    "confidence": None,
                }
            ]

    return {
        "language": language,
        "text": plain.replace("\n", " ").strip(),
        "lines": lines,
        "words": words,
        "engine": {
            "name": "stable-ts-align",
            "model": model_name,
            "method": "forced_align+refine",
            "max_word_dur": max_word_dur,
            "word_dur_factor": word_dur_factor,
        },
    }
