from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def score_timed(timed: dict[str, Any]) -> dict[str, Any]:
    """Heuristic pipeline scorecard (no ground-truth lyrics in Mode B)."""
    lines = timed.get("lines") or []
    words = timed.get("words") or []
    duration = float((timed.get("audio") or {}).get("duration_s") or 0.0)
    text = (timed.get("text") or "").strip()

    issues: list[str] = []
    notes: list[str] = []

    if not lines:
        issues.append("no_lines")
    if not words:
        issues.append("no_words")
    if not text:
        issues.append("empty_transcript")

    # Coverage: last cue end vs duration
    last_end = 0.0
    for w in words:
        last_end = max(last_end, float(w.get("end") or 0))
    for ln in lines:
        last_end = max(last_end, float(ln.get("end") or 0))
    coverage = (last_end / duration) if duration > 0 else None
    if coverage is not None and coverage < 0.5:
        issues.append(f"low_timeline_coverage:{coverage:.2f}")
    elif coverage is not None and coverage < 0.75:
        notes.append(f"moderate_timeline_coverage:{coverage:.2f}")

    # Overlaps / inverted intervals
    inv = 0
    for w in words:
        if float(w.get("end") or 0) < float(w.get("start") or 0):
            inv += 1
    if inv:
        issues.append(f"inverted_word_intervals:{inv}")

    # Gap analysis between consecutive words
    big_gaps = 0
    gap_sum = 0.0
    gap_n = 0
    ordered = sorted(words, key=lambda w: float(w.get("start") or 0))
    for a, b in zip(ordered, ordered[1:]):
        gap = float(b["start"]) - float(a["end"])
        if gap > 0:
            gap_sum += gap
            gap_n += 1
        if gap > 4.0:
            big_gaps += 1
    avg_gap = (gap_sum / gap_n) if gap_n else None
    if big_gaps:
        notes.append(f"word_gaps_gt_4s:{big_gaps}")

    # Confidence if present
    confs = [float(w["confidence"]) for w in words if w.get("confidence") is not None]
    mean_conf = sum(confs) / len(confs) if confs else None
    low_conf = sum(1 for c in confs if c < 0.35)
    if mean_conf is not None and mean_conf < 0.45:
        issues.append(f"low_mean_word_confidence:{mean_conf:.2f}")
    if confs and low_conf / len(confs) > 0.35:
        notes.append(f"many_low_conf_words:{low_conf}/{len(confs)}")

    # Density
    wpm = None
    if duration > 0 and words:
        wpm = len(words) / (duration / 60.0)
        if wpm > 220:
            notes.append(f"very_high_wpm:{wpm:.0f}")
        elif wpm < 20 and len(words) > 5:
            notes.append(f"very_low_wpm:{wpm:.0f}")

    # Simple quality band
    if issues:
        band = "weak"
    elif notes:
        band = "usable_with_cleanup"
    else:
        band = "structurally_ok"

    return {
        "band": band,
        "n_lines": len(lines),
        "n_words": len(words),
        "duration_s": duration,
        "timeline_coverage": coverage,
        "mean_word_confidence": mean_conf,
        "low_conf_words": low_conf if confs else None,
        "avg_inter_word_gap_s": avg_gap,
        "words_per_minute": wpm,
        "issues": issues,
        "notes": notes,
        "language": timed.get("language"),
        "model": (timed.get("engine") or {}).get("model"),
        "mode": timed.get("mode"),
        "transcript_preview": text[:280],
    }


def score_run_dir(run_dir: Path) -> dict[str, Any]:
    timed_path = run_dir / "timed.json"
    timed = json.loads(timed_path.read_text(encoding="utf-8"))
    card = score_timed(timed)
    card["run_dir"] = str(run_dir)
    out_path = run_dir / "scorecard.json"
    out_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return card
