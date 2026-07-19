from __future__ import annotations

"""
Whisper anchor fusion v2 — melisma rescue on a CTC star spine.

Claude lab (song B, 2026-07-19):
  CTC Viterbi sprints through held vowels; Whisper word timestamps track
  melisma. Inside star-flagged windows, DP-match sheet tokens to whisper
  words (char-similarity); matched take whisper timing; unmatched sheet
  tokens abstain; whisper hallucinations match no sheet → ignored.

Greedy matching cascades (a parenthetical ate 'hope') — DP required.
No display lead applied here (zero-lead doctrine).
"""

import difflib
from pathlib import Path
from typing import Any

MIN_SIM = 0.45
GAP = -0.15
# Small inter-phrase stars are normal — ignore for fusion windows
MIN_STAR_S = 2.5
# Instrumental / melisma dumps (song B 64–94 ≈ 30s)
HUGE_STAR_S = 8.0
# CTC sprints melisma *before* dumping remainder into a long star
LOOKBACK_HUGE_S = 14.0
LOOKBACK_MED_S = 4.0
LOOKAHEAD_S = 1.0


def norm_token(t: str) -> str:
    return "".join(
        c for c in t.lower().replace("’", "'") if c.isalpha() or c == "'"
    )


def dp_align(sheet: list[str], asr: list[str]) -> dict[int, int]:
    """Needleman–Wunsch with char-similarity substitution. sheet_idx → asr_idx."""
    n, m = len(sheet), len(asr)
    if n == 0 or m == 0:
        return {}
    sim = [
        [difflib.SequenceMatcher(None, s, a).ratio() for a in asr]
        for s in sheet
    ]
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]  # 1=diag 2=up 3=left
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP
        back[i][0] = 2
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP
        back[0][j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = sim[i - 1][j - 1]
            diag = score[i - 1][j - 1] + (s if s >= MIN_SIM else GAP * 2)
            up = score[i - 1][j] + GAP
            left = score[i][j - 1] + GAP
            score[i][j], back[i][j] = max(
                (diag, 1), (up, 2), (left, 3), key=lambda x: x[0]
            )
    pairs: dict[int, int] = {}
    i, j = n, m
    while i > 0 or j > 0:
        b = back[i][j]
        if b == 1:
            if sim[i - 1][j - 1] >= MIN_SIM:
                pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        else:
            j -= 1
    return pairs


def fusion_windows_from_stars(
    star_spans: list[dict[str, Any]],
    *,
    min_star_s: float = MIN_STAR_S,
    huge_star_s: float = HUGE_STAR_S,
    lookback_huge_s: float = LOOKBACK_HUGE_S,
    lookback_med_s: float = LOOKBACK_MED_S,
    lookahead_s: float = LOOKAHEAD_S,
) -> list[tuple[float, float]]:
    """
    Build fuse windows from large star spans only.

    Melisma advance: CTC packs next lyrics early, then dumps remainder into a
    long star. Fuse sheet words in [star.start - lookback, star.start + ahead].
    Huge stars (instrumental blanks) get longer lookback.
    """
    raw: list[tuple[float, float]] = []
    for s in star_spans:
        a, b = float(s["start"]), float(s["end"])
        dur = b - a
        if dur < min_star_s:
            continue
        lb = lookback_huge_s if dur >= huge_star_s else lookback_med_s
        raw.append((max(0.0, a - lb), a + lookahead_s))
    if not raw:
        return []
    raw.sort()
    merged: list[list[float]] = [[raw[0][0], raw[0][1]]]
    for a, b in raw[1:]:
        if a <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(float(a), float(b)) for a, b in merged]


def _in_star_window(
    start: float,
    end: float,
    star_spans: list[dict[str, Any]],
    *,
    windows: list[tuple[float, float]] | None = None,
) -> bool:
    wins = windows if windows is not None else fusion_windows_from_stars(star_spans)
    for lo, hi in wins:
        if start < hi and end > lo:
            return True
    return False


def _clip_windows_to_segments(
    wins: list[tuple[float, float]],
    segments: list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    """
    Split fusion windows at segment boundaries (sheet-gate
    segments_merged_for_fusion incl. MIX splice cuts). A DP window must never
    straddle a song join — that is the exact bleed the reset exists to stop.
    """
    if not segments:
        return wins
    pieces: list[tuple[float, float]] = []
    for lo, hi in wins:
        for sa, sb in segments:
            a, b = max(lo, float(sa)), min(hi, float(sb))
            if b - a > 0.2:
                pieces.append((a, b))
    return pieces or wins


def fuse_whisper_anchors(
    words: list[dict[str, Any]],
    whisper_words: list[dict[str, Any]],
    *,
    star_spans: list[dict[str, Any]] | None = None,
    # If True and no star spans, fuse nothing (safe). If force_window set, use it.
    force_window: tuple[float, float] | None = None,
    # sheet-gate segments_merged_for_fusion → DP reset boundaries
    segments: list[tuple[float, float]] | None = None,
    drop_parenthetical: bool = True,
    # Lab wreck mode dropped unmatched; product keeps CTC unless strict abstain
    abstain_unmatched: bool = False,
) -> dict[str, Any]:
    """
    Fuse Whisper timings into sheet words inside star-flagged (or force) windows.

    Outside windows: CTC words unchanged.
    Inside: DP match PER WINDOW (reset at window and segment boundaries);
    matched → whisper timing.
    Unmatched: keep CTC by default (safer); set abstain_unmatched to drop.
    Parentheticals always abstain when drop_parenthetical.
    """
    star_spans = list(star_spans or [])
    out = [dict(w) for w in words]
    asr = [
        (float(w["start"]), float(w["end"]), norm_token(w.get("text") or ""))
        for w in whisper_words
        if norm_token(w.get("text") or "")
    ]
    base_wins = (
        [force_window] if force_window is not None
        else fusion_windows_from_stars(star_spans)
    )
    auto_wins = _clip_windows_to_segments(base_wins, segments)

    def in_any_win(w: dict[str, Any]) -> bool:
        s, e = float(w["start"]), float(w["end"])
        return any(s < hi and e > lo for lo, hi in auto_wins)

    win_idx = [i for i, w in enumerate(out) if in_any_win(w)]
    if not win_idx or not asr:
        return {
            "words": out,
            "actions": [],
            "engine": {
                "name": "whisper-anchor-fusion-v2",
                "n_window": 0,
                "n_fused": 0,
                "n_abstained": 0,
                "fusion_windows": auto_wins,
                "n_segments": len(segments or []),
                "note": "no star window or no whisper words",
            },
        }

    # Per-window DP: match resets at every window (and thus segment) boundary,
    # ASR pool clipped to the window's segment — no cross-splice matching.
    fused = 0
    abstained: list[str] = []
    actions: list[dict[str, Any]] = []
    n_lexical = 0
    claimed: set[int] = set()
    for wlo, whi in auto_wins:
        lexical_idx = []
        for i in win_idx:
            if i in claimed:
                continue
            s, e = float(out[i]["start"]), float(out[i]["end"])
            if not (s < whi and e > wlo):
                continue
            claimed.add(i)
            tx = out[i].get("text") or ""
            if drop_parenthetical and tx.startswith("("):
                out[i]["_drop"] = True
                continue
            if not norm_token(tx):
                out[i]["_drop"] = True
                continue
            lexical_idx.append(i)
        if not lexical_idx:
            continue
        n_lexical += len(lexical_idx)

        a_lo, a_hi = wlo - 2.0, whi + 2.0
        if segments:
            for sa, sb in segments:
                if wlo >= float(sa) - 0.01 and whi <= float(sb) + 0.01:
                    a_lo, a_hi = max(a_lo, float(sa)), min(a_hi, float(sb))
                    break
        asr_win = [(s, e, t) for s, e, t in asr if s < a_hi and e > a_lo]
        if not asr_win:
            abstained.extend(str(out[i].get("text")) for i in lexical_idx)
            continue

        pairs = dp_align(
            [norm_token(out[i].get("text") or "") for i in lexical_idx],
            [t for _, _, t in asr_win],
        )
        for li, wi in enumerate(lexical_idx):
            w = out[wi]
            if li in pairs:
                s, e, _ = asr_win[pairs[li]]
                before = [float(w["start"]), float(w["end"])]
                w["start"] = round(s, 4)
                w["end"] = round(e, 4)
                w["duration_s"] = round(e - s, 4)
                w["source"] = "whisper_anchor_fusion"
                fused += 1
                actions.append(
                    {
                        "text": w.get("text"),
                        "reason": "whisper_anchor",
                        "window": [round(wlo, 2), round(whi, 2)],
                        "before": before,
                        "after": [s, e],
                    }
                )
            else:
                abstained.append(str(w.get("text")))
                if abstain_unmatched:
                    w["_drop"] = True
                # else keep CTC timing (product default)

    kept = [w for w in out if not w.get("_drop")]
    for w in kept:
        w.pop("_drop", None)
    # SHEET ORDER IS GROUND TRUTH — never re-sort by time. Cold eval 2026-07-19
    # (mars) showed sorting swapped "God," and "man": a fused word can land
    # before its predecessor, and sorting then reorders the lyric itself.
    # `kept` is already in sheet order; clamp times monotonic instead.
    n_reorder_clamped = 0
    for a, b in zip(kept, kept[1:]):
        if float(b["start"]) < float(a["end"]) - 0.01:
            if float(b["start"]) < float(a["start"]):
                n_reorder_clamped += 1
                b["source"] = str(b.get("source") or "") + "+order_clamped"
            b["start"] = round(float(a["end"]) + 0.02, 4)
            if float(b["end"]) <= float(b["start"]):
                b["end"] = round(float(b["start"]) + 0.1, 4)
            b["duration_s"] = round(float(b["end"]) - float(b["start"]), 4)

    for i, w in enumerate(kept):
        w["id"] = f"W{i}"

    return {
        "words": kept,
        "actions": actions,
        "engine": {
            "name": "whisper-anchor-fusion-v2",
            "n_window": len(win_idx),
            "n_lexical": n_lexical,
            "n_segments": len(segments or []),
            "n_fused": fused,
            "n_reorder_clamped": n_reorder_clamped,
            "n_abstained": len(abstained),
            "abstained_sample": abstained[:16],
            "fusion_windows": auto_wins,
            "min_star_s": MIN_STAR_S,
            "huge_star_s": HUGE_STAR_S,
            "lookback_huge_s": LOOKBACK_HUGE_S,
            "lookback_med_s": LOOKBACK_MED_S,
            "min_sim": MIN_SIM,
            "abstain_unmatched": abstain_unmatched,
            "doctrine": (
                "DP sheet↔whisper in lookback before large stars; "
                "matched←whisper; unmatched keep CTC (or abstain if strict); "
                "whisper-only tokens ignored"
            ),
        },
    }


def run_whisper_words(
    vocal_path: Path,
    *,
    model_name: str = "medium",
    language: str | None = "en",
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Whisper word timestamps on stem (melisma-reliable)."""
    from .whisper_mode_b import transcribe_mode_b

    r = transcribe_mode_b(
        Path(vocal_path),
        model_name=model_name,
        language=language,
        device=device,
    )
    return list(r.get("words") or [])
