from __future__ import annotations

import re
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']+")


def load_lyrics_words(text: str) -> list[str]:
    """Tokenize provided lyrics into display words (keep punctuation attached lightly)."""
    words: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # skip comments / section tags / provenance headers
        if line.startswith("#") or line.startswith("["):
            continue
        for tok in line.split():
            tok = tok.strip()
            if tok:
                words.append(tok)
    return words


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def align_lyrics_to_asr_words(
    lyrics_words: list[str],
    asr_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Mode A: known lyric tokens get times from ASR word stream via sequential fuzzy match.

    - When next lyric word matches next ASR token (normalized), take ASR start/end
    - When ASR inserts garbage, skip ASR tokens
    - When lyric word has no ASR match within a window, interpolate between anchors
    """
    if not lyrics_words:
        return []
    if not asr_words:
        # no anchors — cannot invent times
        return [
            {
                "id": f"W{i}",
                "start": 0.0,
                "end": 0.12,
                "text": w,
                "line_id": "L0",
                "confidence": 0.0,
                "source": "lyrics_only_no_asr",
            }
            for i, w in enumerate(lyrics_words)
        ]

    n_l = len(lyrics_words)
    n_a = len(asr_words)
    # DP: match lyrics i to asr j
    # score 1 if norms equal or prefix, else skip cost
    INF = 10**9
    # path reconstruction via choices
    # Use simple sequential pointer with limited ASR skip
    out: list[dict[str, Any] | None] = [None] * n_l
    j = 0
    for i, lw in enumerate(lyrics_words):
        ln = _norm(lw)
        if not ln:
            continue
        best_j = None
        best_score = -1
        # search window ahead in ASR
        for k in range(j, min(n_a, j + 12)):
            an = _norm(str(asr_words[k].get("text") or ""))
            if not an:
                continue
            score = 0
            if an == ln:
                score = 3
            elif an.startswith(ln) or ln.startswith(an):
                score = 2
            elif len(ln) > 2 and (ln in an or an in ln):
                score = 1
            if score > best_score:
                best_score = score
                best_j = k
            if score == 3:
                break
        if best_j is not None and best_score >= 1:
            aw = asr_words[best_j]
            out[i] = {
                "id": f"W{i}",
                "start": float(aw["start"]),
                "end": float(aw["end"]),
                "text": lw,  # provided lyrics text wins
                "line_id": aw.get("line_id") or "L0",
                "confidence": aw.get("confidence"),
                "source": "lyrics_matched_asr",
            }
            j = best_j + 1
        # else leave None for interpolate

    # interpolate missing
    for i in range(n_l):
        if out[i] is not None:
            continue
        # find prev/next anchors
        prev = next_i = None
        for p in range(i - 1, -1, -1):
            if out[p] is not None:
                prev = p
                break
        for n in range(i + 1, n_l):
            if out[n] is not None:
                next_i = n
                break
        if prev is not None and next_i is not None:
            t0 = float(out[prev]["end"])
            t1 = float(out[next_i]["start"])
            span = max(0.05, t1 - t0)
            # position among gap
            gap_words = next_i - prev
            slot = i - prev
            start = t0 + span * (slot / gap_words)
            end = t0 + span * ((slot + 1) / gap_words)
        elif prev is not None:
            start = float(out[prev]["end"])
            end = start + 0.25
        elif next_i is not None:
            end = float(out[next_i]["start"])
            start = max(0.0, end - 0.25)
        else:
            start, end = 0.0, 0.25
        if end <= start:
            end = start + 0.12
        out[i] = {
            "id": f"W{i}",
            "start": start,
            "end": end,
            "text": lyrics_words[i],
            "line_id": "L0",
            "confidence": None,
            "source": "lyrics_interpolated",
        }

    # build line groups ~ every 8 words or by punctuation
    lines: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    li = 0

    def flush():
        nonlocal li, cur
        if not cur:
            return
        lid = f"L{li}"
        for w in cur:
            w["line_id"] = lid
        lines.append(
            {
                "id": lid,
                "start": float(cur[0]["start"]),
                "end": float(cur[-1]["end"]),
                "text": " ".join(w["text"] for w in cur),
                "confidence": None,
            }
        )
        li += 1
        cur = []

    for w in out:  # type: ignore
        assert w is not None
        cur.append(w)
        t = w["text"]
        if len(cur) >= 8 or t.endswith((".", "?", "!", ",", ";")):
            flush()
    flush()

    return [w for w in out if w is not None]  # type: ignore
