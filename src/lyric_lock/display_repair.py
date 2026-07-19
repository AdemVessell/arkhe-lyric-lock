from __future__ import annotations

"""
Display repair — product-stage last mile on a timed word spine.

Checkpoint origin (song A 32–40s, user ear GOOD 2026-07-18):
  1) Abstain orphan blips (short box + long gap after) — no FLASH lies
  2) Rebind orphans to first rise after a real quiet gap
  3) Gap-fill unplaced short tokens between neighbors (lyric completeness)
  4) Local pack of consecutive words (cut hang / pull late FA starts)

Doctrine:
  - FA/ASR spine keeps identity + order
  - Energy on vocal stem decides draw / re-home / pack
  - Prefer blank or re-home over sharp wrong glyphs
  - Do NOT long-range rebind non-orphan tokens (cascade disasters)

Does NOT require human gold at runtime (gold is the ruler offline).
"""

from pathlib import Path
from typing import Any

import numpy as np

from .edge_physics import _is_function, _load_features, _peak, _slice, word_edge_score


def _mean_e(times: np.ndarray, rms: np.ndarray, t0: float, t1: float) -> float:
    tt, rr = _slice(times, rms, t0, t1)
    return float(np.mean(rr)) if len(rr) else 0.0


def _first_rise(
    times: np.ndarray,
    rms: np.ndarray,
    lo: float,
    hi: float,
    thr: float | None = None,
) -> float:
    tt, rr = _slice(times, rms, lo, hi)
    if len(rr) < 3:
        return lo
    if thr is None:
        thr = max(0.14, float(np.percentile(rr, 55)) * 0.6)
    if float(rr[0]) >= thr:
        return float(tt[0])
    for i in range(1, len(rr)):
        if rr[i - 1] < thr <= rr[i]:
            return float(tt[i])
    return float(tt[int(np.argmax(rr))])


def _last_alive(
    times: np.ndarray,
    rms: np.ndarray,
    s: float,
    hi: float,
    thr: float = 0.16,
    min_dur: float = 0.12,
) -> float:
    tt, rr = _slice(times, rms, s, hi)
    if len(rr) == 0:
        return s + min_dur
    last = s
    dead = 0
    for i in range(len(rr)):
        if rr[i] >= thr:
            last = float(tt[i])
            dead = 0
        else:
            dead += 1
            if dead >= 3 and float(tt[i]) - s > min_dur:
                break
    return max(last + 0.03, s + min_dur)


def _find_post_quiet_onset(
    times: np.ndarray,
    rms: np.ndarray,
    lo: float,
    hi: float,
    *,
    min_quiet: float = 0.20,
    thr_q: float = 0.12,
    thr_on: float = 0.18,
) -> dict[str, float] | None:
    """Longest quiet run then first energy rise after it."""
    tt, rr = _slice(times, rms, lo, hi)
    if len(rr) < 5:
        return None
    best: tuple[float, float, float] | None = None  # score, qdur, onset
    i = 0
    while i < len(rr):
        if rr[i] >= thr_q:
            i += 1
            continue
        j = i
        while j < len(rr) and rr[j] < thr_q:
            j += 1
        qdur = float(tt[min(j, len(tt) - 1)]) - float(tt[i])
        k = j
        onset_t = None
        while k < len(rr):
            if rr[k] >= thr_on:
                onset_t = float(tt[k])
                break
            k += 1
        if onset_t is not None and qdur >= min_quiet * 0.5:
            score = qdur * 2.0 + (0.3 if k < len(rr) and rr[k] > 0.35 else 0.0)
            if best is None or score > best[0]:
                best = (score, qdur, onset_t)
        i = max(j, i + 1)
    if best is None:
        return None
    return {"onset": best[2], "qdur": best[1]}


def _abstain_reasons(
    times: np.ndarray,
    rms: np.ndarray,
    s: float,
    e: float,
    prev_end: float,
    next_s: float,
) -> list[str]:
    dur = e - s
    mid = _mean_e(times, rms, s, e)
    pre = _mean_e(times, rms, max(0.0, s - 0.2), s)
    post = _mean_e(times, rms, e, min(e + 0.25, next_s))
    gap_after = next_s - e
    reasons: list[str] = []
    # Orphan flash: short box then long hole (Doesn't crime)
    if dur < 0.20 and gap_after > 0.70:
        reasons.append("orphan_blip_before_gap")
    if mid < 0.10 and dur < 0.25:
        reasons.append("near_silent_box")
    if dur < 0.22 and pre > 0.35 and gap_after > 0.5 and post < mid * 0.5:
        reasons.append("tail_of_previous_then_gap")
    return reasons


def _gap_fill_one(
    times: np.ndarray,
    rms: np.ndarray,
    text: str,
    prev_end: float,
    next_start: float,
    duration_s: float,
) -> tuple[float, float] | None:
    gap = next_start - prev_end
    if gap < 0.10:
        return None
    lo, hi = prev_end + 0.02, next_start - 0.02
    target = 0.16 if _is_function(text) or len(text) <= 4 else 0.22
    target = min(target, gap - 0.04)
    if target < 0.10:
        return None
    tt, rr = _slice(times, rms, lo, hi)
    s = lo
    if len(rr) > 3 and float(np.max(rr)) > 0.12:
        thr = max(0.14, float(np.max(rr)) * 0.35)
        placed = False
        for k in range(1, len(rr)):
            if rr[k - 1] < thr <= rr[k]:
                s = float(tt[k])
                placed = True
                break
        if not placed:
            s = float(tt[int(np.argmax(rr))])
        if s + target > hi:
            s = max(lo, hi - target)
    else:
        # still emit for lyric completeness (display contract)
        s = lo + max(0.0, (gap - 0.04 - target) * 0.35)
    e = min(s + target, hi)
    if e <= s:
        e = s + 0.12
    return s, e


def _pack_local_chain(
    times: np.ndarray,
    rms: np.ndarray,
    words: list[dict[str, Any]],
    duration_s: float,
    *,
    max_span_s: float = 8.0,
) -> list[dict[str, Any]]:
    """
    Light local pack: cut hangs and pull late starts for consecutive words
    when gap structure looks like a phrase (small gaps).
    Skips words marked freeze_display.
    """
    out = [dict(w) for w in words]
    n = len(out)
    i = 0
    while i < n:
        if out[i].get("freeze_display") or out[i].get("unplaced"):
            i += 1
            continue
        # grow a short run of close words
        j = i
        while j + 1 < n and not out[j + 1].get("freeze_display") and not out[j + 1].get("unplaced"):
            gap = float(out[j + 1]["start"]) - float(out[j]["end"])
            if gap > 0.55:
                break
            if float(out[j + 1]["end"]) - float(out[i]["start"]) > max_span_s:
                break
            j += 1
        if j == i:
            i += 1
            continue
        # pack i..j
        for k in range(i, j + 1):
            if out[k].get("freeze_display"):
                continue
            prev_end = float(out[k - 1]["end"]) if k > i else max(0.0, float(out[k]["start"]) - 0.5)
            next_lim = (
                float(out[k + 1]["start"])
                if k < j
                else (
                    float(out[k + 1]["start"])
                    if k + 1 < n and not out[k + 1].get("unplaced")
                    else duration_s - 0.01
                )
            )
            # freeze next word as hard wall if frozen
            if k + 1 < n and out[k + 1].get("freeze_display"):
                next_lim = min(next_lim, float(out[k + 1]["start"]) - 0.08)

            s0 = float(out[k]["start"])
            e0 = float(out[k]["end"])
            text = out[k].get("text") or ""
            lo = max(prev_end + 0.015, s0 - 0.25)
            hi = min(next_lim - 0.02, e0 + 0.4)
            if hi <= lo + 0.1:
                continue
            thr = 0.18 if not _is_function(text) else 0.14
            s1 = _first_rise(times, rms, lo, min(lo + 0.45, hi), thr=thr)
            s1 = max(s1, prev_end + 0.015)
            max_dur = 0.22 if _is_function(text) or len(text) <= 3 else 0.75
            if text.lower() in ("whatever", "doesn't") or text.lower().startswith("doesn"):
                max_dur = 0.70
            e1 = _last_alive(times, rms, s1, min(s1 + max_dur + 0.15, hi), thr=thr, min_dur=0.12)
            e1 = min(e1, s1 + max_dur, hi)
            if e1 <= s1:
                e1 = s1 + 0.12
            out[k]["start"] = round(s1, 4)
            out[k]["end"] = round(e1, 4)
            out[k]["duration_s"] = round(e1 - s1, 4)
            out[k]["repair_reason"] = (out[k].get("repair_reason") or "") + "+pack"
            out[k]["source"] = (out[k].get("source") or "spine") + "+pack"
        # monotonic within run
        for k in range(i + 1, j + 1):
            if out[k].get("freeze_display") or out[k - 1].get("freeze_display"):
                continue
            if float(out[k]["start"]) < float(out[k - 1]["end"]) + 0.012:
                out[k]["start"] = round(float(out[k - 1]["end"]) + 0.015, 4)
                if float(out[k]["end"]) <= float(out[k]["start"]):
                    out[k]["end"] = round(float(out[k]["start"]) + 0.12, 4)
                out[k]["duration_s"] = round(
                    float(out[k]["end"]) - float(out[k]["start"]), 4
                )
        i = j + 1
    return out


def apply_display_repair(
    words: list[dict[str, Any]],
    vocal_path: Path,
    *,
    pack: bool = True,
) -> dict[str, Any]:
    """
    Apply checkpoint display repair to a word list.

    Returns dict with words, lines, text, actions, engine meta.
    """
    vocal_path = Path(vocal_path).expanduser().resolve()
    times, rms, onset, duration_s = _load_features(vocal_path)
    n = len(words)
    if n == 0:
        return {
            "words": [],
            "lines": [],
            "text": "",
            "repair_actions": [],
            "engine": {"name": "display-repair-v1", "n_actions": 0},
        }

    # --- pass 1: classify + rebind orphans / mark unplaced ---
    provisional: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    for i, w in enumerate(words):
        s0, e0 = float(w["start"]), float(w["end"])
        text = w.get("text") or ""
        prev_end = float(provisional[-1]["end"]) if provisional else 0.0
        next_s = float(words[i + 1]["start"]) if i + 1 < n else duration_s - 0.01
        reasons = _abstain_reasons(times, rms, s0, e0, prev_end, next_s)
        nw = dict(w)
        nw["fa_start"] = s0
        nw["fa_end"] = e0

        if not reasons:
            s1 = max(s0, prev_end + 0.015)
            e1 = max(e0, s1 + 0.10)
            nw["start"] = round(s1, 4)
            nw["end"] = round(e1, 4)
            nw["duration_s"] = round(e1 - s1, 4)
            nw["unplaced"] = False
            nw["source"] = w.get("source") or "spine"
            nw["repair_reason"] = w.get("repair_reason") or "keep"
            provisional.append(nw)
            continue

        if "orphan_blip_before_gap" in reasons:
            hit = _find_post_quiet_onset(
                times, rms, e0 + 0.05, min(duration_s - 0.01, e0 + 2.5), min_quiet=0.25
            )
            if hit is None:
                hit = _find_post_quiet_onset(
                    times,
                    rms,
                    e0 + 0.05,
                    min(duration_s - 0.01, e0 + 2.5),
                    min_quiet=0.12,
                    thr_q=0.14,
                    thr_on=0.15,
                )
            if hit is None:
                nw.update(
                    start=round(s0, 4),
                    end=round(e0, 4),
                    duration_s=round(e0 - s0, 4),
                    unplaced=True,
                    freeze_display=False,
                    source="unplaced",
                    repair_reason="+".join(reasons) + "+no_post_quiet_onset",
                )
                actions.append(
                    {"text": text, "action": "unplaced", "reasons": reasons, "from": [s0, e0]}
                )
                provisional.append(nw)
                continue
            s1 = max(hit["onset"], prev_end + 0.02)
            max_d = 0.34 if not _is_function(text) else 0.28
            if text.lower().startswith("doesn"):
                max_d = 0.32
            e1 = min(
                _last_alive(times, rms, s1, s1 + max_d + 0.15, thr=0.14, min_dur=0.14),
                s1 + max_d,
            )
            nw.update(
                start=round(s1, 4),
                end=round(e1, 4),
                duration_s=round(e1 - s1, 4),
                unplaced=False,
                freeze_display=True,  # protect rebind win from pack clobber
                source="display_repair+orphan_rebind",
                repair_reason="+".join(reasons) + f"+post_quiet_q{hit['qdur']:.2f}",
            )
            actions.append(
                {
                    "text": text,
                    "action": "orphan_rebind",
                    "reasons": reasons,
                    "from": [s0, e0],
                    "to": [s1, e1],
                    "qdur": hit["qdur"],
                }
            )
            provisional.append(nw)
            continue

        # non-orphan abstain → try local only later via gap_fill; mark unplaced for now
        nw.update(
            start=round(s0, 4),
            end=round(e0, 4),
            duration_s=round(e0 - s0, 4),
            unplaced=True,
            source="unplaced",
            repair_reason="+".join(reasons),
        )
        actions.append(
            {"text": text, "action": "unplaced_local", "reasons": reasons, "from": [s0, e0]}
        )
        provisional.append(nw)

    # --- pass 2: gap-fill unplaced between neighbors ---
    for i, w in enumerate(provisional):
        if not w.get("unplaced"):
            continue
        prev_end = 0.0
        for j in range(i - 1, -1, -1):
            if not provisional[j].get("unplaced"):
                prev_end = float(provisional[j]["end"])
                break
        next_start = duration_s - 0.01
        for j in range(i + 1, n):
            if not provisional[j].get("unplaced"):
                next_start = float(provisional[j]["start"])
                break
        filled = _gap_fill_one(
            times, rms, w.get("text") or "", prev_end, next_start, duration_s
        )
        if filled is None:
            continue
        s1, e1 = filled
        w["start"] = round(s1, 4)
        w["end"] = round(e1, 4)
        w["duration_s"] = round(e1 - s1, 4)
        w["unplaced"] = False
        w["source"] = "display_repair+gap_fill"
        w["repair_reason"] = (w.get("repair_reason") or "") + "+gap_fill"
        actions.append(
            {
                "text": w.get("text"),
                "action": "gap_fill",
                "to": [s1, e1],
                "neighbors": [prev_end, next_start],
            }
        )

    # push overlaps after gap fill
    for i in range(1, len(provisional)):
        if provisional[i].get("unplaced") or provisional[i - 1].get("unplaced"):
            continue
        if float(provisional[i]["start"]) < float(provisional[i - 1]["end"]) + 0.012:
            provisional[i]["start"] = round(float(provisional[i - 1]["end"]) + 0.015, 4)
            if float(provisional[i]["end"]) <= float(provisional[i]["start"]):
                provisional[i]["end"] = round(float(provisional[i]["start"]) + 0.12, 4)
            provisional[i]["duration_s"] = round(
                float(provisional[i]["end"]) - float(provisional[i]["start"]), 4
            )

    # --- pass 3: local pack (respect freeze_display) ---
    if pack:
        provisional = _pack_local_chain(times, rms, provisional, duration_s)

    # edge scores
    for w in provisional:
        if w.get("unplaced"):
            continue
        sc = word_edge_score(times, rms, float(w["start"]), float(w["end"]))
        w["edge_pre"] = sc["pre"]
        w["edge_mid"] = sc["mid"]
        w["edge_post"] = sc["post"]
        w["edge_score"] = sc["score"]

    # rebuild lines from placed words
    by: dict[str, list[dict[str, Any]]] = {}
    for w in provisional:
        if w.get("unplaced"):
            continue
        by.setdefault(str(w.get("line_id") or "L0"), []).append(w)
    lines: list[dict[str, Any]] = []
    for lid, wlist in by.items():
        wlist = sorted(wlist, key=lambda x: float(x["start"]))
        if not wlist:
            continue
        lines.append(
            {
                "id": lid,
                "start": float(wlist[0]["start"]),
                "end": float(wlist[-1]["end"]),
                "text": " ".join(x["text"] for x in wlist),
                "confidence": None,
            }
        )
    lines.sort(key=lambda x: float(x["start"]))

    placed = [w for w in provisional if not w.get("unplaced")]
    return {
        "words": provisional,
        "lines": lines,
        "text": " ".join(w["text"] for w in placed),
        "repair_actions": actions,
        "engine": {
            "name": "display-repair-v1",
            "version": "checkpoint_phrase_32_40_2026_07_18",
            "n_actions": len(actions),
            "n_unplaced": sum(1 for w in provisional if w.get("unplaced")),
            "n_orphan_rebind": sum(
                1 for a in actions if a.get("action") == "orphan_rebind"
            ),
            "n_gap_fill": sum(1 for a in actions if a.get("action") == "gap_fill"),
            "pack": pack,
        },
        "user_concept": {
            "name": "display_repair_refuse_rebind_gapfill_pack",
            "applied": True,
            "architecture": (
                "FA/ASR spine identity; vocal energy for abstain, orphan rebind "
                "after quiet, gap-fill completeness, local pack. Gold is offline ruler."
            ),
        },
    }


def apply_display_repair_to_timed(
    timed: dict[str, Any],
    vocal_path: Path,
    *,
    pack: bool = True,
) -> dict[str, Any]:
    """Mutate a timed dict in place-ish: returns new timed with repaired words/lines."""
    result = apply_display_repair(list(timed.get("words") or []), vocal_path, pack=pack)
    out = dict(timed)
    out["words"] = result["words"]
    out["lines"] = result["lines"]
    out["text"] = result["text"]
    out["display_repair"] = {
        "actions": result["repair_actions"],
        "engine": result["engine"],
        "user_concept": result["user_concept"],
    }
    gen = dict(out.get("generator") or {})
    gen["display_repair"] = result["engine"]["name"]
    gen["phase"] = (gen.get("phase") or "timed") + "+display_repair_v1"
    out["generator"] = gen
    return out
