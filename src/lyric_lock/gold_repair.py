from __future__ import annotations

"""
Gold-supervised repair: freeze HIT-like lock, fix EARLY / LATE / HANG / FLASH
using feature signatures learned from human labels (song A opening).

This is the "if it were a hit it wouldn't look like that" judge + fixer.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .edge_physics import (
    _is_function,
    _load_features,
    _peak,
    _slice,
    word_edge_score,
)
from .tempo_map import compute_tempo_map


# Opening sequence gold from user (order matters)
GOLD_OPENING: list[tuple[str, str]] = [
    ("made", "HIT"),
    ("well", "HIT"),
    ("so", "HIT"),
    ("well", "HANG"),
    ("made", "HIT"),
    ("even", "EARLY"),
    ("after", "LATE"),
    ("all", "HIT"),
    ("what's", "HIT"),
    ("more", "LATE"),
    ("we're", "EARLY"),
    ("on", "HIT"),
    ("our", "LATE"),
    ("way", "LATE"),
    ("on", "LATE"),
    ("our", "LATE"),
    ("way", "LATE"),
    ("So", "EARLY"),
    ("whatever", "HANG"),
    ("the", "HIT"),
    ("work", "HIT"),
    ("got", "HIT"),
    ("loud", "HIT"),
    ("Doesn't", "EARLY+FLASH"),
]


def _norm(t: str) -> str:
    t = (t or "").strip().lower()
    for ch in "()[],.!?\"'":
        t = t.replace(ch, "")
    return t


def map_gold_labels(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach gold_label to opening words by sequential text match."""
    out = [dict(w) for w in words]
    gi = 0
    for w in out:
        if gi >= len(GOLD_OPENING):
            w["gold_label"] = None
            continue
        want, lab = GOLD_OPENING[gi]
        wt = _norm(w.get("text") or "")
        wn = _norm(want)
        if wt == wn or wt.startswith(wn[: max(3, len(wn) // 2)]) or wn.startswith(wt[:3]):
            w["gold_label"] = lab
            gi += 1
        else:
            w["gold_label"] = None
    return out


def _hit_stats(words: list[dict[str, Any]]) -> dict[str, float]:
    hits = [w for w in words if w.get("gold_label") == "HIT"]
    if not hits:
        return {
            "dur_med": 0.35,
            "dur_p20": 0.18,
            "dur_p80": 0.55,
            "pre_med": 0.4,
            "post_med": 0.4,
            "mid_med": 0.55,
        }

    def med(key, default=0.3):
        xs = [float(w[key]) for w in hits if w.get(key) is not None]
        return float(np.median(xs)) if xs else default

    durs = [float(w["end"]) - float(w["start"]) for w in hits]
    return {
        "dur_med": float(np.median(durs)),
        "dur_p20": float(np.percentile(durs, 20)),
        "dur_p80": float(np.percentile(durs, 80)),
        "pre_med": med("edge_pre", 0.4),
        "post_med": med("edge_post", 0.4),
        "mid_med": med("edge_mid", 0.55),
    }


def _classify_unlabeled(
    w: dict[str, Any],
    times: np.ndarray,
    rms: np.ndarray,
    stats: dict[str, float],
) -> str:
    """Heuristic miss classes for rest of song (same signatures as gold)."""
    s, e = float(w["start"]), float(w["end"])
    dur = e - s
    sc = word_edge_score(times, rms, s, e)
    pre, mid, post = sc["pre"], sc["mid"], sc["post"]

    # FLASH: very short
    if dur < max(0.14, stats["dur_p20"] * 0.65) and _is_function(w.get("text") or ""):
        return "FLASH"
    if dur < 0.12:
        return "FLASH"

    # EARLY: high pre-energy relative to mid, or long box with weak start
    if pre > mid * 0.95 and pre > stats["pre_med"] * 1.15 and dur > stats["dur_med"] * 1.4:
        return "EARLY"
    if pre > 0.75 and mid < 0.45 and dur > 0.8:
        return "EARLY"

    # HANG: long duration + post not fully dead, or dur >> hit p80
    if dur > stats["dur_p80"] * 1.55 and not _is_function(w.get("text") or ""):
        if post > stats["post_med"] * 0.7 or dur > 0.85:
            return "HANG"

    # LATE: start sits after energy already high for a while — hard to see without context
    # use: mid high but we already started late relative to local peak before start
    tt, rr = _slice(times, rms, s - 0.25, s)
    if len(rr) > 3 and float(np.max(rr)) > mid * 1.05 and float(np.mean(rr[-3:])) > thr_soft(mid):
        if dur < stats["dur_med"] * 1.2:
            return "LATE"

    return "OK"


def thr_soft(mid: float) -> float:
    return max(0.2, mid * 0.55)


def _fix_early(
    times: np.ndarray,
    rms: np.ndarray,
    onset: np.ndarray,
    s: float,
    e: float,
    prev_end: float,
    next_s: float,
    stats: dict[str, float],
) -> tuple[float, float, str]:
    """Push start later to energy/onset; shrink if box ate the whole phrase."""
    peak = _peak(times, rms, s, min(s + 0.5, e))
    thr = max(0.18, peak * 0.42)
    # search forward from s for real onset
    hi = min(e - stats["dur_p20"], s + 0.55, next_s - 0.08)
    tt, rr = _slice(times, rms, s, max(hi, s + 0.05))
    oo = _slice(times, onset, s, max(hi, s + 0.05))[1]
    new_s = s
    for i in range(len(rr)):
        if rr[i] >= thr or (i < len(oo) and oo[i] >= 0.35 and rr[i] >= thr * 0.5):
            new_s = float(tt[i])
            break
    new_s = max(new_s, prev_end + 0.02)
    # target duration near hit median
    target = min(stats["dur_med"] * 1.15, next_s - new_s - 0.03)
    target = max(target, stats["dur_p20"])
    new_e = min(new_s + target, next_s - 0.025, e)
    if new_e <= new_s:
        new_e = new_s + 0.12
    return new_s, new_e, "fix_early"


def _fix_late(
    times: np.ndarray,
    rms: np.ndarray,
    onset: np.ndarray,
    s: float,
    e: float,
    prev_end: float,
    next_s: float,
    stats: dict[str, float],
) -> tuple[float, float, str]:
    """Pull start earlier toward recent onset/energy rise (cap ~180ms)."""
    lo = max(prev_end + 0.02, s - 0.18)
    peak = _peak(times, rms, lo, e)
    thr = max(0.16, peak * 0.40)
    tt, oo = _slice(times, onset, lo, s + 0.02)
    rr = _slice(times, rms, lo, s + 0.02)[1]
    new_s = s
    if len(oo) > 2:
        j = int(np.argmax(oo))
        if oo[j] >= 0.28 and rr[j] >= thr * 0.45:
            cand = float(tt[j])
            if lo <= cand < s:
                new_s = cand
    # also try first rise in window
    if new_s == s:
        for i in range(1, len(rr)):
            if rr[i - 1] < thr <= rr[i]:
                new_s = float(tt[i])
                break
    new_s = max(new_s, prev_end + 0.02)
    # keep similar duration or hit median
    dur = max(e - s, stats["dur_p20"])
    dur = min(dur, stats["dur_p80"] * 1.2)
    new_e = min(new_s + dur, next_s - 0.025)
    if new_e <= new_s:
        new_e = new_s + 0.12
    return new_s, new_e, "fix_late"


def _fix_hang(
    times: np.ndarray,
    rms: np.ndarray,
    s: float,
    e: float,
    next_s: float,
    stats: dict[str, float],
) -> tuple[float, float, str]:
    """Cut end when energy dies; cap duration near hit p80 * 1.3."""
    peak = _peak(times, rms, s, min(s + 0.4, e))
    thr_lo = max(0.12, peak * 0.28)
    tt, rr = _slice(times, rms, s, min(e + 0.05, next_s - 0.02))
    last = s + stats["dur_p20"]
    for i in range(len(rr)):
        if rr[i] >= thr_lo:
            last = float(tt[i])
    cap = s + max(stats["dur_p80"] * 1.35, stats["dur_med"] * 1.5)
    new_e = min(last + 0.04, cap, next_s - 0.025, e)
    new_e = max(new_e, s + stats["dur_p20"])
    return s, new_e, "fix_hang"


def _fix_flash(
    times: np.ndarray,
    rms: np.ndarray,
    onset: np.ndarray,
    s: float,
    e: float,
    prev_end: float,
    next_s: float,
    stats: dict[str, float],
) -> tuple[float, float, str]:
    """
    Doesn't-class: often early AND too short.
    Delay to energy if needed, then enforce min duration from HIT distribution
    without invading next word.
    """
    # first fix early
    s2, e2, _ = _fix_early(times, rms, onset, s, e, prev_end, next_s, stats)
    min_dur = max(0.18, stats["dur_p20"] * 1.05, stats["dur_med"] * 0.55)
    max_dur = min(stats["dur_med"] * 1.1, next_s - s2 - 0.03)
    min_dur = min(min_dur, max(0.14, max_dur))
    e3 = max(e2, s2 + min_dur)
    e3 = min(e3, next_s - 0.02)
    if e3 <= s2:
        e3 = s2 + 0.14
    return s2, e3, "fix_flash"


def gold_repair(
    words: list[dict[str, Any]],
    vocal_path: Path,
    *,
    mix_path: Path | None = None,
) -> dict[str, Any]:
    times, rms, onset, duration_s = _load_features(vocal_path)
    tmap = compute_tempo_map(Path(mix_path) if mix_path else vocal_path)

    # ensure edge features for scoring
    labeled = map_gold_labels(words)
    for w in labeled:
        sc = word_edge_score(times, rms, float(w["start"]), float(w["end"]))
        w["edge_pre"] = sc["pre"]
        w["edge_mid"] = sc["mid"]
        w["edge_post"] = sc["post"]
        w["edge_score"] = sc["score"]

    stats = _hit_stats(labeled)
    actions: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []

    for i, w in enumerate(labeled):
        s0, e0 = float(w["start"]), float(w["end"])
        prev_end = float(out[i - 1]["end"]) if i else 0.0
        next_s = (
            float(labeled[i + 1]["start"])
            if i + 1 < len(labeled)
            else duration_s - 0.01
        )
        # use FA if present as soft anchor
        fa_s = float(w.get("fa_start", s0))
        fa_e = float(w.get("fa_end", e0))

        lab = w.get("gold_label")
        if lab is None:
            lab = _classify_unlabeled(w, times, rms, stats)
            w["inferred_label"] = lab
        else:
            w["inferred_label"] = lab

        reason = "freeze_hit"
        s1, e1 = s0, e0

        if lab == "HIT":
            # freeze — only tiny clamp for overlaps
            s1, e1 = s0, e0
            reason = "freeze_hit"
        elif lab == "EARLY":
            s1, e1, reason = _fix_early(
                times, rms, onset, s0, e0, prev_end, next_s, stats
            )
        elif lab == "LATE":
            s1, e1, reason = _fix_late(
                times, rms, onset, s0, e0, prev_end, next_s, stats
            )
        elif lab == "HANG":
            s1, e1, reason = _fix_hang(times, rms, s0, e0, next_s, stats)
        elif lab in ("EARLY+FLASH", "FLASH"):
            s1, e1, reason = _fix_flash(
                times, rms, onset, s0, e0, prev_end, next_s, stats
            )
        else:
            # OK — light hang/early safety only
            sc = word_edge_score(times, rms, s0, e0)
            if e0 - s0 > stats["dur_p80"] * 1.6 and sc["post"] > stats["post_med"]:
                s1, e1, reason = _fix_hang(times, rms, s0, e0, next_s, stats)
            elif sc["pre"] > sc["mid"] * 1.1 and e0 - s0 > stats["dur_med"] * 1.5:
                s1, e1, reason = _fix_early(
                    times, rms, onset, s0, e0, prev_end, next_s, stats
                )

        s1 = max(s1, prev_end + 0.015)
        e1 = min(e1, next_s - 0.02)
        if e1 <= s1:
            e1 = s1 + max(0.10, stats["dur_p20"] * 0.8)

        spb = tmap.seconds_per_beat_at(s1)
        nw = dict(w)
        nw["start"] = round(s1, 4)
        nw["end"] = round(e1, 4)
        nw["duration_s"] = round(e1 - s1, 4)
        nw["duration_beats"] = round((e1 - s1) / spb, 3) if spb else None
        nw["local_bpm"] = round(tmap.local_bpm_at(s1), 2)
        nw["fa_start"] = fa_s
        nw["fa_end"] = fa_e
        nw["repair_reason"] = reason
        nw["source"] = f"gold_repair+{reason}"
        sc2 = word_edge_score(times, rms, s1, e1)
        nw["edge_pre"] = sc2["pre"]
        nw["edge_mid"] = sc2["mid"]
        nw["edge_post"] = sc2["post"]
        nw["edge_score"] = sc2["score"]
        out.append(nw)
        if reason != "freeze_hit":
            actions.append(
                {
                    "text": w.get("text"),
                    "label": lab,
                    "reason": reason,
                    "before": [s0, e0],
                    "after": [s1, e1],
                }
            )

    # second pass: "[repeated phrase]" chain — if previous is HIT "on" and next are LATE-ish our/way
    for i in range(1, len(out) - 1):
        a, b, c = out[i - 1], out[i], out[i + 1]
        if _norm(a["text"]) == "on" and _norm(b["text"]) == "our" and _norm(c["text"]) == "way":
            # pack tightly after on with hit-like durs
            t = a["end"] + 0.03
            for w, scale in ((b, 0.9), (c, 1.1)):
                d = stats["dur_med"] * scale
                w["start"] = round(t, 4)
                w["end"] = round(min(t + d, (out[out.index(w) + 1]["start"] - 0.02) if out.index(w) + 1 < len(out) else t + d), 4)
                if w["end"] <= w["start"]:
                    w["end"] = round(w["start"] + 0.12, 4)
                w["duration_s"] = round(w["end"] - w["start"], 4)
                w["repair_reason"] = (w.get("repair_reason") or "") + "+on_our_way_pack"
                w["source"] = "gold_repair+on_our_way_pack"
                t = w["end"] + 0.03

    # rebuild lines
    by: dict[str, list] = {}
    for w in out:
        by.setdefault(str(w.get("line_id") or "L0"), []).append(w)
    lines = []
    for lid, wlist in by.items():
        wlist = sorted(wlist, key=lambda x: x["start"])
        lines.append(
            {
                "id": lid,
                "start": wlist[0]["start"],
                "end": wlist[-1]["end"],
                "text": " ".join(x["text"] for x in wlist),
                "confidence": None,
            }
        )
    lines.sort(key=lambda x: x["start"])

    n_gold = sum(1 for w in out if w.get("gold_label"))
    return {
        "words": out,
        "lines": lines,
        "text": " ".join(w["text"] for w in out),
        "tempo_map": tmap.to_dict(),
        "hit_stats": stats,
        "repair_actions": actions,
        "n_gold_labeled": n_gold,
        "user_concept": {
            "name": "gold_hit_miss_judge_and_repair",
            "applied": True,
            "version": "v1_opening_gold_plus_inferred",
            "architecture": (
                "HIT words teach duration/energy cloud. "
                "EARLY/LATE/HANG/FLASH signatures drive explicit fixes. "
                "FA spine times are starting point; gold judge repairs misses."
            ),
            "modules": ["lyric_lock.gold_repair.gold_repair"],
            "gold_labels_source": "user_ear_opening_phrase_made_well_90s",
        },
        "engine": {
            "name": "gold-repair-v1",
            "n_actions": len(actions),
            "hit_stats": stats,
        },
    }
