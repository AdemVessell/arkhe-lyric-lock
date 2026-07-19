from __future__ import annotations
"""Hold-island reassignment: FA spine identity, energy reassigns boundaries."""

from typing import Any
from pathlib import Path
import numpy as np
from .edge_physics import _load_features, _peak, _slice, word_edge_score, _is_function


def _mean_e(times, rms, t0, t1):
    tt, rr = _slice(times, rms, t0, t1)
    return float(np.mean(rr)) if len(rr) else 0.0


def _first_rise(times, rms, t0, t1, thr, min_run=2):
    tt, rr = _slice(times, rms, t0, t1)
    if len(rr) < 3:
        return None
    for i in range(1, len(rr)):
        if rr[i] >= thr and rr[i - 1] < thr:
            run = 1
            j = i
            while j + 1 < len(rr) and rr[j + 1] >= thr * 0.85:
                run += 1
                j += 1
            if run >= min_run:
                return float(tt[i])
    if rr[0] >= thr:
        return float(tt[0])
    return None


def _last_alive(times, rms, t0, t1, thr, max_dead_run=6):
    tt, rr = _slice(times, rms, t0, t1)
    if len(rr) == 0:
        return t0
    last = t0
    dead = 0
    for i in range(len(rr)):
        if rr[i] >= thr:
            last = float(tt[i])
            dead = 0
        else:
            dead += 1
            if dead > max_dead_run and float(tt[i]) - t0 > 0.15:
                break
    return last


def hold_island_reassign(
    words: list[dict[str, Any]],
    vocal_path: Path,
) -> dict[str, Any]:
    times, rms, onset, duration_s = _load_features(vocal_path)
    n = len(words)
    fa = [(float(w["start"]), float(w["end"])) for w in words]
    starts = [fa[i][0] for i in range(n)]
    ends = [fa[i][1] for i in range(n)]
    reasons = ["fa"] * n

    for i in range(n):
        s, e = fa[i]
        prev_e = ends[i - 1] if i else 0.0
        text = words[i].get("text") or ""
        local_peak = _peak(times, rms, max(0, s - 0.3), min(duration_s, e + 0.3))
        thr = max(0.14, local_peak * 0.32)
        thr_hi = max(0.18, local_peak * 0.42)
        reason_parts: list[str] = []

        gap_lo = prev_e + 0.02 if i else max(0.0, s - 0.5)
        gap = s - gap_lo
        if gap > 0.28:
            gap_mean = _mean_e(times, rms, gap_lo, s)
            gap_max = _peak(times, rms, gap_lo, s)
            if gap_max >= thr_hi and gap_mean >= thr * 0.55:
                rise = _first_rise(times, rms, gap_lo, s + 0.05, thr, min_run=2)
                if rise is not None and rise < s - 0.05:
                    new_s = max(rise, gap_lo)
                    if new_s < s - 0.04:
                        starts[i] = new_s
                        reason_parts.append(f"gap_pull:{s - new_s:.2f}")
                        s = new_s

        s = starts[i]
        e = ends[i]
        dur = e - s
        if dur > 0.70:
            t_a, t_b, t_c = s, s + dur / 3, s + 2 * dur / 3
            e1 = _mean_e(times, rms, t_a, t_b)
            e3 = _mean_e(times, rms, t_c, e)
            if e1 < thr * 0.9 and e3 >= thr_hi * 0.85 and e3 > e1 * 1.25:
                rise = _first_rise(times, rms, t_b, e, thr, min_run=2)
                if rise and rise > s + 0.12:
                    starts[i] = rise
                    reason_parts.append(f"blob_delay:{rise - s:.2f}")
                    s = rise
            tt, rr = _slice(times, rms, s, e)
            if len(rr) > 10:
                thr_q = thr * 0.55
                best_rise = None
                in_quiet = False
                quiet_start = None
                for j in range(1, len(rr)):
                    if rr[j] < thr_q:
                        if not in_quiet:
                            in_quiet = True
                            quiet_start = float(tt[j])
                    else:
                        if in_quiet and quiet_start is not None:
                            if float(tt[j]) - quiet_start >= 0.12 and float(tt[j]) > s + 0.15:
                                best_rise = float(tt[j])
                        in_quiet = False
                if best_rise and best_rise > s + 0.2:
                    early_mean = _mean_e(times, rms, s, best_rise)
                    late_mean = _mean_e(times, rms, best_rise, e)
                    if late_mean >= early_mean * 0.85 and (dur > 0.9 or _is_function(text)):
                        starts[i] = best_rise
                        reason_parts.append(f"valley_delay:{best_rise - s:.2f}")
                        s = best_rise

        s = starts[i]
        e = fa[i][1]
        next_lim = fa[i + 1][0] - 0.02 if i + 1 < n else duration_s - 0.01
        alive = _last_alive(times, rms, s, min(e + 0.15, next_lim), thr, max_dead_run=5)
        if e - alive > 0.18:
            ends[i] = min(alive + 0.05, next_lim)
            reason_parts.append(f"blob_cut:{e - ends[i]:.2f}")
        else:
            ends[i] = min(e, next_lim)

        s = starts[i]
        e = ends[i]
        next_s_fa = fa[i + 1][0] if i + 1 < n else duration_s - 0.01
        post_gap = next_s_fa - e
        if post_gap > 0.45 or (e - s) < 0.35:
            hi = min(next_s_fa - 0.05, s + 4.0)
            if hi > e + 0.1:
                if _mean_e(times, rms, e, min(e + 0.25, hi)) >= thr * 0.7 or _peak(times, rms, e, min(e + 0.4, hi)) >= thr_hi:
                    alive2 = _last_alive(times, rms, s, hi, thr * 0.85, max_dead_run=8)
                    cap = 3.8 if not _is_function(text) else 0.55
                    new_e = min(alive2 + 0.06, s + cap, next_s_fa - 0.04)
                    if new_e > e + 0.12:
                        ends[i] = new_e
                        reason_parts.append(f"hold_ext:{new_e - e:.2f}")

        if ends[i] - starts[i] < 0.10:
            ends[i] = starts[i] + 0.10
        if reason_parts:
            reasons[i] = "+".join(reason_parts)

    for i in range(n):
        prev_e = ends[i - 1] if i else 0.0
        if starts[i] < prev_e + 0.015:
            starts[i] = prev_e + 0.015
        if ends[i] <= starts[i] + 0.08:
            ends[i] = starts[i] + 0.10
        if i + 1 < n and ends[i] > starts[i + 1] - 0.02:
            if ends[i] > fa[i + 1][0]:
                starts[i + 1] = max(starts[i + 1], ends[i] + 0.02)
            else:
                ends[i] = max(starts[i] + 0.1, starts[i + 1] - 0.02)

    out = []
    actions = []
    for i, w in enumerate(words):
        nw = dict(w)
        nw["fa_start"] = fa[i][0]
        nw["fa_end"] = fa[i][1]
        nw["start"] = round(starts[i], 4)
        nw["end"] = round(ends[i], 4)
        nw["duration_s"] = round(ends[i] - starts[i], 4)
        nw["source"] = "hold_island_reassign_v1"
        nw["repair_reason"] = reasons[i]
        sc = word_edge_score(times, rms, starts[i], ends[i])
        nw["edge_pre"] = sc["pre"]
        nw["edge_mid"] = sc["mid"]
        nw["edge_post"] = sc["post"]
        nw["edge_score"] = sc["score"]
        out.append(nw)
        if reasons[i] != "fa":
            actions.append({"text": w.get("text"), "reason": reasons[i], "before": [fa[i][0], fa[i][1]], "after": [starts[i], ends[i]]})

    by: dict[str, list] = {}
    for w in out:
        by.setdefault(str(w.get("line_id") or "L0"), []).append(w)
    lines = []
    for lid, wlist in by.items():
        wlist = sorted(wlist, key=lambda x: x["start"])
        lines.append({"id": lid, "start": wlist[0]["start"], "end": wlist[-1]["end"], "text": " ".join(x["text"] for x in wlist)})
    lines.sort(key=lambda x: x["start"])
    return {
        "words": out,
        "lines": lines,
        "text": " ".join(w["text"] for w in out),
        "repair_actions": actions,
        "engine": {"name": "hold-island-reassign-v1", "n_actions": len(actions)},
        "user_concept": {
            "name": "hold_island_reassignment",
            "applied": True,
            "architecture": "FA spine identity; energy reassigns gap/blob/hold boundaries",
        },
    }
