from __future__ import annotations

"""
End physics on a GOOD onset spine (CTC + lead).

NOT the demoted rebind/edge rescue on a bad FA spine.
Here starts stay frozen (or only moved by an earlier global lead);
we only repair ends:
  - trim phrase-final absorb when energy dies
  - extend held words while energy lives (capped)
  - display floor: min on-screen duration without moving starts

Bake-off survivors this targets: made hang, way#1 short, way#2 absorb,
around short, mean/the FLASH (0.08–0.12s).
"""

from pathlib import Path
from typing import Any

import numpy as np

from .edge_physics import _is_function, _load_features, _peak, _slice


def _mean_e(times: np.ndarray, rms: np.ndarray, t0: float, t1: float) -> float:
    tt, rr = _slice(times, rms, t0, t1)
    return float(np.mean(rr)) if len(rr) else 0.0


def _last_alive(
    times: np.ndarray,
    rms: np.ndarray,
    s: float,
    hi: float,
    thr: float,
    max_dead: int = 5,
) -> float:
    tt, rr = _slice(times, rms, s, hi)
    if len(rr) == 0:
        return s + 0.12
    last = s
    dead = 0
    for i in range(len(rr)):
        if rr[i] >= thr:
            last = float(tt[i])
            dead = 0
        else:
            dead += 1
            if dead >= max_dead and float(tt[i]) - s > 0.12:
                break
    return max(last + 0.04, s + 0.10)


def _first_dead(
    times: np.ndarray,
    rms: np.ndarray,
    s: float,
    e: float,
    thr: float,
    min_keep: float = 0.14,
) -> float:
    """Walk from start; return time when energy dies for good (for trim)."""
    return _last_alive(times, rms, s, e + 0.05, thr=thr, max_dead=4)


def end_snap_to_energy(
    words: list[dict[str, Any]],
    vocal_path: Path,
    *,
    # search forward for holds
    max_extend_s: float = 2.8,
    # never extend function words much
    function_max_dur: float = 0.45,
    content_min_for_hold: float = 0.35,
    thr_rel: float = 0.28,
    thr_floor: float = 0.12,
) -> dict[str, Any]:
    """
    Starts frozen. Ends:
      - if energy dies before current end → trim (anti-hang / absorb)
      - if energy continues past end and word looks held → extend to death
    """
    times, rms, onset, duration_s = _load_features(Path(vocal_path))
    out: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    n = len(words)

    for i, w in enumerate(words):
        nw = dict(w)
        s = float(w["start"])
        e = float(w["end"])
        text = w.get("text") or ""
        next_s = (
            float(words[i + 1]["start"])
            if i + 1 < n
            else duration_s - 0.01
        )
        # hard wall before next start
        wall = next_s - 0.025
        if wall <= s + 0.08:
            nw["start"] = round(s, 4)
            nw["end"] = round(max(e, s + 0.10), 4)
            nw["duration_s"] = round(float(nw["end"]) - s, 4)
            out.append(nw)
            continue

        local_peak = _peak(times, rms, s, min(e, s + 0.5))
        thr = max(thr_floor, local_peak * thr_rel)

        # --- trim if end is past energy death ---
        alive = _first_dead(times, rms, s, min(e, wall), thr=thr)
        new_e = e
        reason = "keep_end"

        if e - alive > 0.18 and alive > s + 0.12:
            new_e = min(alive, wall)
            reason = "end_trim_energy_death"

        # --- extend holds: content word, short box, energy still live at end ---
        dur = new_e - s
        energy_at_end = _mean_e(times, rms, max(s, new_e - 0.08), new_e)
        post = _mean_e(times, rms, new_e, min(new_e + 0.25, wall))
        is_fn = _is_function(text)
        max_dur = function_max_dur if is_fn else max_extend_s

        if (
            not is_fn
            and dur < content_min_for_hold
            and (energy_at_end >= thr * 0.85 or post >= thr * 0.75)
        ):
            hi = min(s + max_dur, wall)
            alive2 = _last_alive(times, rms, s, hi, thr=thr * 0.9, max_dead=6)
            if alive2 > new_e + 0.12:
                new_e = min(alive2, wall)
                reason = (
                    "end_extend_hold"
                    if reason == "keep_end"
                    else reason + "+end_extend_hold"
                )
        elif (
            not is_fn
            and energy_at_end >= thr
            and post >= thr * 0.85
            and (wall - new_e) > 0.2
        ):
            # medium boxes that still have live energy into the gap
            hi = min(s + max_dur, wall)
            alive2 = _last_alive(times, rms, new_e - 0.05, hi, thr=thr * 0.85, max_dead=6)
            if alive2 > new_e + 0.15:
                new_e = min(alive2, wall, s + max_dur)
                reason = (
                    "end_extend_live"
                    if reason == "keep_end"
                    else reason + "+end_extend_live"
                )

        new_e = min(max(new_e, s + 0.10), wall)
        nw["start"] = round(s, 4)  # frozen
        nw["end"] = round(new_e, 4)
        nw["duration_s"] = round(new_e - s, 4)
        if reason != "keep_end":
            nw["repair_reason"] = (nw.get("repair_reason") or "") + f"+{reason}"
            nw["source"] = (nw.get("source") or "spine") + "+end_snap"
            actions.append(
                {
                    "text": text,
                    "reason": reason,
                    "before": [s, e],
                    "after": [s, new_e],
                }
            )
        out.append(nw)

    return {
        "words": out,
        "actions": actions,
        "engine": {
            "name": "end-snap-to-energy-v1",
            "n_actions": len(actions),
            "starts_frozen": True,
        },
    }


def _first_energy_rise(
    times: np.ndarray,
    rms: np.ndarray,
    t0: float,
    t1: float,
    thr: float,
    min_run: int = 3,
) -> float | None:
    tt, rr = _slice(times, rms, t0, t1)
    if len(rr) < min_run + 1:
        return None
    run = 0
    for i in range(len(rr)):
        if rr[i] >= thr:
            run += 1
            if run >= min_run:
                return float(tt[i - min_run + 1])
        else:
            run = 0
    return None


def melisma_hold_lock(
    words: list[dict[str, Any]],
    vocal_path: Path,
    *,
    # DEMOTE 2026-07-19: user ear on song B 153633Z — section wrecked.
    # Prefer whisper-anchor fusion + flux snap (Claude lab stack). Off by default.
    # CTC "ate the hold": following content word FA box is a long blob
    blob_min_s: float = 1.55,
    # Held word itself is short (or sits before the blob)
    holder_max_s: float = 1.20,
    max_hold_s: float = 2.60,
    thr_rel: float = 0.30,
    thr_floor: float = 0.12,
    # Holder end → blob start (may pass through tiny function words)
    max_gap_to_blob_s: float = 0.55,
    min_steal_s: float = 0.70,
    victim_max_dur: float = 0.55,
    pad_s: float = 0.04,
    # Optional: pure tight-pack melisma without a blob.
    # OFF by default — over-fires on normal phrasing ("Now we toe…").
    # Blob-steal alone catches the song B "know" / long-sorry failure.
    pack_enable: bool = False,
    pack_max_gap_s: float = 0.08,
    pack_holder_max_s: float = 0.85,
    pack_min_extra_s: float = 1.10,
    pack_max_hold_s: float = 2.20,
    pack_max_victims: int = 3,
) -> dict[str, Any]:
    """
    Multi-note hold lock — v3 blob-steal + tight pack.

    Failure mode (song B 'know'): singer holds a content word across notes;
    CTC assigns a short box then marches the next lyrics (often dumping the
    melisma into a long following blob like 'sorry'). Screen advances while
    ear is still on 'know'.

    Doctrine:
      - identity order frozen
      - prefer STEAL from an over-long following FA blob (high precision)
      - optional short tight-pack claim when energy lives >1.1s past a short box
      - delay only words between holder and end of claim — no whole-song cascade
    """
    times, rms, onset, duration_s = _load_features(Path(vocal_path))
    out = [dict(w) for w in words]
    n = len(out)
    orig_s = [float(w["start"]) for w in out]
    orig_e = [float(w["end"]) for w in out]
    actions: list[dict[str, Any]] = []
    claimed_until = -1.0

    def _thr_around(t_a: float, t_b: float) -> float:
        pk = _peak(times, rms, t_a, t_b)
        return max(thr_floor, pk * thr_rel)

    def _apply_hold(
        i: int,
        new_e: float,
        victims: list[int],
        *,
        reason: str,
        hold_end: float,
    ) -> None:
        nonlocal claimed_until
        s = float(out[i]["start"])
        e = float(out[i]["end"])
        before = [s, e]
        out[i]["end"] = round(new_e, 4)
        out[i]["duration_s"] = round(new_e - s, 4)
        out[i]["repair_reason"] = (out[i].get("repair_reason") or "") + f"+{reason}"
        out[i]["source"] = (out[i].get("source") or "spine") + f"+{reason}"
        claimed_until = max(claimed_until, new_e)

        cursor = new_e + pad_s
        delayed: list[dict[str, Any]] = []
        # Soft wall: original start of first word after victims (or blob remainder)
        wall = (
            orig_s[victims[-1] + 1]
            if victims and victims[-1] + 1 < n
            else duration_s - 0.01
        )
        # Prefer not to slide past the original end of the last victim's island
        if victims:
            wall = max(wall, new_e + 0.35)

        for j in victims:
            old_s, old_e = orig_s[j], orig_e[j]
            if old_s >= new_e - 0.02:
                continue
            d = min(max(0.10, old_e - old_s), victim_max_dur)
            if _is_function(out[j].get("text") or ""):
                d = min(d, 0.36)
            # Leave a little room before wall when many victims
            remain_n = max(1, sum(1 for jj in victims if orig_s[jj] < new_e - 0.02 and jj >= j))
            room = max(0.12, wall - cursor - 0.02)
            d = min(d, room / remain_n)
            ns, ne = cursor, cursor + max(0.10, d)
            out[j]["start"] = round(ns, 4)
            out[j]["end"] = round(ne, 4)
            out[j]["duration_s"] = round(ne - ns, 4)
            out[j]["repair_reason"] = (out[j].get("repair_reason") or "") + "+melisma_delay"
            out[j]["source"] = (out[j].get("source") or "spine") + "+melisma_delay"
            delayed.append(
                {"text": out[j].get("text"), "before": [old_s, old_e], "after": [ns, ne]}
            )
            cursor = ne + pad_s

        # If last victim was a blob that still has energy after hold, keep a tail box
        if victims:
            j = victims[-1]
            if (orig_e[j] - orig_s[j]) >= blob_min_s and cursor < orig_e[j] - 0.25:
                # restore blob residual for the stolen-from word
                out[j]["start"] = round(cursor, 4)
                out[j]["end"] = round(max(cursor + 0.35, min(orig_e[j], cursor + 1.2)), 4)
                out[j]["duration_s"] = round(
                    float(out[j]["end"]) - float(out[j]["start"]), 4
                )

        actions.append(
            {
                "text": out[i].get("text"),
                "reason": reason,
                "before": before,
                "after": [s, new_e],
                "hold_end": hold_end,
                "n_delayed": len(delayed),
                "delayed": delayed[:8],
            }
        )

    i = 0
    while i < n:
        text = out[i].get("text") or ""
        if _is_function(text) or out[i].get("unplaced"):
            i += 1
            continue
        s, e = float(out[i]["start"]), float(out[i]["end"])
        if e <= s or s < claimed_until - 0.05:
            i += 1
            continue
        holder_dur = e - s

        # --- Path A: blob-steal (high precision) ---
        blob_j = None
        if holder_dur <= holder_max_s:
            for j in range(i + 1, min(n, i + 5)):
                gap = orig_s[j] - e
                if gap > max_gap_to_blob_s:
                    break
                jt = out[j].get("text") or ""
                jd = orig_e[j] - orig_s[j]
                if _is_function(jt):
                    continue
                if jd >= blob_min_s:
                    blob_j = j
                    break
                # stop if we hit another content word that isn't a blob
                if jd >= 0.35 and gap > 0.02:
                    # real next content — not this path
                    break

        if blob_j is not None:
            thr = _thr_around(s, min(s + 1.0, orig_e[blob_j]))
            # Search energy from first rise (handles FA start sitting in a quiet gap)
            rise = _first_energy_rise(
                times, rms, s, min(e + 0.35, orig_e[blob_j]), thr * 0.85
            )
            search_from = rise if rise is not None else s
            hi = min(search_from + max_hold_s, orig_e[blob_j] - 0.20, duration_s - 0.01)
            if hi > search_from + 0.4:
                hold_end = _last_alive(
                    times, rms, search_from, hi, thr=thr * 0.88, max_dead=6
                )
                hold_end = min(hold_end + 0.04, hi)
                steal = hold_end - e
                if steal >= min_steal_s and hold_end > e + 0.45:
                    new_e = min(hold_end, s + max_hold_s)
                    # Optionally snap holder start to rise if FA began in silence
                    if rise is not None and rise > s + 0.12 and rise < e:
                        out[i]["start"] = round(rise, 4)
                        s = rise
                        out[i]["duration_s"] = round(new_e - s, 4)
                    victims = list(range(i + 1, blob_j + 1))
                    _apply_hold(
                        i, new_e, victims, reason="melisma_blob_steal", hold_end=hold_end
                    )
                    i = blob_j + 1
                    continue

        # --- Path B: tight-pack short hold (stricter, no blob) ---
        if (
            pack_enable
            and holder_dur <= pack_holder_max_s
            and i + 1 < n
        ):
            pack_gap = orig_s[i + 1] - e
            if 0.0 <= pack_gap <= pack_max_gap_s:
                thr = _thr_around(s, min(s + 0.9, duration_s))
                rise = _first_energy_rise(times, rms, s, min(e + 0.4, duration_s), thr * 0.85)
                search_from = rise if rise is not None else s
                e_post = _mean_e(times, rms, e, min(e + 0.50, duration_s))
                if e_post >= thr * 0.80:
                    hi = min(search_from + pack_max_hold_s, duration_s - 0.01)
                    hold_end = _last_alive(
                        times, rms, search_from, hi, thr=thr * 0.90, max_dead=6
                    )
                    hold_end = min(hold_end + 0.04, hi)
                    if hold_end - e >= pack_min_extra_s:
                        # victims only while original starts inside hold, max few
                        victims = []
                        for j in range(i + 1, min(n, i + 1 + pack_max_victims)):
                            if orig_s[j] < hold_end - 0.12:
                                victims.append(j)
                            else:
                                break
                        # reject if any victim is already a long content box we
                        # would rather leave (blob path should have caught it)
                        if victims and not any(
                            (orig_e[j] - orig_s[j]) >= blob_min_s
                            and not _is_function(out[j].get("text") or "")
                            for j in victims
                        ):
                            # extra guard: require at least one ultra-short FA
                            # in the pack (FLASH march) OR avg victim FA < 0.45
                            vdurs = [orig_e[j] - orig_s[j] for j in victims]
                            if min(vdurs) <= 0.22 or (sum(vdurs) / len(vdurs)) <= 0.45:
                                new_e = min(hold_end, s + pack_max_hold_s)
                                if rise is not None and rise > s + 0.12 and rise < e:
                                    out[i]["start"] = round(rise, 4)
                                    s = rise
                                _apply_hold(
                                    i,
                                    new_e,
                                    victims,
                                    reason="melisma_pack",
                                    hold_end=hold_end,
                                )
                                i = victims[-1] + 1
                                continue

        i += 1

    # Local monotonic only
    for k in range(1, n):
        prev_e = float(out[k - 1]["end"])
        ks = float(out[k]["start"])
        ke = float(out[k]["end"])
        if ks < prev_e + 0.02:
            d = max(0.10, ke - ks)
            out[k]["start"] = round(prev_e + 0.02, 4)
            out[k]["end"] = round(float(out[k]["start"]) + d, 4)
            out[k]["duration_s"] = round(
                float(out[k]["end"]) - float(out[k]["start"]), 4
            )

    return {
        "words": out,
        "actions": actions,
        "engine": {
            "name": "melisma-hold-lock-v3-blob-steal",
            "n_actions": len(actions),
            "n_delayed": sum(int(a.get("n_delayed") or 0) for a in actions),
            "doctrine": (
                "steal melisma from over-long following FA blob; "
                "optional tight-pack when flash-march + live energy; "
                "no whole-song cascade"
            ),
        },
    }


def apply_display_floor(
    words: list[dict[str, Any]],
    *,
    min_display_s: float = 0.28,
    # never push into next word start
    gap_pad: float = 0.03,
) -> dict[str, Any]:
    """
    Display layer only: extend END if duration < floor, never move START.
    If next word is too close, fill up to next_start - pad (may still be short).
    """
    out = [dict(w) for w in words]
    n = len(out)
    actions: list[dict[str, Any]] = []
    for i, w in enumerate(out):
        s = float(w["start"])
        e = float(w["end"])
        dur = e - s
        if dur >= min_display_s:
            continue
        next_s = float(out[i + 1]["start"]) if i + 1 < n else e + min_display_s
        new_e = min(s + min_display_s, next_s - gap_pad)
        if new_e <= e + 0.01:
            # cannot grow; leave (dense phrase)
            continue
        if new_e <= s:
            continue
        w["end"] = round(new_e, 4)
        w["duration_s"] = round(new_e - s, 4)
        w["repair_reason"] = (w.get("repair_reason") or "") + "+display_floor"
        w["source"] = (w.get("source") or "spine") + "+display_floor"
        actions.append(
            {"text": w.get("text"), "before_dur": dur, "after_dur": new_e - s}
        )
    return {
        "words": out,
        "actions": actions,
        "engine": {
            "name": "display-floor-v1",
            "min_display_s": min_display_s,
            "n_actions": len(actions),
            "starts_frozen": True,
        },
    }
