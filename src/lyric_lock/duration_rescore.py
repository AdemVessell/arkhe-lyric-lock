from __future__ import annotations

"""
User doctrine layer: refine forced-align word start/end using
  1) vocal energy (what is actually sounding)
  2) tempo map (BPM / beat times as real-time ruler for duration)
  3) multi-scorer repair loop (agents-as-roles: onset / offset / lag / early)

Does NOT score musical aesthetics or pitch notation.
Does NOT snap every onset to a downbeat for "pretty" video.

v2 tighten: prefer FA when energy is ambiguous; cap hold extension;
push late starts forward only on clear energy rise; cut lagging tails hard.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .tempo_map import TempoMap, compute_tempo_map


def _load_rms(path: Path, *, sr: int = 16000, hop: int = 256):
    import librosa

    y, sr = librosa.load(str(path), sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    if len(rms) > 5:
        kernel = np.ones(5) / 5.0
        rms = np.convolve(rms, kernel, mode="same")
    # normalize
    rms = rms.astype(np.float64)
    peak = float(np.percentile(rms, 95) + 1e-9)
    rms_n = rms / peak
    return times, rms_n, float(len(y) / sr)


def _rms_at(times: np.ndarray, rms: np.ndarray, t: float) -> float:
    if len(times) == 0:
        return 0.0
    i = int(np.clip(np.searchsorted(times, t), 0, len(rms) - 1))
    return float(rms[i])


def _slice(times: np.ndarray, rms: np.ndarray, t0: float, t1: float):
    m = (times >= t0) & (times <= t1)
    if not np.any(m):
        return times[:0], rms[:0]
    return times[m], rms[m]


def _peak_in(times: np.ndarray, rms: np.ndarray, t0: float, t1: float) -> float:
    tt, rr = _slice(times, rms, t0, t1)
    if len(rr) == 0:
        return _rms_at(times, rms, t0)
    return float(np.max(rr))


def refine_start(
    times: np.ndarray,
    rms: np.ndarray,
    fa_start: float,
    prev_end: float,
    next_limit: float,
) -> tuple[float, str]:
    """
    Fix early / late starts.
    - If energy at FA start is still low, push start LATER to first rise (anti-early).
    - If a clear rise sits slightly before FA start, pull earlier a little (anti-lag).
    Prefer not moving more than ~120ms unless strong evidence.
    """
    look_back = 0.12
    look_fwd = 0.20
    t0 = max(prev_end + 0.02, fa_start - look_back)
    t1 = min(next_limit - 0.02, fa_start + look_fwd)
    tt, rr = _slice(times, rms, t0, t1)
    if len(rr) < 3:
        return fa_start, "fa_keep"

    # local peak around FA start
    peak = float(np.max(rr) + 1e-9)
    thr = max(0.18, peak * 0.42)  # stricter than v1
    e_fa = _rms_at(times, rms, fa_start)

    # Case A: FA start is early (quiet) — delay to first crossing thr after/near fa
    if e_fa < thr * 0.85:
        for i in range(len(rr)):
            if tt[i] < fa_start - 0.02:
                continue
            if rr[i] >= thr:
                s = float(tt[i])
                s = min(max(s, prev_end + 0.02), next_limit - 0.05)
                if s > fa_start + 0.01:
                    return s, "start_delayed_until_energy"
                break

    # Case B: clear onset slightly before FA (lagging display) — pull back max 100ms
    # find last rise into thr before fa_start
    best = fa_start
    for i in range(1, len(rr)):
        if tt[i] > fa_start:
            break
        if rr[i - 1] < thr <= rr[i]:
            best = float(tt[i])
    if best < fa_start - 0.02 and fa_start - best <= 0.10:
        best = max(best, prev_end + 0.02)
        return best, "start_pulled_to_onset"

    return fa_start, "fa_keep"


def refine_end(
    times: np.ndarray,
    rms: np.ndarray,
    start: float,
    fa_end: float,
    next_start: float,
    *,
    max_extend: float = 0.35,
    max_hold_extend: float = 1.25,
) -> tuple[float, str]:
    """
    Fix lagging ends and short holds.
    - Cut early when energy collapses (anti-lag after sung).
    - Extend only while energy stays high (true holds), capped.
    Default: stay near FA end; don't invent multi-second tails.
    """
    t_limit = next_start - 0.03
    t_limit = max(t_limit, start + 0.06)
    fa_end = min(max(fa_end, start + 0.06), t_limit)

    peak = _peak_in(times, rms, start, min(start + 0.35, fa_end + 0.1))
    thr_high = max(0.20, peak * 0.45)
    thr_low = max(0.12, peak * 0.28)

    e_fa = _rms_at(times, rms, fa_end)

    # --- cut lagging tail: energy already dead at/before FA end ---
    # search backward from fa_end for last frame above thr_low
    tt, rr = _slice(times, rms, start, fa_end + 0.05)
    if len(rr):
        last_hi = start + 0.06
        for i in range(len(rr)):
            if rr[i] >= thr_low:
                last_hi = float(tt[i])
        # if FA end is well after energy died, cut
        if fa_end - last_hi > 0.08 and e_fa < thr_low:
            end = min(last_hi + 0.05, t_limit)
            end = max(end, start + 0.06)
            return end, "end_cut_energy_dead"

    # --- hold extend: energy still strong past FA end ---
    if e_fa >= thr_high:
        # walk forward while high energy
        t_max = min(t_limit, fa_end + max_hold_extend)
        tt2, rr2 = _slice(times, rms, fa_end, t_max)
        end = fa_end
        low_run = 0
        for i in range(len(rr2)):
            if rr2[i] >= thr_high * 0.85:
                end = float(tt2[i])
                low_run = 0
            else:
                low_run += 1
                if low_run >= 3:
                    break
        end = min(end + 0.04, t_limit)
        if end > fa_end + 0.05:
            # cap mild extends for non-extreme holds
            if end - fa_end > max_extend and e_fa < thr_high * 1.05:
                end = fa_end + max_extend
            return end, "end_extended_hold"

    # --- mild: FA end is fine ---
    return fa_end, "fa_keep"


def _tempo_nudge_end(
    end: float,
    tmap: TempoMap,
    *,
    reason: str,
    max_nudge: float = 0.045,
) -> tuple[float, bool]:
    """Soft subdivision snap only when already near a grid point."""
    if reason == "fa_keep":
        # only tiny snap
        max_nudge = 0.035
    sub = tmap.nearest_subdivision(end, subdiv=2)
    if abs(sub - end) <= max_nudge:
        return float(sub), True
    return end, False


def rescore_word_durations(
    words: list[dict[str, Any]],
    vocal_path: Path,
    *,
    mix_path: Path | None = None,
    tempo_map: TempoMap | None = None,
    n_judge_passes: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vocal_path = vocal_path.expanduser().resolve()
    times, rms, dur = _load_rms(vocal_path)

    if tempo_map is None:
        src = mix_path if mix_path and Path(mix_path).is_file() else vocal_path
        tempo_map = compute_tempo_map(Path(src))

    ordered = sorted(words, key=lambda w: float(w.get("start") or 0))
    # work from FA anchors if present
    for w in ordered:
        if "fa_start" not in w:
            w["fa_start"] = float(w["start"])
            w["fa_end"] = float(w["end"])

    report: dict[str, Any] = {
        "n_words": len(ordered),
        "starts_delayed": 0,
        "starts_pulled": 0,
        "ends_extended": 0,
        "ends_cut": 0,
        "tempo_nudges": 0,
        "judge_passes": n_judge_passes,
        "global_bpm": tempo_map.global_bpm,
        "n_beats": len(tempo_map.beat_times),
        "doctrine": "energy_primary_tempo_duration_prior_v2_tight",
        "agents": [
            "onset_judge",
            "offset_judge",
            "lag_tail_judge",
            "early_flash_judge",
            "tempo_duration_judge",
        ],
        "agent_actions": [],
    }

    # iterative multi-scorer repair
    cur = [dict(w) for w in ordered]
    for pass_i in range(n_judge_passes):
        nxt: list[dict[str, Any]] = []
        for i, w in enumerate(cur):
            fa_s = float(w.get("fa_start", w["start"]))
            fa_e = float(w.get("fa_end", w["end"]))
            prev_end = float(nxt[i - 1]["end"]) if i > 0 else 0.0
            next_start = (
                float(cur[i + 1].get("fa_start", cur[i + 1]["start"]))
                if i + 1 < len(cur)
                else dur - 0.01
            )
            # use previous pass end of next if available after first pass
            if pass_i > 0 and i + 1 < len(cur):
                next_start = float(cur[i + 1]["start"])

            # --- onset judge + early flash judge ---
            s1, s_reason = refine_start(times, rms, fa_s, prev_end, next_start)
            if s_reason == "start_delayed_until_energy":
                report["starts_delayed"] += 1
            elif s_reason == "start_pulled_to_onset":
                report["starts_pulled"] += 1
            report["agent_actions"].append(
                {"pass": pass_i, "word": w.get("text"), "agent": "onset_judge", "action": s_reason}
            )

            # --- offset / lag / hold judges ---
            e1, e_reason = refine_end(
                times, rms, s1, fa_e, next_start
            )
            if e_reason == "end_extended_hold":
                report["ends_extended"] += 1
            elif e_reason == "end_cut_energy_dead":
                report["ends_cut"] += 1
            report["agent_actions"].append(
                {"pass": pass_i, "word": w.get("text"), "agent": "offset_judge", "action": e_reason}
            )

            # --- tempo duration judge (soft) ---
            e2, nudged = _tempo_nudge_end(e1, tempo_map, reason=e_reason)
            e2 = min(max(e2, s1 + 0.05), next_start - 0.02)
            if nudged:
                report["tempo_nudges"] += 1
                report["agent_actions"].append(
                    {
                        "pass": pass_i,
                        "word": w.get("text"),
                        "agent": "tempo_duration_judge",
                        "action": "subdivision_nudge",
                    }
                )

            # lag_tail judge: if still loud after end and next is far, optional small extend already done
            # early_flash: if start energy very low, delay more
            if _rms_at(times, rms, s1) < 0.12 and s1 < fa_s + 0.15:
                s_try, r2 = refine_start(times, rms, s1 + 0.04, prev_end, next_start)
                if r2 == "start_delayed_until_energy" and s_try > s1:
                    s1 = s_try
                    report["agent_actions"].append(
                        {
                            "pass": pass_i,
                            "word": w.get("text"),
                            "agent": "early_flash_judge",
                            "action": "extra_delay",
                        }
                    )
                    e2 = max(e2, s1 + 0.05)

            spb = tempo_map.seconds_per_beat_at(s1)
            nw = dict(w)
            nw["start"] = round(s1, 4)
            nw["end"] = round(e2, 4)
            nw["duration_s"] = round(e2 - s1, 4)
            nw["duration_beats"] = round((e2 - s1) / spb, 3) if spb > 0 else None
            nw["local_bpm"] = round(tempo_map.local_bpm_at(s1), 2)
            nw["source"] = "forced_align+tempo_energy_rescore_v2"
            nw["fa_start"] = fa_s
            nw["fa_end"] = fa_e
            nw["refine_start_reason"] = s_reason
            nw["refine_end_reason"] = e_reason
            nxt.append(nw)
        cur = nxt

    # trim agent_actions log size
    if len(report["agent_actions"]) > 400:
        report["agent_actions"] = report["agent_actions"][:200] + report["agent_actions"][-200:]

    return cur, report


def apply_tempo_energy_rescore(
    timed: dict[str, Any],
    *,
    vocal_path: Path,
    mix_path: Path | None = None,
    n_judge_passes: int = 2,
) -> dict[str, Any]:
    words = list(timed.get("words") or [])
    new_words, report = rescore_word_durations(
        words,
        vocal_path,
        mix_path=mix_path,
        n_judge_passes=n_judge_passes,
    )
    by: dict[str, list] = {}
    for w in new_words:
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

    tmap = compute_tempo_map(Path(mix_path or vocal_path))
    out = dict(timed)
    out["words"] = new_words
    out["lines"] = lines
    out["tempo_map"] = tmap.to_dict()
    out["duration_rescore"] = report
    out["user_concept"] = {
        "name": "tempo_real_time_duration_layer",
        "applied": True,
        "version": "v2_tight_with_judges",
        "modules": [
            "lyric_lock.tempo_map.compute_tempo_map",
            "lyric_lock.duration_rescore.apply_tempo_energy_rescore",
        ],
        "agents_used": report.get("agents"),
        "description": (
            "BPM/beats as real-time duration ruler; vocal energy for "
            "holds/silence; multi-scorer repair (onset/offset/early/lag/tempo)"
        ),
    }
    gen = dict(out.get("generator") or {})
    gen["phase"] = "forced_align+tempo_energy_rescore_v2_judges"
    gen["user_concept_applied"] = True
    out["generator"] = gen
    return out
