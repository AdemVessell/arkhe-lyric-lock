from __future__ import annotations

"""
v3 — Correct architecture for user doctrine:

  SPINE: forced-align word identity + rough times (known lyrics ↔ audio)
  PHYSICS: vocal energy + tempo map refine ONLY edges/duration
  SCORE: onset-early penalty, lag-tail penalty, silence-bleed, hold quality
  VARIANTS: all FA-anchored; pick best per word (or keep FA if scores worse)

NOT: pour lyrics into energy islands (that scramble was unusable).
"""

from pathlib import Path
from typing import Any

import numpy as np

from .tempo_map import TempoMap, compute_tempo_map

# --- function words: almost never true multi-beat holds ---
_FUNCTION = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "it", "is", "so",
    "no", "my", "our", "all", "just", "what", "will", "i", "me", "we", "had", "up",
    "for", "but", "if", "as", "be", "by", "do", "did", "has", "have", "been",
    "was", "were", "are", "am", "not", "out", "from", "with", "this", "that",
    "it's", "there's", "what's", "we're", "doesn't", "don't", "won't",
}


def _core(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalpha() or c == "'")


def _is_function(w: str) -> bool:
    return _core(w) in _FUNCTION


def _load_features(path: Path, *, sr: int = 16000, hop: int = 256):
    import librosa

    y, sr = librosa.load(str(path), sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    if len(rms) > 5:
        rms = np.convolve(rms, np.ones(5) / 5.0, mode="same")
    peak = float(np.percentile(rms, 95) + 1e-9)
    rms = (rms / peak).astype(np.float64)
    onset = onset.astype(np.float64)
    if len(onset):
        onset = onset / (float(np.max(onset)) + 1e-9)
    n = min(len(times), len(rms), len(onset))
    return times[:n], rms[:n], onset[:n], float(len(y) / sr)


def _at(times: np.ndarray, arr: np.ndarray, t: float) -> float:
    i = int(np.clip(np.searchsorted(times, t), 0, len(arr) - 1))
    return float(arr[i])


def _slice(times: np.ndarray, arr: np.ndarray, t0: float, t1: float):
    m = (times >= t0) & (times <= t1)
    return times[m], arr[m]


def _peak(times: np.ndarray, rms: np.ndarray, t0: float, t1: float) -> float:
    tt, rr = _slice(times, rms, t0, t1)
    if len(rr) == 0:
        return _at(times, rms, t0)
    return float(np.max(rr))


def refine_start_strict(
    times: np.ndarray,
    rms: np.ndarray,
    onset: np.ndarray,
    fa_s: float,
    prev_end: float,
    next_limit: float,
) -> tuple[float, str]:
    """
    Anti-early: if still quiet at FA start, delay to energy/onset rise.
    Anti-lag: if clear onset slightly before FA, pull at most 90ms.
    Never move more than ~150ms from FA (spine trust).
    """
    max_delay = 0.16
    max_pull = 0.09
    lo = max(prev_end + 0.015, fa_s - max_pull - 0.02)
    hi = min(next_limit - 0.04, fa_s + max_delay)

    peak = _peak(times, rms, fa_s - 0.05, fa_s + 0.25)
    thr = max(0.16, peak * 0.40)
    e0 = _at(times, rms, fa_s)
    o0 = _at(times, onset, fa_s)

    # --- delay if early flash (quiet at FA) ---
    if e0 < thr * 0.75 and o0 < 0.25:
        tt, rr = _slice(times, rms, fa_s, hi)
        oo = _slice(times, onset, fa_s, hi)[1]
        for i in range(len(rr)):
            if rr[i] >= thr or (i < len(oo) and oo[i] >= 0.35 and rr[i] >= thr * 0.55):
                s = float(tt[i])
                s = float(np.clip(s, prev_end + 0.015, hi))
                if s > fa_s + 0.02:
                    return s, "delay_until_energy"
                break

    # --- mild pull if onset peak just before FA ---
    tt, oo = _slice(times, onset, lo, fa_s + 0.02)
    rr = _slice(times, rms, lo, fa_s + 0.02)[1]
    if len(oo) > 2:
        j = int(np.argmax(oo))
        if oo[j] >= 0.40 and rr[j] >= thr * 0.5:
            cand = float(tt[j])
            if fa_s - max_pull <= cand < fa_s - 0.02:
                cand = max(cand, prev_end + 0.015)
                return cand, "pull_to_onset"

    return fa_s, "keep_fa"


def refine_end_strict(
    times: np.ndarray,
    rms: np.ndarray,
    start: float,
    fa_e: float,
    next_s: float,
    text: str,
    tmap: TempoMap,
) -> tuple[float, str]:
    """
    Anti-lag: cut when energy dies before FA end.
    Holds: extend only while energy stays high; hard caps by word type + beats.
    """
    t_lim = max(start + 0.05, next_s - 0.025)
    fa_e = float(np.clip(fa_e, start + 0.05, t_lim))

    peak = _peak(times, rms, start, min(start + 0.40, fa_e + 0.05))
    thr_hi = max(0.18, peak * 0.48)
    thr_lo = max(0.11, peak * 0.26)
    e_fa = _at(times, rms, fa_e)
    spb = tmap.seconds_per_beat_at(start)

    # function words: almost never hold
    if _is_function(text):
        max_ext = min(0.12, 0.35 * spb)
    else:
        # content: up to ~1.2 beats or 0.55s mild; strong hold up to 2 beats
        max_ext = min(0.55, 1.2 * spb)
        max_hold = min(1.15, 2.0 * spb)

    # --- cut dead energy (lag after sung) ---
    tt, rr = _slice(times, rms, start, min(fa_e + 0.08, t_lim))
    last_live = start + 0.05
    for i in range(len(rr)):
        if rr[i] >= thr_lo:
            last_live = float(tt[i])
    if fa_e - last_live > 0.07 and e_fa < thr_lo:
        end = min(last_live + 0.04, t_lim)
        end = max(end, start + 0.05)
        return end, "cut_dead_energy"

    # --- hold extend only if still hot at FA end ---
    if e_fa >= thr_hi and not _is_function(text):
        t_max = min(t_lim, fa_e + max_hold)
        tt2, rr2 = _slice(times, rms, fa_e, t_max)
        end = fa_e
        cold = 0
        for i in range(len(rr2)):
            if rr2[i] >= thr_hi * 0.82:
                end = float(tt2[i])
                cold = 0
            else:
                cold += 1
                if cold >= 3:
                    break
        end = min(end + 0.03, t_lim)
        if end > fa_e + 0.04:
            # don't let mild energy create multi-second lag
            if end - fa_e > max_ext and e_fa < thr_hi * 1.15:
                end = fa_e + max_ext
            return end, "extend_hold"

    # slight trim if FA end is soft but not dead
    if e_fa < thr_lo * 1.1:
        end = min(fa_e, last_live + 0.05) if "last_live" in dir() else fa_e
        end = max(min(end, t_lim), start + 0.05)
        if end < fa_e - 0.03:
            return end, "trim_soft_tail"

    return min(fa_e, t_lim), "keep_fa"


def word_edge_score(
    times: np.ndarray,
    rms: np.ndarray,
    start: float,
    end: float,
) -> dict[str, float]:
    """
    Reality math we care about for lyric video:
    - pre-onset should be quiet (anti-early)
    - span should carry energy (support)
    - post-offset should be quiet (anti-lag)
    """
    pre0, pre1 = start - 0.12, start - 0.02
    post0, post1 = end + 0.02, end + 0.14
    mid0, mid1 = start, end

    def mean_e(a, b):
        tt, rr = _slice(times, rms, a, b)
        if len(rr) == 0:
            return 0.0
        return float(np.mean(rr))

    pre = mean_e(pre0, pre1)
    mid = mean_e(mid0, mid1)
    post = mean_e(post0, post1)
    # higher better
    score = (mid * 1.4) - (pre * 1.1) - (post * 1.25)
    if end - start < 0.05:
        score -= 0.8
    if end <= start:
        score -= 2.0
    return {
        "score": score,
        "pre": pre,
        "mid": mid,
        "post": post,
        "dur": end - start,
    }


def apply_edge_physics(
    fa_words: list[dict[str, Any]],
    vocal_path: Path,
    *,
    mix_path: Path | None = None,
) -> dict[str, Any]:
    """
    FA spine + three edge variants; pick best edges per word by reality score.
    """
    times, rms, onset, duration_s = _load_features(vocal_path)
    tmap = compute_tempo_map(Path(mix_path) if mix_path else vocal_path)

    # normalize FA list
    base: list[dict[str, Any]] = []
    for i, w in enumerate(sorted(fa_words, key=lambda x: float(x["start"]))):
        s, e = float(w["start"]), float(w["end"])
        if e <= s:
            e = s + 0.08
        base.append(
            {
                "id": w.get("id", f"W{i}"),
                "text": (w.get("text") or "").strip(),
                "fa_start": s,
                "fa_end": e,
                "line_id": w.get("line_id", "L0"),
                "confidence": w.get("confidence"),
            }
        )

    def run_variant(name: str, start_fn, end_fn) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, w in enumerate(base):
            prev_end = out[i - 1]["end"] if i else 0.0
            next_s = base[i + 1]["fa_start"] if i + 1 < len(base) else duration_s - 0.01
            s, sr = start_fn(w, prev_end, next_s)
            e, er = end_fn(w, s, next_s)
            e = min(max(e, s + 0.05), next_s - 0.02)
            s = max(s, prev_end + 0.012)
            if e <= s:
                e = s + 0.05
            spb = tmap.seconds_per_beat_at(s)
            sc = word_edge_score(times, rms, s, e)
            out.append(
                {
                    "id": w["id"],
                    "text": w["text"],
                    "start": round(s, 4),
                    "end": round(e, 4),
                    "duration_s": round(e - s, 4),
                    "duration_beats": round((e - s) / spb, 3) if spb else None,
                    "local_bpm": round(tmap.local_bpm_at(s), 2),
                    "fa_start": w["fa_start"],
                    "fa_end": w["fa_end"],
                    "line_id": w["line_id"],
                    "confidence": w.get("confidence"),
                    "source": f"fa_spine+{name}",
                    "refine_start_reason": sr,
                    "refine_end_reason": er,
                    "edge_score": sc["score"],
                    "edge_pre": sc["pre"],
                    "edge_mid": sc["mid"],
                    "edge_post": sc["post"],
                }
            )
        return out

    # Variant A: pure FA
    def A_start(w, prev, nxt):
        return w["fa_start"], "fa"

    def A_end(w, s, nxt):
        return min(w["fa_end"], nxt - 0.02), "fa"

    # Variant B: strict energy edges
    def B_start(w, prev, nxt):
        return refine_start_strict(
            times, rms, onset, w["fa_start"], prev, nxt
        )

    def B_end(w, s, nxt):
        return refine_end_strict(
            times, rms, s, w["fa_end"], nxt, w["text"], tmap
        )

    # Variant C: B + soft tempo nudge on end only
    def C_start(w, prev, nxt):
        return B_start(w, prev, nxt)

    def C_end(w, s, nxt):
        e, r = B_end(w, s, nxt)
        sub = tmap.nearest_subdivision(e, subdiv=2)
        if abs(sub - e) <= 0.04 and s + 0.05 < sub < nxt - 0.02:
            return sub, r + "+tempo_nudge"
        return e, r

    variants = {
        "A_fa_raw": run_variant("A_fa_raw", A_start, A_end),
        "B_energy_edges": run_variant("B_energy_edges", B_start, B_end),
        "C_energy_tempo": run_variant("C_energy_tempo", C_start, C_end),
    }

    # Per-word pick by edge_score — independent (FA neighbors already limited edges).
    # Do NOT cascade-push later words; that created multi-second drift.
    n = len(base)
    chosen: list[dict[str, Any]] = []
    pick_counts = {k: 0 for k in variants}
    for i in range(n):
        best_name = None
        best_w = None
        best_sc = -1e9
        for name, ws in variants.items():
            sc = ws[i]["edge_score"]
            if sc > best_sc:
                best_sc = sc
                best_name = name
                best_w = dict(ws[i])
        assert best_w is not None and best_name is not None
        best_w["chosen_variant"] = best_name
        best_w["source"] = f"fa_spine+pick_{best_name}"
        pick_counts[best_name] += 1
        chosen.append(best_w)

    # Gentle overlap resolve: prefer cutting previous end (anti-lag) over
    # delaying next start (which cascades and breaks the whole timeline).
    for i in range(1, n):
        prev, cur = chosen[i - 1], chosen[i]
        if cur["start"] < prev["end"] - 0.005:
            # cut prev end to just before cur start
            new_prev_end = cur["start"] - 0.015
            if new_prev_end > prev["start"] + 0.05:
                prev["end"] = round(new_prev_end, 4)
                prev["duration_s"] = round(prev["end"] - prev["start"], 4)
                prev["refine_end_reason"] = (prev.get("refine_end_reason") or "") + "+overlap_cut"
            else:
                # only then nudge current start slightly (cap 40ms)
                cur["start"] = round(min(cur["start"] + 0.04, prev["end"] + 0.02), 4)
                if cur["end"] <= cur["start"]:
                    cur["end"] = round(cur["start"] + 0.06, 4)
                cur["duration_s"] = round(cur["end"] - cur["start"], 4)

    # global scores for proof
    def total_score(ws):
        return float(sum(w["edge_score"] for w in ws) / max(len(ws), 1))

    variant_totals = {k: total_score(v) for k, v in variants.items()}
    variant_totals["per_word_pick"] = total_score(chosen)

    # lines
    by: dict[str, list] = {}
    for w in chosen:
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

    # stats vs FA
    delayed = sum(1 for w in chosen if w["start"] > w["fa_start"] + 0.04)
    pulled = sum(1 for w in chosen if w["start"] < w["fa_start"] - 0.03)
    cut = sum(1 for w in chosen if w["end"] < w["fa_end"] - 0.05)
    ext = sum(1 for w in chosen if w["end"] > w["fa_end"] + 0.05)

    return {
        "words": chosen,
        "lines": lines,
        "text": " ".join(w["text"] for w in chosen),
        "tempo_map": tmap.to_dict(),
        "variant_totals": variant_totals,
        "pick_counts": pick_counts,
        "physics_report": {
            "starts_delayed": delayed,
            "starts_pulled": pulled,
            "ends_cut": cut,
            "ends_extended": ext,
            "doctrine": "fa_spine_energy_tempo_edge_physics_v3",
            "global_bpm": tmap.global_bpm,
            "n_beats": len(tmap.beat_times),
        },
        "user_concept": {
            "name": "tempo_energy_duration_on_identity_spine",
            "applied": True,
            "version": "v3_fa_spine_edge_physics",
            "architecture": (
                "Forced-align keeps lyric identity + rough times. "
                "Energy+tempo only refine start/end (duration reality). "
                "Multi-variant edge checks; pick by pre/mid/post energy math."
            ),
            "modules": [
                "lyric_lock.edge_physics.apply_edge_physics",
                "lyric_lock.tempo_map.compute_tempo_map",
                "lyric_lock.forced_align (spine input)",
            ],
            "not_used": "reality_align island packing (rejected — unusable scramble)",
        },
        "engine": {
            "name": "edge-physics-v3",
            "method": "per_word_pick_ABC",
            "variants": list(variants.keys()) + ["per_word_pick"],
        },
    }
