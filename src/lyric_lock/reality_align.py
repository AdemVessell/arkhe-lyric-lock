from __future__ import annotations

"""
Reality-first lyric timing (user doctrine, proper architecture).

Primary measurements (reality):
  - Vocal energy / onset strength over time (what is sounding)
  - Tempo map: beat times + local BPM (musical time → seconds)

Known lyrics:
  - Ordered word sequence that must occupy real spans

Not primary:
  - Forced-align boxes as master clock (those are a *variant check*)

Pipeline position:
  stem + tempo + energy  →  place lyrics into measured vocal islands
  variants scored by energy fit  →  pick best placement
  FA is one candidate, not the authority
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .mode_a import load_lyrics_words
from .tempo_map import TempoMap, compute_tempo_map


@dataclass
class Island:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _load_features(vocal_path: Path, *, sr: int = 16000, hop: int = 256):
    import librosa

    y, sr = librosa.load(str(vocal_path), sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    # smooth rms
    if len(rms) > 5:
        rms = np.convolve(rms, np.ones(5) / 5.0, mode="same")
    # normalize
    peak = float(np.percentile(rms, 95) + 1e-9)
    rms_n = (rms / peak).astype(np.float64)
    onset = onset.astype(np.float64)
    if len(onset) > 0:
        onset = onset / (float(np.max(onset)) + 1e-9)
    # match lengths
    n = min(len(times), len(rms_n), len(onset))
    return times[:n], rms_n[:n], onset[:n], float(len(y) / sr)


def _rms_at(times: np.ndarray, rms: np.ndarray, t: float) -> float:
    i = int(np.clip(np.searchsorted(times, t), 0, len(rms) - 1))
    return float(rms[i])


def detect_vocal_islands(
    times: np.ndarray,
    rms: np.ndarray,
    *,
    thr: float = 0.14,
    min_dur: float = 0.18,
    merge_gap: float = 0.22,
) -> list[Island]:
    """Contiguous regions where vocal energy is 'on' — the reality of singing."""
    above = rms >= thr
    islands: list[Island] = []
    i = 0
    n = len(times)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        t0, t1 = float(times[i]), float(times[min(j - 1, n - 1)])
        # pad slightly
        t0 = max(0.0, t0 - 0.02)
        t1 = t1 + 0.04
        if t1 - t0 >= min_dur:
            islands.append(Island(t0, t1))
        i = j

    # merge close islands
    if not islands:
        return islands
    merged = [islands[0]]
    for isl in islands[1:]:
        prev = merged[-1]
        if isl.start - prev.end <= merge_gap:
            merged[-1] = Island(prev.start, max(prev.end, isl.end))
        else:
            merged.append(isl)
    return merged


def _word_weight(w: str) -> float:
    """Relative sung duration prior (not FA). Longer for content words / holds cues."""
    core = "".join(c for c in w.lower() if c.isalpha() or c == "'")
    if not core:
        return 0.5
    # base by length
    wgt = 0.55 + 0.12 * min(len(core), 10)
    # parenthetical ohs etc.
    if "oh" in core and len(core) <= 4:
        wgt *= 1.35
    # function words slightly shorter
    if core in {
        "a", "the", "and", "to", "of", "in", "on", "it", "is", "so", "no", "my",
        "our", "all", "just", "what", "will", "i", "me", "we", "had", "up",
    }:
        wgt *= 0.72
    return max(0.35, wgt)


def _allocate_in_island(
    words: list[str],
    island: Island,
    times: np.ndarray,
    rms: np.ndarray,
    onset: np.ndarray,
    tmap: TempoMap,
) -> list[dict[str, Any]]:
    """
    Place a run of words inside one energy island.
    Duration ∝ weight, scaled to island length; boundaries prefer onset peaks / energy dips.
    """
    if not words:
        return []
    weights = np.array([_word_weight(w) for w in words], dtype=np.float64)
    weights /= weights.sum()
    dur = island.duration
    # minimum per word
    min_w = min(0.09, dur / max(len(words) * 1.5, 1))
    raw = weights * dur
    # ensure mins
    raw = np.maximum(raw, min_w)
    raw *= dur / raw.sum()

    # cumulative boundaries (linear first)
    bounds = [island.start]
    acc = island.start
    for d in raw:
        acc += float(d)
        bounds.append(min(acc, island.end))
    bounds[-1] = island.end

    # snap internal bounds to nearest energy valley or onset in a window
    for b in range(1, len(bounds) - 1):
        c = bounds[b]
        win0, win1 = c - 0.08, c + 0.08
        m = (times >= win0) & (times <= win1)
        if not np.any(m):
            continue
        tt = times[m]
        # prefer local minimum of rms among candidates with some onset nearby
        rr = rms[m]
        oo = onset[m] if len(onset) == len(rms) else rms[m]
        # score: low energy + high onset change = boundary
        score = -rr + 0.35 * oo
        j = int(np.argmax(score))
        snapped = float(tt[j])
        # keep order
        lo = bounds[b - 1] + min_w * 0.5
        hi = bounds[b + 1] - min_w * 0.5
        bounds[b] = float(np.clip(snapped, lo, hi))

    # soft tempo: if a boundary is within 40ms of a subdivision, nudge
    for b in range(1, len(bounds) - 1):
        sub = tmap.nearest_subdivision(bounds[b], subdiv=2)
        if abs(sub - bounds[b]) <= 0.04:
            lo = bounds[b - 1] + min_w * 0.5
            hi = bounds[b + 1] - min_w * 0.5
            if lo < sub < hi:
                bounds[b] = sub

    out: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            e = s + min_w
        spb = tmap.seconds_per_beat_at(s)
        out.append(
            {
                "text": w,
                "start": round(s, 4),
                "end": round(e, 4),
                "duration_s": round(e - s, 4),
                "duration_beats": round((e - s) / spb, 3) if spb > 0 else None,
                "local_bpm": round(tmap.local_bpm_at(s), 2),
                "source": "reality_island_pack",
                "island": [island.start, island.end],
            }
        )
    return out


def pack_lyrics_into_islands(
    lyric_words: list[str],
    islands: list[Island],
    times: np.ndarray,
    rms: np.ndarray,
    onset: np.ndarray,
    tmap: TempoMap,
) -> list[dict[str, Any]]:
    """
    Assign consecutive lyric words to vocal islands by capacity (seconds).
    Math: island.duration vs sum of weight-prior durations at local BPM.
    """
    if not lyric_words:
        return []
    if not islands:
        # fallback: single island from global energy
        return _allocate_in_island(
            lyric_words,
            Island(0.0, tmap.duration_s),
            times,
            rms,
            onset,
            tmap,
        )

    # estimated seconds per weight unit from median local spb
    spb = tmap.seconds_per_beat_at(islands[0].start)
    # target: ~0.55 beats per unit weight
    sec_per_weight = spb * 0.55

    words_out: list[dict[str, Any]] = []
    wi = 0
    n = len(lyric_words)

    for li, isl in enumerate(islands):
        if wi >= n:
            break
        # how many words fit in this island?
        capacity = isl.duration
        # leave last island to absorb remainder
        if li == len(islands) - 1:
            chunk = lyric_words[wi:]
            placed = _allocate_in_island(chunk, isl, times, rms, onset, tmap)
            for j, p in enumerate(placed):
                p["id"] = f"W{wi + j}"
                p["line_id"] = f"I{li}"
            words_out.extend(placed)
            wi = n
            break

        # take words until estimated duration fills ~90% of island
        chunk: list[str] = []
        est = 0.0
        while wi + len(chunk) < n:
            w = lyric_words[wi + len(chunk)]
            add = _word_weight(w) * sec_per_weight
            # always take at least 1 word per island if any left
            if chunk and est + add > capacity * 0.92:
                break
            chunk.append(w)
            est += add
            # don't overstuff small islands
            if est >= capacity * 0.88 and len(chunk) >= 1:
                break
        if not chunk:
            chunk = [lyric_words[wi]]
        placed = _allocate_in_island(chunk, isl, times, rms, onset, tmap)
        for j, p in enumerate(placed):
            p["id"] = f"W{wi + j}"
            p["line_id"] = f"I{li}"
        words_out.extend(placed)
        wi += len(chunk)

    # leftover words if islands ran out
    if wi < n and words_out:
        last_end = words_out[-1]["end"]
        tail_isl = Island(last_end, tmap.duration_s)
        placed = _allocate_in_island(
            lyric_words[wi:], tail_isl, times, rms, onset, tmap
        )
        base = wi
        for j, p in enumerate(placed):
            p["id"] = f"W{base + j}"
            p["line_id"] = "I_tail"
        words_out.extend(placed)

    return words_out


def energy_fit_score(words: list[dict[str, Any]], times: np.ndarray, rms: np.ndarray) -> float:
    """Higher = word spans cover more energy, less energy left in gaps."""
    if not words or len(rms) == 0:
        return -1e9
    total = float(np.sum(rms) + 1e-9)
    covered = 0.0
    for w in words:
        s, e = float(w["start"]), float(w["end"])
        m = (times >= s) & (times <= e)
        if np.any(m):
            covered += float(np.sum(rms[m]))
    # penalty for ultra-short / inverted
    pen = 0.0
    for w in words:
        d = float(w["end"]) - float(w["start"])
        if d < 0.05:
            pen += 0.5
        if d <= 0:
            pen += 2.0
    # order penalty
    for a, b in zip(words, words[1:]):
        if float(b["start"]) + 1e-4 < float(a["end"]):
            pen += 1.0
    return covered / total - 0.02 * pen


def reality_align(
    vocal_path: Path,
    lyrics_text: str,
    *,
    mix_path: Path | None = None,
    fa_words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build timing from reality (energy islands + tempo), score variants, pick best.

    Variants:
      A) reality island pack (primary doctrine)
      B) forced-align candidate (if provided) — check only
      C) hybrid: island pack but snap each word center toward FA if within 80ms
    """
    vocal_path = vocal_path.expanduser().resolve()
    mix = Path(mix_path) if mix_path else vocal_path
    times, rms, onset, duration_s = _load_features(vocal_path)
    tmap = compute_tempo_map(mix)

    # adaptive island threshold from rms distribution
    thr = float(np.clip(np.percentile(rms, 40), 0.08, 0.22))
    islands = detect_vocal_islands(times, rms, thr=thr)
    lyric_words = load_lyrics_words(lyrics_text)

    variants: dict[str, list[dict[str, Any]]] = {}

    # A — reality pack
    variants["reality_island"] = pack_lyrics_into_islands(
        lyric_words, islands, times, rms, onset, tmap
    )

    # B — FA check
    if fa_words:
        # normalize FA to same text length if possible
        variants["forced_align"] = [
            {
                "id": w.get("id", f"W{i}"),
                "text": w.get("text"),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "duration_s": float(w["end"]) - float(w["start"]),
                "source": "forced_align_check",
                "line_id": w.get("line_id", "L0"),
                "duration_beats": None,
                "local_bpm": round(tmap.local_bpm_at(float(w["start"])), 2),
            }
            for i, w in enumerate(fa_words)
        ]

    # C — hybrid: for each reality word, if FA word same index exists and close, blend
    if fa_words and variants["reality_island"]:
        hyb = []
        fa = variants["forced_align"]
        for i, rw in enumerate(variants["reality_island"]):
            nw = dict(rw)
            nw["source"] = "hybrid_reality_fa"
            if i < len(fa):
                fs, fe = fa[i]["start"], fa[i]["end"]
                # only blend if FA roughly in same ballpark
                mid_r = 0.5 * (rw["start"] + rw["end"])
                mid_f = 0.5 * (fs + fe)
                if abs(mid_r - mid_f) <= 0.35:
                    # trust energy-island span more for duration; FA more for onset if onset peak agrees
                    o_r = _rms_at(times, rms, rw["start"])
                    o_f = _rms_at(times, rms, fs)
                    start = fs if o_f >= o_r else rw["start"]
                    # end: earlier of the two if both past energy drop — prefer min end to reduce lag
                    e_r = _rms_at(times, rms, rw["end"])
                    e_f = _rms_at(times, rms, fe)
                    end = fe if e_f < e_r else rw["end"]
                    # keep order: start < end
                    if end <= start:
                        end = start + 0.08
                    nw["start"] = round(float(start), 4)
                    nw["end"] = round(float(end), 4)
                    nw["duration_s"] = round(nw["end"] - nw["start"], 4)
                    nw["fa_start"] = fs
                    nw["fa_end"] = fe
            hyb.append(nw)
        # enforce non-overlap
        for i in range(1, len(hyb)):
            if hyb[i]["start"] < hyb[i - 1]["end"]:
                hyb[i]["start"] = hyb[i - 1]["end"] + 0.02
                if hyb[i]["end"] <= hyb[i]["start"]:
                    hyb[i]["end"] = hyb[i]["start"] + 0.08
                hyb[i]["duration_s"] = round(hyb[i]["end"] - hyb[i]["start"], 4)
        variants["hybrid"] = hyb

    scores = {
        name: energy_fit_score(ws, times, rms) for name, ws in variants.items() if ws
    }
    best_name = max(scores, key=scores.get)
    best_words = variants[best_name]

    # assign sequential ids / line groups by islands already
    for i, w in enumerate(best_words):
        w["id"] = w.get("id") or f"W{i}"
        if "duration_beats" not in w or w["duration_beats"] is None:
            spb = tmap.seconds_per_beat_at(w["start"])
            w["duration_beats"] = round(w["duration_s"] / spb, 3) if spb else None
        if "local_bpm" not in w:
            w["local_bpm"] = round(tmap.local_bpm_at(w["start"]), 2)

    # lines by island line_id
    by: dict[str, list] = {}
    for w in best_words:
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

    return {
        "words": best_words,
        "lines": lines,
        "text": " ".join(w["text"] for w in best_words),
        "tempo_map": tmap.to_dict(),
        "islands": [{"start": i.start, "end": i.end, "duration": i.duration} for i in islands],
        "island_threshold": thr,
        "variant_scores": scores,
        "chosen_variant": best_name,
        "user_concept": {
            "name": "reality_first_tempo_energy_lyric_math",
            "applied": True,
            "architecture": (
                "Primary: vocal energy islands + tempo map duration priors. "
                "Lyrics packed into measured reality. "
                "FA is a scored check variant, not master clock."
            ),
            "modules": [
                "lyric_lock.reality_align.reality_align",
                "lyric_lock.tempo_map.compute_tempo_map",
            ],
            "variants_evaluated": list(scores.keys()),
            "chosen_variant": best_name,
        },
        "engine": {
            "name": "reality-align",
            "method": best_name,
            "n_islands": len(islands),
            "n_variants": len(scores),
        },
    }
