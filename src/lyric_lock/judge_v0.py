from __future__ import annotations

"""
Judge v0 — agreement certifier (Claude lab 2026-07-19).

Per-word score:
  CTC confidence + 2.0 * energy_support − 1.5 if star-adjacent

Validated region-level vs user ear (song B):
  wreck median −3.25 vs verses −1.77 (clean separation).
Word-level end-defect detection is weak (needs judge v0.1 end features).

Runtime use: annotate words; region medians drive retry / fail-loudly.
Not a ship gate alone at v0.
"""

from pathlib import Path
from typing import Any

import numpy as np


def _load_rms_20ms(path: Path) -> tuple[np.ndarray, float]:
    import wave

    path = Path(path)
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        y = (
            np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(
                np.float32
            )
            / 32768.0
        )
        if f.getnchannels() > 1:
            y = y.reshape(-1, f.getnchannels()).mean(axis=1)
    hop = max(1, int(sr * 0.02))
    n = len(y) // hop
    if n < 1:
        return np.zeros(1, dtype=np.float64), 0.02
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1))
    rms = rms / (float(np.percentile(rms, 95)) + 1e-9)
    return rms.astype(np.float64), hop / float(sr)


def score_words(
    words: list[dict[str, Any]],
    vocal_path: Path,
    star_spans: list[dict[str, Any]] | None = None,
    *,
    energy_thr: float = 0.12,
    star_adj_s: float = 0.3,
) -> dict[str, Any]:
    rms, dt = _load_rms_20ms(Path(vocal_path))
    n = len(rms)
    star_spans = list(star_spans or [])
    out = [dict(w) for w in words]
    scores: list[float] = []

    for w in out:
        s, e = float(w["start"]), float(w["end"])
        i0 = int(s / dt)
        i1 = max(int(e / dt), i0 + 1)
        seg = rms[max(0, i0) : min(n, i1)]
        e_sup = float(np.mean(seg >= energy_thr)) if len(seg) else 0.0
        conf = w.get("confidence")
        if conf is None:
            conf = -6.0
        conf = float(conf)
        star_adj = False
        for sp in star_spans:
            ss, se = float(sp["start"]), float(sp["end"])
            if (
                abs(s - se) < star_adj_s
                or abs(e - ss) < star_adj_s
                or (s < se and e > ss)
            ):
                star_adj = True
                break
        score = conf + 2.0 * e_sup - (1.5 if star_adj else 0.0)
        w["judge_v0"] = round(score, 3)
        w["judge_energy_support"] = round(e_sup, 3)
        w["judge_star_adjacent"] = bool(star_adj)
        scores.append(score)

    med = float(np.median(scores)) if scores else 0.0
    # simple region split: lowest-quartile mean as "suspect"
    q25 = float(np.percentile(scores, 25)) if scores else 0.0

    return {
        "words": out,
        "engine": {
            "name": "judge-v0-agreement",
            "n_words": len(scores),
            "median_score": round(med, 3),
            "p25_score": round(q25, 3),
            "formula": "conf + 2*energy_support - 1.5*star_adj",
            "validation_note": (
                "song B wreck vs verses: median −3.25 vs −1.77 (ear-gated). "
                "Word-level end defects need v0.1."
            ),
        },
        "region_hint": {
            "suspect_if_median_below": -2.5,
            "median": round(med, 3),
        },
    }
