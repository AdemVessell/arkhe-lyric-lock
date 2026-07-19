from __future__ import annotations

"""
Post-fusion boundary layers (Claude lab song B, 2026-07-19):

1) Onset snap (zero lead): if a word box starts in silence, advance start to
   first stem RMS rise (rms>=0.12) within +1.0s. Corrects Whisper early lean
   across gaps.

2) Flux boundary snap: transitions off held words (>=0.8s, contiguous gap
   <0.15s) snap successor start (and held end) to spectral-flux peak.
   Loudness gates cannot see these (e.g. you→know was 570ms early).
   Forward-only; held word keeps screen until successor onset.
"""

from pathlib import Path
from typing import Any

import numpy as np


def _load_rms_20ms(path: Path) -> tuple[np.ndarray, float]:
    """Normalized RMS at 20ms hop (Claude lab SNAP recipe)."""
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


def _load_flux_novelty(path: Path) -> tuple[np.ndarray, float]:
    """Spectral flux novelty (local median-normalized). 25ms win / 10ms hop."""
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
    win = max(8, int(sr * 0.025))
    hop = max(1, int(sr * 0.010))
    nf = max(0, (len(y) - win) // hop)
    if nf < 4:
        return np.zeros(1, dtype=np.float64), hop / float(sr)
    # vectorized frames
    shape = (nf, win)
    strides = (y.strides[0] * hop, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides).copy()
    frames *= np.hanning(win)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    logm = np.log1p(mag * 100.0)
    flux = np.maximum(logm[1:] - logm[:-1], 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    # local median normalize
    half = 100
    nov = np.empty_like(flux)
    for i in range(len(flux)):
        lo, hi = max(0, i - half), min(len(flux), i + half)
        med = float(np.median(flux[lo:hi]))
        nov[i] = flux[i] / (med + 1e-6)
    return nov.astype(np.float64), hop / float(sr)


def apply_onset_snap(
    words: list[dict[str, Any]],
    vocal_path: Path,
    *,
    thr: float = 0.12,
    max_advance_s: float = 1.0,
) -> dict[str, Any]:
    rms, dt = _load_rms_20ms(Path(vocal_path))
    n = len(rms)
    out = [dict(w) for w in words]
    actions: list[dict[str, Any]] = []

    def r_at(t: float) -> float:
        i = int(t / dt)
        return float(rms[min(max(i, 0), n - 1)])

    for w in out:
        s = float(w["start"])
        e = float(w["end"])
        # mean of first ~80ms
        pre = float(np.mean([r_at(s + k * dt) for k in range(4)]))
        if pre >= thr:
            continue
        limit = min(s + max_advance_s, e - 0.05)
        t = s
        while t < limit and r_at(t) < thr:
            t += dt
        if t > s + 0.02 and t < limit:
            new_s = round(t - dt, 4)
            snap_ms = int((new_s - s) * 1000)
            w["start"] = new_s
            w["duration_s"] = round(float(w["end"]) - new_s, 4)
            w["snap_ms"] = snap_ms
            w["source"] = (w.get("source") or "spine") + "+onset_snap"
            actions.append(
                {"text": w.get("text"), "snap_ms": snap_ms, "to": new_s}
            )

    return {
        "words": out,
        "actions": actions,
        "engine": {
            "name": "onset-snap-v1",
            "n_actions": len(actions),
            "thr": thr,
            "max_advance_s": max_advance_s,
            "doctrine": "silent start → first RMS rise; zero lead",
        },
    }


def apply_flux_boundary_snap(
    words: list[dict[str, Any]],
    vocal_path: Path,
    *,
    hold_min_s: float = 0.8,
    max_gap_s: float = 0.15,
    thr: float = 1.6,
) -> dict[str, Any]:
    nov, dt = _load_flux_novelty(Path(vocal_path))
    n = len(nov)
    out = [dict(w) for w in words]
    actions: list[dict[str, Any]] = []

    def peaks_in(a: float, b: float) -> list[tuple[float, float]]:
        i0, i1 = int(a / dt), int(b / dt)
        found: list[tuple[float, float]] = []
        for i in range(max(1, i0), min(n - 1, i1)):
            if nov[i] > thr and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]:
                found.append((i * dt, float(nov[i])))
        return found

    for i in range(1, len(out)):
        p, w = out[i - 1], out[i]
        p_s, p_e = float(p["start"]), float(p["end"])
        w_s, w_e = float(w["start"]), float(w["end"])
        gap = w_s - p_e
        if gap >= max_gap_s:
            continue
        if (p_e - p_s) < hold_min_s:
            continue
        pk = peaks_in(w_s - 0.10, min(w_s + 0.90, w_e - 0.10))
        if not pk:
            continue
        best_t, best_v = max(pk, key=lambda x: x[1])
        if best_t <= w_s + 0.06:
            continue
        new_t = round(best_t - 0.02, 4)
        if new_t <= p_s + 0.12:
            continue
        before_p, before_w = p_e, w_s
        p["end"] = new_t
        p["duration_s"] = round(new_t - p_s, 4)
        w["start"] = new_t
        w["duration_s"] = round(w_e - new_t, 4)
        w["flux_snap_ms"] = int((new_t - before_w) * 1000) or 1
        p["source"] = (p.get("source") or "spine") + "+flux_hold"
        w["source"] = (w.get("source") or "spine") + "+flux_snap"
        actions.append(
            {
                "from": p.get("text"),
                "to": w.get("text"),
                "before": [before_p, before_w],
                "after": new_t,
                "flux": best_v,
            }
        )

    return {
        "words": out,
        "actions": actions,
        "engine": {
            "name": "flux-boundary-snap-v1",
            "n_actions": len(actions),
            "hold_min_s": hold_min_s,
            "max_gap_s": max_gap_s,
            "thr": thr,
            "doctrine": (
                "forward-only: held word keeps screen until successor "
                "flux-peak onset"
            ),
        },
    }
