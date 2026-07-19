from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TempoMap:
    """Musical time → real time (seconds). Not an aesthetic score."""

    beat_times: np.ndarray  # seconds
    global_bpm: float
    duration_s: float

    def local_bpm_at(self, t: float) -> float:
        bt = self.beat_times
        if len(bt) < 2:
            return float(self.global_bpm)
        # interval around t
        i = int(np.searchsorted(bt, t))
        i0 = max(0, i - 1)
        i1 = min(len(bt) - 1, i0 + 1)
        if i1 <= i0:
            return float(self.global_bpm)
        dt = float(bt[i1] - bt[i0])
        if dt <= 1e-4:
            return float(self.global_bpm)
        return 60.0 / dt

    def seconds_per_beat_at(self, t: float) -> float:
        bpm = self.local_bpm_at(t)
        return 60.0 / max(bpm, 1.0)

    def nearest_beat(self, t: float) -> float:
        bt = self.beat_times
        if len(bt) == 0:
            return t
        j = int(np.argmin(np.abs(bt - t)))
        return float(bt[j])

    def nearest_subdivision(self, t: float, subdiv: int = 2) -> float:
        """Nearest 1/subdiv of a beat (subdiv=2 → eighths in 4/4 feel)."""
        bt = self.beat_times
        if len(bt) < 2:
            return t
        # find surrounding beats
        i = int(np.searchsorted(bt, t))
        if i <= 0:
            return float(bt[0])
        if i >= len(bt):
            return float(bt[-1])
        b0, b1 = float(bt[i - 1]), float(bt[i])
        span = b1 - b0
        if span <= 1e-6:
            return b0
        # candidates: b0 + k/subdiv * span
        best = b0
        best_d = abs(t - b0)
        for k in range(subdiv + 1):
            c = b0 + (k / subdiv) * span
            d = abs(t - c)
            if d < best_d:
                best_d = d
                best = c
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_bpm": float(self.global_bpm),
            "n_beats": int(len(self.beat_times)),
            "beat_times_s": [float(x) for x in self.beat_times.tolist()],
            "duration_s": float(self.duration_s),
            "role": "real_time_clock_only",
        }


def compute_tempo_map(
    audio_path: Path,
    *,
    sr: int = 22050,
) -> TempoMap:
    """
    Beat grid in real seconds from the mix (or stem).
    Used only to convert rhythmic duration ↔ wall-clock.
    """
    import librosa

    audio_path = audio_path.expanduser().resolve()
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    duration_s = float(len(y) / sr)
    # beat_track: units=time returns beat locations in seconds
    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    # librosa may return tempo as array
    if hasattr(tempo, "__len__"):
        tempo = float(np.asarray(tempo).reshape(-1)[0])
    else:
        tempo = float(tempo)
    beat_times = np.asarray(beat_times, dtype=np.float64)
    if beat_times.size == 0:
        # synthetic grid from global tempo
        spb = 60.0 / max(tempo, 1.0)
        beat_times = np.arange(0.0, duration_s, spb)
    return TempoMap(beat_times=beat_times, global_bpm=tempo, duration_s=duration_s)
