from __future__ import annotations

"""
Path B sheet-completeness + suspect-span gate (Claude lab reel_gate 2026-07-19).

Validated (song C pair on demo-4 reel stem):
  incomplete sheet → FAIL-LOUD 0–17.9s with heard-text naming missing lines
  fixed sheet      → that span quiet

Classes:
  MISSING_LYRICS       — heard content matches no sheet line → FAIL LOUD
  REPEAT_OF_KNOWN_LINE — matches a sheet line or adjacent line-pair (vamp)
  AD_LIB / UNTRANSCRIBABLE — ≤2 heard words or empty → ABSTAIN, not fail-loud

Review 2026-07-19 (Claude ACCEPT + 3 fixes):
  1) prune unreachable classify branches
  2) multi-line vamp match (adjacent line pairs)
  3) engineered splice detector on MIX (digital zeros ≥0.1s) as mandatory
     fusion cuts — not stem (demucs blurs zeros)

Auto-segment:
  silence dips alone over-segments; merge_segments_for_fusion uses
  dips + large stars + MIX splices (+ optional judge windows).
"""

import difflib
import json
from pathlib import Path
from typing import Any

import numpy as np

MIN_STAR_S = 2.0
MIN_ENERGY_FRAC = 0.35
DIP_S = 1.2
MIN_SEG_S = 20.0
ADLIB_MAX_WORDS = 2
REPEAT_SIM = 0.55
LARGE_STAR_S = 2.5
# Engineered splice: digital-zero run on MIX (validated reel: 0.22s × 3 joins)
SPLICE_MIN_ZERO_S = 0.1
SPLICE_ABS_MAX = 0.5 / 32768.0  # below 1 LSB after float convert


def norm(t: str) -> str:
    return "".join(
        c for c in t.lower().replace("’", "'") if c.isalpha() or c in "' "
    )


def sheet_lines(text: str) -> list[str]:
    return [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith(("#", "["))
    ]


def sheet_match_candidates(sheet_norms: list[str]) -> list[str]:
    """Single lines + concatenated adjacent pairs (multi-line vamps)."""
    cands = list(sheet_norms)
    for i in range(len(sheet_norms) - 1):
        pair = f"{sheet_norms[i]} {sheet_norms[i + 1]}".strip()
        if pair:
            cands.append(pair)
    return cands


def best_sheet_sim(heard: str, sheet_norms: list[str]) -> float:
    h = norm(heard)
    if not h:
        return 0.0
    return max(
        (
            difflib.SequenceMatcher(None, h, c).ratio()
            for c in sheet_match_candidates(sheet_norms)
        ),
        default=0.0,
    )


def _decode_with_ffmpeg(path: Path, sr: int | None) -> tuple[np.ndarray, int]:
    """Fallback decoder: stdlib `wave` only handles PCM. Real-world masters are
    often WAVE_FORMAT_EXTENSIBLE (tag 65534), 24-bit, float, or not WAV at all.
    ffmpeg is already a hard dependency, so decode through it.
    """
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not on PATH; cannot decode audio for gate")
    out_sr = sr or 44100
    cmd = [exe, "-v", "error", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
           "-ac", "1", "-ar", str(out_sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return y, out_sr


def _load_mono_float(path: Path, *, target_sr: int | None = 16000) -> tuple[np.ndarray, int]:
    import wave

    path = Path(path)
    try:
        with wave.open(str(path), "rb") as f:
            if f.getsampwidth() != 2:
                raise wave.Error(f"unsupported sample width {f.getsampwidth()}")
            sr = f.getframerate()
            y = (
                np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(
                    np.float32
                )
                / 32768.0
            )
            if f.getnchannels() > 1:
                y = y.reshape(-1, f.getnchannels()).mean(axis=1)
    except Exception:
        # EXTENSIBLE / 24-bit / float / non-WAV → decode via ffmpeg
        y, sr = _decode_with_ffmpeg(path, target_sr)
        return y.astype(np.float32), sr
    if target_sr is not None and sr != target_sr:
        import torch
        import torchaudio

        wf = torch.from_numpy(y).unsqueeze(0)
        wf = torchaudio.functional.resample(wf, sr, target_sr)
        y = wf.squeeze(0).numpy()
        sr = target_sr
    return y.astype(np.float32), sr


def _load_mono16k(path: Path) -> tuple[np.ndarray, int]:
    return _load_mono_float(path, target_sr=16000)


def rms_track(y: np.ndarray, hop: int = 320) -> np.ndarray:
    n = len(y) // hop
    if n < 1:
        return np.zeros(1, dtype=np.float64)
    r = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1))
    return (r / (float(np.percentile(r, 95)) + 1e-9)).astype(np.float64)


def detect_engineered_splices(
    mix_path: Path,
    *,
    min_zero_s: float = SPLICE_MIN_ZERO_S,
    abs_max: float = SPLICE_ABS_MAX,
) -> list[dict[str, Any]]:
    """
    Digital-zero runs ≥ min_zero_s on the MIX = engineered splices.

    MUST use mix audio, not stem (demucs blurs zeros; stem zeros = rests).
    Self-disables on continuous audio (returns []).

    Validated on demo-4 reel: three 0.22s joins at 46.00 / 92.22 / 138.44.
    """
    # Native rate — do not resample before zero detection (preserves exact zeros)
    y, sr = _load_mono_float(Path(mix_path), target_sr=None)
    if len(y) < int(sr * min_zero_s):
        return []
    zero = np.abs(y) <= abs_max
    min_run = max(1, int(round(min_zero_s * sr)))
    edge_margin_s = 0.5  # ignore file head/tail pads
    flank_s = 0.05  # require real audio on both sides (engineered join)
    flank_n = max(1, int(round(flank_s * sr)))
    flank_thr = abs_max * 8  # clearly above digital zero
    splices: list[dict[str, Any]] = []
    i, n = 0, len(zero)
    dur_s = n / float(sr)
    while i < n:
        if not zero[i]:
            i += 1
            continue
        j = i
        while j < n and zero[j]:
            j += 1
        run = j - i
        if run >= min_run:
            t0 = i / float(sr)
            t1 = j / float(sr)
            mid = 0.5 * (t0 + t1)
            # skip leading/trailing silence of continuous masters
            if t0 < edge_margin_s or t1 > dur_s - edge_margin_s:
                i = j
                continue
            # both flanks must carry energy (splice between content, not rest)
            pre = y[max(0, i - flank_n) : i]
            post = y[j : min(n, j + flank_n)]
            if len(pre) < flank_n // 2 or len(post) < flank_n // 2:
                i = j
                continue
            if float(np.mean(np.abs(pre))) < flank_thr:
                i = j
                continue
            if float(np.mean(np.abs(post))) < flank_thr:
                i = j
                continue
            splices.append(
                {
                    "start": round(t0, 4),
                    "end": round(t1, 4),
                    "mid": round(mid, 4),
                    "dur_s": round(t1 - t0, 4),
                }
            )
        i = j
    return splices


def auto_segments_silence(
    rms: np.ndarray,
    *,
    dip_s: float = DIP_S,
    min_seg_s: float = MIN_SEG_S,
    frame_s: float = 0.02,
) -> list[tuple[float, float]]:
    """
    Silence-dip segments only. Lab: over-segments (7/4 songs) — prefer
    merge_segments_for_fusion before fusion DP reset.
    """
    quiet = rms < 0.06
    bounds: list[float] = []
    i, n = 0, len(rms)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * frame_s >= dip_s and i > 0 and j < n:
                bounds.append(((i + j) / 2) * frame_s)
            i = j
        else:
            i += 1
    segs: list[tuple[float, float]] = []
    lo = 0.0
    for b in bounds:
        if b - lo >= min_seg_s:
            segs.append((lo, b))
            lo = b
    segs.append((lo, n * frame_s))
    return segs


def merge_segments_for_fusion(
    silence_segs: list[tuple[float, float]],
    star_spans: list[dict[str, Any]] | list[tuple[float, float]],
    *,
    judge_windows: list[tuple[float, float]] | None = None,
    splice_mids: list[float] | None = None,
    large_star_s: float = LARGE_STAR_S,
    min_seg_s: float = 25.0,
    # Splices are mandatory hard cuts — min length may be shorter
    splice_min_seg_s: float = 8.0,
) -> list[tuple[float, float]]:
    """
    PATH_B fusion DP reset boundaries:
      soft: silence dips + large-star edges + judge windows
      hard: engineered MIX splices (digital zeros) — always cut, no DP bleed
    """
    soft_cuts: list[float] = []
    for a, _b in silence_segs:
        if a > 0.5:
            soft_cuts.append(float(a))
    for sp in star_spans:
        if isinstance(sp, dict):
            s, e = float(sp["start"]), float(sp["end"])
        else:
            s, e = float(sp[0]), float(sp[1])
        if e - s >= large_star_s and s > 1.0:
            soft_cuts.append(s)
    for w in judge_windows or []:
        lo = float(w[0])
        if lo > 1.0:
            soft_cuts.append(lo)

    hard_cuts = sorted(
        set(round(float(m), 3) for m in (splice_mids or []) if float(m) > 0.5)
    )

    end = silence_segs[-1][1] if silence_segs else (
        max(hard_cuts + soft_cuts) + min_seg_s if (hard_cuts or soft_cuts) else 0.0
    )

    # Build cuts: hard splices always; soft only if far enough from hard + min_seg
    all_ordered = sorted(set(round(c, 3) for c in soft_cuts + hard_cuts))
    hard_set = set(hard_cuts)

    segs: list[tuple[float, float]] = []
    lo = 0.0
    for c in all_ordered:
        need = splice_min_seg_s if c in hard_set else min_seg_s
        # always honor hard splice if we're past a tiny pad
        if c in hard_set and c - lo >= 1.0:
            segs.append((lo, c))
            lo = c
        elif c - lo >= need:
            segs.append((lo, c))
            lo = c
    if end - lo >= 5.0:
        segs.append((lo, end))
    elif segs:
        segs[-1] = (segs[-1][0], end)
    else:
        segs.append((0.0, end))
    return segs


def classify_suspect(
    heard: str,
    best_line_sim: float,
    *,
    repeat_sim: float = REPEAT_SIM,
) -> str:
    """
    MISSING_LYRICS | REPEAT_OF_KNOWN_LINE | AD_LIB

    Pre-filter (evaluate_star_span) already requires min duration + energy.
    Pruned unreachable span_dur / energy_frac branches (Claude review).
    """
    heard_s = (heard or "").strip()
    n_words = len([w for w in heard_s.split() if w])

    if best_line_sim >= repeat_sim and n_words >= 2:
        return "REPEAT_OF_KNOWN_LINE"
    if n_words >= 3 and best_line_sim < repeat_sim:
        return "MISSING_LYRICS"
    # empty or ≤2 tokens (Yeah / Thanks / hum)
    return "AD_LIB"


def action_for_class(cls: str) -> str:
    if cls == "MISSING_LYRICS":
        return "FAIL_LOUD"
    if cls == "REPEAT_OF_KNOWN_LINE":
        return "AUTO_RECOVERABLE"
    return "ABSTAIN"  # AD_LIB


def star_spans_from_ctc(
    vocal_16k: Path,
    sheet_text: str,
    *,
    device: str = "cpu",
) -> list[dict[str, float]]:
    """Run star CTC and return active star spans only."""
    from .ctc_align import ctc_align_words

    fa = ctc_align_words(
        Path(vocal_16k),
        sheet_text,
        display_lead_s=0.0,
        star=True,
        drop_parenthetical=True,
        device=device,
    )
    return list(fa.get("star_spans") or [])


def evaluate_star_span(
    y: np.ndarray,
    sr: int,
    rms: np.ndarray,
    s: float,
    e: float,
    sheet_norms: list[str],
    whisper_model: Any | None,
) -> dict[str, Any] | None:
    frame_s = 0.02
    i0, i1 = int(s / frame_s), int(e / frame_s)
    efrac = float(np.mean(rms[i0:i1] >= 0.12)) if i1 > i0 else 0.0
    dur = e - s
    if dur < MIN_STAR_S or efrac < MIN_ENERGY_FRAC:
        return None

    heard = ""
    if whisper_model is not None:
        seg = y[int(s * sr) : int(e * sr)]
        if len(seg) > sr * 0.2:
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tpath = tmp.name
            with wave.open(tpath, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
                w.writeframes(pcm.tobytes())
            try:
                heard = (
                    whisper_model.transcribe(tpath, language="en", fp16=False)
                    .get("text")
                    or ""
                ).strip()
            finally:
                Path(tpath).unlink(missing_ok=True)

    best = best_sheet_sim(heard, sheet_norms)
    cls = classify_suspect(heard, best)
    return {
        "span": [round(s, 3), round(e, 3)],
        "energy_frac": round(efrac, 3),
        "heard": heard,
        "best_line_sim": round(best, 3),
        "class": cls,
        "action": action_for_class(cls),
    }


def run_sheet_gate(
    vocal_path: Path,
    sheet_text: str,
    *,
    mix_path: Path | None = None,
    star_spans: list[dict[str, Any]] | None = None,
    whisper_model_name: str = "medium",
    use_whisper: bool = True,
    device: str = "cpu",
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Full gate report: segments, MIX splices, merged segments, suspects.
    Pass mix_path for engineered-splice cuts (required for mashup DP safety).

    sections: the per-section windows the spine was aligned in, when sectioned.
      REQUIRED for a sound verdict on a sectioned run. The gate reasons from
      star spans, and stars only exist inside an aligned window — so audio
      BETWEEN sections produces no star and is invisible to the gate. That
      silence reads as "nothing to flag" when it is really "never examined".
      Given sections, the inter-section gaps are evaluated on the same path as
      stars (energy floor, then ASR, then classification), so uncovered sung
      audio still fails loudly.
    """
    y, sr = _load_mono16k(Path(vocal_path))
    rms = rms_track(y)
    silence_segs = auto_segments_silence(rms)

    if star_spans is None:
        star_spans = star_spans_from_ctc(Path(vocal_path), sheet_text, device=device)

    splices: list[dict[str, Any]] = []
    splice_mids: list[float] = []
    if mix_path is not None and Path(mix_path).is_file():
        splices = detect_engineered_splices(Path(mix_path))
        splice_mids = [float(s["mid"]) for s in splices]

    merged = merge_segments_for_fusion(
        silence_segs, star_spans, splice_mids=splice_mids
    )
    lines_n = [norm(l) for l in sheet_lines(sheet_text)]

    wmodel = None
    if use_whisper:
        import whisper

        wmodel = whisper.load_model(whisper_model_name)

    # Audio no section claimed. Evaluated exactly like a star span so an
    # unexamined window cannot masquerade as a clean one.
    dur_s = len(y) / float(sr)
    section_gaps: list[dict[str, float]] = []
    if sections:
        wins = sorted(
            (max(0.0, float(s["t0"])), min(dur_s, float(s["t1"]))) for s in sections
        )
        cursor = 0.0
        for t0, t1 in wins:
            if t0 - cursor > MIN_STAR_S:
                section_gaps.append({"start": round(cursor, 3), "end": round(t0, 3)})
            cursor = max(cursor, t1)
        if dur_s - cursor > MIN_STAR_S:
            section_gaps.append({"start": round(cursor, 3), "end": round(dur_s, 3)})

    suspects: list[dict[str, Any]] = []
    for sp in star_spans:
        s, e = float(sp["start"]), float(sp["end"])
        v = evaluate_star_span(y, sr, rms, s, e, lines_n, wmodel)
        if v:
            suspects.append(v)
    for sp in section_gaps:
        s, e = float(sp["start"]), float(sp["end"])
        v = evaluate_star_span(y, sr, rms, s, e, lines_n, wmodel)
        if v:
            v["origin"] = "section_gap"
            suspects.append(v)
    suspects.sort(key=lambda v: v["span"][0])

    fail_loud = [v for v in suspects if v["action"] == "FAIL_LOUD"]
    abstain = [v for v in suspects if v["action"] == "ABSTAIN"]
    recover = [v for v in suspects if v["action"] == "AUTO_RECOVERABLE"]

    return {
        "schema": "arkhe.lyric_lock.suspect_spans/v1",
        "vocal_path": str(Path(vocal_path).resolve()),
        "mix_path": str(Path(mix_path).resolve()) if mix_path else None,
        "engineered_splices": splices,
        "section_gaps_examined": section_gaps,
        "sectioned": bool(sections),
        "segments_silence_only": [list(s) for s in silence_segs],
        "segments_merged_for_fusion": [list(s) for s in merged],
        "segment_note": (
            "use segments_merged_for_fusion for fusion DP reset: "
            "silence dips + large stars + MIX digital-zero splices "
            "(splices mandatory; self-disable if none on continuous audio)"
        ),
        "star_spans": star_spans,
        "suspects": suspects,
        "summary": {
            "n_fail_loud": len(fail_loud),
            "n_abstain_adlib": len(abstain),
            "n_auto_recoverable": len(recover),
            "n_engineered_splices": len(splices),
            "pass": len(fail_loud) == 0,
        },
        "fail_loud": fail_loud,
        "lab_validation_2026_07_19": {
            "faith_incomplete": "FAIL-LOUD 0-17.9s validated",
            "faith_fixed": "that span quiet",
            "ad_lib_class": "AD_LIB abstain not fail-loud",
            "repeat_tier": "REPEAT_OF_KNOWN_LINE + adjacent line-pairs",
            "splice_detector": "MIX digital zeros ≥0.1s → hard fusion cuts",
        },
    }


def write_suspect_spans(report: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path
