from __future__ import annotations

"""
Mode A spine: torchaudio MMS_FA CTC forced alignment on a vocal stem.

Bake-off (2026-07-18, song A 0–90s, human gold 29w):
  bias-corrected onset MAE 92ms (no-star) / 89ms (star) vs 540ms stable-ts.

Convergence 2026-07-19 (Claude lab + user ear):
  - DEFAULT display lead = 0.0 (sung onset). −0.239 RETIRED — spacebar
    anticipation in gold, caused pre-waiting on ear QA.
  - Star-per-line (with_star) is the default: phrase-final hangs eaten by
    star tokens; honest abstention spans on off-sheet audio.

Gotchas:
  - MMS_FA dict includes CTC blank '-' as a key — filter hyphens from targets.
  - Lyric text must not outrun the audio slice (slice-tail guard).
  - Parenthetical / backing "(...)" lines can be excluded from lead targets.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np

# RETIRED 2026-07-19: was −0.239 (LEAD240). Autopsy: gold spacebar anticipation.
# Display target = sung onset. Keep symbol for CLI override / A-B only.
RETIRED_DISPLAY_LEAD_S = -0.239
DEFAULT_DISPLAY_LEAD_S = 0.0


def _lyrics_lines(text: str, *, drop_parenthetical: bool = True) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if drop_parenthetical:
            if re.fullmatch(r"\(.*\)", line):
                continue
            line = re.sub(r"\([^)]*\)", " ", line).strip()
            if not line:
                continue
        lines.append(line)
    return lines


def lyrics_words(
    text: str,
    *,
    drop_parenthetical: bool = True,
) -> list[str]:
    words: list[str] = []
    for line in _lyrics_lines(text, drop_parenthetical=drop_parenthetical):
        words.extend(line.split())
    return words


def lyrics_line_tokens(
    text: str,
    *,
    drop_parenthetical: bool = True,
) -> list[list[str]]:
    """Per-line token lists (for star-per-line targets)."""
    return [
        line.split()
        for line in _lyrics_lines(text, drop_parenthetical=drop_parenthetical)
    ]


def _load_wav_mono16k(path: Path):
    import torch
    import torchaudio

    path = path.expanduser().resolve()
    try:
        wf, sr = torchaudio.load(str(path))
        if sr != 16000:
            wf = torchaudio.functional.resample(wf, sr, 16000)
            sr = 16000
        if wf.size(0) > 1:
            wf = wf.mean(0, keepdim=True)
        return wf, 16000
    except Exception:
        import wave

        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() > 1:
                raw = raw.reshape(-1, w.getnchannels()).mean(axis=1)
            wf = torch.from_numpy(raw.astype(np.float32) / 32768.0).unsqueeze(0)
            if sr != 16000:
                wf = torchaudio.functional.resample(wf, sr, 16000)
            return wf, 16000


def _normalize_token(word: str, dictionary: dict[str, int]) -> str:
    """Keep only chars present in MMS dict; drop CTC blank '-' (id 0)."""
    w = word.lower().replace("’", "'")
    chars: list[str] = []
    for c in w:
        if c not in dictionary:
            continue
        if dictionary[c] == 0:  # CTC blank
            continue
        chars.append(c)
    return "".join(chars)


def _rebuild_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for w in words:
        by.setdefault(str(w.get("line_id") or "L0"), []).append(w)
    lines: list[dict[str, Any]] = []
    for lid, wlist in by.items():
        wlist = sorted(wlist, key=lambda x: float(x["start"]))
        if not wlist:
            continue
        lines.append(
            {
                "id": lid,
                "start": float(wlist[0]["start"]),
                "end": float(wlist[-1]["end"]),
                "text": " ".join(x["text"] for x in wlist),
                "confidence": None,
            }
        )
    lines.sort(key=lambda x: float(x["start"]))
    return lines


def ctc_align_words(
    audio_wav: Path,
    lyrics_text: str,
    *,
    device: str = "cpu",
    display_lead_s: float | None = DEFAULT_DISPLAY_LEAD_S,
    drop_parenthetical: bool = True,
    # Star-per-line: MMS_FA with_star, '*' between lyric lines
    star: bool = True,
    # Slice-tail: refuse to pack leftover lyric past audio end
    tail_margin_s: float = 0.35,
    min_word_frames: int = 1,
    # Per-section windows: bound the blast radius of a lost alignment
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    CTC forced alignment (MMS_FA) of known lyrics onto audio (prefer vocal stem @ 16k).

    display_lead_s: global shift on starts/ends (default 0.0 — sung onset).
      Pass RETIRED_DISPLAY_LEAD_S (−0.239) only for A/B autopsy, not product.
    star: insert '*' between lyric lines (default True). Active star spans
      are returned as honest-abstention instruments for fusion windows.
    sections: optional list of {"name", "t0", "t1", "lines": [str, ...]}. Each
      section aligns inside its own audio window instead of one monotonic pass
      over the whole track.

      Why this exists: forced alignment must consume every token in order and
      has no way to resync. One stretch of bad acoustic evidence — a section
      the sheet does not cover, a dense passage, a long vamp — makes the
      Viterbi path cram every remaining token into sub-second boxes, and the
      damage runs to the end of the song. Observed on "Straight Ghostin"
      (2026-07-19): the verse tail broke at ~52s and 130 words smeared across
      the remaining 120s, including two words holding 40s between them.

      Sections bound that: a section can only wreck itself. Windows should come
      from an ASR structure pass, then be checked. Star tokens still wrap every
      line, so slack at the window edges is absorbed rather than forced.

      Default None = single whole-song pass (unchanged behaviour).
    """
    import torch
    import torchaudio

    audio_wav = audio_wav.expanduser().resolve()
    line_tokens = lyrics_line_tokens(
        lyrics_text, drop_parenthetical=drop_parenthetical
    )
    if not line_tokens:
        raise ValueError("lyrics empty after strip / parenthetical filter")

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=bool(star))
    model.eval()
    if device and device != "cpu":
        try:
            model = model.to(device)
        except Exception:
            device = "cpu"
    dictionary = bundle.get_dict(star="*" if star else None)

    waveform, sr = _load_wav_mono16k(audio_wav)
    assert sr == 16000
    dur_s = float(waveform.size(1) / sr)

    # Split the work into aligned segments. No sections = one whole-song
    # segment, which reproduces the single-pass path exactly.
    if sections:
        segments = []
        li = 0
        for sec in sections:
            sec_lines = lyrics_line_tokens(
                "\n".join(sec["lines"]), drop_parenthetical=drop_parenthetical
            )
            if not sec_lines:
                continue
            segments.append(
                {
                    "name": str(sec.get("name") or f"S{len(segments)}"),
                    "t0": max(0.0, float(sec["t0"])),
                    "t1": min(dur_s, float(sec["t1"])),
                    "line_tokens": sec_lines,
                    "line_base": li,
                }
            )
            li += len(sec_lines)
        if not segments:
            raise ValueError("sections produced no alignable lines")
    else:
        segments = [
            {
                "name": None,
                "t0": 0.0,
                "t1": dur_s,
                "line_tokens": line_tokens,
                "line_base": 0,
            }
        ]

    dropped: list[str] = []
    words_out: list[dict[str, Any]] = []
    star_spans: list[dict[str, float]] = []
    section_report: list[dict[str, Any]] = []
    wi = 0
    n_alignable = 0

    for seg in segments:
        # Build target sequence: optional leading/trailing stars + per-line words
        # Each entry: (display_text or "*", norm_token, line_id or None)
        entries: list[tuple[str, str, str | None]] = []
        for li, toks in enumerate(seg["line_tokens"]):
            lid = f"L{seg['line_base'] + li}"
            if star and not entries:
                entries.append(("*", "*", None))
            for w in toks:
                n = _normalize_token(w, dictionary)
                if n:
                    entries.append((w, n, lid))
                else:
                    dropped.append(w)
            if star:
                entries.append(("*", "*", None))

        keep = [(rw, nw, lid) for rw, nw, lid in entries if nw]
        if not any(rw != "*" for rw, _, _ in keep):
            raise ValueError("no alignable words after MMS_FA dict filter")
        n_alignable += sum(1 for rw, _, _ in keep if rw != "*")

        wf = waveform[:, int(seg["t0"] * sr) : int(seg["t1"] * sr)]
        if wf.size(1) < sr // 10:
            raise ValueError(
                f"section {seg['name']!r} window {seg['t0']}-{seg['t1']}s "
                "is too short to align"
            )

        with torch.inference_mode():
            emission, _ = model(wf.to(next(model.parameters()).device))

        targets = torch.tensor(
            [[dictionary[c] for _, nw, _ in keep for c in nw]],
            dtype=torch.int32,
            device=emission.device,
        )
        aligned, scores = torchaudio.functional.forced_align(
            emission, targets, blank=0
        )
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0])

        ratio = wf.size(1) / emission.size(1) / float(sr)  # sec per frame
        offset = seg["t0"]
        si = 0
        seg_confs: list[float] = []
        for rw, nw, lid in keep:
            n = len(nw)
            group = spans[si : si + n]
            si += n
            if not group:
                continue
            start = float(group[0].start) * ratio + offset
            end = float(group[-1].end) * ratio + offset
            conf = float(np.mean([float(s.score) for s in group]))
            if rw == "*":
                if end - start > 0.10:
                    star_spans.append(
                        {"start": round(start, 4), "end": round(end, 4)}
                    )
                continue
            seg_confs.append(conf)
            words_out.append(
                {
                    "id": f"W{wi}",
                    "start": start,
                    "end": end,
                    "text": rw,
                    "confidence": round(conf, 4),
                    "source": (
                        ("ctc_mms_fa_star" if star else "ctc_mms_fa")
                        + ("_seg" if sections else "")
                    ),
                    "duration_s": round(end - start, 4),
                    "line_id": lid or "L0",
                    **({"section": seg["name"]} if seg["name"] else {}),
                }
            )
            wi += 1
        if seg["name"]:
            section_report.append(
                {
                    "name": seg["name"],
                    "t0": round(seg["t0"], 3),
                    "t1": round(seg["t1"], 3),
                    "n_words": len(seg_confs),
                    "mean_confidence": (
                        round(float(np.mean(seg_confs)), 4) if seg_confs else None
                    ),
                }
            )

    # --- slice-tail guard ---
    hard_end = max(0.0, dur_s - tail_margin_s)
    filtered: list[dict[str, Any]] = []
    tail_dropped = 0
    for w in words_out:
        if w["start"] >= hard_end:
            tail_dropped += 1
            continue
        if w["end"] > dur_s:
            w["end"] = dur_s
        if w["end"] <= w["start"]:
            w["end"] = min(dur_s, w["start"] + 0.08)
        w["duration_s"] = round(float(w["end"]) - float(w["start"]), 4)
        filtered.append(w)
    words_out = filtered

    # --- optional global display lead (default 0) ---
    lead = 0.0 if display_lead_s is None else float(display_lead_s)
    if lead != 0.0:
        for w in words_out:
            w["start"] = round(max(0.0, float(w["start"]) + lead), 4)
            w["end"] = round(max(w["start"] + 0.04, float(w["end"]) + lead), 4)
            w["duration_s"] = round(float(w["end"]) - float(w["start"]), 4)
            w["source"] = (w.get("source") or "ctc_mms_fa") + "+lead"
        # shift star spans with the same constant so fusion windows stay coherent
        star_spans = [
            {
                "start": round(max(0.0, float(s["start"]) + lead), 4),
                "end": round(max(0.0, float(s["end"]) + lead), 4),
            }
            for s in star_spans
        ]

    for i, w in enumerate(words_out):
        w["id"] = f"W{i}"

    lines = _rebuild_lines(words_out)
    text = " ".join(w["text"] for w in words_out)
    method = (
        "ctc_forced_align"
        + ("+star_per_line" if star else "")
        + ("+per_section_windows" if sections else "")
    )

    return {
        "words": words_out,
        "lines": lines,
        "text": text,
        "star_spans": star_spans,
        "engine": {
            "name": "torchaudio-mms-fa",
            "bundle": "MMS_FA",
            "torchaudio": torchaudio.__version__,
            "method": method,
            "input": "audio_16k_mono",
            "device": device,
            "star": bool(star),
            "n_star_spans_active": len(star_spans),
            "display_lead_s": lead,
            "drop_parenthetical": drop_parenthetical,
            "tail_margin_s": tail_margin_s,
            "n_raw_lyric_tokens": sum(len(t) for t in line_tokens),
            "n_alignable": n_alignable,
            "sectioned": bool(sections),
            "sections": section_report,
            "n_dropped_unalignable": len(dropped),
            "n_tail_dropped": tail_dropped,
            "dropped_unalignable_sample": dropped[:12],
            "calibration": {
                "global_onset_shift_s": lead,
                "basis": (
                    "zero lead (sung onset) — default 2026-07-19; "
                    "−0.239 retired as gold spacebar anticipation"
                ),
            },
            "weights": {
                "note": "Phase-3b TOOL_ADMISSION; ~1.18GB torch hub checkpoint",
                "hub_path_hint": "~/.cache/torch/hub/checkpoints/model.pt",
            },
        },
    }
