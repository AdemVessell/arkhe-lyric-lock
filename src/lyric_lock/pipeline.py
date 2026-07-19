from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .export_srt import write_srt
from .hashutil import sha256_file
from .mode_a import align_lyrics_to_asr_words, load_lyrics_words
from .preprocess import probe_duration_s, to_whisper_wav
from .whisper_mode_b import transcribe_mode_b


def _run_id(stem: str, *, mode: str, separated: bool) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:48]
    tag = f"mode{mode}"
    if separated:
        tag += "_stem"
    return f"{ts}_{safe}_{tag}"


def run_pipeline(
    audio: Path,
    out_root: Path,
    *,
    model_name: str = "large-v3",
    language: str | None = None,
    device: str | None = None,
    fixture_id: str | None = None,
    separate: bool = False,
    demucs_model: str = "htdemucs",
    demucs_device: str = "cpu",
    lyrics_path: Path | None = None,
    force_align: bool = True,
    # Mode A engine: ctc (MMS_FA, default post bake-off) | stable_ts (variant check)
    align_engine: str = "ctc",
    # CTC display-lead: default 0.0 (sung onset). −0.239 RETIRED 2026-07-19.
    display_lead_s: float | None = None,
    # Star-per-line MMS_FA (default on for CTC)
    ctc_star: bool = True,
    # Whisper anchor fusion v2 in star-flagged windows
    fuse_anchors: bool = True,
    whisper_fuse_model: str = "medium",
    # Onset snap (silent starts) + flux boundary snap (held-word transitions)
    onset_snap: bool = True,
    flux_snap: bool = True,
    # Judge v0 agreement scores (annotate only)
    judge: bool = True,
    # Path B sheet-gate preflight (Mode A CTC): suspects + fusion DP segments
    gate: bool = True,
    # FAIL LOUD on MISSING_LYRICS (abort run); False = warn and continue
    gate_enforce: bool = True,
    # End physics on good spine
    end_snap: bool = True,
    # DEMOTE 2026-07-19: melisma blob-steal wrecked song B ear section
    melisma_hold: bool = False,
    display_floor_s: float = 0.28,
    # Legacy display-repair (orphan rebind) — OFF by default on CTC path
    display_repair: bool | None = None,
    display_pack: bool = True,
    drop_parenthetical: bool = True,
) -> Path:
    """
    Mode B: lyrics_path is None → free ASR text + times.
    Mode A: lyrics_path provided → forced alignment of known lyrics (default CTC),
            or legacy fuzzy ASR-paste if force_align=False.
    """
    audio = audio.expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio not found: {audio}")

    mode = "A" if lyrics_path else "B"
    run_dir = out_root / _run_id(audio.stem, mode=mode, separated=separate)
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    source_for_asr = audio
    separate_meta: dict[str, Any]

    if separate:
        from .separate import separate_vocals

        sep_out = work / "demucs"
        print(f"  separating vocals ({demucs_model}) …")
        vocals_path = separate_vocals(
            audio,
            sep_out,
            model=demucs_model,
            device=demucs_device,
        )
        stable = work / "vocals.wav"
        shutil.copy2(vocals_path, stable)
        source_for_asr = stable
        separate_meta = {
            "enabled": True,
            "tool": "demucs",
            "model": demucs_model,
            "device": demucs_device,
            "vocals_path": str(stable),
            "runtime": "ArkheCleanroom/lyric-lock",
        }
        print(f"  vocals → {stable}")
    else:
        separate_meta = {"enabled": False}

    wav = work / "audio_16k_mono.wav"
    to_whisper_wav(source_for_asr, wav)

    duration = probe_duration_s(audio)
    digest = sha256_file(audio)

    asr_words: list[dict[str, Any]] = []
    asr_lines: list[dict[str, Any]] = []
    lyrics_meta: dict[str, Any] | None = None
    result_engine: dict[str, Any]
    star_spans: list[dict[str, Any]] = []
    fusion_meta: dict[str, Any] | None = None
    onset_meta: dict[str, Any] | None = None
    flux_meta: dict[str, Any] | None = None
    judge_meta: dict[str, Any] | None = None

    if lyrics_path and force_align:
        lyrics_path = lyrics_path.expanduser().resolve()
        raw = lyrics_path.read_text(encoding="utf-8")
        (run_dir / "lyrics_provided.txt").write_text(raw, encoding="utf-8")
        eng = (align_engine or "ctc").lower().replace("-", "_")
        if eng in ("ctc", "mms_fa", "mms", "torchaudio"):
            from .ctc_align import DEFAULT_DISPLAY_LEAD_S, ctc_align_words

            lead = (
                DEFAULT_DISPLAY_LEAD_S
                if display_lead_s is None
                else float(display_lead_s)
            )
            print(
                f"  CTC forced-align (MMS_FA) star={ctc_star} "
                f"display_lead={lead:.3f}s …"
            )
            fa = ctc_align_words(
                wav,
                raw,
                device=device or "cpu",
                display_lead_s=lead,
                drop_parenthetical=drop_parenthetical,
                star=bool(ctc_star),
            )
            words = fa["words"]
            lines = fa["lines"]
            full_text = fa.get("text") or " ".join(w["text"] for w in words)
            result_engine = fa["engine"]
            star_spans = list(fa.get("star_spans") or [])
            lyrics_meta = {
                "path": str(lyrics_path),
                "n_tokens": len(load_lyrics_words(raw)),
                "n_forced_words": len(words),
                "method": result_engine.get("method") or "ctc-mms-fa",
                "align_engine": "ctc",
                "n_matched": len(words),
                "n_interpolated": 0,
                "n_tail_dropped": (fa.get("engine") or {}).get("n_tail_dropped"),
                "n_star_spans": len(star_spans),
                "display_lead_s": lead,
            }
            phase = "mode-a-ctc-mms-fa"
            if ctc_star:
                phase += "+star"
            if display_repair is None:
                display_repair = False  # end_snap path, not rebind rescue
        else:
            print("  forced-aligning provided lyrics (stable-ts align+refine) …")
            from .forced_align import forced_align_words

            fa = forced_align_words(
                wav,
                raw,
                model_name=model_name,
                language=language or "en",
                device=device or "cpu",
            )
            words = fa["words"]
            lines = fa["lines"]
            full_text = fa.get("text") or " ".join(w["text"] for w in words)
            result_engine = fa["engine"]
            lyrics_meta = {
                "path": str(lyrics_path),
                "n_tokens": len(load_lyrics_words(raw)),
                "n_forced_words": len(words),
                "method": "stable-ts-forced-align+refine",
                "align_engine": "stable_ts",
                "n_matched": len(words),
                "n_interpolated": 0,
            }
            phase = "mode-a-forced-align-stable-ts"
            if display_repair is None:
                display_repair = True
    elif lyrics_path:
        # legacy fuzzy path (kept for A/B comparison only)
        result = transcribe_mode_b(
            wav,
            model_name=model_name,
            language=language,
            device=device,
        )
        asr_words = result.get("words") or []
        asr_lines = result.get("lines") or []
        result_engine = result["engine"]
        lyrics_path = lyrics_path.expanduser().resolve()
        raw = lyrics_path.read_text(encoding="utf-8")
        (run_dir / "lyrics_provided.txt").write_text(raw, encoding="utf-8")
        lyric_tokens = load_lyrics_words(raw)
        aligned_words = align_lyrics_to_asr_words(lyric_tokens, asr_words)
        by_line: dict[str, list[dict[str, Any]]] = {}
        for w in aligned_words:
            by_line.setdefault(str(w.get("line_id") or "L0"), []).append(w)
        lines = []
        for lid, wlist in by_line.items():
            wlist = sorted(wlist, key=lambda x: float(x["start"]))
            lines.append(
                {
                    "id": lid,
                    "start": float(wlist[0]["start"]),
                    "end": float(wlist[-1]["end"]),
                    "text": " ".join(w["text"] for w in wlist),
                    "confidence": None,
                }
            )
        lines.sort(key=lambda x: float(x["start"]))
        words = aligned_words
        full_text = " ".join(w["text"] for w in words)
        lyrics_meta = {
            "path": str(lyrics_path),
            "n_tokens": len(lyric_tokens),
            "n_matched": sum(
                1 for w in words if w.get("source") == "lyrics_matched_asr"
            ),
            "n_interpolated": sum(
                1 for w in words if w.get("source") == "lyrics_interpolated"
            ),
            "method": "legacy-fuzzy-asr-paste",
        }
        phase = "mode-a-lyrics-aligned-legacy"
    else:
        result = transcribe_mode_b(
            wav,
            model_name=model_name,
            language=language,
            device=device,
        )
        asr_words = result.get("words") or []
        asr_lines = result.get("lines") or []
        result_engine = result["engine"]
        words = asr_words
        lines = asr_lines
        full_text = result.get("text") or ""
        phase = (
            "stem-whisper-mode-b" if separate else "baseline-whisper-mode-b"
        )

    timed: dict[str, Any] = {
        "schema": "arkhe.lyric_lock.timed/v0",
        "mode": mode,
        "fixture_id": fixture_id,
        "audio": {
            "path": str(audio),
            "sha256": digest,
            "duration_s": duration,
        },
        "separation": separate_meta,
        "asr_audio": {
            "path": str(source_for_asr),
            "kind": "vocals_stem" if separate else "full_mix",
        },
        "lyrics_provided": lyrics_meta,
        "engine": result_engine,
        "language": language or "en",
        "text": full_text,
        "lines": lines,
        "words": words,
        "asr_words_raw": asr_words if (mode == "A" and asr_words) else None,
        "needs_review": [],
        "generator": {
            "name": "lyric-lock-v0",
            "version": __version__,
            "phase": phase,
        },
    }

    # Spine snapshot before post-spine stages
    (run_dir / "timed_spine.json").write_text(
        json.dumps(timed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if display_repair is None:
        display_repair = mode == "B"

    repair_meta: dict[str, Any] | None = None
    end_meta: dict[str, Any] | None = None
    melisma_meta: dict[str, Any] | None = None
    floor_meta: dict[str, Any] | None = None
    feature_audio = source_for_asr

    def _rebuild_lines_from_words(
        ws: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        by: dict[str, list] = {}
        for w in ws:
            if w.get("unplaced"):
                continue
            by.setdefault(str(w.get("line_id") or "L0"), []).append(w)
        out_lines: list[dict[str, Any]] = []
        for lid, wlist in by.items():
            wlist = sorted(wlist, key=lambda x: float(x["start"]))
            if not wlist:
                continue
            out_lines.append(
                {
                    "id": lid,
                    "start": float(wlist[0]["start"]),
                    "end": float(wlist[-1]["end"]),
                    "text": " ".join(x["text"] for x in wlist),
                    "confidence": None,
                }
            )
        out_lines.sort(key=lambda x: float(x["start"]))
        txt = " ".join(w["text"] for w in ws if not w.get("unplaced"))
        return out_lines, txt

    # Path B sheet-gate preflight: suspects + splice/star segments before fusion.
    # Reuses this run's star spans; MIX (original audio) drives splice cuts.
    gate_meta: dict[str, Any] | None = None
    gate_segments: list[tuple[float, float]] | None = None
    if (
        gate
        and mode == "A"
        and words
        and star_spans
        and (lyrics_meta or {}).get("align_engine") == "ctc"
    ):
        from .sheet_gate import run_sheet_gate, write_suspect_spans

        print("  Path B sheet gate (preflight) …")
        gate_report = run_sheet_gate(
            wav,
            raw,
            mix_path=audio,
            star_spans=star_spans,
            whisper_model_name=whisper_fuse_model,
            device=device or "cpu",
        )
        write_suspect_spans(gate_report, run_dir / "suspect_spans.json")
        gate_segments = [
            (float(a), float(b))
            for a, b in gate_report.get("segments_merged_for_fusion") or []
        ]
        summary = gate_report.get("summary") or {}
        gate_meta = {
            "enforced": bool(gate_enforce),
            "summary": summary,
            "n_segments_for_fusion": len(gate_segments),
            "n_engineered_splices": summary.get("n_engineered_splices"),
        }
        for v in gate_report.get("fail_loud") or []:
            print(
                f"  GATE FAIL-LOUD {v['span'][0]:.2f}-{v['span'][1]:.2f}s "
                f"MISSING_LYRICS heard: {str(v.get('heard'))[:70]!r}"
            )
        if not summary.get("pass", True) and gate_enforce:
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "mode": mode,
                        "audio": str(audio),
                        "phase": "mode-a-GATE-FAILED",
                        "gate": gate_meta,
                        "note": (
                            "sheet gate FAIL LOUD: sung regions without lyric "
                            "coverage — fix sheet (see suspect_spans.json) or "
                            "re-run with gate_enforce=False / --gate-warn"
                        ),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"sheet gate FAIL LOUD: {summary.get('n_fail_loud')} sung "
                f"region(s) missing from lyrics — {run_dir / 'suspect_spans.json'}"
            )

    # Convergence stack (Mode A CTC): fuse → onset snap → flux → end_snap → floor
    # Melisma blob-steal is DEMOTEd (default off).
    if mode == "A" and words and (lyrics_meta or {}).get("align_engine") == "ctc":
        if fuse_anchors and star_spans:
            # Star spans flag hard windows; no stars → skip (nothing to fuse)
            from .whisper_fusion import fuse_whisper_anchors, run_whisper_words

            print(
                f"  whisper-anchor-fusion v2 "
                f"(model={whisper_fuse_model}, stars={len(star_spans)}) …"
            )
            try:
                ww = run_whisper_words(
                    feature_audio,
                    model_name=whisper_fuse_model,
                    language=language or "en",
                    device=device,
                )
                (run_dir / "whisper_fuse_words.json").write_text(
                    json.dumps(ww, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                fu = fuse_whisper_anchors(
                    words,
                    ww,
                    star_spans=star_spans,
                    segments=gate_segments,
                    drop_parenthetical=drop_parenthetical,
                )
                words = fu["words"]
                fusion_meta = fu
                print(
                    f"  fusion: fused={fu['engine']['n_fused']} "
                    f"abstained={fu['engine']['n_abstained']} "
                    f"window={fu['engine']['n_window']}"
                )
                phase = (phase or "") + "+whisper_fusion_v2"
                result_engine = {
                    **(result_engine or {}),
                    "method": (
                        str((result_engine or {}).get("method") or "")
                        + "+whisper_anchor_fusion_v2"
                    ),
                }
            except Exception as e:
                print(f"  fusion SKIPPED ({type(e).__name__}: {e})")
                fusion_meta = {"engine": {"name": "whisper-anchor-fusion-v2", "error": str(e)}}

        if onset_snap:
            from .onset_flux_snap import apply_onset_snap

            print("  onset-snap (silent starts → first RMS rise) …")
            osn = apply_onset_snap(words, feature_audio)
            words = osn["words"]
            onset_meta = osn
            print(f"  onset-snap: actions={osn['engine']['n_actions']}")
            phase = (phase or "") + "+onset_snap"

        if flux_snap:
            from .onset_flux_snap import apply_flux_boundary_snap

            print("  flux-boundary-snap (held-word transitions) …")
            flx = apply_flux_boundary_snap(words, feature_audio)
            words = flx["words"]
            flux_meta = flx
            print(f"  flux-snap: actions={flx['engine']['n_actions']}")
            phase = (phase or "") + "+flux_snap"

        if judge:
            from .judge_v0 import score_words as judge_score_words

            print("  judge-v0 (agreement certifier) …")
            jg = judge_score_words(words, feature_audio, star_spans)
            words = jg["words"]
            judge_meta = jg
            print(
                f"  judge: median={jg['engine']['median_score']} "
                f"p25={jg['engine']['p25_score']}"
            )
            phase = (phase or "") + "+judge_v0"

        lines, full_text = _rebuild_lines_from_words(words)
        timed["words"] = words
        timed["lines"] = lines
        timed["text"] = full_text
        from .ctc_align import DEFAULT_DISPLAY_LEAD_S as _ZERO_LEAD

        timed["star_spans"] = star_spans
        timed["convergence"] = {
            "display_lead_s": (
                _ZERO_LEAD if display_lead_s is None else float(display_lead_s)
            ),
            "lead_doctrine": "zero lead — sung onset (−0.239 retired)",
            "fusion": (fusion_meta or {}).get("engine"),
            "onset_snap": (onset_meta or {}).get("engine"),
            "flux_snap": (flux_meta or {}).get("engine"),
            "judge_v0": (judge_meta or {}).get("engine"),
        }
        timed["generator"]["phase"] = phase
        # refresh spine snapshot after convergence layers that change identity
        # (fusion may drop words) — keep timed_spine as pre-fusion CTC
        timed["engine"] = result_engine

    # End-snap → (optional demoted melisma) → display floor on Mode A spine
    if (end_snap or melisma_hold or display_floor_s) and words and mode == "A":
        from .end_physics import (
            apply_display_floor,
            end_snap_to_energy,
            melisma_hold_lock,
        )

        if end_snap:
            print("  end-snap-to-energy (starts frozen) …")
            es = end_snap_to_energy(words, feature_audio)
            words = es["words"]
            end_meta = es
            print(f"  end-snap: actions={es['engine']['n_actions']}")
            phase = (phase or "") + "+end_snap"

        if melisma_hold:
            print(
                "  melisma-hold-lock (DEMOTEd default-off; "
                "user ear wrecked song B) …"
            )
            mh = melisma_hold_lock(words, feature_audio)
            words = mh["words"]
            melisma_meta = mh
            print(
                f"  melisma: holds={mh['engine']['n_actions']} "
                f"delayed={mh['engine']['n_delayed']}"
            )
            phase = (phase or "") + "+melisma_hold_DEMOTED"

        if display_floor_s and display_floor_s > 0:
            print(f"  display-floor min={display_floor_s:.2f}s (ends only) …")
            fl = apply_display_floor(words, min_display_s=float(display_floor_s))
            words = fl["words"]
            floor_meta = fl
            print(f"  display-floor: actions={fl['engine']['n_actions']}")
            phase = (phase or "") + "+display_floor"

        lines, full_text = _rebuild_lines_from_words(words)
        timed["words"] = words
        timed["lines"] = lines
        timed["text"] = full_text
        timed["end_physics"] = {
            "end_snap": (end_meta or {}).get("engine"),
            "melisma_hold": (melisma_meta or {}).get("engine"),
            "melisma_status": "DEMOTED_default_off_2026_07_19",
            "display_floor": (floor_meta or {}).get("engine"),
            "n_end_actions": (end_meta or {}).get("engine", {}).get("n_actions"),
            "n_melisma_actions": (melisma_meta or {}).get("engine", {}).get(
                "n_actions"
            ),
            "n_floor_actions": (floor_meta or {}).get("engine", {}).get("n_actions"),
        }
        timed["generator"]["phase"] = phase

    if display_repair and words:
        from .display_repair import apply_display_repair_to_timed

        print("  display-repair (orphan rebind + gap-fill + pack) …")
        timed = apply_display_repair_to_timed(
            timed, feature_audio, pack=display_pack
        )
        words = timed["words"]
        lines = timed["lines"]
        full_text = timed.get("text") or full_text
        repair_meta = timed.get("display_repair")
        eng = (repair_meta or {}).get("engine") or {}
        print(
            f"  display-repair: actions={eng.get('n_actions')} "
            f"orphan_rebind={eng.get('n_orphan_rebind')} "
            f"gap_fill={eng.get('n_gap_fill')} "
            f"unplaced={eng.get('n_unplaced')}"
        )
        phase = (phase or "") + "+display_repair_v1"
        timed["generator"]["phase"] = phase

    (run_dir / "timed.json").write_text(
        json.dumps(timed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if repair_meta is not None:
        (run_dir / "display_repair.json").write_text(
            json.dumps(repair_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # SRT from placed words only
    placed_words = [w for w in words if not w.get("unplaced")]
    write_srt(lines, run_dir / "lyrics.srt")
    write_srt(
        [
            {"start": w["start"], "end": w["end"], "text": w["text"]}
            for w in placed_words
        ],
        run_dir / "lyrics.words.srt",
    )

    placed_for_meta = [w for w in words if not w.get("unplaced")]
    first_word = float(placed_for_meta[0]["start"]) if placed_for_meta else None
    meta = {
        "run_dir": str(run_dir),
        "mode": mode,
        "fixture_id": fixture_id,
        "audio": str(audio),
        "asr_audio": str(source_for_asr),
        "separate": separate,
        "lyrics_path": str(lyrics_path) if lyrics_path else None,
        "align_engine": (lyrics_meta or {}).get("align_engine") if lyrics_meta else None,
        "display_lead_s": (
            0.0 if display_lead_s is None else float(display_lead_s)
        ),
        "ctc_star": bool(ctc_star),
        "fuse_anchors": bool(fuse_anchors),
        "onset_snap": bool(onset_snap),
        "flux_snap": bool(flux_snap),
        "judge": bool(judge),
        "end_snap": bool(end_snap),
        "melisma_hold": bool(melisma_hold),
        "melisma_status": "DEMOTED_default_off_2026_07_19",
        "display_floor_s": display_floor_s,
        "display_repair": bool(display_repair),
        "display_pack": bool(display_pack),
        "drop_parenthetical": bool(drop_parenthetical),
        "n_star_spans": len(star_spans),
        "sha256": digest,
        "duration_s": duration,
        "model": model_name,
        "language": language or "en",
        "n_lines": len(lines),
        "n_words": len(words),
        "n_words_placed": len(placed_for_meta),
        "first_word_start_s": first_word,
        "lyrics_meta": lyrics_meta,
        "display_repair_engine": (repair_meta or {}).get("engine") if repair_meta else None,
        "convergence": (timed.get("convergence") if isinstance(timed, dict) else None),
        "end_physics": {
            "end_snap": (end_meta or {}).get("engine") if end_meta else None,
            "melisma_hold": (melisma_meta or {}).get("engine") if melisma_meta else None,
            "melisma_status": "DEMOTED_default_off_2026_07_19",
            "display_floor": (floor_meta or {}).get("engine") if floor_meta else None,
        },
        "gate": gate_meta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fcp_import": "lyrics.srt",
        "phase": phase,
        "note": (
            "Mode A — forced align known lyrics to vocal (stable-ts)"
            if mode == "A" and force_align
            else (
                "Mode A — legacy fuzzy paste"
                if mode == "A"
                else (
                    "Mode B + vocal stem then Whisper"
                    if separate
                    else "Mode B baseline — Whisper only"
                )
            )
            + ("; display-repair v1 on" if display_repair else "; display-repair off")
        ),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "transcript.txt").write_text(full_text + "\n", encoding="utf-8")
    return run_dir


# backwards-compatible name
def run_mode_b(*args: Any, **kwargs: Any) -> Path:
    return run_pipeline(*args, **kwargs)
