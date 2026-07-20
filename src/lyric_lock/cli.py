from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import get_fixture, list_fixtures
from .export_video import render_lyric_video
from .paths import OUT
from .pipeline import run_pipeline
from .score_run import score_run_dir


def _cmd_list(_: argparse.Namespace) -> int:
    fixtures = list_fixtures()
    for fx in fixtures:
        audio = Path(fx["audio"])
        exists = "OK" if audio.is_file() else "MISSING"
        print(
            f"{fx['id']:24}  p={fx.get('priority', '?')}  "
            f"~{fx.get('duration_s_approx', '?')}s  [{exists}]  {fx.get('title', '')}"
        )
    return 0


def _load_sections(path: Path | None) -> list[dict] | None:
    """Read + validate a --sections spec. Bad windows are worse than none."""
    if not path:
        return None
    secs = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(secs, list) or not secs:
        raise SystemExit("--sections: expected a non-empty JSON list")
    for i, s in enumerate(secs):
        missing = {"t0", "t1", "lines"} - set(s)
        if missing:
            raise SystemExit(f"--sections[{i}]: missing {sorted(missing)}")
        if float(s["t1"]) <= float(s["t0"]):
            raise SystemExit(f"--sections[{i}]: t1 must be > t0")
        if not s["lines"]:
            raise SystemExit(f"--sections[{i}]: empty lines")
    for i, (a, b) in enumerate(zip(secs, secs[1:])):
        if float(b["t0"]) < float(a["t0"]):
            raise SystemExit(f"--sections[{i+1}]: windows must be in time order")
    return secs


def _cmd_run(args: argparse.Namespace) -> int:
    fixture_id = None
    if args.fixture:
        fx = get_fixture(args.fixture)
        audio = Path(fx["audio"])
        fixture_id = fx["id"]
        if not audio.is_file():
            print(f"error: fixture audio missing: {audio}", file=sys.stderr)
            return 2
    elif args.audio:
        audio = Path(args.audio)
    else:
        print("error: provide --fixture or --audio", file=sys.stderr)
        return 2

    out_root = Path(args.out) if args.out else OUT
    lyrics_path = Path(args.lyrics) if args.lyrics else None
    mode = "A" if lyrics_path else "B"
    print(f"Mode {mode} run")
    print(f"  audio:  {audio}")
    print(f"  model:  {args.model}")
    print(f"  separate: {args.separate}")
    if lyrics_path:
        print(f"  lyrics: {lyrics_path}")
    print(f"  out:    {out_root}")

    # display_repair: None = engine default (off for CTC, on for stable_ts/B)
    dr_flag = getattr(args, "display_repair", None)
    if getattr(args, "no_display_repair", False):
        dr_flag = False
    elif getattr(args, "force_display_repair", False):
        dr_flag = True

    lead = getattr(args, "display_lead", None)
    if lead is not None:
        lead = float(lead)

    run_dir = run_pipeline(
        audio,
        out_root,
        model_name=args.model,
        language=args.language,
        device=args.device,
        fixture_id=fixture_id,
        separate=bool(args.separate),
        demucs_model=args.demucs_model,
        demucs_device=args.demucs_device,
        lyrics_path=lyrics_path,
        align_engine=getattr(args, "align_engine", "ctc") or "ctc",
        display_lead_s=lead,
        ctc_star=not bool(getattr(args, "no_ctc_star", False)),
        fuse_anchors=not bool(getattr(args, "no_fuse_anchors", False)),
        whisper_fuse_model=str(
            getattr(args, "whisper_fuse_model", None) or "medium"
        ),
        onset_snap=not bool(getattr(args, "no_onset_snap", False)),
        flux_snap=not bool(getattr(args, "no_flux_snap", False)),
        judge=not bool(getattr(args, "no_judge", False)),
        gate=not bool(getattr(args, "no_gate", False)),
        gate_enforce=not bool(getattr(args, "gate_warn", False)),
        end_snap=not bool(getattr(args, "no_end_snap", False)),
        # melisma DEMOTEd: only on with --force-melisma-hold
        melisma_hold=bool(getattr(args, "force_melisma_hold", False)),
        display_floor_s=float(getattr(args, "display_floor", 0.28) or 0.0),
        display_repair=dr_flag,
        display_pack=not bool(getattr(args, "no_display_pack", False)),
        drop_parenthetical=not bool(getattr(args, "keep_parenthetical", False)),
        sections=_load_sections(getattr(args, "sections", None)),
        collapse_check=not bool(getattr(args, "no_collapse_check", False)),
        collapse_fail=not bool(getattr(args, "collapse_warn", False)),
    )

    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    print(f"done → {run_dir}")
    print(f"  lines={meta['n_lines']}  words={meta['n_words']}  lang={meta.get('language')}")
    if meta.get("first_word_start_s") is not None:
        print(f"  first_word_start_s={meta['first_word_start_s']}")
    print(f"  FCP: import {run_dir / 'lyrics.srt'}")

    if args.video or args.score:
        card = score_run_dir(run_dir)
        print(f"  score band={card['band']} coverage={card.get('timeline_coverage')}")
        for issue in card.get("issues") or []:
            print(f"  issue: {issue}")
        for note in card.get("notes") or []:
            print(f"  note:  {note}")

    if args.video:
        audio = Path(meta["audio"])
        title = audio.stem
        if fixture_id:
            try:
                title = get_fixture(fixture_id).get("title") or title
            except KeyError:
                pass
        mix = Path(meta["audio"])
        style = args.video_style
        suffix = {"karaoke": "karaoke", "word": "word", "line": "line"}.get(
            style, style
        )
        mp4 = run_dir / f"lyric_video_{suffix}.mp4"
        print(f"  rendering lyric video ({style}) → {mp4}")
        render_lyric_video(
            mix,
            run_dir / "timed.json",
            mp4,
            style=style,
            title=title,
        )
        print(f"  video: {mp4}")
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    timed = run_dir / "timed.json"
    meta_path = run_dir / "run_meta.json"
    if not timed.is_file():
        print(f"error: missing {timed}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    audio = Path(args.audio) if args.audio else Path(meta["audio"])
    card = score_run_dir(run_dir)
    print(json.dumps(card, indent=2, ensure_ascii=False))
    mp4 = Path(args.out) if args.out else (run_dir / "lyric_video_karaoke.mp4")
    style = args.video_style
    render_lyric_video(
        audio,
        timed,
        mp4,
        style=style,
        title=args.title or audio.stem,
    )
    print(f"video ({style}) → {mp4}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    card = score_run_dir(run_dir)
    print(json.dumps(card, indent=2, ensure_ascii=False))
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    """Apply display-repair to an existing run directory."""
    from .display_repair import apply_display_repair_to_timed
    from .export_srt import write_srt

    run_dir = Path(args.run_dir).expanduser().resolve()
    spine = run_dir / "timed_spine.json"
    timed_path = spine if spine.is_file() else (run_dir / "timed.json")
    if not timed_path.is_file():
        print(f"error: missing timed.json in {run_dir}", file=sys.stderr)
        return 2
    timed = json.loads(timed_path.read_text(encoding="utf-8"))
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    if args.vocals:
        vocals = Path(args.vocals).expanduser().resolve()
    else:
        cand = run_dir / "work" / "vocals.wav"
        if cand.is_file():
            vocals = cand
        else:
            vocals = Path(meta.get("asr_audio") or meta.get("audio") or "")
    if not vocals.is_file():
        print(f"error: vocals/feature audio not found: {vocals}", file=sys.stderr)
        return 2

    print(f"display-repair on {timed_path.name}  features={vocals}")
    repaired = apply_display_repair_to_timed(
        timed, vocals, pack=not bool(args.no_pack)
    )
    (run_dir / "timed.json").write_text(
        json.dumps(repaired, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dr = repaired.get("display_repair") or {}
    (run_dir / "display_repair.json").write_text(
        json.dumps(dr, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    words = [w for w in repaired.get("words") or [] if not w.get("unplaced")]
    lines = repaired.get("lines") or []
    write_srt(lines, run_dir / "lyrics.srt")
    write_srt(
        [{"start": w["start"], "end": w["end"], "text": w["text"]} for w in words],
        run_dir / "lyrics.words.srt",
    )
    eng = dr.get("engine") or {}
    print(
        f"done actions={eng.get('n_actions')} orphan={eng.get('n_orphan_rebind')} "
        f"gap_fill={eng.get('n_gap_fill')} unplaced={eng.get('n_unplaced')}"
    )
    if args.video:
        audio = Path(meta.get("audio") or vocals)
        mp4 = run_dir / "lyric_video_word.mp4"
        render_lyric_video(
            audio,
            run_dir / "timed.json",
            mp4,
            style="word",
            title=audio.stem + " · display-repair",
        )
        print(f"video → {mp4}")
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    """Path B sheet gate — lab-validated song C fail-loud + AD_LIB abstain."""
    from .sheet_gate import run_sheet_gate, write_suspect_spans

    vocal = Path(args.vocal)
    lyrics = Path(args.lyrics)
    if not vocal.is_file():
        print(f"error: vocal missing: {vocal}", file=sys.stderr)
        return 2
    if not lyrics.is_file():
        print(f"error: lyrics missing: {lyrics}", file=sys.stderr)
        return 2
    sheet = lyrics.read_text(encoding="utf-8")
    print(f"Path B sheet gate")
    print(f"  vocal:  {vocal}")
    print(f"  lyrics: {lyrics}")
    mix = Path(args.mix) if getattr(args, "mix", None) else None
    report = run_sheet_gate(
        vocal,
        sheet,
        mix_path=mix,
        use_whisper=not bool(args.no_whisper),
        whisper_model_name=str(args.whisper_model or "medium"),
    )
    out = Path(args.out) if args.out else (vocal.parent / "suspect_spans.json")
    write_suspect_spans(report, out)
    summ = report.get("summary") or {}
    print(f"  silence segs: {len(report.get('segments_silence_only') or [])}")
    print(f"  merged segs:  {len(report.get('segments_merged_for_fusion') or [])}")
    splices = report.get("engineered_splices") or []
    if mix:
        print(f"  mix splices:  {len(splices)}  {[s.get('mid') for s in splices]}")
    else:
        print("  mix splices:  (no --mix; splice detector off)")
    print(
        f"  fail_loud={summ.get('n_fail_loud')}  "
        f"ad_lib={summ.get('n_abstain_adlib')}  "
        f"repeat={summ.get('n_auto_recoverable')}  "
        f"pass={summ.get('pass')}"
    )
    for v in report.get("fail_loud") or []:
        print(
            f"  FAIL-LOUD {v['span'][0]:.1f}-{v['span'][1]:.1f}s  "
            f"heard={v.get('heard', '')[:80]!r}"
        )
    for v in report.get("suspects") or []:
        if v.get("class") == "REPEAT_OF_KNOWN_LINE":
            print(
                f"  REPEAT   {v['span'][0]:.1f}-{v['span'][1]:.1f}s  "
                f"sim={v.get('best_line_sim')}  heard={v.get('heard', '')[:60]!r}"
            )
        if v.get("class") == "AD_LIB":
            print(
                f"  AD_LIB   {v['span'][0]:.1f}-{v['span'][1]:.1f}s  "
                f"heard={v.get('heard', '')[:40]!r}"
            )
    print(f"  wrote {out}")
    return 0 if summ.get("pass") else 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lyric_lock",
        description="lyric-lock-v0: audio → timed lyrics (Mode B first)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="List Desktop fixtures from catalog")
    pl.set_defaults(func=_cmd_list)

    pr = sub.add_parser("run", help="Run Mode B (no provided lyrics)")
    pr.add_argument("--fixture", help="Fixture id from fixtures/catalog.json")
    pr.add_argument("--audio", help="Path to audio file")
    pr.add_argument("--model", default="large-v3", help="Whisper model (default large-v3)")
    pr.add_argument("--language", default=None, help="Force language code (e.g. en, ja)")
    pr.add_argument("--device", default=None, help="torch device override (cpu/cuda/mps)")
    pr.add_argument("--out", default=None, help="Output root (default: ./out)")
    pr.add_argument(
        "--lyrics",
        default=None,
        help="Path to provided lyrics .txt (Mode A — force text onto ASR times)",
    )
    pr.add_argument(
        "--separate",
        action="store_true",
        help="Vocal stem via cleanroom demucs before Whisper",
    )
    pr.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs model name (default htdemucs)",
    )
    pr.add_argument(
        "--demucs-device",
        default="cpu",
        help="Demucs device (cpu/cuda/mps)",
    )
    pr.add_argument(
        "--video",
        action="store_true",
        help="After timing, render ffmpeg lyric video (karaoke word-lock default)",
    )
    pr.add_argument(
        "--no-gate",
        action="store_true",
        help="Skip Path B sheet-gate preflight (Mode A CTC)",
    )
    pr.add_argument(
        "--gate-warn",
        action="store_true",
        help="Sheet gate warns on MISSING_LYRICS instead of aborting run",
    )
    pr.add_argument(
        "--sections",
        type=Path,
        help=(
            "JSON: [{name,t0,t1,lines:[...]}] — align each section in its own "
            "audio window. Bounds a lost alignment to one section instead of "
            "smearing to the end of the track."
        ),
    )
    pr.add_argument(
        "--no-collapse-check",
        action="store_true",
        help="Skip the spine-collapse check that runs before the sheet gate",
    )
    pr.add_argument(
        "--collapse-warn",
        action="store_true",
        help="Spine collapse warns instead of aborting the run",
    )
    pr.add_argument(
        "--score",
        action="store_true",
        help="Write/print heuristic scorecard after run",
    )
    pr.add_argument(
        "--video-style",
        choices=("karaoke", "word", "line"),
        default="karaoke",
        help="lyric video burn-in: karaoke (default), word-at-a-time, or CC line",
    )
    pr.add_argument(
        "--align-engine",
        choices=("ctc", "stable_ts"),
        default="ctc",
        help="Mode A spine: ctc=MMS_FA (default, bake-off winner) | stable_ts=variant check",
    )
    pr.add_argument(
        "--display-lead",
        type=float,
        default=None,
        help="Global onset shift seconds (default 0.0 sung onset; −0.239 RETIRED)",
    )
    pr.add_argument(
        "--no-ctc-star",
        action="store_true",
        help="Disable MMS_FA star-per-line (default: star on)",
    )
    pr.add_argument(
        "--no-fuse-anchors",
        action="store_true",
        help="Skip Whisper anchor fusion v2 in star windows",
    )
    pr.add_argument(
        "--whisper-fuse-model",
        default="medium",
        help="Whisper model for anchor fusion (default medium)",
    )
    pr.add_argument(
        "--no-onset-snap",
        action="store_true",
        help="Skip onset snap (silent starts → first RMS rise)",
    )
    pr.add_argument(
        "--no-flux-snap",
        action="store_true",
        help="Skip flux boundary snap on held-word transitions",
    )
    pr.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip judge-v0 agreement scores",
    )
    pr.add_argument(
        "--no-end-snap",
        action="store_true",
        help="Skip end-snap-to-energy (Mode A)",
    )
    pr.add_argument(
        "--force-melisma-hold",
        action="store_true",
        help="Enable DEMOTEd melisma blob-steal (default OFF — ear wreck)",
    )
    pr.add_argument(
        "--display-floor",
        type=float,
        default=0.28,
        help="Min on-screen duration seconds; extends ends only (0=off)",
    )
    pr.add_argument(
        "--no-display-repair",
        action="store_true",
        help="Force off orphan-rebind display-repair",
    )
    pr.add_argument(
        "--force-display-repair",
        action="store_true",
        help="Force on orphan-rebind display-repair (even on CTC)",
    )
    pr.add_argument(
        "--no-display-pack",
        action="store_true",
        help="Run display-repair but skip local phrase pack",
    )
    pr.add_argument(
        "--keep-parenthetical",
        action="store_true",
        help="Keep full-line (backing) parentheticals as lead targets",
    )
    pr.set_defaults(func=_cmd_run)

    prep = sub.add_parser(
        "repair",
        help="Apply display-repair to an existing run (timed_spine.json or timed.json)",
    )
    prep.add_argument("run_dir", help="Path to out/<run_id>")
    prep.add_argument(
        "--vocals",
        default=None,
        help="Vocal/feature wav (default: work/vocals.wav or asr_audio)",
    )
    prep.add_argument(
        "--no-pack",
        action="store_true",
        help="Skip local phrase pack",
    )
    prep.add_argument(
        "--video",
        action="store_true",
        help="Re-render word-style lyric video after repair",
    )
    prep.set_defaults(func=_cmd_repair)

    pv = sub.add_parser(
        "video",
        help="Render ffmpeg lyric video from an existing run dir (karaoke default)",
    )
    pv.add_argument("run_dir", help="Path to out/<run_id>")
    pv.add_argument("--audio", default=None, help="Override audio path")
    pv.add_argument("--out", default=None, help="Output mp4 path")
    pv.add_argument("--title", default=None)
    pv.add_argument(
        "--video-style",
        choices=("karaoke", "word", "line"),
        default="karaoke",
        help="karaoke=phrase with word advance (default); word=one word; line=CC",
    )
    pv.set_defaults(func=_cmd_video)

    ps = sub.add_parser("score", help="Heuristic scorecard for a run dir")
    ps.add_argument("run_dir", help="Path to out/<run_id>")
    ps.set_defaults(func=_cmd_score)

    pg = sub.add_parser(
        "gate",
        help="Path B sheet-completeness gate → suspect_spans.json (fail-loud / AD_LIB / repeat)",
    )
    pg.add_argument(
        "--vocal",
        required=True,
        help="16k mono vocal stem wav (or any mono wav; resampled)",
    )
    pg.add_argument("--lyrics", required=True, help="Lyric sheet text file")
    pg.add_argument(
        "--mix",
        default=None,
        help="Full mix wav for engineered-splice detector (digital zeros; not stem)",
    )
    pg.add_argument(
        "--out",
        default=None,
        help="Write suspect_spans.json here (default: next to vocal)",
    )
    pg.add_argument(
        "--no-whisper",
        action="store_true",
        help="Skip whisper heard-text (classes degraded)",
    )
    pg.add_argument("--whisper-model", default="medium")
    pg.set_defaults(func=_cmd_gate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
