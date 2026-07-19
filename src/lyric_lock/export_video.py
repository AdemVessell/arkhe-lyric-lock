from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


class VideoExportError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise VideoExportError("ffmpeg not found on PATH")
    return exe


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\u2019")
        .replace("%", "%%")
    )


def _ass_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _ass_text_escape(text: str) -> str:
    # ASS special chars
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", " ")
    )


def write_ass_from_timed(
    timed: dict[str, Any],
    path: Path,
    *,
    style: str = "karaoke",
) -> Path:
    """
    Burn-in ASS for lyric video.

    style:
      - karaoke (default): phrase stays up; words advance with \\k timing (true lyric-video feel)
      - word: one large word at a time center-screen
      - line: CC-style full line cues (FCP-ish, NOT lyric-video default)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # PrimaryColour = white (active), SecondaryColour = dim gray (upcoming)
    # Karaoke uses primary as "sung" and secondary as "not yet" depending on effect.
    header = """[Script Info]
Title: lyric-lock-v0
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,72,&H00FFFFFF,&H00AAAAAA,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,100,100,140,1
Style: Word,Arial,96,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,5,0,5,80,80,0,1
Style: Line,Arial,56,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,80,80,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    words = list(timed.get("words") or [])
    lines = list(timed.get("lines") or [])

    if style == "line":
        for cue in lines:
            text = _ass_text_escape((cue.get("text") or "").strip())
            if not text:
                continue
            start = float(cue["start"])
            end = float(cue["end"])
            if end <= start:
                end = start + 0.08
            if len(text) > 42:
                mid = len(text) // 2
                sp = text.rfind(" ", 0, mid + 10)
                if sp > 10:
                    text = text[:sp] + "\\N" + text[sp + 1 :]
            events.append(
                f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},Line,,0,0,0,,{text}"
            )

    elif style == "word":
        # One word at a time, large center.
        # Acoustic end from forced-align is truth. Do not hold through
        # inter-phrase silence (no "sit waiting for next line").
        min_readable = 0.12
        ordered = sorted(
            [w for w in words if (w.get("text") or "").strip()],
            key=lambda w: float(w.get("start") or 0),
        )
        for i, cue in enumerate(ordered):
            text = _ass_text_escape((cue.get("text") or "").strip())
            start = float(cue["start"])
            end = float(cue["end"])
            if end <= start:
                end = start + min_readable
            elif end - start < min_readable:
                # only pad ultra-short onsets; never past next word
                end = start + min_readable
            if i + 1 < len(ordered):
                next_start = float(ordered[i + 1]["start"])
                # hard stop before next lyric; silence between phrases = blank screen
                end = min(end, next_start - 0.03)
                if end <= start:
                    end = min(start + 0.05, next_start)
            events.append(
                f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},Word,,0,0,0,,{text}"
            )

    else:  # karaoke — default lyric-video
        by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for w in words:
            by_line[str(w.get("line_id") or "L?")].append(w)

        # Preserve line order by first word start
        ordered_ids = sorted(
            by_line.keys(),
            key=lambda lid: float(by_line[lid][0]["start"]) if by_line[lid] else 0.0,
        )
        for lid in ordered_ids:
            wlist = sorted(by_line[lid], key=lambda w: float(w.get("start") or 0))
            if not wlist:
                continue
            start = float(wlist[0]["start"])
            end = float(wlist[-1]["end"])
            if end <= start:
                end = start + 0.2
            # pad end slightly so last word finish is visible
            end = max(end, start + 0.25)

            parts: list[str] = []
            for i, w in enumerate(wlist):
                wt = _ass_text_escape((w.get("text") or "").strip())
                if not wt:
                    continue
                ws = float(w["start"])
                we = float(w["end"])
                if we <= ws:
                    we = ws + 0.08
                # \\k duration in centiseconds
                k_cs = max(1, int(round((we - ws) * 100)))
                # gap to next word absorbed into this \\k so the phrase stays continuous
                if i + 1 < len(wlist):
                    gap = float(wlist[i + 1]["start"]) - we
                    if 0 < gap < 0.6:
                        k_cs += int(round(gap * 100))
                sep = " " if parts else ""
                parts.append(f"{sep}{{\\k{k_cs}}}{wt}")

            if not parts:
                continue
            # {\\k} karaoke: secondary→primary as each word is hit
            body = "{\\kf0}" + "".join(parts)
            events.append(
                f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},Karaoke,,0,0,0,,{body}"
            )

    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return path


def render_lyric_video(
    audio: Path,
    timed_path: Path,
    out_mp4: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    style: str = "karaoke",
    title: str | None = None,
    # backwards-compat
    word_level: bool | None = None,
) -> Path:
    """
    ffmpeg lyric plate: animated dark field + ASS burn-in + source audio.

    Default style is karaoke (word-by-word within phrases) — lyric video, not CC.
    """
    if word_level is True:
        style = "word"
    elif word_level is False and style == "karaoke":
        # explicit old flag False meant line mode
        pass

    ffmpeg = require_ffmpeg()
    audio = audio.expanduser().resolve()
    timed = json.loads(timed_path.read_text(encoding="utf-8"))
    out_mp4 = out_mp4.expanduser().resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    ass_path = out_mp4.with_suffix(".ass")
    write_ass_from_timed(timed, ass_path, style=style)

    ass_esc = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    label = _escape_drawtext(title or Path(timed.get("audio", {}).get("path", "lyrics")).stem)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio),
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={fps}",
        "-filter_complex",
        (
            f"[1:v]format=yuv420p,"
            f"geq=r='14+10*sin(X/100+T*0.35)':g='10+8*sin(Y/120+T*0.25)':b='32+14*sin((X+Y)/160+T*0.45)'[bg];"
            f"[bg]drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial.ttf:"
            f"text='{label}':fontsize=34:fontcolor=white@0.5:"
            f"x=(w-text_w)/2:y=80[titled];"
            f"[titled]ass='{ass_esc}'[v]"
        ),
        "-map",
        "[v]",
        "-map",
        "0:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Fallback: word SRT if present, else line SRT
        for name in ("lyrics.words.srt", "lyrics.srt"):
            srt = timed_path.parent / name
            if srt.is_file():
                break
        else:
            raise VideoExportError(f"ffmpeg failed:\n{proc.stderr[-3000:]}")
        srt_esc = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        cmd2 = [
            ffmpeg,
            "-y",
            "-i",
            str(audio),
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x101018:s={width}x{height}:r={fps}",
            "-filter_complex",
            f"[1:v]subtitles='{srt_esc}':force_style='Fontsize=48,PrimaryColour=&H00FFFFFF,Outline=3,Alignment=5'[v]",
            "-map",
            "[v]",
            "-map",
            "0:a",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
        if proc2.returncode != 0:
            raise VideoExportError(
                "ffmpeg lyric video failed (primary + fallback):\n"
                f"--- primary ---\n{proc.stderr[-2000:]}\n"
                f"--- fallback ---\n{proc2.stderr[-2000:]}"
            )
    if not out_mp4.is_file() or out_mp4.stat().st_size < 1000:
        raise VideoExportError(f"output missing or tiny: {out_mp4}")
    return out_mp4
