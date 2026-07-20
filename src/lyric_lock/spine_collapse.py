"""Spine-collapse detector — runs BEFORE the sheet gate.

The gate answers "does the sheet cover the sung audio?" by looking at where the
spine placed words. That question is only meaningful if the spine is sane. When
forced alignment loses the plot it does not leave a hole — it *smears*, stretching
a few words across tens of seconds and crushing the rest into sub-second boxes.
Smear looks like coverage. So a collapsed spine walks straight past fail-loud.

Observed on "Straight Ghostin" (2026-07-19): single-pass CTC broke at ~52s and
crammed 130 words into the remaining audio, with "straight, narrow" holding
91.6→131.4s on its own. The gate reported exactly one 2.2s finding and missed
both real coverage holes, because the smear appeared to cover them.

This module looks at the spine alone — no ASR, no ground truth, no ear — and
reports the shapes a healthy spine does not have.

Signals
-------
stretched   a word holding far longer than the song's own typical word
crushed     a run of words at/below the floor, packed shoulder to shoulder
repeat_drift  when the sheet repeats a block (chorus), the internal rhythm of
            each instance should match. Onsets measured relative to each
            instance's own start should agree across instances. This one is the
            sharpest: it needs no thresholds tuned to a genre, only the sheet's
            own repetition, and it caught the failure above instantly.

Thresholds are multiples of the song's own median, not absolute seconds — a
ballad and a rap do not share a scale.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _norm(t: str) -> str:
    return "".join(c for c in t.lower() if c.isalnum() or c == "'")


def find_repeat_blocks(
    words: list[dict[str, Any]], *, min_lines: int = 4
) -> list[list[list[dict[str, Any]]]]:
    """Group repeated line-sequences (choruses) by their text.

    Returns a list of blocks; each block is a list of instances; each instance
    is the list of its words. Only blocks appearing 2+ times are returned.
    """
    lines: list[tuple[str, list[dict]]] = []
    for w in words:
        lid = w.get("line_id")
        if lines and lines[-1][0] == lid:
            lines[-1][1].append(w)
        else:
            lines.append((lid, [w]))

    sigs = [" ".join(_norm(w["text"]) for w in ws) for _, ws in lines]
    by_sig: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sigs):
        by_sig[s].append(i)

    # seed on the first repeated line, then extend the run while it keeps matching
    seeds = [idxs for s, idxs in by_sig.items() if len(idxs) > 1 and s]
    if not seeds:
        return []
    blocks: list[list[list[dict]]] = []
    used: set[int] = set()
    for idxs in sorted(seeds, key=lambda v: v[0]):
        if idxs[0] in used:
            continue
        span = 1
        while True:
            nxt = span + 1
            ok = all(
                i + nxt <= len(sigs)
                and sigs[i : i + nxt] == sigs[idxs[0] : idxs[0] + nxt]
                for i in idxs
            )
            if not ok:
                break
            span = nxt
        if span < min_lines:
            continue
        for i in idxs:
            used.update(range(i, i + span))
        blocks.append(
            [[w for _, ws in lines[i : i + span] for w in ws] for i in idxs]
        )
    return blocks


def detect(
    words: list[dict[str, Any]],
    *,
    stretch_factor: float = 8.0,
    crush_floor_s: float = 0.10,
    crush_run: int = 6,
    drift_tol_s: float = 0.35,
    drift_frac: float = 0.34,
) -> dict[str, Any]:
    """Report collapse signals. Never raises on shape — returns findings."""
    findings: list[dict[str, Any]] = []
    words = [w for w in words if w.get("start") is not None]
    if len(words) < 12:
        return {"collapsed": False, "findings": [], "note": "too few words"}

    durs = np.array([float(w["end"]) - float(w["start"]) for w in words])
    med = float(np.median(durs)) or 0.01

    for w, d in zip(words, durs):
        if d > med * stretch_factor:
            findings.append(
                {
                    "kind": "stretched",
                    "span": [round(float(w["start"]), 3), round(float(w["end"]), 3)],
                    "text": w["text"],
                    "detail": f"{d:.1f}s is {d/med:.0f}x the median word ({med:.2f}s)",
                }
            )

    run: list[dict] = []
    for w, d in zip(words, durs):
        if d <= crush_floor_s:
            run.append(w)
            continue
        if len(run) >= crush_run:
            findings.append(
                {
                    "kind": "crushed",
                    "span": [
                        round(float(run[0]["start"]), 3),
                        round(float(run[-1]["end"]), 3),
                    ],
                    "text": " ".join(x["text"] for x in run[:8]),
                    "detail": f"{len(run)} consecutive words at/below {crush_floor_s}s",
                }
            )
        run = []
    if len(run) >= crush_run:
        findings.append(
            {
                "kind": "crushed",
                "span": [
                    round(float(run[0]["start"]), 3),
                    round(float(run[-1]["end"]), 3),
                ],
                "text": " ".join(x["text"] for x in run[:8]),
                "detail": f"{len(run)} consecutive words at/below {crush_floor_s}s",
            }
        )

    repeat_checked = 0
    for block in find_repeat_blocks(words):
        if len(block) < 2:
            continue
        # onset of each word relative to its own instance start
        rel = []
        for inst in block:
            base = float(inst[0]["start"])
            rel.append([float(w["start"]) - base for w in inst])
        n = min(len(r) for r in rel)
        if n < 6:
            continue
        repeat_checked += 1
        spreads = [max(r[i] for r in rel) - min(r[i] for r in rel) for i in range(n)]
        bad = [i for i, s in enumerate(spreads) if s > drift_tol_s]
        if len(bad) > n * drift_frac:
            worst = int(np.argmax(spreads))
            findings.append(
                {
                    "kind": "repeat_drift",
                    "span": [
                        round(min(float(i[0]["start"]) for i in block), 3),
                        round(max(float(i[-1]["end"]) for i in block), 3),
                    ],
                    "text": " ".join(w["text"] for w in block[0][:6]),
                    "detail": (
                        f"{len(bad)}/{n} words drift >{drift_tol_s}s across "
                        f"{len(block)} repeats of this block "
                        f"(worst {spreads[worst]:.1f}s on {block[0][worst]['text']!r})"
                    ),
                }
            )

    collapsed = any(f["kind"] in ("stretched", "repeat_drift") for f in findings)
    return {
        "collapsed": bool(collapsed),
        "findings": findings,
        "median_word_s": round(med, 4),
        "repeat_blocks_checked": repeat_checked,
        "thresholds": {
            "stretch_factor": stretch_factor,
            "crush_floor_s": crush_floor_s,
            "crush_run": crush_run,
            "drift_tol_s": drift_tol_s,
            "drift_frac": drift_frac,
        },
    }
