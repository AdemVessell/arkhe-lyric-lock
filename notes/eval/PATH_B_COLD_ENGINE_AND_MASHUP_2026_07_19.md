# PATH B — Cold engine, mashups, and what Fable should know (2026-07-19)

**Audience:** Grok + Fable (Claude lab) + human.  
**Status:** Strategy + **lab-validated gate** (Claude 2026-07-19) + Grok module port.

---

## PATH B GATE — LAB RESULT (2026-07-19, Claude)

**Source:** `~/Desktop/Arkhē Claude Lab/99_WORKING/LYRIC_LOCK_SPINE_BAKEOFF_20260718/reel/`  
(`reel_gate.py`, `out/suspect_spans.json`)

| # | Result |
|---|--------|
| 1 | **Sheet-completeness VALIDATED** on song C pair: incomplete → FAIL-LOUD 0–17.9s with heard-text naming missing lines; fixed sheet → that span quiet. |
| 2 | Gate also found real holes: **"[uncovered line]"** ~97.4s (song A), **"[uncovered line]"** ~181s (song B). Sheet updated in `fixtures/lyrics/demo_reel.txt`. |
| 3 | **Repeat-matcher works:** 152.5–165.0s = repeat of "song B the same name…" → auto-recoverable, no human. |
| 4 | **AD_LIB / UNTRANSCRIBABLE** class required: span &lt;4s or ≤2 heard words, or empty+energy → **abstain**, NOT fail-loud. v1 over-alarmed on Yeah/Thanks/hum. |
| 5 | **Silence-only auto-seg over-segments** (7 segs / 4 songs). Merge with large-star + judge windows before fusion DP reset. |

**Grok port:** `src/lyric_lock/sheet_gate.py` + CLI `lyric_lock gate --vocal … --lyrics … [--mix …]`

### Sheet-gate review (Claude ACCEPT, 2026-07-19) — applied in Grok

1. **classify_suspect pruned** — unreachable `span_dur` / `energy_frac` branches removed (pre-filter covers min dur + energy).
2. **Repeat match** uses single lines **and adjacent line-pairs** (multi-line vamps).
3. **Engineered splice detector** on **MIX** (digital-zero runs ≥0.1s) → **mandatory** fusion cuts. Validated reel joins: 46.00 / 92.22 / 138.44. Stem forbidden for this (demucs blurs zeros). Self-disables when no zeros (continuous songs).

---

## Two products (do not collapse)

| | **A — Demo montage** | **B — Cold engine (chosen direction)** |
|--|--|--|
| How | Run each song Mode A separately; offset + concat timestamps | One engine, one continuous audio (+ sheet if Mode A), no hand segmentation required |
| Mashup | Presentation layer only | Stress test of generalization |
| Human | Optional polish | OK to step in when judge fails loud — not shame |
| Goal | Looks great for share | Close as possible to zero-runtime human; tweak surface when needed |

**User decision (2026-07-19):** pursue **B** — get as close as possible; human tweak-in is acceptable.

---

## Why individual clips beat the demo reel mashup

Reel: `Desktop/demo_reel.wav`  
(song C → song E V5 → song A → song B; ~46s sung each; dip silence between; slow-mo song E V2 removed.)

| Factor | Singles | Mashup |
|--------|---------|--------|
| Align assumption | 1 song ↔ 1 sheet ↔ continuous time | 4 prods + artificial dips + stitched sheet |
| Sheet quality | Song-specific / iterated | Uneven; song C was incomplete at first |
| Iteration | Ear-gated per track | One-shot compile |
| Stem / energy | Continuous song | Splice edges every ~45s confuse snaps/stars |
| Fusion DP | One neighborhood | Can bleed across boundaries if windows large |
| Eval feel | Local misses | Reads as “whole system regressed” |

**Not the diagnosis:** “CTC forgot song A / song B.”  
**Is the diagnosis:** mashup violates single-performance assumptions; singles were tuned; reel was one cold pass.

### song C blank (concrete bug, fixed once)

- First reel Mode A run: `20260719T014346Z_demo_reel_modeA_stem` (karaoke).
- Star span **0–17.9s** while vocals live → blank screen.
- Cause: sheet only had “[sheet line]”; reel song C starts ~15s into original on **“[sung line]”**.
- Fix: expanded `fixtures/lyrics/demo_reel.txt`; re-align on **same stem** → first word ~1.1s “Hey”.
- Word video: `out/20260719T015609Z_lustre_4_reel_WORD_song Cfix/` (word-by-word, not karaoke).

Karaoke on first reel run was operator default (CLI default), not product doctrine for QA — user prefers **word** for ear check.

---

## Current cold stack (post-convergence)

See also: `docs/CONVERGENCE_2026_07_19.md`, Claude lab  
`~/Desktop/Arkhē Claude Lab/99_WORKING/LYRIC_LOCK_SPINE_BAKEOFF_20260718/REPORT_SPINE_BAKEOFF_2026_07_18.md`.

```
stem → CTC+star (display lead = 0) → whisper fusion (star windows)
     → onset snap → flux boundary snap → judge v0
     → end_snap → display_floor
```

| Item | Status |
|------|--------|
| Lead −0.239s | **RETIRED** (gold spacebar anticipation; pre-waiting) |
| Melisma blob-steal | **DEMOTED** default off |
| CTC MMS_FA | **KEPT** spine |
| Star-per-line | default on |
| Human at runtime | optional polish after fail-loud — goal still autonomy |

---

## Path B — get closer (priority)

1. ~~**Sheet completeness fail-loud**~~ **LAB VALIDATED** + module `sheet_gate.py` (MISSING_LYRICS / REPEAT / AD_LIB).
2. **Soft auto-segment for fusion:** silence dips + large stars + judge windows (`merge_segments_for_fusion`) — silence alone over-segs; use merged for DP reset.
3. **Judge drives compute:** OK / soft→fusion / hard→retry window / still bad→suspect spans + abstain.
4. **Human airlock:** timed.json / SRT / word video remain editable (FCP or timing UI).
5. **Eval for B:** same knobs on (a) held-out singles (b) demo reel reel; measure gap + human minutes on fails.
6. **Wire gate into pipeline** (optional preflight before ship) — next hands-on.

**Do not for B:** concat four hand-tuned Mode A runs and call it generalization (that’s A).

---

## Shareable object? (honest)

| Layer | Shareable now? |
|-------|----------------|
| Doctrine + demotions + notes | Yes |
| Code tree `lyric-lock-v0` | Yes as research tree |
| Standalone OSS package | **No** — no pins, README still Phase-1 Mode B, Desktop paths, dual-lab process |
| Masters / full gold | Private unless stripped |

See chat summary 2026-07-19: process is shareable *research*; not yet fork-clean product.

---

## Key paths (this workstation)

| What | Path |
|------|------|
| Build root | `projects/sovereign-rebuilds/builds/lyric-lock-v0/` |
| Convergence | `docs/CONVERGENCE_2026_07_19.md` |
| This note | `notes/eval/PATH_B_COLD_ENGINE_AND_MASHUP_2026_07_19.md` |
| Reel audio | `Desktop/demo_reel.wav` |
| Reel lyrics | `fixtures/lyrics/demo_reel.txt` |
| Reel run (karaoke, song C blank) | `out/20260719T014346Z_demo_reel_modeA_stem/` |
| Reel run (word, song C fix) | `out/20260719T015609Z_lustre_4_reel_WORD_song Cfix/` |
| Claude bake-off | `Desktop/Arkhē Claude Lab/99_WORKING/LYRIC_LOCK_SPINE_BAKEOFF_20260718/` |

---

## Fable read-along

Read **this note** + `docs/CONVERGENCE_2026_07_19.md` + bake-off REPORT + lab `reel/`.  
Do not re-litigate retired lead or demoted melisma without new ear evidence.  
Gate module: `src/lyric_lock/sheet_gate.py`.  
CLI: `PYTHONPATH=src python3 -m lyric_lock gate --vocal <stem.wav> --lyrics fixtures/lyrics/demo_reel.txt`.  
Next: wire gate preflight into Mode A pipeline; use `segments_merged_for_fusion` for fusion DP reset.
