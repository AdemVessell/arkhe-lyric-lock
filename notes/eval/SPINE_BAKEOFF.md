# SPINE BAKE-OFF — CTC (MMS_FA) vs stable-ts — 2026-07-18

**Operator:** Claude (Fable 5), Arkhē Claude Lab. Read-only against `Arkhē_Grok/…/lyric-lock-v0`; all writes here.
**Lever:** raw spine swap only. No display repair, no edge physics, no rebind, nothing tuned on gold.
**Ruler:** `human_gold_APPROVED_PARTIAL.json` (29 words, song A 0–90s, user ear QA'd).

## Verdict

**CTC forced alignment (torchaudio MMS_FA, on the existing demucs vocal stem) decisively beats the stable-ts spine.**
Bias-corrected onset scatter drops from **540ms → 92ms MAE** (raw spine vs raw spine). It also beats the fully
repaired shipped best (317ms) with zero repair layers. And it runs in **7.6s on CPU** for 90s of audio
(vs minutes for Whisper medium) — iteration speed changes class.

Second finding: **both spines independently converge on a ~+235ms median offset vs gold**
(repaired +233ms, CTC +239ms, stable-ts +311ms). Two unrelated aligners agreeing on the same constant
says the offset lives in the **gold capture / display-lead preference**, not in either aligner.
It is one calibration number, not an alignment failure.

## Scoreboard (29/29 matched, all spines, identical sequential matcher)

| metric | stable_ts_raw | gold_run_repaired | **ctc_mms_fa** |
|---|---|---|---|
| median signed onset | +311ms | +233ms | **+239ms** |
| MAE onset (raw) | 669ms | 407ms | **283ms** |
| **MAE onset (bias-corrected)** | 540ms | 317ms | **92ms** |
| MAE end | 460ms | 636ms | 445ms |
| onset ≤80ms (bias-corr) | 7/29 | 10/29 | **18/29** |
| onset ≤150ms (bias-corr) | 9/29 | 16/29 | **23/29** |
| onset ≤250ms (bias-corr) | 16/29 | 21/29 | **27/29** |

Worst-case CTC outlier: +701ms (`way`, third motif). Worst-case stable-ts: +2505ms (same motif) and
−1549ms (`even`). CTC's failure mode is mild lateness; stable-ts's is multi-second displacement in both directions.

## Method

- Engine: `torchaudio.pipelines.MMS_FA` (wav2vec2 CTC, 1.18GB weights, one-time download from
  `dl.fbaipublicfiles.com`) + `torchaudio.functional.forced_align` + `merge_tokens`. System Python,
  torchaudio 2.8.0, CPU.
- Input: `vocals_16k_mono.wav` (the demucs stem already produced by run `20260718T090143Z`), and the
  same `lyrics_provided.txt` the stable-ts spine used. Same words in, same words out.
- Gotcha logged: MMS_FA's dictionary exposes the CTC blank `-` as a real key; hyphens in
  `(Oh-oh-oh-oh)` must be filtered or `forced_align` rejects the targets.
- Scoring: greedy in-order text match, identical for every spine. Bias correction = subtract each
  spine's own median signed error; nothing fitted beyond that single median.

## What the per-word table shows

- CTC deltas are **tightly clustered** around +240ms across the whole slice — verse, motif chain, chorus.
- The `[repeated phrase] ×3` motif that broke FA (+2.0–2.5s) is **fixed** in instances 1–2 (+127…+325ms)
  and merely late-ish in instance 3 (+418…+701ms).
- The phrase-initial EARLY disasters (`even` −1549, `we're` −931, `So` −869, `Doesn't` −957) are **gone** —
  CTC places all four onsets +208…+593ms, same sign, correctable by the constant.
- The long `made` hold (6.14→9.52s) is boxed as a 3.4s duration event — consistent with the ledger
  claim that long holds become accurate when the aligner is real. Watch test will confirm.

## Caveats (hostile, on my own result)

1. **One song, one 90s slice, 29 gold words.** Same overfit trap the repair layers fell into.
   Nothing here is "default" until a second song's gold slice reproduces both the scatter win and
   the ~240ms constant.
2. The −239ms calibration constant was measured **on this gold**. Using it as a product constant is
   legitimate only if it is stable across songs (as gold-capture latency should be), not per-song fitted.
3. **Ends are still weak** (445ms MAE): CTC snaps to phoneme boundaries; held-vowel decay and HANG/FLASH
   remain an end-physics problem. Note: energy-based end repair was demoted *on a bad spine*; on a
   92ms spine it is a fresh, untested experiment — not covered by the DEMOTE doctrine's negative result.
4. Residual 92ms scatter is near the noise floor of spacebar-captured gold itself. The ruler now needs
   the drag-trim nudge pass before it can measure further progress.

## Artifacts

| File | Role |
|---|---|
| `out/ctc_spine.json` | Raw CTC spine (123 words, timed/v0-compatible) |
| `out/ctc_spine_LEAD240.json` | CTC spine with single global −239ms shift (only lever applied) |
| `out/ctc_word_RAW.mp4` | Watch test A — raw CTC, word style, full-mix audio |
| `out/ctc_word_LEAD240.mp4` | Watch test B — lead-shifted CTC |
| `out/scoreboard.json` | Full summary + per-word deltas, all three spines |
| `ctc_align.py`, `score_spines.py` | Reproducible; no lyric-lock code modified |

## EAR QA RESULT (user, 2026-07-18, both videos watched)

**Gate passed.** User transcribed both videos word-by-word through ~47s (44 words — beyond the 29-word gold).

| | RAW | LEAD240 |
|---|---|---|
| flagged words | 11 | **5** |
| flags removed by the −239ms shift | — | even, after, all, on, our, way (all "late") |

Every **late** flag disappears under the single constant → **calibration confirmed by ear**, and the
constant holds on words *outside* the gold slice (30–44). Onset class: solved on this song.

**Surviving flags are all duration/end class, none onset:**

| flag | spine box | diagnosis |
|---|---|---|
| made #5 "hangs" | 5.90–9.28 (3.38s) | phrase-final absorb: CTC stretches last word into inter-phrase space |
| way #1 "short" | 14.40–14.86 (0.46s) | held word cut early (opposite direction) |
| it "slightly off", all "lilts" | 0.14s boxes | dense-phrase micro-jitter, ≤150ms class |
| around "too short" | 46.69–47.61 | hold stolen by following "(Oh-oh-oh-oh)" backing token |
| mean (omitted in user's transcript, both files) | 37.69–37.81 (**0.12s**) | FLASH — invisible at 30fps; "the" next door is 0.08s |

Same absorb pattern, unflagged (QA stopped ~47s): way #2 = 3.72s, "way)" = **11.6s** (absorbed the
instrumental break). Also: words 80–122 (second-chorus tail) are compressed into 85–90s with
0.05–0.12s boxes — the 90s audio slice ends mid-song while the lyric text continues, so CTC packs the
leftover text at the edge. Slice-tail artifact, must be handled (trim targets to slice or star token).

## CROSS-SONG PASS — song B (lustre V1, full 166.7s) — 2026-07-18/19

User ear on song 2: first full section perfect, wreck begins at the held-out "[held phrase]" —
next lyrics advance erroneously. Diagnosis (energy islands + Mode B ASR cross-check, `song_b/`):

1. **Melisma advance** (root cause A): third "[held phrase]" is sung stretched over ~7s
   (Mode B hears: I 55.1 / hope 57.6 / you 61.1 / know 61.5). CTC Viterbi sprints through held
   vowels (each remaining token costs 1 frame) and fires "Now we go" ~9s early. Star tokens do NOT
   fix this — wildcard emission is cheap everywhere, so the path still advances fast and dumps the
   remainder into star.
2. **Sheet-vs-performance mismatch** (root cause B): performance vamps an extra "Now we go" (~88–90s)
   plus ad-libs not in the lyric text. A monotonic aligner *must* misspend tokens on off-sheet audio.

**Star-per-line variant** (`ctc_align.py --star`, MMS_FA with_star):

| effect | evidence |
|---|---|
| Made-well regression | **none** — med +239 identical, scatter 89ms vs 92ms, ≤150ms 23/29 unchanged |
| Phrase-final hangs **fixed free** | made 3.38s→0.54s, way#2 3.73s→0.42s, "(way)" 11.6s→0.12s — stars eat the absorb |
| Sheet-mismatch region | **honest abstention**: star span 64–94.8s = blank screen instead of 30s of wrong words |
| Star spans as instrument | each active span is an automatic "unresolved audio" flag → where human anchors go |
| Not fixed | melisma advance (above), FLASH words, slice-tail confetti, held-word-cut-short |

**Melisma rescue — Whisper anchor fusion (v2, built same night).** User ear rejected the star-only
video ("not good in the long parts"). Key discovery from re-ASR of the wreck window (44–130s,
whisper medium on stem, `song_b/out/wreck_asr_words.json`):

- The lyric sheet was RIGHT all along (3 reps + "Now we go"; the "extra Now we go at 88s" was a
  90s-slice artifact garbling "song B" — corrected). The wreck is pure melisma: "I" held 2.7s,
  "know" held 3.8s / 4.2s.
- **Whisper's word timestamps track melisma holds correctly** — the exact regime where CTC Viterbi
  sprints. The engines fail in opposite directions.

Mechanism (`song_b/fuse_anchors.py`): CTC star spine everywhere + inside the flagged window,
sheet tokens ↔ Whisper words via monotonic DP alignment (char-similarity substitution, gaps allowed);
matched tokens take Whisper timing; "(...)" backing tokens always abstain; Whisper hallucinations
("Brexit", "LänderTV" in the outro instrumental) match no sheet token → auto-rejected. Result:
23/23 lexical words fused; timeline now mirrors the performance (reps with real holds, Now-we-go at
67.6, 20s instrumental blank, We-shared-the-same-way at 90.5). Greedy matching v1 failed (a "(oh)"
consumed the "hope" anchor and cascaded) — DP required.

Tier structure this establishes for hard regions: **1) CTC (confident) → 2) Whisper anchors
(melisma, flagged by star spans) → 3) human taps (only if both fail)**. Autonomy preserved;
the machine plays every card before asking for the user's hands.

## AUTONOMY TURN — 2026-07-19

User ear on adaptive-lead video: "decent, not a pass — things sit pre-waiting." Verdict: earliness
relative to the voice (not off-rhythm). User restated the goal: **zero human at runtime.**

**Lead-constant autopsy:** the −239ms was derived from spacebar gold on songs the user knows —
it captured *anticipation*, not display preference. Enshrined too fast; ear caught it. Doctrine:
display target is the sung onset. Three clocks distinguished: capture (taps, early) / acoustic
(consonant) / perceptual hit (vowel). Constants must come from the signal, not from taps.

**Lever A (`ctc_star_FUSED_SNAP.json` / `song_b_SNAP.mp4`):** zero display lead + onset snap —
words whose boxes start in silence advance to first stem-energy rise (4 words snapped: I'm +400ms,
Now +199ms, We +79ms, I +80ms). Whisper's early lean across gaps corrected mechanically.

**Lever B — Judge v0 (agreement-as-certifier, replaces the human tier):**
Features per word: CTC confidence + energy support − star adjacency. Tested against existing ear labels:

| test | result |
|---|---|
| song B: wreck region vs passed verses | median judge −3.25 vs −1.77 — **clean separation** |
| Made-well: his 5 word-level flags vs clean | −1.27 vs −0.70 — right direction, **weak** (flags were END defects; features are onset-trust) |

Conclusion: region-level self-judging is real now (drives retry/fail-loudly); word-level end-defect
detection needs end features (|box_end − energy_death|, |box_dur − island_dur|) → judge v0.1.

**Runtime tier doctrine (human removed):** CTC → whisper-anchor fusion → self-judge → retry with
more compute (ensemble decodes, re-separation, star-span repeat-matching) → **fail loudly, don't ship**.
Human ear remains only in the development loop until the judge reaches parity, then retires.

## PATH B — GATE BUILD + GROK MODULE REVIEW (2026-07-19, `reel/`)

**Gate built and validated** (`reel/reel_gate.py`, `reel/out/suspect_spans.json`): song C pair passes
(incomplete sheet → FAIL-LOUD 0–17.9s naming the missing lines; fixed sheet → quiet). Bonus: found
two more genuine sheet holes ("[uncovered line]" 97.4s, "[uncovered line]" 181s) + repeat-matcher correctly
auto-recovered the song B vamp (sim 0.68). Refinement fed to Grok: AD_LIB/UNTRANSCRIBABLE class.

**Grok's `sheet_gate.py` review:** song Cful port, AD_LIB class + 3-word override + segment merge all in.
Findings: (a) two dead-code branches in `classify_suspect` (span_dur clause, energy_frac fallback —
unreachable); (b) repeat-match compares single sheet lines only — multi-line vamps can dip <0.55 and
misclassify as MISSING; match line-pairs too. (c) Smoke test: merged segmentation 7→6 segments but
window 62.3–97.4 still straddles the ~92s song join → fusion DP bleed risk remains.

**Splice detector (closes c):** the reel's joins are digital-zero runs in the MIX — exactly 0.22s at
46.00 / 92.22 / 138.44 (all three joins, centisecond-exact). Zero-runs ≥0.1s on the mix = mandatory
segment cuts; detector self-disables on continuous audio (no natural zeros). Stem is the wrong signal
for splice detection (demucs blurs the zeros; stem zeros also occur at vocal rests).

## Recommended next levers (post-QA order, one at a time)

1. ~~User watch test~~ **DONE — passed.** Onset class solved (this song); survivors are end/duration class.
2. **End-snap to vocal energy** (single mechanism, both directions): phrase-final ends trimmed at stem
   energy death (fixes made-hang / way#2 / way-paren absorb), held-word ends extended to energy death
   (fixes way#1, around). This is energy repair *on a good spine* — new experiment, not covered by
   DEMOTE_REBIND's negative result, which was spine-rescue.
3. **Display floor** (~0.25–0.30s min on-screen, display layer only, never move starts): kills the
   mean/the FLASH class.
4. **Slice-tail guard**: align only the lyric text the slice actually sings (or MMS_FA star token for
   overflow); refuse to pack leftover text into the last seconds.
5. **Backing/parenthetical class**: `(...)` lines excluded from lead-vocal targets or styled separately.
6. **Cross-song validation**: second song's gold slice — scatter + constant stability — before default.
7. **Promote CTC engine into `lyric_lock`** (Grok's tree, Grok's call); stable-ts demotes to variant check.
