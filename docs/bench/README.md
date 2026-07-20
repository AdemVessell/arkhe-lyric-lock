# lyric-lock repair bench (v1)

Human airlock for the autonomous pipeline. Loads a run's `timed.json` + the song
audio, shows where the machine was unsure, and makes corrections take seconds.

**Local only.** Single HTML file, no server, no network. Audio never leaves the
machine (masters stay private). Open `index.html` in Safari or Chrome.

## Load

| Field | What |
|---|---|
| audio | the song `.wav` (mix or stem) |
| timed.json | from any run dir, e.g. `out/<run_id>/timed.json` — carries the judge scores |
| suspect_spans | same run dir — the gate's region findings (fail-loud / abstain / repeat) |

Cold-eval runs to try it on:
`out/20260719T042939Z_make_it_all_align_modeA_stem/timed.json` (+ `~/Desktop/make it all align.wav`)

## Read the display — post-run findings surface automatically

- **Word chips** colored by the run's own `judge_v0` scores, calibrated **per song
  by percentile** (worst 15% red, next 20% amber). Fixed cutoffs do not transfer:
  in the 2026-07-19 cold eval, judge medians ran +0.72 (align) to −2.25 (mars),
  so one absolute threshold flagged 3% of one song and 36% of another. Percentiles
  always mean "the worst of *this* song".
- **Waveform bands** show the sheet gate's regions: red = FAIL-LOUD (sung audio
  with no lyric coverage), amber = abstain (ad-lib / untranscribable), green =
  auto-recovered repeat. Hover text shows what the machine *heard* there.
- **next flag** <kbd>F</kbd> walks every finding in time order — judge-flagged words
  and gate regions merged — selecting each and cueing playback 1s before it.
  Auto-recovered repeats are skipped (nothing to fix).
- **Summary line** counts everything: words, judge flags, gate spans, thresholds used.
- **Live stage** under the timeline shows the current word large and in time — exactly what the
  video will display — coloured by the machine's own confidence, and reading `(silence)` when
  nothing should be on screen. If it says `(silence)` while you hear singing, that is a miss.
- **Onset ticks** appear beneath the waveform once you zoom in past ~14 s of span: the actual
  detected attacks. A word edge that doesn't sit on a tick is the error, visible at a glance.

Workflow: press <kbd>F</kbd>, listen, fix or accept, press <kbd>F</kbd> again. You
only ever review what the machine doubts.

## The loop, in one sentence

**Drag to select a part → space plays just that part on repeat → hold `J` through each word
as you hear it.** Click anywhere to let the selection go.

## Repair tools

| Tool | Use |
|---|---|
| **zoom** | mouse wheel at the cursor · `+` / `−` · `0` fits the whole song · "zoom to selection" |
| **pan** | alt-drag, or shift-wheel. Playback auto-scrolls the view when zoomed in |
| **move a word** | drag its block in the lane |
| **top and tail** | drag a block's left/right edge handle. Edges snap to detected onsets within 100 ms — hold shift to place freely |
| drag on waveform | select a region (or click a chip; shift-click for a range) |
| **chunk move** | select several words, then drag inside the selection — the whole group moves together, keeping lengths and gaps |
| **add a word** | `A` (or the + word button) inserts at the playhead; double-click empty timeline also adds there |
| **delete** | `⌫` removes the selected words |
| **rename** | double-click a word chip or its block |
| **capture** <kbd>C</kbd> | hold <kbd>space</kbd> through each word as you hear it — taps snap to the nearest detected onset, so rough timing is fine |
| nudge ±20 / ±100ms | shift selected words |
| ripple: on | a nudge also shifts everything after it (fixes "the rest of the verse is late") |
| snap starts to onsets | pull selected word starts onto detected onsets (±450ms search; leaves anything further alone rather than guessing) |
| lyrics box | edit the words in the region and apply — spreads them across the region span |
| speed 0.5×/0.75× | slow playback for dense passages (the fast-rap case) |
| <kbd>⌘Z</kbd> | undo (60 levels) |

**Tap latency** slider subtracts your reaction time before snapping. Default 90ms.
This is the one setting that matters for capture accuracy — if your captures land
consistently late, raise it. (Onset snap already absorbs most of it: in testing,
taps 130ms late landed within 10ms of the true onset.)

## Export

- **timed.json** — repaired, same schema; feed back to `lyric_lock video` to re-render
- **srt** — word-level captions for Final Cut (File → Import → Captions)

## Doctrine

The bench exists for the residual the machine honestly flags — not as the primary
path. If you find yourself repairing most of a song, that is an engine result to
record, not a workflow. Capture never becomes ground truth for constants: taps
carry anticipation (see three-clocks note in the bake-off report), which is why
they snap to signal onsets instead of being written raw.
