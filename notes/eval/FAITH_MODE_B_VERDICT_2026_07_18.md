# song C Mode B verdict — 2026-07-18

**User judgment:** not good. Front singing missing. Overall merely barely okay.  
**We agree.** This is a **baseline failure** against the lyric-video bar, not a packaging issue.

## What we shipped

| Artifact | Notes |
|----------|--------|
| Whisper medium Mode B timings | First word ~**31.6 s** |
| Karaoke mp4 | Word advance works as *display*; content wrong/missing early |
| Line CC video | Wrong product model (fixed default to karaoke) |

## Failures (Effect track)

1. **Intro miss** — user hears singing before lyrics appear; system silent until ~32 s  
2. **Accuracy only barely okay** where lyrics do appear (Mode B free ASR on full mix)  
3. **Not autonomous lyric-video quality** — would need heavy human cleanup  

## Mechanism evidence

- Full-mix energy is **not** silent in 0–30 s (RMS active from t≈0; louder ~8–30 s).  
- Whisper **medium on forced 0–40 s head** still first-word ~**30.0 s** (`Oh, that I want love…`).  
  → Not only a “full song skip” bug; **medium does not hear usable lyric content early** on this mix.  
- Whisper **base** full run had cues from ~**15.4 s** but garbled text — hears *something*, wrong words.  
- Tweaking no_speech thresholds **did not** fix intro; one re-run got **worse** (coverage collapse).  

## Doctrine check

| Goal | Status |
|------|--------|
| Word-by-word lyric video display | Renderer fixed (karaoke) — necessary, not sufficient |
| Real-time lock to sung vocal | **Failed** on song C Mode B |
| Whisper alone as golden path | **Falsified** for this track |

## What this means for the project

Phase 1 did its job: **Whisper-on-mix Mode B is not the product.**

Next levers (priority order):

1. **Vocal stem** (Demucs-class, cleanroom admit) then re-ASR/align  
2. **Mode A** — provided lyrics + forced alignment (when user is ready)  
3. **Chunked / multi-pass** timing with merge (intro window special-cased only after stem proves signal)  
4. Larger model only *after* stem — large-v3 on same mix may not fix deaf intro  

## Score (honest)

| Dimension | Score |
|-----------|-------|
| Pipeline plumbing (audio→json→srt→mp4) | OK |
| Lyric-video *form* (word lock UI) | OK after karaoke fix |
| Capture of full performance | **Fail** |
| Word accuracy | Barely okay |
| Ready for FCP/autonomous release | **No** |
