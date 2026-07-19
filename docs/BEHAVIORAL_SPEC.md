# Behavioral Spec — lyric-lock-v0

## Primary behavior

Given a path to audio containing singing, produce a portable timed-lyric package on the **media timeline** (seconds).

### Mode B (active first)

| Input | Output |
|-------|--------|
| Audio file | `timed.json` (canonical) |
| (no lyrics file) | `lyrics.srt` (FCP / universal) |
| | `run_meta.json` (provenance) |
| | optional work WAV in `out/` |

Later: Mode A adds optional `lyrics.txt` forced alignment path.

## timed.json schema (v0)

```json
{
  "schema": "arkhe.lyric_lock.timed/v0",
  "mode": "B",
  "audio": { "path": "...", "sha256": "...", "duration_s": 0.0 },
  "engine": { "name": "whisper", "model": "large-v3", "word_timestamps": true },
  "language": "en",
  "text": "full transcript...",
  "lines": [
    { "id": "L0", "start": 1.23, "end": 4.56, "text": "line text", "confidence": null }
  ],
  "words": [
    { "id": "W0", "start": 1.23, "end": 1.50, "text": "word", "line_id": "L0", "confidence": null }
  ],
  "needs_review": []
}
```

## SRT / video behavior

- **`lyrics.srt`**: line cues for FCP / CC import (NLE handshake)  
- **`lyrics.words.srt`**: word cues (debug / alternate)  
- **`timed.json` words**: canonical for **lyric video** (word-by-word lock)  
- **ffmpeg lyric video default**: **karaoke** style — phrase on screen, words advance in time  
  - not closed-caption sentence blocks  
  - alternate: `word` (one word center) or `line` (CC)  
- Times are wall-clock on the source audio (no beat-snapping in Phase 1)

## Acceptance (Phase 1)

1. CLI completes on a Desktop fixture without provided lyrics  
2. SRT imports into Final Cut Pro (File → Import → Captions)  
3. Human ear-lock score recorded in notes (1–5)  
4. No third-party app source vendored into `src/`  

## Later acceptance (not Phase 1)

- Tempo-map-assisted duration for held vowels  
- Mode A lyrics-given alignment  
- ITT export  
- Frontier multi-view repair loop  
