# lyric-lock-v0 — capability ledger (not lost in iteration)

What is **in the object** (code + docs), not just session chat.

## Durable capabilities (as of 2026-07-18)

| Capability | Where | Status |
|------------|--------|--------|
| Mode B free ASR timing | `whisper_mode_b.py`, `pipeline.py` | Working baseline |
| Vocal stem (Demucs cleanroom) | `separate.py`, `~/ArkheCleanroom/lyric-lock` | Admitted + wired |
| Mode A **forced align** (known lyrics → acoustic word times) | `forced_align.py` (stable-ts align+refine) | **Core precision path** |
| **Tempo map + energy duration rescore** (user doctrine) | `tempo_map.py`, `duration_rescore.py` | **Your layer** — BPM/beats as real-time ruler + vocal energy for holds/silence |
| Legacy fuzzy Mode A (deprecated) | `mode_a.py` | Kept only for A/B; not default |
| Word-at-a-time lyric video | `export_video.py` `style=word` | Primary display |
| Karaoke phrase+word advance | `export_video.py` `style=karaoke` | Optional style |
| Line SRT for FCP | `export_srt.py` | NLE handshake |
| Scorecard heuristics | `score_run.py` | Draft quality flags |
| Fixtures + your lyrics | `fixtures/catalog.json`, `fixtures/lyrics/` | song D gold text |
| Doctrine / display research | `docs/`, `notes/LYRIC_VIDEO_DISPLAY_PATTERNS.md` | Recorded |

## Learning locked in (product)

1. Free Whisper on mix ≠ lyric-video timing product  
2. Stem helps signal; does not invent lead vocal where demucs empties intro  
3. **Fuzzy paste + interpolate causes race/lilt** — rejected for Mode A default  
4. **Forced align is the duration engine** (start/end of each sung word)  
5. Display must not hold words through **lyrical silence**  
6. Long holds can be accurate when aligner is real (seen on prior karaoke / FA)  

## Default recommended path

```text
audio + lyrics.txt
  → demucs vocals (cleanroom)
  → stable-ts forced align + refine
  → timed.json (word start/end)
  → word-at-a-time video (no silence linger)
  → also lyrics.srt for FCP
```

## Still open (polish)

- Sub-100ms onset exactness on hard consonants  
- Trailing words sometimes fail FA at song end  
- Tempo map as optional prior (not aesthetic scoring)  
- Windowed FA for weak intro regions  
