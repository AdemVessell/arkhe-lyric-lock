# song C stem pass — 2026-07-18

## What we did

1. Admitted **demucs 4.0.1** in `~/ArkheCleanroom/lyric-lock/.venv` (not in product tree)  
2. Separated **htdemucs** two-stem vocals from `song C.wav`  
3. Whisper **medium** on stem → karaoke mp4 on **full mix** audio  

## Comparison

| Run | First word | Words | Coverage | Notes |
|-----|------------|-------|----------|--------|
| Mix medium (prior) | ~31.6s | 49 | ~1.0 | Barely okay body; missing intro |
| Stem + bad decode knobs | 30.0s | 22 | 0.45 | Truncated — knobs harmful |
| **Stem + vanilla decode** | **30.0s** | **51** | **0.82** | Fuller text; **intro still fail** |

## Stem physics (important)

Demucs `vocals.wav` energy:

- **0–14s:** ~−80 dB (effectively empty)  
- **~15s+:** real vocal energy appears  

So if the user hears “singing” in the true front of the **mix**, either:

- it is not isolated as lead vocal by htdemucs, or  
- it is non-vocal melodic content / FX / very soft V that demucs killed  

Whisper on stem still starts ~**30s** even though stem has energy from ~15s — **remaining ASR miss on soft/early vocal.**

## Verdict

**Stem path is infrastructure success, quality partial fail.**

- Helped: cleaner body text vs garbled base; pipeline works  
- Did **not** solve: front singing / lyric-video “good” bar  
- Next real lever: **Mode A (provided lyrics + force align)** and/or better early-window ASR on stem 15–40s merge  

## Artifact

`out/20260718T072230Z_song C_modeB_stem/lyric_video_karaoke.mp4`
