# LYRIC-LOCK CONVERGENCE — 2026-07-19

Sync of Grok tree (`lyric-lock-v0`) to Claude lab state under  
`~/Desktop/Arkhē Claude Lab/99_WORKING/LYRIC_LOCK_SPINE_BAKEOFF_20260718/`.

## Ear-gated doctrine changes

| Item | Status | Reason |
|------|--------|--------|
| Display lead **−0.239s** | **RETIRED** | Autopsy: spacebar anticipation in gold (user knows songs), not display preference. Caused pre-waiting. **Display target = sung onset, zero lead.** |
| Melisma blob-steal v3 (`153633Z`) | **DEMOTED** (default off) | User ear: section wrecked. |
| CTC MMS_FA port (`150114Z`) | **KEPT** as spine engine | Bias-corrected onset class win stands. |

## Adopted stack (Claude lab)

1. **Star-per-line CTC** (`ctc_align.py`, `star=True` default)  
   MMS_FA `with_star`, `*` between lyric lines. Phrase-final hangs eaten; active star spans = honest abstention instruments. Made-well: 89ms vs 92ms (no regression).

2. **Whisper anchor fusion v2** (`whisper_fusion.py`)  
   In star-flagged windows: DP-match sheet tokens ↔ whisper-medium-on-stem words; matched take whisper timing; unmatched abstain; whisper hallucinations auto-rejected. Greedy matching cascades — DP required.

3. **Onset snap** (`onset_flux_snap.py`)  
   Words starting in silence advance to first stem RMS ≥ 0.12 within +1.0s.

4. **Flux boundary snap** (`onset_flux_snap.py`)  
   Transitions off held words (≥0.8s, contiguous) snap to spectral-flux peak. Forward-only; held word keeps screen until successor onset.

5. **Judge v0** (`judge_v0.py`)  
   `conf + 2*energy_support − 1.5*star_adj`. Region-level validated (wreck −3.25 vs verses −1.77). Annotates words; not sole ship gate.

## Product pipeline order (Mode A CTC)

```
stem → CTC+star (lead=0) → whisper fusion (star windows)
     → onset snap → flux snap → judge v0
     → end_snap → display_floor
     → (melisma OFF)
```

## CLI

| Flag | Default |
|------|---------|
| `--display-lead` | `0.0` (omit for default) |
| star | on (`--no-ctc-star` off) |
| fuse anchors | on (`--no-fuse-anchors` off) |
| onset / flux snap | on |
| judge | on |
| melisma | **off** (`--force-melisma-hold` only) |

## Reference artifacts (Claude lab, do not re-pull)

- Made-well: `…/LYRIC_LOCK_SPINE_BAKEOFF_20260718/out/ctc_spine_star.json`
- song B FLUX stack: `…/song_b/out/ctc_star_FUSED_FLUX.json` + `song_b_FLUX.mp4`
- Fusion recipe source: `…/song_b/fuse_anchors.py` (+ session one-shots for snap/flux/judge)

## Three clocks

1. **Capture** — taps / spacebar (early on known songs)  
2. **Acoustic** — consonant / energy onset  
3. **Perceptual** — vowel hit  

Constants must come from the **signal**, never from taps alone.
