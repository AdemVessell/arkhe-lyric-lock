# Tool Admission — lyric-lock-v0

Workstation trust: only admitted tools may be invoked by the orchestrator.

## Admitted (Phase 1)

| Tool | Version / location | Role | Notes |
|------|-------------------|------|-------|
| ffmpeg / ffprobe | Homebrew `/opt/homebrew/bin` | decode, mono 16 kHz work WAV | Already present |
| openai-whisper | pip `20250625` | Mode B transcription + word timestamps | Already present; Python API |
| Python 3 + torch | system / user | run Whisper | Already present |
| stdlib | — | hash, json, pathlib, argparse | Owned path |

## Admitted (Phase 2 — stem path)

| Tool | Version / location | Role | Notes |
|------|-------------------|------|-------|
| demucs | PyPI `demucs==4.0.1` in `~/ArkheCleanroom/lyric-lock/.venv` | Vocal stem (`htdemucs`, two-stems=vocals) | **Not** vendored into `src/`; orchestrator shells to cleanroom Python only |

## Admitted (Phase 3 — forced alignment)

| Tool | Version / location | Role | Notes |
|------|-------------------|------|-------|
| stable-ts | user pip `stable-ts==2.19.1` (system Python + existing openai-whisper) | Mode A **variant check** forced align + refine | Demoted from default after CTC bake-off; keep for A/B |

## Admitted (Phase 3b — CTC spine, default Mode A)

| Tool | Version / location | Role | Notes |
|------|-------------------|------|-------|
| torchaudio MMS_FA | system/user torchaudio (≥2.1; bake-off used 2.8.0) | Mode A **default** CTC forced align of known lyrics on vocal stem | Weights ~**1.18GB**, one-time download via `torchaudio.pipelines.MMS_FA.get_model()` → typically `~/.cache/torch/hub/checkpoints/model.pt` (Facebook public files). **Not** vendored into `src/`. |
| torchaudio.functional.forced_align + merge_tokens | same | CTC path decode | Filter CTC blank `'-'` from lyric targets (dict exposes blank as key). |

**Bake-off (2026-07-18):** bias-corrected onset MAE **92ms** vs stable-ts **540ms** on song A 0–90s gold (29w).  
**Lead autopsy (2026-07-19):** **−239ms RETIRED** — was spacebar anticipation in gold, not display preference; caused pre-waiting. **Default lead = 0.0 (sung onset).** See `docs/CONVERGENCE_2026_07_19.md` for star + whisper fusion + flux stack.

**Source of truth report:** `~/Desktop/Arkhē Claude Lab/99_WORKING/LYRIC_LOCK_SPINE_BAKEOFF_20260718/REPORT_SPINE_BAKEOFF_2026_07_18.md`

## Not admitted yet (require explicit gate)

| Candidate | Why deferred |
|-----------|----------------|
| whisperX | Optional alt aligner; CTC default first |
| madmom / BeatNet | Tempo map — later |
| schufo/lyrics-aligner | Research tree — quarantine only if ever used |
| Any `curl \| bash` installer | Forbidden |

## Cleanroom policy

- Future OSS installs: isolated venv under e.g. `~/ArkheCleanroom/lyric-lock/` (outside product tree)  
- No `git clone` of third-party apps into `builds/lyric-lock-v0/src`  
- Our code owns exports and orchestration  

## Inspiration (docs only — not runtime)

- Academic lyrics-to-audio alignment / DALI  
- Commercial shape: separate → transcribe → timestamp (e.g. AudioShake-class pipelines)  
- Consumer AI lyric video tools (marketing claims; not trusted as solved)  
