# Third-party components and licences

Our code (`src/`, `bench/`) is MIT — see LICENSE. It calls the following, which
are **not** ours and carry their own terms. Check these before commercial use.

| Component | Role | Licence | Note |
|---|---|---|---|
| torchaudio `MMS_FA` bundle (weights) | CTC forced alignment | **CC-BY-NC 4.0 (non-commercial)** | The default aligner. Commercial use requires swapping in a permissively licensed CTC aligner (e.g. a wav2vec2 CTC model under MIT/Apache) — the interface in `ctc_align.py` is model-agnostic. |
| torch / torchaudio | runtime | BSD-3-Clause | |
| openai-whisper | melisma re-timing, gate transcription | MIT | model weights MIT |
| demucs (htdemucs) | vocal separation | MIT | weights MIT |
| ffmpeg | decode / render | LGPL or GPL depending on build | invoked as an external binary, not linked |
| stable-ts | legacy alignment spine (A/B only) | MIT | |

No third-party application source is vendored into this tree.
