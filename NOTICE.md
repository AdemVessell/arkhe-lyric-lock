# Third-party components and licences

Our application code (`src/`, `bench/`) is MIT — see [LICENSE](LICENSE). It calls
the following components, which are **not** ours and retain their own licences.
The MIT licence for our code does not extend to third-party model weights or
clear the default workflow for commercial use.

| Component | Role | Licence | Note |
|---|---|---|---|
| torchaudio `MMS_FA` bundle (weights) | CTC forced alignment | **[CC-BY-NC 4.0 (non-commercial)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)** | Default aligner; downloaded separately, not bundled. Commercial use needs separate permission for MMS or a replacement with appropriate commercial-use rights. See the attribution and replacement status below. |
| torch / torchaudio | runtime | BSD-3-Clause | |
| openai-whisper | melisma re-timing, gate transcription | MIT | model weights MIT |
| demucs (htdemucs) | vocal separation | MIT | weights MIT |
| ffmpeg | decode / render | LGPL or GPL depending on build | invoked as an external binary, not linked |
| stable-ts | legacy alignment spine (A/B only) | MIT | |

## MMS_FA attribution and use

MMS_FA comes from Meta's Massively Multilingual Speech (MMS) project:
Vineel Pratap et al., *Scaling Speech Technology to 1,000+ Languages* (2023).

- [Official torchaudio MMS_FA model documentation](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html)
- [Upstream MMS project, licence notice and citation](https://github.com/facebookresearch/fairseq/blob/100cd91db19bb27277a06a25eb4154c805b10189/examples/mms/README.md#license)
- [Full CC-BY-NC 4.0 licence, including attribution conditions and warranty disclaimer](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)

No MMS weights are included in this repository. On first use, torchaudio
downloads the weights if they are not already cached. They remain subject to
their own licence. The BSD licence for the torch/torchaudio runtime does not
replace the MMS model's licence.

CC-BY-NC permits non-commercial sharing under its terms; it is not a
personal-use-only licence. Downloading the weights separately or running them
locally does not grant commercial-use permission. The default MMS-backed
workflow is not cleared for commercial use. Obtain separate permission for MMS
or use a replacement with appropriate commercial-use rights.

## Replacement status

The current CTC backend in `src/lyric_lock/ctc_align.py` explicitly selects
MMS_FA. **A replacement CTC backend permitting commercial use has not been
implemented or validated here.** Replacing it requires engineering and
validation of alignment and sheet checks; it is not currently a configurable
checkpoint swap. The legacy `stable_ts` option is not a validated equivalent
for the complete workflow, and the standalone sheet gate still uses MMS_FA.

No third-party application source is vendored into this tree.
