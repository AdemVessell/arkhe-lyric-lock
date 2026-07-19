# Arkhē Lyric Lock

**Give it a song and its lyrics. It gives you the timing — and tells you where it isn't sure.**

Lyric-video timing that runs locally, scores its own output, refuses to ship
work it doesn't trust, and hands the leftovers to a repair bench where fixes
take seconds.

---

## Why this exists

Syncing lyrics to audio is a stubbornly manual job. Most people reach for a
transcription model (Whisper and friends), and on sung material it stumbles:
held vowels stretch past what speech models expect, dense mixes mask consonants,
ad-libs and vamps aren't in any sheet. You end up hand-typing timestamps.

**The fix is a different tool, not a bigger model.** Transcription asks *what was
sung* — a hard question. Forced alignment is handed the lyrics and asks only
*when does each word occur* — a much easier one. Run it on a separated vocal
stem and the numbers change completely:

| spine | mean onset error (bias-corrected) | words within 150 ms |
|---|---|---|
| attention alignment (stable-ts) | 540 ms | 9 / 29 |
| **CTC alignment (MMS_FA) on vocal stem** | **92 ms** | **23 / 29** |

Measured against human-tapped ground truth on the same audio, same lyrics.
Alignment runs in ~8 s for 90 s of audio on CPU.

## What it does

```
audio + lyrics.txt
   ↓  demucs separates the vocal
   ↓  CTC forced alignment places every word of your sheet
   ↓  a second pass re-times held notes the aligner rushes through
   ↓  onsets snap to where the voice actually starts
   ↓  the run scores itself and flags what it doubts
timed.json · word SRT · line SRT · reference video
```

`timed.json` and the SRTs import straight into Final Cut, Premiere, Resolve or
After Effects. **The styling stays yours** — the built-in video is a plain
reference render, not the product.

## The honesty layer

This is the part that isn't standard:

- **Every word carries a confidence score**, and the thresholds are computed
  per song — absolute cutoffs don't transfer between a ballad and a rap.
- **Sung regions your sheet doesn't cover are named out loud**, quoting what was
  heard there, and the run *aborts* rather than silently showing a blank screen.
- **Ad-libs and untranscribable vocalisations abstain** instead of being forced
  into lyrics they don't match.
- **Repeated lines that the sheet already contains are auto-recovered** without
  bothering you.
- When it can't place a word honestly, it **shows nothing rather than a lie**.

## Install

```bash
python3 -m pip install -r requirements.txt      # torch, torchaudio, openai-whisper
# demucs is best kept in its own venv:
python3 -m venv ~/.venvs/lyric-lock && ~/.venvs/lyric-lock/bin/pip install demucs==4.0.1
```

ffmpeg must be on PATH. First run downloads the MMS_FA aligner weights (~1.2 GB).

## Use

```bash
PYTHONPATH=src python3 -m lyric_lock run \
  --audio song.wav --lyrics sheet.txt --separate \
  --video --video-style word
```

Outputs land in `out/<run_id>/`: `timed.json`, `lyrics.srt`, `lyrics.words.srt`,
`suspect_spans.json` (the findings), `run_meta.json`, and the reference video.

Useful flags: `--gate-warn` (report missing-lyric regions instead of aborting),
`--no-fuse-anchors`, `--align-engine stable_ts` (the losing spine, kept for A/B).

Check a sheet without a full run:

```bash
PYTHONPATH=src python3 -m lyric_lock gate --vocal stem.wav --lyrics sheet.txt --mix song.wav
```

## The repair bench

`bench/index.html` — open it in a browser, no server, nothing leaves your machine.
Load the audio, `timed.json` and `suspect_spans.json` from a run.

Flagged words arrive pre-coloured. Press <kbd>F</kbd> to jump to the next thing
the machine doubted. Fix it by holding <kbd>space</kbd> through the word (your tap
snaps to the real onset — rough timing is fine), nudging ±20 ms, rippling a whole
phrase, or editing the lyrics for that region. Export back to `timed.json` or SRT.

Most songs need none of this. It's ready anyway.

## Honest limits

- **It needs the lyric text.** It finds the timing, not the words. Lyrics-free
  transcription of singing is not solved — see `notes/eval/` for our own
  falsified attempts.
- **Very fast dense rap still jitters**; word boundaries blur and function words
  collapse below the display floor.
- **Backing vocals and parentheticals are skipped by design.**
- Validated on a handful of songs by ear, not on a public benchmark. Treat the
  numbers above as measured-here, not as a leaderboard claim.

## Negative results

`notes/eval/` keeps the failures on purpose — the falsified free-transcription
path, the demoted repair mechanisms, the display-lead constant that turned out to
be captured human anticipation rather than a real preference. The reasoning that
produced the working stack is in `notes/eval/SPINE_BAKEOFF.md`.

## Licensing

Our code is MIT (`LICENSE`). The models are not ours and carry their own terms —
notably **MMS_FA aligner weights are CC-BY-NC (non-commercial)**. See `NOTICE.md`
before any commercial use; a permissively licensed aligner can be swapped in.

## Credit

Built as an Arkhē research object: inspired-by, own implementation, negative
results kept visible. Ground-truth timing labels by ear.
