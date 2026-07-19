# Final Cut Pro Handshake

## Goal

Prove timestamps by importing into Final Cut Pro on the same song timeline.

## Apple-supported caption imports

| Format | Extension | Phase |
|--------|-----------|-------|
| SubRip | `.srt` | **Phase 1** |
| iTunes Timed Text | `.itt` | later |
| CEA-608 | `.scc` | optional later |

Source: Apple Support — Import captions into Final Cut Pro (CEA-608 / iTT / SRT).

## Test procedure

1. New FCP project; import source audio (or video with that audio)  
2. Place clip on timeline from time 0 (or note offset)  
3. **File → Import → Captions…** → select `lyrics.srt` from run `out/`  
4. Play and score ear-lock (heard word vs caption)  
5. Log score + cleanup notes in `notes/eval/`  

## Known limits

- FCP captions are **line-oriented**, not karaoke word-flash  
- For word-highlight music video, use `timed.json` words with a separate renderer / frontier AI path  
- If audio does not start at timeline 0, shift captions or align clip start  

## Phase 1 pass criteria

- SRT imports without error  
- Captions appear during vocal sections  
- Human records whether timing is usable / early / late  
