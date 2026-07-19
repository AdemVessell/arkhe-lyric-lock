# DEMOTE: evidence-gate rebind path (2026-07-18)

## Verdict
**Going backward.** User ear QA: rebind v1–v3b is **worse** than pre-rebind auto on 32–40s (Doesn't / phrase). Not a product path.

## What we tried
- Abstain low-evidence / orphan blip
- Rebind to activity islands
- Post-quiet onset rebind
- Cascades that displaced whole phrases (v3)

## Outcome
- Paper ms sometimes "closer"
- **Watch test: no improvement → way worse**
- Same failure class as hold-island / anti-hang thrashing: energy stories on a bad FA spine do not create ear-lock

## Baseline to keep (do not use rebind as default)
| Artifact | Path | Role |
|----------|------|------|
| Human gold (partial, QA'd) | `tools/timing-ui/exports/human_gold_APPROVED_PARTIAL.json` | **Truth / ruler** |
| Human gold video | `out/20260718T104928Z_made_well_HUMAN_GOLD_TRIM/` | Visual truth |
| Last auto before rebind dig | `out/20260718T111752Z_made_well_ANTI_HANG/` | Auto baseline (still imperfect) |
| Scoreboard | `notes/eval/SCOREBOARD_HUMAN_GOLD_PARTIAL_2026_07_18.md` | Metrics |

## Demoted runs (research only, not pipeline default)
- `out/*EVIDENCE_GATE*`
- `out/*REBIND*`

## Doctrine update
1. **Do not ship rebind-as-default.**
2. **Stop stacking energy re-home knobs** until spine changes or human residual is the product path.
3. Negative results stay visible (this file).
4. Next real options only: better FA spine bake-off, or hybrid UI on gold — not rebind v4.

