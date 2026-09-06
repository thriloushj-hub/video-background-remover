# run_2026-09-06_fix55 — does the 5.5 recovery seed actually recover anyone?

A single-variable A/B. Same clip, same settings, same VM recipe; the only
difference from `run_2026-09-05b/full_out/1917_alpha.mp4` is commit `332e64a`,
the recovery seed. `--mode alpha` only, because the question is about the matte
and encoding three modes buys nothing here.

    1917: 724f @24.000  1 shot  bad_frames=0  reentries=11  1739.3s

## The answer: it works, partially, and the mechanism matches the outcome

`bench/check_full_clip_solos.py`, both directions, unchanged thresholds:

| | before | after 5.5 |
|---|---|---|
| frames where v1 holds a region v2 does not | 97 | **72** (−26%) |
| mean fraction of frame | 0.00364 | **0.00147** (−60%) |
| worst single frame | 0.1228 | **0.0498** (−59%) |
| frames at ≥2% of frame | 43 | **18** (−58%) |
| frames where v2 holds a region v1 does not | 146 | **146** (unchanged) |
| its mean fraction | 0.00339 | **0.00339** (unchanged) |

**The two longest bursts are gone.** Before: eleven bursts including 540–551
and 596–607, twelve frames each. After: eight bursts, and both of those are
absent.

**And they are exactly the ones the fix targeted.** Comparing the local passes'
own seed lines, the anchors at **f544 and f600** went from
`seeding 1 object(s) (auto-split)` to `seeding 2 object(s) (explicit)` — the
blurred runner is now in the seed. Every other anchor seeded the same count as
before. Two anchors changed, and the two bursts inside their windows are the
two that disappeared.

**The v2-only side did not move at all**, to five decimal places. That is the
specificity check: the change touched the recovery path and nothing else, and
did not quietly widen the cast.

## What is left, and it is two different things

18 frames still at ≥2%, in bursts 490–493, 505–506, 508, 574–576, 611,
637–641, 656, 704.

1. **490–508 has no local pass at all.** The first accepted scan hit is f536;
   at f496 the scan rejected the box itself (box cov 0.292, **mask cov 0.881** —
   he was 88% covered at that moment, so "uncovered" was the wrong word for him
   then). Nothing fires, so nothing can be recovered. That is a scan-density and
   scan-threshold question, not a seeding one.
2. **574–576, 611, 637–641, 656 sit inside windows whose pass already seeded
   two objects** — before *and* after. He is seeded at the anchor and the
   tracker loses him again within a few frames of heavy blur. That is the hard
   half, and no threshold fixes it.

So: the seeding half of 5.5 is fixed and measured. The tracking half is not,
and pretending otherwise would be the fifth entry in HANDOFF §7.

## Caveat on the handover package

`_to_send/outputs/full_length/1917_*` and the 1917 side-by-side were rendered
**before** this fix, from `run_2026-09-05b`. They are consistent with each
other and with the numbers published beside them; they simply predate 5.5.
Re-rendering 1917 in all three output modes is one ~30-minute pass.
