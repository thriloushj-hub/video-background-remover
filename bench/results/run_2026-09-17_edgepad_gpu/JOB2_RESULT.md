# Job 2 -- the engine's loss: THERE IS NO MULTIPLIER (GPU, 18 Sep 2026)

Harness: `bench/probe_engine_deficit.py` (unit-tested; 226 pass on the VM
before the run). Fresh A100, 11 segments, ~75 min: the seven frame-edge
windows, the three 2 Sep additions (bilibili_w499_570, dlh, es2), and
`bilibili_2shot.mp4` as the positive control with an extra `la0` arm
(lookahead_frames 0 -- the configuration the "engine loses 243 rows" case was
recorded on). Arms otherwise: `base` (shipped defaults) and `band` (5.10b).

The quantity: `deficit = alpha_short - seed_short` at the shot's seed frame,
on the SEED's own solid columns, per edge -- one instrument, one run, same
columns for both sides. Numbers: `job2_deficit.json` and `j2b_gb.txt` in this
folder's zip (`job2_2026-09-18.zip`, landing in ~/Downloads unverified);
everything below was read off the VM first.

## 1. At the seed frame, the engine follows its seed. 22 edge rows, base arm:

| worst deficits | px |
|---|---|
| 1917 bottom | +18 |
| dlh bottom | +16 |
| ipman bottom | +14 |
| 1917 top | +11 |
| codylexi left | +10 |
| everything else (17 rows) | +8 to **-2** |

Median **+0.5 px**, none over 30, and microsoft's bottom is **-2** -- the
engine reaches slightly PAST its seed. On the **band** arm all 22 rows are
0-over-10, median 0: given a seed that reaches the edge, the engine keeps it
there. The frame-edge defect is decided at the seed, full stop.

## 2. The 243-row "engine loss" reproduces as a SEED loss, exactly as the 8 Sep retraction said.

bilibili_2shot, shot 2 (the boots shot), global bottom row:

| arm | seed bottom | alpha bottom f0.. |
|---|---|---|
| la0 (no lookahead) | **604** | 587, 598, 608, 626 ... |
| base (5.7 default) | 604 at f56; 5.7 re-seeds | **866**, 879, 885, 899 ... |
| band | 604 | 866, 879, 885, 900 ... |

Given a seed reaching 604 the engine returns 587 -- **17 rows short, not
243**. The docstring of `bench/probe_engine_coverage.py` still motivated Job 2
with `seed 825 -> ... -> alpha 588`, i.e. 243 rows "lost inside the engine" --
**but 825 was the JPEG-corrupted measurement that the 8 Sep 01:30 callout
retracted** (lossless decode gives 596; JPEG at q100 gives 825), and the same
callout already recorded "the engine returns 587 from a seed of 596 -- it
follows faithfully". The retraction fixed the callout and never reached the
docstring, and the docstring is what the next plan was written from. Docstring
now fixed.

## 3. The "1.5-2.5x" premise is dead too.

It came from comparing the 14-15 Sep CPU seed reach against alpha reach from
different runs on different columns -- two instruments. With one instrument the
multiplier is <=18 px absolute, not a ratio, and only on the motion-blur clips.

## 4. Caveats, so nobody over-reads this

- Per-frame deficits beyond the first ~3 frames measure SUBJECT MOTION leaving
  the seed-anchored columns, not the engine (ipman "+785" is the fighter moving
  off the bottom edge). The seed-frame value is the claim.
- The edge-anchored probe cannot see an interior deficit on a truncated seed --
  the control exposed exactly that (its edge rows were silently absent), which
  is why shot 2 was measured by global bottom row as well. Any future interior
  claim needs the global measure, not this probe alone.
- 11 segments, seed frames only. Nothing here is about tracking losses
  mid-shot (5.5's territory).

## What Job 2 means for the plan

There is no engine-side loss to chase. The remaining frame-edge work is
entirely the seed, the fix for that is 5.10b (band pad, GPU-validated,
BAND_RESULT.md), and the only open item is the default flip + re-render.
