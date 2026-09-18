# 5.10b -- the band pad: all of the win, none of the regressions (GPU, 17 Sep 2026)

Same harness, same four windows, same three-arm shape as Job 1 earlier today
(`README.md` in this folder), with the middle arm run as
`maskrcnn_edge_pad=64, maskrcnn_edge_pad_band=True`. Fresh A100, **223 tests
passed on the VM first**, md5 of `vbgr/seed.py` and `bench/ab_edge_pad_final.py`
identical to local, torch 2.11.0+cu128. Command:

    python -u bench/ab_edge_pad_final.py --band --config pipe.json \
        --out ab_band edge/{microsoft,ipman,interview,tryguys}*.mp4

Results zip `ab_band_2026-09-17.zip` downloaded from the VM; **landing in
~/Downloads not verified** (this session had no access to that folder). Every
number below was read off the VM first.

## Why a band

Job 1 established that the full pad fixes the bottom edge on the final alpha and
**breaks side edges**: tryguys' right edge 0.608 -> 0.523, worse on 67 of 72
frames, the yellow jacket's shoulder dropped -- and already 33 px short in the
SEED against 6 px shipped. Cause of the break not established; the pad also
moved interior pixels (ipman's unexplained churn, a blurred fist lost on f6).

Two variants were screened on CPU first, 7 clips x 3 frames x 4 edges, seed
level, against the shipped seed:

| arm | mean edge delta | edge cases worse by >0.05 | verdict |
|---|---|---|---|
| replicate 64, all edges (shipped candidate) | +0.441 | **5 of 50** | the Job 1 regression |
| **reflect** 101 instead of replicate | +0.615 | 0 of 49 | **REJECTED** |
| replicate 64, bottom only | +0.360 | 7 of 50 | rejected |
| **band: shipped mask, padded mask only inside a 64 px border band** | +0.486 | **0 of 50** | **KEEP** |

**Reflect looked best on every edge and is catastrophic.** On microsoft the
mirrored figure at the left edge turns into a person-shaped continuation and the
seed swallows **56% of the frame** -- the whole room -- with interior flip 0.58.
It is the leak this project has retracted five findings to, and a metric that
only looks at edges cannot see it. Bottom-only padding is not a subset of the
defect either: it still changes the whole detector input, and it made
microsoft's top edge and interview's right edge WORSE (-0.19, -0.12).

**The band keeps the pad for the one thing it is for.** Both passes run; per
box, the shipped mask is kept everywhere except a 64 px band along the frame
border, where the two are OR-ed. By construction: no edge can be worse than
shipped, the interior is the shipped interior, and the cast is the shipped cast
(a box the unpadded pass did not match stays unmatched, so nothing new can enter
and nothing can merge).

## The GPU A/B, final alpha

edge alpha, mean over 72 frames, shared columns

| clip | edge | pad0 | **band** | full pad (Job 1) | frames better/worse |
|---|---|---|---|---|---|
| microsoft | bottom | 0.479 | **0.984** | 0.984 | 72 / 0 |
| microsoft | top | 0.716 | 0.717 | 0.713 | 0 / 0 |
| microsoft | left | 0.999 | 0.999 | 0.989 | 0 / 0 |
| ipman | bottom | 0.869 | **0.904** | 0.904 | 10 / 0 |
| ipman | top, right | - | unchanged | unchanged | 0 / 0 |
| interview | bottom | 0.166 | **0.737** | 0.730 | 72 / 0 |
| interview | right | 0.455 | **0.633** | 0.633 | 72 / 0 |
| interview | top | 0.051 | 0.052 | 0.051 | 0 / 0 |
| tryguys | bottom | 0.960 | **0.989** | 0.981 | 29 / 0 |
| tryguys | **right** | 0.608 | **0.613** | **0.523** | **7 / 0** |
| tryguys | left | 0.631 | 0.637 | 0.633 | 4 / 0 |
| tryguys | top | 0.399 | 0.402 | 0.402 | 0 / 0 |

**Not one edge on any clip is worse on any frame.** `w90` (the dilation
control) moves by at most 1 px. Reach goes to 0 px short on every edge the pad
touches.

## Interior, integrity, cast

| clip | band churn (mean / max) | control churn | regions pad0 holds and the arm does not |
|---|---|---|---|
| microsoft | 0.00013 / 0.00017 | 0.00004 | 0 |
| ipman | 0.00078 / 0.00601 | 0 | **0** (full pad lost the f6 fist) |
| interview | 0.00257 / 0.00592 | 0 | 0 |
| tryguys | 0.00022 / 0.00035 | 0 | **0** (full pad lost the shoulder on 2 frames) |

Object counts identical in all arms (2/2/2/3), `bad_frames` 0, seed frame 0.
Interior churn is **not** exactly 0 as it is at the seed: `erode_dilate` after
the union spreads band content a few px inward, and the engine propagates from
the changed seed. It is 3-6x smaller than the full pad's and, on ipman, no
longer loses anything.

## What this supports, and what it does not

- **The seed gain survives to the final alpha**, on every edge where a subject
  is cut by the frame: bottom on all four clips, interview's right edge.
- **The band form has no measured cost** on these four clips: no edge worse, no
  region lost, cast unchanged, interior churn at or near the run-to-run floor.
- Recommendation: **turn `maskrcnn_edge_pad = 64` and
  `maskrcnn_edge_pad_band = True` on by default.** Thriloush's call.
- **Four clips, 72-frame windows.** Before a delivery re-render this should be
  swept over the other clips (the seed-level screen is cheap and needs no GPU:
  7 clips x 3 frames took ~16 CPU minutes). microsoft's shipped bottom edge is
  not even stable run to run (0.479 vs 0.325), so per-clip deltas below ~0.05
  should not be read as anything.
- **Why the full pad breaks a side edge is still unexplained.** The band avoids
  it rather than explaining it.

---

## Addendum, same evening: no simpler arm survives the screen

`bench/screen_pad_variants.py`, CPU, 7 clips x 3 frames x 4 edges, product
seeder path, scored against the shipped seed. The question was whether the
two-pass band could be replaced by a single pad that does not retreat -- a flat
grey pad (which `bench/probe_pad_content.py` showed does not retreat on
tryguys' right edge) or a narrower 32 px replicate.

| arm | mean edge delta | edge cases worse by >0.05 | cases better by >0.05 | frames losing a region | interior flip |
|---|---|---|---|---|---|
| replicate 32 | +0.415 | **8 of 49** | 34 | **1** (microsoft f24) | 0.0070 |
| replicate 64 | +0.441 | **5 of 50** | 37 | 0 | 0.0089 |
| grey 64 | +0.161 | **2 of 50** | 26 | 0 | 0.0081 |
| band + replicate 32 | +0.460 | **0 of 50** | 34 | 0 | 0.0000 |
| **band + replicate 64** | **+0.486** | **0 of 50** | **39** | 0 | 0.0000 |
| band + grey 64 | +0.172 | **0 of 50** | 27 | 0 | 0.0000 |

**Neither simpler arm is clean.** Grey padding fixes tryguys' right edge (0.02
-> 0.58 at the seed, 0 px short) exactly as the content probe predicted, but it
recovers far less elsewhere -- less than half the mean gain, 26 wins against 39
-- and it still loses two edge cases. Replicate 32 regresses eight, including
microsoft's top edge 0.51 -> 0.00, and drops a region on one frame. Worst case
per arm: replicate 32 -0.511, replicate 64 -0.538, grey 64 -0.361.

**Every band arm is 0-of-50 worse by construction, and replicate 64 is the
strongest of the three.** That is the arm already validated on GPU, so the CPU
screen adds no reason to change it and one reason to keep it.

Caveat kept from above: this is the seed, 7 clips x 3 frames. Only the GPU A/B
on four of those clips says anything about the final alpha.
