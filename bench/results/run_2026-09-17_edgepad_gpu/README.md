# 5.10 Job 1 -- edge_pad 0 vs 64 on the FINAL alpha (GPU, 17 Sep 2026)

A100, fresh VM, code zip 81 entries, **219 passed on the VM first**, md5 of
`vbgr/seed.py` and `bench/ab_edge_pad_final.py` identical to local.
Product path (`Pipeline.run_clip`), `lookahead_frames` default, output_mode
alpha. Clips: the four 72-frame windows the CPU work used
(`_to_delete/edgeprobe_2026-09-14/`). Three arms per clip: pad0, pad64, pad0
again (control). ~3.2 min per arm, ~40 min total.

Harness: `bench/ab_edge_pad_final.py`. Full output (json, look-sheets, uint8
alphas): `ab_edgepad_2026-09-17.zip`, 83 MB, downloaded from the VM --
**landing in ~/Downloads not yet verified**. The numbers below were read off
the VM before download.

## Verdict: the default does NOT flip. pad 64 fixes the bottom edge and breaks a side edge.

## Frame edge, final alpha (mean over frames; shared columns)

edge alpha / share absent / median px short / width 90 px in

| clip | edge | cols | pad0 | pad64 | pad0 repeat | delta (frames better/worse) |
|---|---|---|---|---|---|---|
| microsoft | bottom | 910 | 0.479 / 0.51 / 5.5 / 954 | **0.984** / 0.02 / 0.0 / 954 | 0.325 / 0.67 / 17.4 / 954 | **+0.505** (72/0) |
| microsoft | top | 480 | 0.716 | 0.713 | 0.716 | -0.002 (0/3) |
| microsoft | left | 960 | 0.999 | 0.989 | 0.999 | -0.010 (0/4) |
| ipman | bottom | 528 | 0.869 / 0.13 / 0.7 | 0.904 / 0.09 / 0.0 | 0.869 | +0.035 (10/0; up to +0.86) |
| ipman | top / right | | unchanged | | | 0/0 |
| interview | bottom | 1618 | 0.166 / 0.87 / 3.8 / 1656 | **0.730** / 0.26 / 0.0 / 1660 | 0.166 | **+0.564** (72/0) |
| interview | right | 507 | 0.455 | 0.633 | 0.455 | +0.179 (72/0) |
| tryguys | bottom | 1845 | 0.964 | 0.981 | 0.964 | +0.017 (7/0) |
| tryguys | left / top | | unchanged | | | 0/0 |
| tryguys | **right** | 756 | 0.608 / 0.39 / **0.0** | **0.523** / 0.47 / **5.7** | 0.608 | **-0.086 (0/67)** |

- `w90` flat everywhere (dilation control clean).
- Control: bit-identical on ipman, interview, tryguys. **microsoft's shipped
  edge is itself unstable run to run** (0.479 vs 0.325, max per-frame 0.263);
  pad64 at 0.984 clears both runs by 0.5+.

## Pictures (soft alpha over magenta)

- microsoft f36, interview f36: pad0 leaves a magenta band along the bottom of
  frame under the subjects; pad64 holds solid to the edge. The fix, as intended.
- **tryguys, right edge: pad64 drops the yellow jacket's shoulder** where it
  meets the frame edge, a region pad0 holds -- on 67 of 72 frames.
  `evidence_tryguys_pad64_drops_shoulder.jpg` (source | pad0 | pad64, f0).
  It is already in the SEED: at the seed frame the right edge is 6 px short at
  pad0 and **33 px short at pad64**. The CPU sweep measured the BOTTOM edge only,
  so it could not see this. Mechanism not established.
- **ipman f6: pad64 drops the motion-blurred fist** (0.70% of frame, 90 px from
  the top, not at the edge) that pad0 holds softly. One frame; control is
  bit-identical, so the change is caused by the pad.

## Interior and integrity

| clip | pad64 churn (mean / max) | control churn | regions pad0 holds, pad64 not |
|---|---|---|---|
| microsoft | 0.00023 / 0.00064 | 0.00004 | 0 |
| ipman | 0.00129 / 0.01131 | 0 | 1 frame (f6, 0.0070) |
| interview | 0.00354 / 0.00661 | 0 | 0 |
| tryguys | 0.00030 / 0.00091 | 0 | 2 frames (f0, f11, max 0.0052) = the shoulder |

No region appears only at pad64. Object counts identical in every arm
(2/2/2/3), no bad frames, seed frame 0 everywhere.

## Engine loss at the seed frame (Job 2's quantity)

seed px short -> alpha px short, bottom edge: microsoft 15 -> 13,
ipman 13 -> 24, interview 15 -> 16, tryguys 6 -> 7. At pad64 all 0 -> 0.
**On these windows the engine does not multiply the seed's shortfall at the
seed frame, except ipman (1.8x).** The 1.5-2.5x figure was not measured this
way; Job 2 has to measure across frames, not only the seed frame.
