# 5.9 — re-binarising a truncated mask at a lower cutoff (10 Sep)

A100, `198 passed` on the VM first, `bilibili_2shot.mp4` byte-identical to the
committed segment. Two runs: a seed-level A/B (~35 s) and a matting A/B through
the pipeline (131 s).

## The mechanism

Mask R-CNN returns a soft probability map per instance; `> 0.5` is what turns it
into a mask, and that line was hardcoded. A dark boot on a dark floor scores
under it, so the boot is not in the seed at all — and a shot inherits its seed
for its whole length. 5.9 re-binarises **only the boxes whose mask stops short**,
once, at 0.25, and keeps the result only if it reaches further **and** does not
spill outside the box.

## Seed level — it fires on every frame, and the guard is not doing the work

Shot 2's first four frames, `masks_from_boxes` off vs on:

| frame | tail | fill | spill (share of box) |
|---|---|---|---|
| 0 | 0.0806 → **0.0717** | 0.3981 → **0.4590** | 0.000 → 0.001 |
| 1 | 0.0838 → **0.0717** | 0.3877 → **0.4475** | 0.000 → 0.001 |
| 2 | 0.0642 → **0.0544** | 0.4032 → **0.4626** | 0.000 → 0.001 |
| 3 | 0.3215 → **0.2412** | 0.3400 → **0.3940** | 0.000 → 0.000 |

**About 15% more mask inside the box on every frame, and spill stays at 0.001 of
the box** — so this is recovering the subject, not bleeding into background.
That distinction is the whole safety argument and it is measured, not assumed.

## Through the pipeline — real, and modest

| | seed frame | alpha bottom, f0 | shot mean alpha |
|---|---|---|---|
| shot 1 (control) | 0 → 0 | 1065 → 1065 | 0.23264 → 0.23244 (**−0.00020**) |
| shot 2 | 2 → 2 | 866 → **891** | 0.10138 → **0.10517** (**+0.00379**) |

The control moves by 0.0002 and the affected shot by 0.0038 — about 4x the
0.00089 per-frame run-to-run noise measured on butter, so the gain is real. It
is also **not a full fix**: the trailing boot gets more alpha, not solid alpha.

## State

Default **OFF** (`maskrcnn_mask_recover`). 5.7b was turned on from a plausible
argument and had to be reverted; this one has seed-level numbers behind it but
has not been swept across the delivery, and a lower cutoff applied everywhere is
exactly the kind of change that can fatten a matte on clips nobody looked at.
The sweep is 3 minutes and it is the gate that should decide.
