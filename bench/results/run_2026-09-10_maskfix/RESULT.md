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

---

## Cutoff sweep, and it retracts the reading above

Same two frames, `maskrcnn_mask_binarise` swept with no recovery logic:

| cutoff | tail (f2) | fill (f2) | spill (share of box) |
|---|---|---|---|
| 0.50 | 0.0642 | 0.4032 | 0.0000 |
| 0.30 | 0.0566 | 0.4499 | 0.0010 |
| 0.25 | 0.0544 | 0.4626 | 0.0013 |
| 0.15 | 0.0501 | 0.4939 | 0.0021 |
| 0.10 | 0.0479 | 0.5124 | 0.0026 |
| 0.05 | 0.0457 | 0.5377 | 0.0032 |

Fill rises 33% and spill stays under 0.4% of the box, with **no knee anywhere**.
A quantity that grows smoothly with no natural stopping point is a warning, not
a result, and `seed_cutoff_adds_a_rim.jpg` says why.

**Green is the mask at 0.5; red is everything 0.10 adds. The red is a rim — a
few pixels wide, following the whole silhouette, coat and leg and boot alike.
The trailing boot's sole is outside BOTH.**

So the lower cutoff **dilates the boundary; it does not recover the missing
region**. That is exactly why fill grows without a knee: the growth is
proportional to perimeter, not to anything about the boot. And it explains the
+0.00379 downstream as a slightly fatter matte rather than a recovered subject.

**The spill guard is weaker than it looks**, and this is the honest limit of the
mechanism: it only sees growth OUTSIDE the box. A rim thickening *inside* the
box is invisible to it, which is most of what happens here.

## Verdict

**5.9 stays off, and should not be turned on.** It buys boundary dilation on
every clip in exchange for nothing on the defect it was written for -- and edge
quality is the axis the client named. Two fixes in the seed-threshold family
have now failed for the same underlying reason: the boot is not a low-probability
region the mask head is unsure about, it is **absent from the instance at any
threshold**.

Where that leaves it, honestly:
1. A different seeder for that box -- SAM 2/3 prompted with a point at the
   bottom of the detection box -- which is a real mechanism change, not a knob.
2. Prompt the matting engine directly on the missing region.
3. Accept it and say so in the package: the boots are recovered from absent to
   held-but-soft, which is what the shipped render already does.
