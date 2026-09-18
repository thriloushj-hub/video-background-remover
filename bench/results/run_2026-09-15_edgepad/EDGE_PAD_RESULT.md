# 5.10 -- the frame edge: final CPU state, 16 Sep 2026

Everything here is CPU, off the clips decoded from their own mp4 (not the
committed JPEGs -- s7 entry 28). Harnesses: `bench/sweep_edge_pad.py`,
`bench/check_subject_integrity.py`.

## The defect (established 14-15 Sep, unchanged)

Where a subject is cut by the edge of frame the seed mask **stops short of it**.
It is truncation, not softness: on microsoft's bottom edge, of the 475 columns
solidly foreground 80-100 px up, **81.3% are below alpha 0.05 at the bottom row**
and 16.4% are above 0.5. "Mean alpha 0.52 in the bottom 8 rows" was a mean over
a bimodal distribution.

Cause, confirmed in torchvision's own source: `expand_masks(mask, padding=1)`
zero-pads Mask R-CNN's 28x28 ROI mask and `paste_mask_in_image` bilinearly
resizes it to the box, leaving a ramp ~1 mask cell (= box_dim/28) on EVERY box
edge. Harmless where that edge sits in background; at the frame edge the person
fills the box to the border. Second, additive contributor: the boxes themselves
stop **2-11 px short** of the frame edge.

Set-wide at pad 0: silhouette width at the bottom row is **0 px on all 7 clips**
and `absent` is **1.00 on all 7**.

## What was tried, and what survives

| arm | edge alpha | real subjects lost | verdict |
|---|---|---|---|
| shipped | 0.011 - 0.331 | - | the defect |
| **pad 64 (replicate, all edges)** | **0.712 - 0.956** | **0** | **KEEP** |
| source fix (replicate in `expand_masks`) | 0.540 - 0.700 | 0 | rejected |
| proportional `h x 0.10` | 0.920 - 0.974 | 0 lost, **1 MERGED** | rejected |
| asymmetric (pad only touched edges) | 0.128 - 0.945 | 0 | rejected, broken |

### pad 64 -- kept
Recovery on all seven clips, reach 0 px short everywhere, `absent` 0.00-0.12.
Dilation controls clean: `w0/w90` lands 0.769-0.963 and **never exceeds 1**
(a dilating fix climbs past 1); `w90`, the body 90 px up, is **flat on every
clip**. Recommended value: 64. Reach is fixed by 16, confidence plateaus by 64,
128 buys +0.016.

### source fix -- rejected
Swapping torchvision's zero pad for a replicate pad removes the ramp without
touching the detector input, so the detection count never moves (interview
5.7 -> 5.7 against pad 64's 5.7 -> 7.7) and interior churn is 2-3x lower. But it
only fixes the RAMP, not the box shortfall, so recovery is materially worse
(microsoft 0.700 vs 0.956; ipman still 5.7 px short at absent 0.31). And it
fires at every box edge, so it dilates laterally: `w90` grows -- tryguys
1874 -> 1907, interview 1638 -> 1674 -- where pad 64 keeps it flat. Combining it
with a small pad was worse than pad 64 alone on 3 of 4 clips. Also a shipping
liability: it monkeypatches a third-party internal and would break silently on
a torchvision upgrade.

### proportional h x 0.10 -- rejected
Derived from the band being a fixed 4.51% of the SQUARE 576 alpha grid, so it
eats `0.0451 x (H + 2P)` rows and the required pad scales with frame height
(`P >= 0.0517 x H`). It measured better than fixed 64 on all four affected clips
(interview +0.082, and visibly less magenta wash over dark hair -- the axis Runbo
named). **It also merges a 5%-of-frame person into the neighbouring instance on
interview f0.** That is a 3.2a regression: one obj_id per person is the largest
correctness fix here, and a merged pair is the configuration that cost ipman a
fighter at frame 7. An edge gain does not buy that.
Two further honesty notes: `h x 0.15` is measurably WORSE than `h x 0.10` on all
four, so there is a real optimum and not a plateau; and with only two frame
heights in this set (960, 1080), "10% of height" is **indistinguishable from a
~100 px constant** -- the proportional form is better reasoning, not proven.

### asymmetric -- rejected, and the failure is instructive
Pad only the frame edges a detection's box actually touches. It gates on
`box within 12 px of the edge` -- **but the defect IS that boxes stop 2-11 px
short**, so it is gating on the signal the bug corrupts. On 1917 it padded
nothing and edge alpha collapsed 0.797 -> 0.128. A wider tolerance (~1 mask cell)
is the obvious repair and is untested.

## The check that was missing, and the two errors it caught

`bench/check_subject_integrity.py`. Every frame-edge metric in this project
conditions on already-held foreground, so **none of them can see a subject
disappear**. Adding the check produced both a false alarm and a real finding:

- **1917's "2 lost subjects" is not a loss.** Both are trench bystanders at
  height ratio **0.247 and 0.490** against the seed gate's 0.60 -- the figures
  3.3g/3.3h drop on purpose. Every gate-passing subject is held at pad 64 on
  every frame. Looking at the crop settles it: heavily blurred background
  soldiers. In the full-frame picture pad 64 also drops a small detached
  floating fragment that the shipped path keeps, which is cleaner, not worse.
- **interview's disputed 5%-of-frame person** (area 0.0506) is height ratio
  **0.416** -- also a bystander the gate drops -- and it is **held** at pad 64.
  Only `h x 0.10` merges it.

So: **pad 64 loses no gate-passing subject on any of 7 clips x 3 frames.**

## Still not established

Everything above is the **SEED**. The matting engine follows it and loses another
1.5-2.5x on top, so **a better final alpha is not demonstrated** and needs a GPU
A/B (edge_pad 0 vs 64, final alpha, magenta look-sheets). That run is the only
thing between here and a decision on the default.

ipman's interior churn at pad 64 (4.68%, against a ~1.1% noise floor from a 2-8
px pad) remains unexplained. A scale-change hypothesis -- torchvision pins the
longer side to `max_size=1333`, so a 64 px pad puts the subject at ~94% of its
previous detection scale -- **died on its own control**: the arithmetic is
identical for microsoft, which barely moves.

## Retracted from this session

- "interview has a remaining 74-89 px notch the pad does not fix" -- **wrong**.
  75-87% of those pixels are background: at the bottom of frame the sleeve slopes
  diagonally and pale floor appears below it. The metric conditioned on a column
  being solid 80-100 px up and silently assumed a vertical silhouette.
- "the notch is a dark jacket against a dark background, i.e. 5.9's mechanism"
  -- **wrong, and backwards**. That strip measures Laplacian variance **211.5**;
  the region where the seed works perfectly measures **7.3**.
- "h x 0.10 cuts off a person" -- **overstated**. It merges a bystander at 79.7%
  coverage, which is still a reason to reject it, but it does not delete anyone.

---

## Addendum, 16 Sep: the asymmetric repair works

The first asymmetric attempt gated on `box within 12 px of the frame edge`,
which is the signal the bug corrupts (boxes stop 2-11 px short), so it padded
nothing. Repaired by widening the trigger to **5% of the frame dimension** --
comfortably larger than the shortfall it must survive.

| clip | sym 64 edge a | **asym2 edge a** | sym churn | **asym2 churn** |
|---|---|---|---|---|
| microsoft | **0.956** | 0.945 | **1.09%** | 1.16% |
| ipman | 0.890 | **0.914** | 4.68% | **3.69%** |
| interview | **0.838** | 0.819 | 2.89% | **2.79%** |
| tryguys | 0.908 | 0.908 | 1.22% | 1.22% |
| 1917 | 0.797 | **0.850** | 7.25% | **5.84%** |
| codylexi | 0.712 | **0.757** | 1.46% | **1.27%** |
| jensen | **0.797** | 0.794 | 1.62% | **1.14%** |
| **set mean** | 0.842 | **0.855** | 2.89% | **2.45%** |

**Real LOST 0 and real MERGED 0 for both arms**, measured against the seed
gate's height test.

Better edge on 4 of 7, worse on 2 (marginally), tied on 1. Lower churn on 5 of
7. The gains land where they should: **ipman and 1917, the two clips with the
worst interior churn, improve most** (4.68 -> 3.69 and 7.25 -> 5.84), which is
the mechanism doing what it is for -- do not perturb an edge that does not need
it. That also chips at the one open caveat, ipman's unexplained churn, without
explaining it.

### Cost, and a way to make it free

As written this needs a detector pass on the raw frame first, to decide which
edges to pad -- so **2x the seeder's detection cost**. That is avoidable and the
architecture already hands us what is needed: `masks_from_boxes(frame, boxes)`
is called with boxes **from the pipeline's own detector**, so `_instances` can
decide which edges need padding from those boxes rather than running Mask R-CNN
twice. Zero extra cost, and it keeps the existing contract that this seeder only
answers "what is the mask inside this box".

**Not implemented.** pad 64 symmetric is what is in the tree (default 0). This
is a second mechanism and it is Thriloush's call whether it goes in, especially
since the margin is small and the GPU A/B has not run on either arm.
