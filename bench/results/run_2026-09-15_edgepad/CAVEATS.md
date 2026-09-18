# 5.10 follow-up: the two caveats from the 15 Sep sweep, 15 Sep

## Caveat 1 -- where does the added mask area actually go?

Added / removed area at pad 64 vs pad 0, split by whether it lands within 64 px
of a frame edge (the band the fix targets) or in the interior. 7 clips, 3
frames each.

| clip | added @edge | added interior | removed | net | edge share of added |
|---|---|---|---|---|---|
| microsoft | 2.56% | 0.50% | 1.44% | +1.61% | **83.5%** |
| tryguys | 1.98% | 0.66% | 1.48% | +1.16% | 74.9% |
| codylexi | 2.11% | 0.84% | 0.69% | +2.26% | 71.6% |
| interview | 3.44% | 1.68% | 1.24% | +3.88% | 67.1% |
| jensen | 1.82% | 1.04% | 0.62% | +2.24% | 63.6% |
| ipman | 3.68% | 3.13% | 1.54% | +5.27% | 54.0% |
| 1917 | 4.61% | 4.83% | 2.94% | +6.49% | **48.8%** |

## Caveat 1, the control: is the interior movement jitter or dilation?

A pad far SMALLER than the ~30 px mask-cell ramp cannot be recovering the edge
band, so whatever it moves in the interior is the detector's own sensitivity to
an input perturbation. That is the noise floor.

| clip | pad | interior churn | edge added |
|---|---|---|---|
| ipman | 2 | 1.08% | 0.67% |
| ipman | 4 | 1.07% | 1.00% |
| ipman | 8 | 1.83% | 1.90% |
| ipman | **64** | **4.59%** | 3.68% |
| 1917 | 2 | 2.55% | 1.90% |
| 1917 | 4 | 2.33% | 1.95% |
| 1917 | 8 | **6.60%** | 1.80% |
| 1917 | **64** | **6.96%** | 4.61% |
| microsoft | 2 | 0.26% | 0.63% |
| microsoft | 4 | 0.52% | 0.94% |
| microsoft | 8 | 0.65% | 1.85% |
| microsoft | **64** | **1.09%** | 2.56% |

**1917's interior movement is essentially all jitter**: an 8 px pad already
produces 6.60% against pad 64's 6.96%, and a 2 px pad produces 2.55%. That clip
is the heavy motion-blur one and its detector output is unstable to any
perturbation. **microsoft is small throughout** (floor 0.3-0.7%, pad 64 1.09%).

**ipman is the one that does not resolve**: floor ~1.1%, pad 64 4.59%, and it
scales with pad width. About 3.5% of interior movement is caused by the large
pad and is NOT explained.

A hypothesis that does NOT survive its own control: torchvision's
`GeneralizedRCNNTransform` pins the longer side to `max_size=1333`, so a 64 px
pad puts the subject at ~94% of its previous detection scale. But that
arithmetic is identical for microsoft (1920x960 -> 2048x1088, same 0.694 ->
0.651 scale change) and microsoft barely moves. So the scale change cannot be
the explanation on its own, and no mechanism is claimed.

## Caveat 2 -- the detection count moving on interview (5.7 -> 7.7)

Resolved, and it is benign. Every instance the padded run adds is a small
background figure in the middle of the frame:

- interview f0, the four NEW instances: heights **71, 106, 79, 109 px** against
  a tallest subject of 1144 px -- height ratios **0.06-0.10**, against the seed
  gate's **0.60** threshold. **None of them touches any frame edge**, so none is
  a phantom created by the replicated pad.
- interview f48: pad 64 finds 8 against pad 0's 7, and **every one matches an
  existing pad-0 instance** at IoU 0.80-0.96. That count difference is jitter
  around the 0.80 score cut, not a new subject.
- 1917 f0 goes the OTHER way, **3 -> 2**. The count moves in both directions,
  which is the signature of score jitter rather than a systematic effect.
- 1917 f24's one NEW instance is 183 px against 862 -- ratio 0.21, also far
  below the gate.

**So no extra subject is created.** And structurally it could not become one
anyway: in the product path `masks_from_boxes` is handed boxes from the
pipeline's OWN detector, and this seeder only answers "what is the mask inside
this box" -- an extra Mask R-CNN instance matters only if it wins the
`match_iou` contest against a real box. That was the reasoning offered
yesterday; this is the measurement behind it.
