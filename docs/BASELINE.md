# v1 baseline

Produced by running `bench/run_bench.py score` over the delivered `output vids/`
set. This is the number to beat. Raw data: `bench/results/v1_baseline.json`.

```
python bench/run_bench.py score --source "output vids" --run v1_sam3_matanyone2 \
    --alpha-from green --short-side 400 --stride 4 \
    --out bench/results/v1_baseline.json
```

| clip | frames | edge_soft ↑ | edge_sharp ↑ | temporal ↓ | dropouts ↓ | frag ↓ | area_cv ↓ |
|---|---|---|---|---|---|---|---|
| microsoft | 567 | 0.1712 | 0.6720 | **0.0352** | 0 | 2.000 | **0.0678** |
| dance | 363 | 0.1849 | 0.6995 | 0.1087 | 1 | 3.055 | 0.1414 |
| shakira | 369 | 0.2153 | 0.6795 | 0.1412 | 0 | 2.484 | 0.4354 |
| butter | 382 | 0.2166 | 0.6923 | 0.1312 | 1 | 2.896 | **0.5292** |
| ipman | 325* | 0.2022 | 0.6823 | — | — | 1.561 | 0.2289 |
| 1917 | 724 | 0.2704 | 0.6803 | 0.1507 | **4** | 1.713 | **0.4439** |

\* `ipman` is missing 176 of its 501 source frames, so alpha and source are not
time-aligned and every motion-compensated metric is invalid for it. It is scored
without motion compensation and the temporal/dropout columns are omitted rather
than filled with a fabricated number. **Fix the frame drop before using ipman as
a benchmark clip at all.**

## Caveats — read these before quoting the numbers

1. **Alpha is recovered by inverting a green composite.** v1 emitted no alpha
   channel, so there is nothing else to score. `compose.alpha_from_greenscreen`
   works in YCrCb (so a dark subject is not confused with the key) but it
   slightly over-softens genuinely hard edges. This inflates `edge_soft` for
   every clip roughly equally. Only compare a v2 run scored *the same way*, or
   re-score both from a real alpha pass.
2. **Scored at a 400 px short side.** Full 1080p needs ~5 GB per clip just to
   hold the arrays. Every run in a comparison must use the same `--short-side`;
   `compare` warns if they differ.
3. **Six clips, not the full sixteen.** The scoring run was killed by the memory
   limit on this machine partway through `jensen`. On a normal box, run the whole
   `bench/clips.txt`. Use `--max-frames` if memory is tight.
4. **These are relative metrics.** There is no ground truth. `edge_soft = 0.27`
   is not "good" or "bad" in the abstract; it is only meaningful against another
   run on the same clip at the same resolution.

## What the numbers say

The metrics separate the clips you flagged as broken from the ones that work,
without anyone watching a video:

- **`microsoft` is the control.** `temporal` 0.035 and `area_cv` 0.068 — a stable
  talking head, which is what the v1 pipeline was tuned on and what it does well.
- **`butter` has 6.5× the area instability of `microsoft`** (`area_cv` 0.529).
  The contact sheet (`sheets/butter.png`) shows why, and it is **not** primarily
  motion blur: the matte swings between including almost nothing (frame 279,
  area 0.047) and swallowing large opaque slabs of background (frames 126, 162,
  234, area up to 0.553). That is mask leakage plus a lost track, in a crowded
  multi-person music video. The relevant v2 defences are the appearance gate in
  `seed.drop_background_regions` and the gate's "area exploded" rejection —
  not the motion path.
- **`1917` is the only clip with multiple dropout events (4)**, with
  `area_cv` 0.444. The contact sheet shows a background soldier included at
  frame 18 (area 0.437) who is gone by frame 360 (area 0.111). That is the
  subject-selection and re-entry failure, and it is exactly what the ReID layer
  targets.
- **`edge_sharp` is nearly identical across all clips (0.672–0.700).** Read
  together with the modest `edge_soft`, that says the v1 boundaries are
  consistently *soft-ish but not detailed* — which is the expected signature of
  a good propagator working from a segmentation-quality seed, and the thing a
  dedicated matting head should move.

## Targets for v2

Not promises — targets, so there is something to falsify:

| metric | v1 | v2 target | mechanism |
|---|---|---|---|
| `ipman` frame count | 325/501 | 501/501 | frame-parity assertions |
| `1917` dropouts | 4 | ≤ 1 | ReID re-entry + memory gate |
| `butter` area_cv | 0.529 | ≤ 0.30 | appearance gate + memory gate |
| `edge_soft` (mean) | 0.211 | +15% | SAM2Matting head + flow-adaptive trimap |
| `temporal` (mean) | 0.128 | −20% | flow-guided temporal stabiliser |
| `microsoft` (control) | — | no regression | — |

If a change does not move one of these, it is not worth the complexity it costs.
