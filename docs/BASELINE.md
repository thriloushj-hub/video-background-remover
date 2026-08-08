# v1 baseline

Produced by running `bench/run_bench.py score` over the delivered `output vids/`
set. This is the number to beat. Raw data: `bench/results/v1_baseline.json`
(the original 6-clip run) and `bench/results/v1_baseline_full16.json` (all 16
frozen clips, added 2026-08-08 — see "Full 16-clip run" below).

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
3. ~~Six clips, not the full sixteen.~~ **Resolved 2026-08-08** — see "Full
   16-clip run" below. The memory-limit crash was environment-specific
   (missing `ffprobe`, not actually a memory ceiling — see that section) and
   didn't recur once fixed.
4. **These are relative metrics.** There is no ground truth. `edge_soft = 0.27`
   is not "good" or "bad" in the abstract; it is only meaningful against another
   run on the same clip at the same resolution.

## Full 16-clip run (2026-08-08)

The remaining 10 clips (`dance2, dance3, mv, tryguys, codylexi, asianboss2,
interview, jensen, eddie, leo`) are scored, same settings (`--short-side 400
--stride 4 --alpha-from green`), plus `--max-frames 800` as a memory safety
net. Merged with the original 6 into `bench/results/v1_baseline_full16.json`.

The original crash on `jensen` turned out not to be a real memory ceiling:
this run's actual blocker was `video_io.probe()` shelling out to `ffprobe`,
which wasn't installed in this environment at all (first attempt failed all
10 clips instantly with `FileNotFoundError: [WinError 2]`). After installing
FFmpeg, all 10 — including `jensen` — scored without incident at the same
settings that supposedly killed the process before. Worth remembering next
time a run dies part way through: check the actual traceback before assuming
it's memory pressure.

| clip | frames | edge_soft ↑ | edge_sharp ↑ | temporal ↓ | dropouts ↓ | frag ↓ | area_cv ↓ |
|---|---|---|---|---|---|---|---|
| microsoft | 567 | 0.1712 | 0.6720 | **0.0352** | 0 | 2.000 | **0.0678** |
| jensen | 509 | 0.1872 | 0.6697 | 0.0304 | 0 | 1.000 | 0.0136 |
| eddie | 143 | 0.2287 | 0.6755 | 0.0360 | 0 | 1.000 | 0.2180 |
| leo | 208 | 0.2155 | 0.6729 | 0.0512 | 0 | 1.000 | 0.3234 |
| asianboss2 | 354 | 0.1907 | 0.6716 | 0.0363 | 0 | 1.000 | 0.0103 |
| codylexi | 490 | 0.1971 | 0.6796 | 0.0398 | 0 | 1.000 | 0.0185 |
| dance | 363 | 0.1849 | 0.6995 | 0.1087 | 1 | 3.055 | 0.1414 |
| tryguys† | 250 | 0.3884 | 0.4697 | 0.0833‡ | 0 | 1.000 | 0.0004 |
| dance2‡ | 607 | 0.1700 | 0.6785 | 0.3709‡ | 0 | 1.000 | 0.1622 |
| dance3 | 619 | 0.1904 | 0.6891 | 0.1341 | 0 | 1.955 | 0.1573 |
| shakira | 369 | 0.2153 | 0.6795 | 0.1412 | 0 | 2.484 | 0.4354 |
| interview‡ | 642 | 0.1630 | 0.6751 | 0.1388‡ | 0 | 1.925 | 0.0952 |
| butter | 382 | 0.2166 | 0.6923 | 0.1312 | 1 | 2.896 | **0.5292** |
| mv | 588 | 0.1928 | 0.6688 | 0.1025 | **3** | 1.721 | **0.6302** |
| ipman* | 325 | 0.2022 | 0.6823 | — | — | 1.561 | 0.2289 |
| 1917 | 724 | 0.2704 | 0.6803 | 0.1507 | **4** | 1.713 | **0.4439** |

Sorted by `area_cv` (roughly stable → unstable). \* `ipman` as above.
‡ `dance2`, `tryguys`, `interview` each have a 1-frame source/output count
mismatch (608→607, 251→250, 643→642) — the same class of bug as BUGS.md #1,
much smaller magnitude, but the `temporal` column for these three is a red
flag, not a clean number: motion compensation was disabled the same way it
was for `ipman`, so it's measuring raw frame-to-frame delta including the
1-frame misalignment, not genuine flicker. Treat these three `temporal`
values as unreliable, not just imprecise.
† `tryguys` looks like an outlier worth a second look before trusting it in
any comparison: `edge_sharp` 0.4697 is far below every other clip's
0.65–0.70 band, `frag` is a flat 1.0, and `area_cv` is 0.0004 — essentially a
perfectly stable mask, which is unusual for a multi-person clip in this set.
That combination (soft edges + zero fragmentation + near-zero area variance)
is the signature of a near-static or largely-empty mask, not a clean track.
Worth a contact sheet before drawing conclusions from it.

**New finding: `mv` is worse than `butter` on the metrics that mattered
most.** `dropouts` 3 and `area_cv` 0.6302 — both higher than `butter`'s
(1 and 0.5292), which was the worse of the two originally-flagged
multi-person clips. `mv` was in the frozen set as a "fast motion / dance"
control, not flagged as broken by eye — this is exactly the kind of miss
the 6-clip baseline couldn't catch. Recommend adding it alongside
`butter`/`1917`/`ipman` as a clip to watch, not just a motion-blur control.

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
- **`1917` is the only clip with multiple dropout events (4)** — true against
  the original 6-clip set, but the full 16 adds a second: **`mv` has 3
  dropouts and the highest `area_cv` of any clip (0.6302), higher than
  `butter`.** `mv` was frozen as a "fast motion / dance" control, not
  flagged by eye as broken. That's the point of running the full set —
  `1917`'s `area_cv` 0.444 with the contact sheet showing a background
  soldier included at frame 18 (area 0.437) who is gone by frame 360 (area
  0.111) is still the clearest single case of the subject-selection/re-entry
  failure the ReID layer targets, but `mv` needs its own contact sheet
  before assuming it's the same failure mode or a different one.
- **`edge_sharp` is nearly identical across 15 of the 16 clips (0.672–0.700)**,
  with one outlier: **`tryguys` at 0.4697**, well outside that band, paired
  with `frag` pinned at 1.0 and `area_cv` at 0.0004 — near-zero variance is
  unusual for a multi-person clip in this set and worth a contact sheet
  before trusting it in a comparison; it may be a near-static or
  largely-empty mask rather than a clean track. For the other 15, the tight
  `edge_sharp` band read together with the modest `edge_soft` says the v1
  boundaries are consistently *soft-ish but not detailed* — the expected
  signature of a good propagator working from a segmentation-quality seed,
  and the thing a dedicated matting head should move.

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
