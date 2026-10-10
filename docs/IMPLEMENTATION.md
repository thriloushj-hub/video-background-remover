# vbgr2 — implementation guide

For whoever puts this into production. Everything Runbo reviewed on 4 Oct
(and the pure-script re-drop of 7 Oct) comes from the recipe below.

## What it does

Video in, same video out with the background removed: transparent WebM
(VP9 + alpha + source audio), green-screen MP4 with audio, and the matte on
its own. Per shot it detects people once (YOLOv8x + Mask R-CNN), tracks and
mattes them with SAM2Matting (SAM 3 tracker + matting head), re-finds people
who enter or come back, smooths edges over time, decontaminates edge colour,
and writes the outputs. No per-clip tuning: one config for every clip.

## The ship recipe

```
vbgr2 run -c configs/production.yaml -i <input_dir> -o <output_dir> --mode all
# or one file:
vbgr2 one <clip.mp4> -c configs/production.yaml -o <output_dir> --mode all
```

`configs/production.yaml` is the complete config, written out from the code
on 9 Oct. It is the code's defaults plus `io.edge_finish: hard_round` (the
crisp, rounded edge Runbo approved). **Do not use `configs/default.yaml`** —
it is from 23 Aug and no longer matches the pipeline.

Set two paths in it for your machine: `matting.repo_dir` (SAM2Matting clone)
and `matting.checkpoint` (`SAM2Matting-SAM3.pt`).

One difference from the reviewed files, so nobody is surprised: those were
rendered with `edge_finish: soft` and the hard_round edge was applied
afterwards on CPU; the production config applies it inside the pipeline
(same function, `vbgr/refine.py: edge_finish`). Expect the same look, not
byte-identical files. The green MP4 also uses the pipeline's green (0,177,64)
rather than v1's lighter green used in the review set.

## Install (Linux, NVIDIA GPU)

```
pip install -e .[gpu]
pip install ultralytics            # needed by the detector; easy to miss
git clone https://github.com/FudanCVL/SAM2Matting third_party/SAM2Matting
pip install -e third_party/SAM2Matting
# checkpoint: SAM2Matting-SAM3.pt from Hugging Face FudanCVL/SAM2Matting
#   -> checkpoints/SAM2Matting-SAM3.pt
# ffmpeg with libvpx-vp9 must be on PATH (transparent WebM)
vbgr2 selftest                     # CPU-only sanity check
python -m pytest tests             # 270 pass on CPU
```

YOLOv8x and torchvision's Mask R-CNN weights download on first run. A log
line `[reid] DINOv3 unavailable ... classical descriptor` is expected and
harmless.

## Hardware, speed, cost

- Developed and run on Colab A100 (40 GB and 80 GB). A 72-frame 4K window
  peaked at 3.8 GB, so memory is not the constraint; time is.
- **About 2.5–3.3 s per frame on an A100** (all 18 delivery clips: 9,640
  frames in 6.8 GPU-hours on 7 Sep; 8 h 45 min on 26 Sep with the current
  "both" tracking mode). That is roughly **1–1.3 A100-hours per minute of
  24 fps video**. This is far from real time and is the main production cost.
- The VP9-with-alpha encode is CPU-bound and adds a few minutes per long
  clip after the log goes quiet.

## Known limits (what users will see)

From the 18-clip delivery, pure script output:
- Very fast motion and heavy motion blur (1917, ipman, leo): holes in limbs,
  and on 1917 some background soldiers are kept with the main subject.
- First few frames of a shot can miss part of a person at the frame edge
  (tryguys); an occasional single frame drops a small piece (microsoft).
- Dark clothing against a dark floor can lose part of a knee (tryguys).
- Fine flyaway hair is thinner than v1 (es2).
- A small background patch can stick to hair for a second or two (interview).
- Only people are picked as subjects (person detector). A held object is
  kept only if the matte attaches it to the person.

## Licensing — read before shipping commercially

SAM2Matting is CC BY-NC-SA 4.0 (non-commercial) and ultralytics/YOLO is
AGPL-3.0. Runbo's brief said licensing was not a constraint for this build;
for a paid product someone should confirm that. `--require-commercial`
makes the pipeline refuse non-commercial engines. Details: docs/LICENSING.md.

## What is not the product

`bench/` (benchmarks, comparison tooling, `bench/candidates/` experiments),
`compare_app/`, `notebooks/`, `_to_send*`, `_to_delete/`. None of it is
imported by `vbgr/`.

## Next improvements (not in this version)

1. Temporal-consistency pass on edges (flow-aligned smoothing over
   neighbouring frames) — main fix for frame-to-frame instability.
2. Warm-up backfill for the first frames of a shot, and gap-frame fill from
   both neighbours — fix the edge-of-frame and single-frame drops.
3. SAM 3 seeder in place of Mask R-CNN.
Each needs a full GPU re-render and a by-eye review before it ships.
