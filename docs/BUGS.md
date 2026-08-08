# Defects found in the v1 outputs

These came out of auditing the 19 clips in `output vids/` before writing any
new model code. None of them are model-quality problems, and all of them would
survive any amount of model swapping. Worth fixing regardless of which engine
you land on.

## 1. `ipman` output is 176 frames short of `ipman.mp4` — and it is a file
## mismatch, not a frame-dropping bug

```
ipman.mp4         501 frames   24/1
ipman_matte.mp4   325 frames   1131/50
```

**Diagnosed, not guessed.** Each matte frame was matched back to its source
frame by normalised cross-correlation over the non-green (subject) pixels. The
method was validated on two clips whose alignment is known good: `butter`
scores a median NCC of 0.986 at offset 0 and `1917` scores 0.953. On `ipman`
the best constant offset is **160**, with a median NCC of 0.909, and 172 of the
confident matches land on exactly that offset.

So the matte covers **source frames 160 through 484**:

| | |
|---|---|
| missing at the start | frames 0-159, 160 frames, 0.00s to 6.67s |
| missing at the end | frames 485-500, 16 frames, 20.21s to 20.88s |
| **total** | **176, which is exactly the deficit** |

Verified by eye as well: matte frame 0 is pixel-for-pixel source frame 160, and
matte frame 324 is source frame 484 (`ipman_edges.png`).

**Why this is probably not a code bug.** The missing frames are contiguous at
*both ends*, and neither cut lands on a shot boundary (frame 160 sits inside
shot 57-215, frame 484 inside shot 391-500). A pipeline that loses frames would
be expected to drop a whole shot, or to fail partway and truncate once. Losing
a clean block off the front and a smaller block off the back, at arbitrary
frame numbers, is the signature of the matte having been rendered from a
**shorter cut of the clip** than the `ipman.mp4` that later ended up in the
same folder.

**What this does and does not change.**

* The output still is not time-aligned with `ipman.mp4`, so `ipman` cannot be
  scored against that source, and it cannot be windowed for a like-for-like
  comparison. That part stands.
* But "we have a frame-dropping bug" is the wrong headline. The likely fix is
  to re-export the matte from the full clip, or to store the exact source cut
  alongside every output.
* Confirming this properly needs the v1 Colab notebook. **Checked: it is not
  in this repo.** `notebooks/vbgr2_colab.ipynb` is the *v2* runner — it
  imports `vbgr.pipeline.Pipeline`, sets `cfg.motion.enabled`, runs the
  ablation grid — none of which existed when `output vids/` was produced. The
  original YOLO → SAM 3 → MatAnyone 2 notebook referenced in the handoff doc
  (README's "Successor to the YOLO → SAM 3 → MatAnyone 2 notebook") was never
  checked into this repo, only its output was. So the "wrong source cut"
  theory is still just the best-supported hypothesis from the NCC evidence
  above, not a confirmed root cause, and closing this properly means getting
  hold of that original notebook (or the person who ran it) rather than
  anything in this checkout. Don't spend more analysis time on this from
  inside the repo — the next step is external.

*Guard in v2 regardless:* `shots.shot_ranges` asserts that shot ranges tile
`[0, n)` exactly, `VideoWriter.close(expect=n)` raises `FrameCountMismatch`
unless the count matches, and `Pipeline` asserts frame parity after every
stage. None of that would have caught a mismatched input file, so v2 also
writes the source filename and frame count into the output filename.

## 2. Frame rate is round-tripped through a float

| clip | source | output |
|---|---|---|
| butter | 24000/1001 | 1199/50 |
| interview | 30000/1001 | 1497/50 |
| dlh | 60/1 | 5999/100 |
| ipman | 24/1 | 1131/50 |

23.976 became 23.98, 29.97 became 29.94, 60 became 59.99. Harmless for a
preview; not harmless when the output has to cut against the original, and it
desyncs audio over a long clip.

*Fix in v2:* `video_io` carries an exact `Fraction` end to end and encodes with
`-vsync passthrough`. The selftest asserts `24000/1001` survives a round trip.

## 3. Outputs are padded to a multiple of 16

```
1917.mp4         1920x804      1917_matte.mp4    1920x816
dance.mp4        1920x1080     dance_matte.mp4   1920x1088
```

The matte is 8–12 lines taller than its own source, so it is not pixel-aligned
with the footage it came from. Anyone compositing this in an NLE gets a
sub-pixel offset, and the benchmark harness has to resample to compare at all.

*Fix in v2:* dimensions are preserved; where a model requires a multiple of 16
the padding is added before inference and cropped off after.

## 4. The `output vids` batch emits no alpha channel, and the one file that
## claims to is not transparent

Two separate things here, and an earlier draft of this doc conflated them.

**4a. `transparency.webm` does not render as transparent.** It is in the
`bg_remove_samples` set, it declares `alpha_mode=1` in the container, and it
still decodes fully opaque: a green background with dark smears through it.
Confirmed in a normal video player, not just in tooling. Those smears are the
signature of writing un-decontaminated foreground colour into regions where
alpha is zero, where the colour is mathematically undefined. So a transparent
export path exists but is not producing a usable file.

**4b. The `output vids` batch has no alpha at all.** The delivered
`_matte.mp4` files there are **green composites**, not mattes. That means:

* the alpha cannot be re-keyed, only re-composited over green;
* any edge error is baked in permanently;
* the benchmark harness has to *invert* the green composite to recover an
  approximate alpha, which is lossy (see `compose.alpha_from_greenscreen`).

*Fix in v2:* `output_mode: all` writes green, a real alpha pass, and a
VP9/WebM with a true alpha channel.

## 5. Audio is dropped

Every output is video-only. For a batch tool whose output has to be cut against
the original, that is a workflow problem.

*Fix in v2:* `video_io.mux_audio` copies the source audio track onto the
rendered video.

## 6. No foreground decontamination

The composites use the observed pixel as the foreground colour, so the old
background is mixed into the boundary band and then composited *again* over
green. This produces a fringe that no amount of alpha improvement can remove.
It is most visible on `butter` and `shakira`.

*Fix in v2:* `decontaminate.py`, multi-level fast foreground estimation, on by
default.

---

## What the metrics say about the model-quality problems

From `bench/results/v1_baseline.json` (see `docs/BASELINE.md`). The
reference-free metrics separate the clips you flagged as broken from the ones
that work, which is the main evidence that the harness measures something real:

| clip | edge_soft | temporal | dropouts | area_cv | |
|---|---|---|---|---|---|
| microsoft | 0.171 | **0.035** | 0 | **0.068** | stable talking head |
| dance | 0.185 | 0.109 | 1 | 0.141 | |
| butter | 0.217 | 0.131 | 1 | **0.529** | you flagged this |
| shakira | 0.215 | 0.141 | 0 | 0.435 | |
| 1917 | 0.270 | 0.151 | **4** | **0.444** | you flagged this |
| ipman | 0.202 | — | — | 0.229 | you flagged this; 176 frames missing |

`area_cv` (coefficient of variation of mask area) is 6.5x higher on `butter`
than on `microsoft`, and `1917` is the only clip with multiple dropout events.
Those are the limb-keying-out and track-instability failures, made countable.
