# vbgr2 — video background remover

Successor to the YOLO → SAM 3 → MatAnyone 2 notebook. Same shape, three things
added that the notebook structurally could not do, plus a way to prove whether
any of it actually helped.

```bash
pip install -e ".[gpu]"
python -m vbgr.cli selftest          # CPU only, no models, no downloads
python -m vbgr.cli engines           # what's wired, and under what licence
python -m vbgr.cli run -i clips/ -o results/
```

---

## Where this stands — 7 September 2026

**All eighteen delivery clips now run end to end at their full length**, not the
72-frame windows everything was measured on until 5 Sep: **9,640 frames, 56
shots, 6.8 GPU-hours, zero failures, zero bad frames, no subject lost on any
clip.** It writes real per-pixel alpha, a transparent WebM that is actually
transparent (round-tripped at alpha correlation 1.0000), a composite over a
still or moving background, and carries source audio through with the matte
bit-identical.

**On the metrics that can be compared to v1 it does not beat v1, and does not
lose to it.** Nothing tears, and boundary disagreement is a sub-pixel rim across
the set. The comparison against the *other* previous attempt (RVM, re-run here
on the same windows) found something more useful than a table: **the metric
suite cannot separate the two previous attempts either**, which is why
"measurably better" has been so hard to demonstrate.

**Running at full length was worth it, and the reason is unflattering.**
Thirteen of the eighteen clips hold exactly the same people as v1 — zero
disagreement across 6,297 frames. The other five turned up **two genuine v2
defects that eighteen benchmark windows could not see**, both on the failure
modes named in the original brief:

- **1917** drops motion-blurred runners crossing close to camera. Half fixed and
  measured (worst frame 12.3% → 5.0%); the rest is diagnosed, not papered over.
  See `bench/results/run_2026-09-06_fix55/`.
- **bilibili** drops a subject's boots for 17 frames after a cut. Traced to the
  mask stage rather than the detector — the person box is right and the mask is
  up to 298 px short — which is why the fix that worked on 1917 provably cannot
  work here. Not fixed. See `bench/results/run_2026-09-07_rest11/`.

A third difference, on `dance`, is an accepted design decision rather than a
bug: one further-back dancer falls below the subject-size gate, at a cost now
measured at 11 frames on one clip.

Nine of the eighteen clips turn out to contain shot cuts, which was not knowable
from single-shot windows — including `microsoft`, the static talking-head
control. Per-shot re-seeding carries all 56.

Numbers are in `bench/results/`, one directory per run, each with its own
README. `python bench/audit_readme_numbers.py` re-derives every published figure
from the committed JSONs and expects `FAILURES: 0`.

## Start here

1. **Run `python -m vbgr.cli selftest`.** ~15 seconds, no GPU, no downloads. If
   it fails, don't start a batch.
2. **Run `python -m pytest tests/ -q`.** Expect **170 passed**, about twenty
   seconds. A lower number means a stale copy of the source.
3. **Read `docs/BUGS.md`.** Six defects in the v1 outputs that have nothing to
   do with model quality and would survive any model swap. One of them — `ipman`
   is missing 176 of its 501 frames — means part of what reads as "the matting
   broke on fast motion" in that clip is actually *missing frames*.
4. **Baseline, then change one thing at a time.** `bench/` exists so that
   "better" is a number and not a vibe.

> **Licensing was answered on 2 Sep: aim for the best model, treat licensing as
> a commercial question later.** `docs/LICENSING.md` still holds the facts, and
> they have not changed — the two highest-quality engines are non-commercial and
> the default detector is AGPL. The detector is swappable in about twenty lines
> whenever that becomes live.

---

## The diagnosis

Your handoff doc already reached the right architecture — SAM 3 for a clean
seed, MatAnyone 2 for a soft temporally-stable matte — and the three problems
you were left with are all the *same* problem wearing different clothes:

> **A memory propagator can only propagate. It cannot discover, and it cannot
> forget.**

- **Fast, low-contrast limbs (`ipman`).** The kicking leg was never in the seed,
  so there is nothing to propagate. Worse, a blurred limb is *genuinely
  semi-transparent* — those pixels are a real mix of subject and background —
  and a 4-pixel trimap band cannot represent a 30-pixel smear. The model is
  never even asked about the region where the answer lives.
- **Re-entry after occlusion (`1917`).** v1 decided "is this person new?" by
  overlap with the current matte. That is *presence*, not *identity*. A subject
  who is occluded and returns is a stranger; if they return behind someone
  else's matte, the overlap test says "already covered" and they are never
  re-seeded at all.
- **Progressive drift (the stomach that keyed out).** `DRIFT_ANCHOR_WARMUP` was
  the right instinct but treats the symptom. The cause is that the tracker
  writes *every* prediction into memory, including the ones made while the
  subject is occluded or blurred. Those get matched against forever.

So v2 adds, in order of expected payoff:

| Fix | Where | Addresses |
|---|---|---|
| Flow-adaptive trimap + per-frame matting head | `vbgr/motion.py`, `refine.py` | `ipman`, `butter` |
| Flow-warped alpha prior | `vbgr/motion.py` | `ipman`, `butter` |
| Memory quality gate | `vbgr/gate.py` | drift, `1917` |
| DINOv3 ReID re-prompting | `vbgr/reid.py` | `1917` re-entry |
| Foreground decontamination | `vbgr/decontaminate.py` | fringe on every clip |
| Frame-parity + exact fps I/O | `vbgr/video_io.py` | `ipman`'s 176 missing frames |
| Reference-free regression harness | `bench/` | "nothing felt satisfying" |

---

## How the fixes work

### Motion blur — widen the band where, and only where, things move fast

The unknown band of the trimap is set per pixel:

```
width(p) = clip(base + gain · |flow(p)|, base, max)     # 4 → 40 px
```

Implemented with a distance transform rather than dilation, because you cannot
vary a structuring element per pixel but you *can* threshold a distance field
against a per-pixel width map. Static edges keep a 4 px band and stay crisp; a
limb moving 30 px/frame gets a ~31 px band and the matting head is finally asked
about the smear.

That is only half of it, because the head still needs a starting point. So the
previous frame's alpha is warped forward by optical flow and blended in, with
weight proportional to *both* how fast the region is moving and how
photometrically consistent the warp is. Fast limb → strong prior. Static
background → no prior at all. Occlusion boundary → warp is inconsistent, trust
drops to zero, no smearing.

```python
warped, trust = motion.flow_alpha_prior(prev_alpha, prev, curr, flow)
alpha = motion.blend_with_prior(alpha, warped, trust, flow_mag)
trimap = motion.flow_adaptive_trimap(alpha, flow_mag)
alpha = refine.refine_band(alpha, curr, trimap, engine.matte_frame)
```

The whole thing is gated on the frame's 95th-percentile flow magnitude, so on a
talking-head clip it never fires.

> Note on the v1 attempt: the doc says a motion re-seed "mostly failed". I think
> that is because it unioned a *hard* SAM 3 mask into the matte — which cannot
> represent blur by construction, so it could not have worked. Widening the band
> and re-running a matting head is a different operation with a different
> failure mode.

### Re-entry — identity, not presence

`vbgr/reid.py` keeps a small feature pool per identity (DINOv3 embeddings of
background-suppressed crops) and matches detections against it with a global
Hungarian assignment, not greedily — greedy matching swaps identities between
two similar people standing next to each other, which is *the* classic ReID
failure.

Three deliberate design choices:

- **Only high-quality observations are banked.** Bank a frame from the middle of
  an occlusion and the pool slowly becomes a picture of the occluder.
- **Ambiguous matches start a new track.** A wrong merge is unrecoverable; a
  wrong split costs one extra matting pass.
- **Cheap gates first.** The detector runs every N frames; the promptable
  segmenter only runs on boxes the current matte fails to explain. On a
  fixed-cast clip the entire re-entry system costs one YOLO pass every 8 frames.

The bidirectional recovery from v1 is kept and extended: forward pass, reverse
pass seeded at the last frame, and a local forward+backward pass anchored where
ReID is most confident for anyone present at neither end.

### Drift — score frames before they enter memory

`vbgr/gate.py` scores every prediction on four independent signals before it is
committed: area ratio vs. running median, IoU against the flow-warped previous
alpha, boundary binariness, and fragmentation. No single one is reliable, which
is why there are four.

A rejected frame is re-run with `commit_to_memory=False` where the engine
supports it. After `lost_after` consecutive rejects the track is declared lost
and handed to the re-prompting layer.

```
REJECT score=0.000 area=0.0000 flowIoU=0.000 bin=1.000 cc=0 (track collapsed)
```

`DRIFT_ANCHOR_WARMUP` is kept — it is still the cheapest defence — but it is now
a backstop rather than the only one.

### Decontamination

Compositing is `out = a·F + (1-a)·newBG`. If you use the observed pixel as `F`
you are compositing `a·(a·F + (1-a)·oldBG) + (1-a)·newBG`, so the old background
bleeds through the whole boundary band. Multi-level fast foreground estimation
(Germer et al. 2020) solves for the true `F`. Pure NumPy, ~0.7 s per 1080p frame.

Verified against a synthetic composite with known `F` and `B`: converges to
0.3% error, and the shipping defaults remove ~60% of the fringe error at
0.72 s/frame (see `tests/test_core.py::test_decontamination_recovers_a_known_foreground`).

---

## Measuring it

There is **no ground-truth alpha** for `1917`/`ipman`/`butter` — they are film
clips. MAD, MSE, Grad, Conn and dtSSD all need GT, so on your benchmark set they
are unusable. That is why nothing accumulated: every change was argued visually.

`bench/metrics.py` therefore provides reference-free metrics that need no GT:

| metric | measures | direction |
|---|---|---|
| `edge_soft` | fraction of boundary-band pixels that are fractional | higher = less "cut-out" |
| `edge_sharp` | mean \|∇α\| on the band | distinguishes *soft* from merely *blurry* |
| `temporal` | motion-compensated \|α_t − warp(α_{t−1})\| | lower = less flicker |
| `dropouts` | area collapse → recovery events | the ipman failure, counted |
| `frag` | mean significant connected components | lower is better |
| `area_cv` | coefficient of variation of mask area | lower = more stable track |

These are only meaningful *relative to another run on the same clip*. They are a
regression harness, not an absolute score. Reference-based metrics are also
implemented for when you point this at VideoMatte240K or VM108.

The evidence that they measure something real: run on your existing v1 outputs,
they separate exactly the clips you flagged as broken from the ones that work.

```
   clip  frames  edge_soft  edge_sharp  temporal  dropouts   frag  area_cv
microsoft   567     0.1712      0.6720   0.03521         0  2.000   0.0678   <- stable
    dance   363     0.1849      0.6995   0.10867         1  3.055   0.1414
   butter   382     0.2166      0.6923   0.13123         1  2.896   0.5292   <- you flagged
  shakira   369     0.2153      0.6795   0.14116         0  2.484   0.4354
     1917   724     0.2704      0.6803   0.15070         4  1.713   0.4439   <- you flagged
    ipman   325     0.2022      0.6823         -         -  1.561   0.2289   <- you flagged
```

`area_cv` is 6.5× higher on `butter` than on `microsoft`; `1917` is the only
clip with multiple dropout events. Nobody had to squint at a video to see that.

```bash
# score an existing directory of outputs (no GPU)
python bench/run_bench.py score --source "output vids" --run v1 \
    --alpha-from green --out bench/results/v1.json

# run the pipeline over the frozen set and score it
python bench/run_bench.py run --clips bench/clips.txt --run v2_sam2matting \
    --out bench/results/v2.json

python bench/run_bench.py compare bench/results/v1.json bench/results/v2.json

# where did it go wrong? contact sheet of the worst frames, not uniform samples
python bench/contact_sheet.py sheet --source "output vids" --clip 1917 \
    --out sheets/1917.png
```

`bench/clips.txt` is the frozen set. Don't change it casually — the value is
that runs stay comparable over time. Note it includes four *stable* talking-head
clips as a control group: if those regress, a change is trading general quality
for hard-case quality and you need to know that.

---

## Engines

```
key             name                    licence                 mode       commercial
matanyone2      MatAnyone 2             NTU S-Lab License 1.0   streaming  NO
sam2matting     SAM2Matting             CC BY-NC-SA 4.0         sequence   NO
rvm             RobustVideoMatting      GPL-3.0                 streaming  yes
passthrough     flow propagation        MIT (this repo)         streaming  yes
existing_alpha  replay                  n/a                     sequence   yes
```

**Streaming vs sequence is forced by upstream, not invented here.** MatAnyone 2
exposes `InferenceCore.step(image)` — genuinely frame-at-a-time, so memory
gating and mid-shot re-prompting work. SAM2Matting exposes
`init_state(video_path=dir)` + `propagate_in_video(state)` — a generator over a
whole frame directory with no supported interruption point. It therefore
declares no `memory_gate` capability and the pipeline compensates by chunking
the timeline. Pretending otherwise would produce an adapter that silently does
nothing.

Both adapters are written against the actual upstream inference scripts,
including the two details that are easy to get wrong:

- SAM2Matting wants the seed as **288×288 logits** built as `(m > 0.005)·20 − 10`.
  A native-resolution 0/1 mask silently produces a near-empty matte.
- MatAnyone 2 wants **RGB, CHW, float [0,1]** — not BGR HWC uint8 — and warmup
  works by prepending copies of frame 0 and *discarding* those outputs. Getting
  the discard wrong is how you end up N frames short.

Swapping engine is a config string:

```yaml
matting:
  engine: sam2matting     # or matanyone2, rvm
```

```bash
vbgr2 run -i clips/ --require-commercial
# PermissionError: engine 'SAM2Matting' is licensed 'CC BY-NC-SA 4.0'
```

---

## Layout

```
vbgr/
  config.py         all knobs, dataclasses, YAML-loadable
  video_io.py       frame-exact decode/encode, exact rational fps, audio mux
  shots.py          cut detection; asserts ranges tile every frame
  detect.py         YOLO person/prop detection and subject selection
  seed.py           SAM 3 seeding: concept + text + interactive, gated, holes
  motion.py         optical flow, flow-adaptive trimap, warped alpha prior
  gate.py           memory quality gate
  reid.py           DINOv3 identity pool, Hungarian matching, coverage tests
  refine.py         guided filter, band refinement, temporal stabilisation
  decontaminate.py  multi-level foreground estimation, spill suppression
  compose.py        green/alpha/BGRA output; alpha recovery from a green comp
  pipeline.py       orchestration
  engines/          base ABC + matanyone2, sam2matting, rvm, passthrough
bench/              metrics, harness, contact sheets, frozen clip list
tests/              31 CPU-only tests, no models required
docs/               BUGS.md, LICENSING.md, BASELINE.md
```

Every fix in this repo lives **outside** `vbgr/engines/`. That is deliberate: if
the licence question lands on "commercial" and the engine has to be replaced,
none of the motion, ReID, gating, decontamination or harness work is wasted.

---

## Honest status

**Verified here, on real data, CPU-only:** flow estimation and warping,
flow-adaptive trimap widening, the memory gate's accept/reject decisions,
foreground decontamination (against a synthetic composite with known F and B),
temporal stabilisation, shot splitting, compositing round-trip, video I/O frame
parity and exact fps, subject selection, ReID matching and assignment, and the
whole benchmark harness — which was run over 6 of your real clips to produce the
baseline table above. 31/31 tests pass; `selftest` passes.

Two bugs were caught this way and are fixed: `boundary_distance` returned zero
everywhere (making the entire frame "unknown"), and the subject-selection
confidence floor reintroduced the side-entrant bug the height gate exists to fix.

**Not verified, because this sandbox has no GPU and no checkpoints:** the
SAM2Matting and MatAnyone 2 adapters, the SAM 3 seeder, the YOLO detector, and
the DINOv3 feature extractor. They are written against the real upstream APIs
(I read the actual inference scripts, not just the READMEs) and use signature
introspection to tolerate version drift, but expect to spend an hour wiring on
first run. `vbgr/engines/base.py::filter_kwargs` exists for that reason. Start
with `--engine rvm`, which needs only `torch.hub`, to confirm the pipeline
plumbing before touching the research repos.

**Expected gain, honestly:** the frame-parity, fps, decontamination and harness
work I'd bet on. The motion-blur fix I'd bet on for `butter` and moderately for
`ipman` — but `ipman` is fast motion *plus* very low contrast (dark leg, dark
set), and low contrast is the harder half; SAM2Matting's own claims are the main
reason to expect improvement there. The ReID layer should fix `1917`'s re-entry
outright. None of this is a substitute for running the benchmark.
