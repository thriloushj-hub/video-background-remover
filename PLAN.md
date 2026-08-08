# vbgr2 — project plan

Written 2026-08-08. Reflects the repo at commit `8c9a2b2` (docs/BUGS.md
settlement + contact sheets), before the 16-clip baseline run.

## What this is

A video background remover, v2 of a prior YOLO → SAM 3 → MatAnyone 2 Colab
notebook whose output (`output vids/`, 19 clips) shipped with three known
failure modes: fast/low-contrast limbs keying out (`ipman`), lost re-entry
after occlusion (`1917`), and progressive drift where stable regions
gradually key out (multiple clips, worst on `butter`).

The bet this repo makes: those three failures are one underlying problem —
*"a memory propagator can only propagate; it cannot discover, and it cannot
forget"* (README) — and the fix is three targeted mechanisms (flow-adaptive
trimap + warped alpha prior, DINOv3 ReID re-entry, a memory quality gate)
layered around whichever matting engine does the frame-by-frame work,
plus a foreground-decontamination pass and a reference-free benchmark
harness so "better" is a measured number instead of a vibe.

## What it's trying to prove

Concretely, per `docs/BASELINE.md`'s target table:

| metric | v1 | v2 target |
|---|---|---|
| `ipman` frame count | 325/501 | 501/501 |
| `1917` dropouts | 4 | ≤ 1 |
| `butter` area_cv | 0.529 | ≤ 0.30 |
| `edge_soft` (mean) | 0.211 | +15% |
| `temporal` (mean) | 0.128 | −20% |
| `microsoft` (control) | — | no regression |

These are relative, reference-free metrics (no ground-truth alpha exists for
any of these film clips) — the proof is "the number moved in the right
direction and the control didn't move," not an absolute score.

## Architecture, in one line

Everything that isn't the matting model itself — motion, ReID, memory
gating, decontamination, video I/O, the benchmark — lives outside
`vbgr/engines/`, deliberately, so a licensing decision that forces an engine
swap doesn't waste the rest of the system. See `docs/LICENSING.md`.

---

## Status

### Done and verified (CPU-only, real data, no GPU needed)

All of this has passing tests (38 tests in `tests/test_core.py`, plus
`python -m vbgr.cli selftest`) and/or was run against the real delivered
`output vids/` clips:

- **`video_io.py`** — frame-exact decode/encode, exact rational fps
  end-to-end (`Fraction`, not float), audio mux. Fixes BUGS.md #2.
- **`shots.py`** — cut detection; asserts shot ranges tile every frame
  exactly.
- **`motion.py`** — optical flow, flow-adaptive trimap width, flow-warped
  alpha prior with photometric-consistency trust weighting.
- **`gate.py`** — memory quality gate (area ratio, flow-IoU, boundary
  binariness, fragmentation); accept/reject decisions verified.
- **`reid.py`** — DINOv3-embedding identity pool, Hungarian (not greedy)
  matching, coverage tests. *Matching logic* is tested; the DINOv3
  embeddings themselves are not (see Blocked, below — DINOv3 needs a GPU to
  actually extract features).
- **`decontaminate.py`** — multi-level fast foreground estimation, verified
  against a synthetic composite with known F/B (0.3% error), ~60% fringe
  reduction at 0.72s/frame.
- **`refine.py`** — guided filter, band refinement, temporal stabilisation.
- **`compose.py`** — green/alpha/BGRA compositing round-trip, plus
  `alpha_from_greenscreen` (needed only because v1 delivered no alpha).
- **`bench/`** — the whole reference-free metrics harness (`edge_soft`,
  `edge_sharp`, `temporal`, `dropouts`, `frag`, `area_cv`), contact-sheet
  tooling, compare/regression tooling. Run over 6 of the 16 frozen clips to
  produce `bench/results/v1_baseline.json` (see Blocked below for why not
  all 16 yet).
- **Two real bugs caught by this work and fixed**: `boundary_distance`
  returning zero everywhere (whole frame reads as "unknown"), and a
  subject-selection confidence floor that reintroduced the side-entrant bug
  the height gate exists to fix.
- **Documentation forensics**: `docs/BUGS.md` — six defects in the v1
  outputs diagnosed from evidence (NCC frame-matching for the `ipman`
  offset, not guessed), each with a specific v2 fix pointed at it. Contact
  sheets (`sheets/butter.png`, `sheets/1917.png`) committed as the visual
  evidence for two of them.

### In progress / partially settled

- **BUGS.md #1 (`ipman`'s 176 missing frames)** — NCC evidence strongly
  supports "rendered from a shorter cut of the clip," but the notebook that
  would confirm it isn't in this repo. `notebooks/vbgr2_colab.ipynb` is the
  *v2* runner (imports `vbgr.pipeline.Pipeline`), not the original v1
  handoff notebook — checked 2026-08-08, see BUGS.md for the detail. This is
  now blocked on something outside the repo, not on more analysis time
  inside it (see Blocked).
- **Baseline coverage** — 6 of 16 frozen clips scored
  (`bench/results/v1_baseline.json`), killed by a memory limit partway
  through `jensen`. A run to cover the remaining 10 is starting now
  (`--max-frames`/`--short-side` to stay under the memory ceiling, or
  batched and merged). This needs no GPU — it's reference-free scoring of
  already-delivered `output vids/`, not inference.

### Not started / blocked

- **The engine adapters have never been executed.** `sam2matting.py` and
  `matanyone2.py` are written against the real upstream inference scripts
  (README claims the actual scripts were read, not just READMEs, and that
  signature-introspection via `filter_kwargs` exists specifically to
  tolerate version drift) but zero lines of either have run.
  **Blocked on: GPU/Colab access.** SAM2Matting-SAM3 wants ~24GB for 1080p.
- **YOLO detection (`detect.py`)** — logic is written, untested against a
  real model. Blocked on GPU access (or at minimum a CPU-slow YOLOv8 run,
  which is possible but not attempted).
- **SAM 3 seeding (`seed.py`)** — same: written, not run. Blocked on GPU.
- **DINOv3 feature extraction for ReID** — the *matching* algorithm
  (Hungarian assignment, ambiguous-match-starts-new-track policy) is tested
  with synthetic embeddings; real DINOv3 embeddings have never been
  extracted. Blocked on GPU.
- **The full v2 pipeline, end-to-end, on real footage** — has never run.
  Everything above composes into `pipeline.py`, which is CPU-tested with
  the `passthrough`/synthetic paths but not with a real matting engine in
  the loop. This is the big unknown: the individual pieces are tested in
  isolation, the whole assembled system is not.
- **The licensing decision** — three options laid out in
  `docs/LICENSING.md` (research-only with SAM2Matting/MatAnyone2, pay for
  commercial terms, or build a from-scratch commercial-safe engine —
  "a multi-month project, not an afternoon"). **Undecided, and it gates
  which engine work is worth doing.** If the answer ends up "commercial,"
  most SAM2Matting-specific tuning is wasted; none of the surrounding
  system (motion/ReID/gate/decontam/harness) is, by design.
- **`ipman`'s missing-frames root cause** — see above; needs the original
  notebook or the person who ran it.

---

## Honest uncertainty

Things I don't actually know, stated plainly rather than papered over:

- **Whether the SAM2Matting/MatAnyone2 adapters work at all.** They're
  written against the real scripts, which is better evidence than nothing,
  but "read the source carefully" and "ran without error" are different
  claims. README already flags "expect to spend an hour wiring on first
  run" — that's optimistic-case language; wiring against a research repo's
  actual checkpoint format and shape conventions routinely turns up
  surprises no amount of source-reading catches (dtype/device mismatches,
  an undocumented required kwarg, a checkpoint schema that changed since
  the script was last read). I'd budget more like a day, not an hour, for
  first successful inference on *each* engine, and I could be wrong in
  either direction — there's no way to tighten this without a GPU run.
- **Whether the motion-blur fix actually helps `ipman`.** README says this
  directly: "ipman is fast motion *plus* very low contrast... low contrast
  is the harder half; SAM2Matting's own claims are the main reason to
  expect improvement there." Translation: the flow-adaptive trimap widens
  the region the matting head gets asked about, but whether the head *gives
  a good answer* in that region on a dark leg against a dark set is an
  open question this repo's CPU-only work cannot resolve. This could
  under-deliver on `ipman` even if it clearly helps `butter`.
- **Whether the ReID layer fixes `1917` "outright," as claimed.** The
  matching logic is sound and tested in isolation, but real DINOv3
  embeddings on real footage (motion blur, lighting changes across the
  occlusion, partial occlusion at re-entry) could be noisier than the
  synthetic test embeddings. I'd treat this claim as "should help
  substantially," not "will fix it," until there's a real run.
  - **Timeline for the GPU phase is a bigger unknown than the timeline for
  anything done so far.** Wiring three untested adapters, running the full
  16-clip benchmark on real inference (not just scoring existing outputs),
  and then iterating on whatever the numbers say — that last part in
  particular has no fixed size. If the first real run mostly confirms the
  targets, this is fast. If `ipman` stays broken because the low-contrast
  problem is genuinely harder than the trimap fix addresses, this could
  mean a second iteration on the matting head itself, which is a
  materially bigger job. I don't have enough signal yet to bound this.

---

## Rough sequencing

Not a committed schedule — a realistic ordering given what's blocked on
what. Time estimates are guesses, flagged as such, for a single person
working on this alongside other things.

1. **Now, no GPU needed:**
   - Full 16-clip baseline scoring (in progress as of this doc).
   - Decide licensing (docs/LICENSING.md, three options). This is a
     business/legal call, not an engineering one — it has no time estimate
     because it doesn't depend on this repo at all, but it's the thing
     most likely to be *accidentally* on the critical path if left until
     GPU time is already booked.
   - Optional, cheap: try to track down the original v1 notebook (BUGS.md
     #1) — bounded, low-cost, resolves one open question.

2. **First GPU session (Colab, per `notebooks/vbgr2_colab.ipynb`):**
   - Environment + checkpoint setup, `selftest`, `engines` — mechanical,
     should be quick (< 1 hour) if the install steps in the notebook are
     accurate.
   - Get **one** engine (start with `rvm` per the README's own advice —
     "needs only `torch.hub`" — to validate pipeline plumbing before
     touching the research repos) running end-to-end on one clip.
   - Then SAM2Matting and/or MatAnyone2, whichever the licensing decision
     favors. Guess: a day each for first successful inference, could be
     faster, could be much slower — see Honest uncertainty above.

3. **Once one real engine runs end-to-end:**
   - Run the full 16-clip benchmark for real (`bench/run_bench.py run`),
     not just scoring existing outputs.
   - Compare against `bench/results/v1_baseline.json` (once extended to 16
     clips) using `bench/run_bench.py compare`.
   - Run the ablation grid in the notebook (motion/reid/gate/decontam on
     vs. off) to see which mechanisms are actually earning their keep, per
     clip category (broken/motion-blur/multi-person/control).

4. **Iterate on whatever the numbers say.** Unknown size — see Honest
   uncertainty. If the targets in `docs/BASELINE.md` are mostly hit, this
   is a short tuning pass. If `ipman` or the low-contrast case stays broken,
   this could mean going back to the matting-head problem itself, which is
   a different and bigger job than anything currently in `vbgr/`.

5. **Only if licensing lands on "ship commercially, build your own"**:
   the multi-month path in `docs/LICENSING.md` option 3 — train a matting
   head on a permissively-licensed dataset, pair with a permissively
   licensed tracker (RT-DETR or Faster R-CNN swap-in for YOLO is a ~20-line
   change per LICENSING.md). Not sequenced further here because it's
   contingent on a decision that hasn't been made.

---

## What would change this plan

- **Licensing decision** determines whether step 2+ targets SAM2Matting/
  MatAnyone2 tuning or a from-scratch engine build — different multi-month
  vs multi-week shapes.
- **First real GPU run** determines whether the "day per engine" guess in
  step 2 was optimistic or pessimistic, and whether the motion-blur fix's
  `ipman` uncertainty resolves favorably.
- **Finding (or not finding) the original v1 notebook** closes BUGS.md #1
  but doesn't block anything else — it's informational, not gating.
