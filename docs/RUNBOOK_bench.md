# Running the full v2-vs-v1 benchmark

Everything needed is in **`vbgr2/bench_payload/`** — four zips, ~21 MB total:
`vbgr_code.zip`, `clips_1.zip`, `clips_2.zip`, `clips_3.zip`.
They are already split under Colab's 10 MB per-file upload limit.

**Rebuild `vbgr_code.zip` with `python bench/build_code_zip.py`** — never by
hand. It enumerates `vbgr/`, `bench/`, `tests/` and the root scripts, verifies
content *and* membership, and prints what it added or dropped. As of 31 Aug it
is **58 entries** (2 Sep).

The hand-built version it replaced refreshed the contents of a frozen 34-entry
list and verified those entries by hash — a check that cannot notice a file
that *should* be present and is not. On 31 Aug the VM reported **118 tests
passed** against 126 locally, because a test written after the list was frozen
had never shipped. No `vbgr/` source file was ever missing, so nothing measured
was invalidated, but only by luck.

## Why this exists as a bundle

The Colab VM recycled four times on 2026-08-16 and each cycle costs a full
setup. This packages the whole run so it goes in one pass.

## Two upload gotchas, both learned the hard way

1. **Colab's sidebar upload silently does nothing while the "Ensure that your
   files are saved elsewhere" warning dialog is open.** Three uploads reported
   success and never appeared on the VM. Dismiss the dialog with **OK first**,
   then upload. Each file must be under 10 MB, which is why the clips are in
   three zips.
2. **Re-uploading a file the VM already has saves it as `vbgr_code (1).zip`.**
   Unzipping `vbgr_code.zip` then runs the OLD code and the run looks perfectly
   fine. Always `ls -la vbgr_code*.zip` and copy the new file over the old one
   before unzipping. Step 4 below does this for you.

Code can also go onto a VM with **zero clicks** — Colab runs Monaco and exposes
it, so a cell's contents can be set directly with
`monaco.editor.getModels()[i].setValue(code)`. That removes the upload step for
code, which is the part that changes every iteration. The clips still need one
human action per fresh VM. See `vbgr_colab_access.md`.

## Steps

1. Open the v4 notebook, connect an **A100**.

2. Run cells **1, 2, 2b, 3** (GPU check, install, imports, checkpoints).
   About 5 minutes on a fresh VM.

3. Files panel → upload, dismissing the warning dialog first:
   `vbgr_code.zip`, `clips_1.zip`, `clips_2.zip`, `clips_3.zip`

4. Unpack, defusing the `(1).zip` trap first:

       !cd /content && ls -la vbgr_code*.zip
       !cd /content && [ -f "vbgr_code (1).zip" ] && mv "vbgr_code (1).zip" vbgr_code.zip; \
         unzip -o -q vbgr_code.zip && mkdir -p win \
         && unzip -o -q -d win clips_1.zip && unzip -o -q -d win clips_2.zip \
         && unzip -o -q -d win clips_3.zip && ls win | wc -l

   Expect `15`.

5. **Run the test suite while you are here.** This is the only environment with
   torch, so `tests/test_subject_loss.py` cannot run anywhere else — off-GPU it
   skips and you see 75 instead of 85.

       !cd /content && pip -q install pytest && python -m pytest tests/ -q

   Expect **151 passed** as of 2 Sep. A lower number usually means an old
   `vbgr_code.zip` unpacked — check the count before trusting the run, not
   after.

   Expect **85 passed**.

6. Run the benchmark:

       !cd /content && python vbgr_bench.py

   About 10 minutes: Mask R-CNN seed plus one SAM2Matting pass per clip, plus
   writing the alphas. Add `--fix` to also run the motion-fix arm (roughly
   triples the time). Pass clip names to run a subset:
   `python vbgr_bench.py 1917 butter`.

7. **Get the results off the VM before it recycles.** This is the step that has
   been skipped twice, and it is why the 20 Aug run was nearly lost and why
   `hole` still cannot be validated:

       !cd /content && zip -q -r bench_out_results.zip bench_out && ls -la bench_out_results.zip
       from google.colab import files; files.download('/content/bench_out_results.zip')

   Roughly 35 MB. It lands in your Downloads folder. Unzip it into
   `vbgr2/bench/results/run_<date>/`.

## What you get back

- `bench_out/bench_results.json` — the table, plus each clip's `object_areas`,
  the per-subject coverage that `subj_lost_at` was computed from.
- `bench_out/<clip>/sheet_off.jpg` — a 24-frame contact sheet per clip.
- `bench_out/<clip>/alpha/*.png` — **new**. Every frame's alpha as an 8-bit PNG
  at ≤960px. This is the artefact that lets a metric change be re-scored
  without another GPU run. Structural questions survive the downscale;
  boundary-detail ones do not, so `edge_soft` and `sharpness` must still come
  off the full-resolution run.

## Reading the table

**Do not read the printed table as a v2-vs-v1 result.** `area_cv` and
`mean_area` measure *coverage*, and v1's matte over-includes, so v2 scoring
lower on both is neither a regression nor proof of an improvement. On the
23 Aug run that table read as a 15/15 loss while the pictures showed the
opposite. The runner prints this warning itself now.

Score the run properly, on CPU, on your own machine:

    python bench/score_disagreement.py --run bench/results/run_<date>

That splits the disagreement into **halo** (v1 keying in background — v2 right
to be smaller), **hole** (v2 tore a subject it is holding — a real defect) and
an **excess rim width**. `halo` high with `hole` ~0 is the only shape of result
that supports a v2-beats-v1 claim. Full reasoning in
`Coverage_Disagreement.md`.

`edge_soft` and `sharpness` are **not** comparable to the v1 column at all —
v1's alpha is recovered from a green composite through `clip((d-12)/28)`, and
that ramp pins sharpness to 0.654-0.673 on every clip regardless of content.
See `V1_Window_Baselines.md`.

`subj_lost_at` is the frame after which the matte stops holding all the seeded
subjects; `never` is the pass condition. It reads the tracker's per-object
areas, and returns `never` rather than guessing if those are unavailable.

`area_cv` is kept as a *drift* diagnostic only. For instability use
`area_jitter`, which is immune to the camera moving — butter's dolly-back
scores `area_cv` 0.563 against `area_jitter` 0.070, and this project has read
that clip's camera move as matte instability three separate times.

## What the runner refuses to do, precisely

`cut_guard` aborts a clip if `compute_scene_cuts` returns more than one shot
start. It does **not** abort on a cut that was suppressed by `min_shot_len`
(12 frames): a cut closer than that to either end of the window is dropped from
the shot starts and only produces a printed WARNING.

That is not hypothetical. On both the 20 and 23 Aug runs ipman printed:

    [shots] WARNING: 1 detected cut(s) at [62] suppressed by min_shot_len=12.

62 is 10 frames from the end of a 72-frame window, so it is suppressed and the
run continues. The contact sheet shows **continuous action across frame 62** —
same two men, same framing, mid-kick — so this looks like a false positive from
the fast kick, the absdiff guard firing on rapid motion the way it used to fire
on lighting ramps.

**If you see that warning again, do not change `min_shot_len` or the threshold
to silence it.** Two things are genuinely unresolved and both need deciding
rather than tuning: whether the runner should abort on a suppressed cut, and
whether the detector should be firing at 62 at all.

---

## If you are new: the whole chain, in order

This runbook covers the **benchmark** path only. The full set of documents a
new person needs, in reading order:

1. **`HANDOFF.md`** (vault) — what the project is, how it is worked, where
   everything lives, the gotchas, and the decisions outstanding. Start here.
2. **`tracker.md`** (vault, Notion-synced) — task state and the daily log.
3. **This runbook** — the 15-clip benchmark, cell by cell.
4. **`Bench_Run_Cells.md`** (vault) — the same thing as a paste-and-go Colab
   checklist with an expected output at every step, plus **§7b** for the
   product pipeline (`vbgr.cli run`), **§7c** for a single clip, and **§7d**
   for the `| grep` buffering trap.
5. **`Colab_Runbook.md`** and **`vbgr_colab_access.md`** (vault) — the account
   split, and how code gets onto a VM without a file picker.

**Two things are not in any document and have to be handed over directly:** the
Colab Pro+ login that owns the GPU, and the v1 delivery folder (the `_matte.mp4`
files), which lives outside the repo and is what every v1 baseline is computed
from.

## Which clip set is canonical (settled 25 Aug)

**`bench_payload/` is canonical. `bench_bundle/` has been moved to
`_to_delete/session_2026-08-25/` — delete it.**

There were two of them, both claiming the same 15 windows, and they were
different encodes: `1917_w4-76.mp4` (1,859,840 B, 4.95 Mbit/s) against
`1917_w4_76.mp4` (2,234,889 B, 5.96 Mbit/s), and so on for all 15. Note even
the naming differed, hyphen against underscore.

Not cosmetic — seed detection differs between them (1917 3→1 kept against
2→1; interview 5 detected against 7). The scored metrics still agreed to about
0.002, so nothing already concluded is invalidated.

Payload wins on three counts: it is the higher-bitrate encode, it is what the
23 Aug run on disk actually used, and this runbook already points at it. Frame
counts and dimensions are identical between the two, and `bench/clips.txt` —
the frozen list itself — is untouched.

`bench_payload/` is gitignored (21 MB of encoded video); the frozen list it
encodes is tracked.
