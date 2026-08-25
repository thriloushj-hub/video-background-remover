# Running the full v2-vs-v1 benchmark

Everything needed is in **`vbgr2/bench_payload/`** — four zips, ~21 MB total:
`vbgr_code.zip`, `clips_1.zip`, `clips_2.zip`, `clips_3.zip`.
They are already split under Colab's 10 MB per-file upload limit.

## Why this exists as a bundle

The Colab VM recycled four times on 2026-08-16 and each cycle costs a full
setup. This packages the whole run so it goes in one pass.

## Upload gotcha, learned the hard way

Colab's sidebar upload silently does nothing while the **"Ensure that your
files are saved elsewhere"** warning dialog is open. Three uploads failed to
that before I spotted it. Dismiss the dialog with **OK first**, then upload.
Each file must be under 10 MB, which is why the clips are in three zips.

## Steps

1. Open the v4 notebook, connect an **A100**.
2. Run cells **1, 2, 2b, 3** (GPU check, install, imports, checkpoints).
   About 5 minutes on a fresh VM.
3. Files panel -> upload, dismissing the warning dialog first:
   `vbgr_code.zip`, `clips_1.zip`, `clips_2.zip`, `clips_3.zip`
4. In a new cell:

       !cd /content && unzip -o -q vbgr_code.zip && mkdir -p win \
         && unzip -o -q -d win clips_1.zip && unzip -o -q -d win clips_2.zip \
         && unzip -o -q -d win clips_3.zip && ls win | wc -l

   Expect `15`.

5. Run it:

       !cd /content && python vbgr_bench.py

   About 8 minutes: Mask R-CNN seed plus one SAM2Matting pass per clip.
   Add `--fix` to also run the motion-fix arm (roughly triples the time).
   Pass clip names to run a subset: `python vbgr_bench.py 1917 butter`.

6. Results land in `/content/bench_out/bench_results.json`, with a contact
   sheet per clip at `/content/bench_out/<clip>/sheet_off.jpg`.

## Reading the table

`area_cv`, `dropouts` and `mean_area` are comparable to the v1 column.
`edge_soft` and `sharpness` are **not** — v1's alpha is recovered from a green
composite through `clip((d-12)/28)`, and that ramp pins sharpness to 0.654-0.673
on every clip regardless of content. See V1_Window_Baselines.md.

`subj_lost_at` is the frame after which the matte stops holding all the seeded
subjects. `never` is the pass condition, and it is the only column that
separates a cleaner matte from a lost person.

## What the runner refuses to do

It re-checks every window for shot cuts and aborts on one, because two of the
three original windows turned out to straddle cuts and their numbers were void.

---

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
