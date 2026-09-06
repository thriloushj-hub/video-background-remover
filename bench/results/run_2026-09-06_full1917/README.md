# run_2026-09-06_full1917 — 1917 re-rendered at full length on the 5.5 fix

The A/B in `run_2026-09-06_fix55/` proved the recovery seed works but was
`--mode alpha` only, so the handover package was still carrying 1917 media
rendered before the fix. This is the same clip run again on `bb2634b` with
`--mode all`, which is what `_to_send/outputs/full_length/1917_*` now holds.

    1917: 724f @24.000  1 shot  bad_frames=0  reseeds=0  reentries=11  1849.6s

Same summary line as the 5 Sep run. `1917_transparent.webm` comes back with
Matroska `alpha_mode=1`, which is the field that means anything for VP9 alpha
(see HANDOFF §6).

## It reproduced the A/B exactly, on a different VM and a different day

`bench/check_full_clip_solos.py`, both directions, unchanged thresholds:

| | A/B (`run_2026-09-06_fix55`) | this run |
|---|---|---|
| frames where v1 holds a region v2 does not | 72 | 72 |
| mean fraction of frame | 0.001471 | 0.001471 |
| frames where v2 holds a region v1 does not | 146 | 146 |
| its mean fraction | 0.003392 | 0.003392 |

And it goes further than the summary: **all 205 frames on which either side
holds something the other does not are identical frame-for-frame**, per-frame
fractions included. Two A100s, two days, one matte. The 5 Sep run reproduced
the 2 Sep edge scores at the *table* level; this is the first reproduction at
pixel level.

## Why butter and ipman were not re-rendered

Measured rather than assumed, because "we rendered two of three clips on
different builds" is exactly the kind of thing that comes back later.

**butter cannot be affected.** 5.5 only changes what happens when a recovery
pass is handed a seed box the mask head will not mask — which the engine prints
as `seeding 1 object(s) (auto-split)`. Across butter's **27** recovery seedings
in `run_2026-09-05b/full3.log` that line appears **zero** times.

**ipman is affected by 0.00004 of frame.** It was re-rendered on this VM and
compared to the shipped alpha frame by frame *on the VM* (`ipman_alpha.mp4`
uploaded, both decoded, absolute difference per frame):

    501 frames   mean_abs 2.7e-05   frames with any pixel > 0.1: 30
    all 30 are frames 0-23; worst frame 4,124 px = 0.0027 of frame

A boundary nudge in the first second, not a subject. So 5.5 is a 1917 fix, and
the package says so.

## What did not come off the VM, and why it did not matter

The ipman re-render itself. `files.download` fired, the cell went green,
`~/Downloads` stayed empty — the repeat-download trap in HANDOFF §6, third
occurrence, and a full notebook reload did not clear it. It cost nothing
because the comparison above was computed **on the VM before anything was
downloaded**. That is now the rule rather than the workaround.

## Files

| | |
|---|---|
| `1917_alpha.mp4`, `1917_green_av.mp4`, `1917_transparent.webm` | gitignored (media); copies are in `_to_send/outputs/full_length/` |
| `1917_solos.json` | per-frame both-ways decomposition |
| `1917_sheet.jpg` | the four worst remaining frames — f492, f493, f637, f638 |
| `render1917.log` | the full run log, including every `[reentry]` line |

The sheet is worth looking at: the residual is the blurred runner crossing
close to camera, held by v1 and dropped by v2, at exactly the two clusters
`run_2026-09-06_fix55/README.md` predicted would survive — f490–508, where no
local pass fires at all, and f637–641, where he is seeded and the tracker loses
him again in the blur.
