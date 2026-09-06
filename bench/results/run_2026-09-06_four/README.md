# run_2026-09-06_four — four more clips at full length, and bilibili's boots

Chosen against Runbo's own brief rather than by size: he named "subject coming
into frame after being hidden" and "motion blur / fast motion" for attempt 2,
and "edge detection or accuracy" for attempt 1.

| clip | frames | shots | why this one | GPU |
|---|---|---|---|---|
| bilibili | 1372 | **13** | a different person in most shots — a new subject entering frame thirteen times | 3419.9s |
| es2 | 879 | 1 | flying curly hair at 60fps — the edge axis | 2425.4s |
| dlh | 661 | 1 | 60fps full-body fast motion | 1687.9s |
| dance3 | 619 | 1 | two dancers, entrant behaviour | 1654.0s |

`bad_frames=0`, `reseeds=0`, `reentries=0`, zero failures on all four. Total
2h33m of A100, against a 2.5h estimate made from measured frame counts.

## Three of the four are clean, and clean means zero

`bench/check_full_clip_solos.py`, both directions, unchanged thresholds:

| clip | v1 holds, v2 does not | v2 holds, v1 does not |
|---|---|---|
| bilibili | 6 frames, max 0.0302 | **0** |
| es2 | **0** | **0** |
| dlh | **0** | **0** |
| dance3 | **0** | **0** |

Across es2, dlh and dance3 — **2,158 frames** — there is not one standalone
region either matte holds that the other does not. Sampled boundary agreement
is **IoU 0.956 (dance3), 0.977 (dlh), 0.983 (es2)** with mean areas within
0.003 of v1's, so the zeros are two mattes agreeing, not two empty mattes. The
alphas were checked for exactly that before the zeros were believed.

`es2_hair_edge.jpg` is the one worth looking at: frame 450 at full resolution
over a flat mid-tone. Both hold every curl and the dangling earring; v1 carries
a slightly lighter rim. A tie at the hardest edge in the delivery.

## bilibili: v2 drops a subject's boots at a cut

**The finding.** At the cut on **f1159** the shot changes to a woman framed from
the waist down — legs and tall black boots against a dark floor. v1 holds the
whole figure. **v2 holds the dress and legs and drops both boots from the ankle
down**, for ~17 frames, then recovers on its own.

`bilibili_cut_loss.jpg` — f1158 / f1159 / f1165 / f1170 / f1176 / f1185, with
**no brightness floor in the composite**, so alpha 0 renders black. That matters:
the default `panel()` in `check_full_clip_solos.py` shows the source at 25% under
alpha 0, and on the first contact sheet the boots looked *present* in v2. They
are not. A picture with a floor in it is a picture that lies.

**Measured over the whole clip** (`bilibili_cov.json`): 50 frames where v1 holds
>0.5% of frame that v2 does not, 42 above 1%, and 16 frames at IoU < 0.80.

**Every burst but one starts exactly on a detected cut.** Accepted shot starts
are 0, 94, 160, 271, 436, 499, 728, 838, 961, 1072, 1103, 1159, 1292; the loss
bursts begin at 160, 436, 1072, 1103, 1159 and 1292. Most last two to four
frames. The one at 1159 lasts seventeen.

**It starts at seeding, not at tracking.** The loss is already present on f1159
itself — the first frame of the shot, the frame the seed is built from. Same
family as 5.5: the person is detected and the part of them that is dark against
a dark background does not reach the mask. **Which of the two seeding stages is
responsible is NOT established** — that needs a run with the seed masks
persisted, and it has not been done. Nothing further is claimed.

`[shots] WARNING: 4 detected cut(s) at [95, 968, 1104, 1300] suppressed by
min_shot_len=12` also appears on this clip. It is **not** the cause — the
suppressed cuts are not where the losses are — but it is the first time that
warning has fired on real full-length footage, and 1104 sits one frame off an
accepted boundary.

## What this run says about the other eleven clips

Two defects in seven full-length clips, both invisible to eighteen benchmark
windows, both on the failure modes named in the brief. The windows say the set
is clean. The full clips say five of seven are clean and two have specific,
reproducible problems. Eleven clips have still never been looked at outside a
3-second slice, at ~4,500 frames and ~3.2 GPU-hours.

## Files

| | |
|---|---|
| `*_alpha.mp4` | the four mattes (gitignored; copies in `_to_send/outputs/full_length/`) |
| `es2_transparent.webm`, `es2_green_av.mp4` | es2 in all three modes |
| `side_by_side/` | source \| v1 \| v2 over one background, all four |
| `*_solos.json` | per-frame both-ways decomposition |
| `bilibili_cov.json` | per-frame area and IoU for bilibili |
| `bilibili_cut_loss.jpg`, `bilibili_sheet.jpg`, `zoom_bilibili_f1170.png` | the boot loss |
| `es2_hair_edge.jpg` | the edge case at full resolution |
| `run_alpha.log`, `run_all.log` | the two batch logs |

`bilibili` was scored in the cloud container rather than on the Windows VM: the
solos pass takes ~5 minutes on 1372 frames of 1080p and the device shell caps
out at two.
