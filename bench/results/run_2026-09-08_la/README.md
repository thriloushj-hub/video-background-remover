# 5.7 — the look-ahead seed, validated on GPU (8 Sep 2026)

A100-SXM4-40GB, fresh VM, torch 2.11.0+cu128, code at `5dbc4aa`
(`185 passed` on the VM before the run, so this is today's code and not a stale
zip). One command, both arms, **133.1 s**:

    python bench/probe_lookahead_seed.py --clip bilibili_2shot.mp4 \
        --config pipe.json --arm off --arm lookahead:4 \
        --out-json bench/results/run_2026-09-08_la/probe_lookahead.json

`bilibili_2shot.mp4` is source frames 1103–1174. Shot detection finds the cut at
index 56, so shot 2 here is the shipped shot 2 — the sixteen frames where v2
drops a subject's boots.

## The result

**Shot 2 frame 0: `587` → `866`.** The pre-registered expectation was "about
830"; it beat it.

| | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | … | f15 |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 587 | 598 | 607 | 627 | 634 | 638 | 650 | 661 | … | 797 |
| lookahead:4 | 866 | 879 | 885 | 900 | 916 | 923 | 965 | 959 | … | 1079 |

**Every one of the sixteen frames gains**, +273 to +353 rows, mean **+307.5**.
The last four reach 1079, the bottom of frame. Box bottom is 894.

## Why it moved, from the run's own output

    shot 2  seeded from frame 2 of this shot: frame 0 mask tail 0.332, frame 2 0.060
    shot 2  backfilled 2 frame(s) before the seed

The probe measures the tails **independently of the pipeline** and gets
`0.3311, 0.0850, 0.0597, 0.3212` for the shot's first four frames — reproducing
the 5.7 commit's CPU figures (0.333 / 0.080 / 0.054 / 0.332) to within 0.006 by
a separate path. Frame 0 is above the 0.15 gate; frame 2 beats it by 0.272,
well over the 0.10 minimum gain. Frames 0 and 1 are covered by the backward
pass, which is why they gain too rather than keeping the bad seed.

## The control

**Shot 1 is bit-identical between the two arms** (`alpha_bottom` equal on all
56 frames), and it says why it stayed put:

    shot 1  look-ahead: frame 0 mask tail 0.010 <= 0.150; not scanning

That line is new. The 8 Sep 01:24 arm (`../run_2026-09-08_engine/probe_la.json`)
returned this same "nothing changed" on shot 2 with **no note at all**, which
could not be distinguished from the arm never being applied. It is now
distinguishable, and this run's shot 1 is what a genuine decline looks like.

## What this does NOT establish

**One clip, one shot.** `lookahead_frames` stays at its default of **0** and
should stay there until this runs over the eighteen delivered clips — nine of
which contain cuts, 56 shots in total. The failure mode to look for is the
mirror of the one it fixes: a shot where frame 0 is genuinely representative and
a later frame is not (an entrant arriving, a subject leaving), where moving the
seed would cost coverage rather than gain it. Nothing here rules that out; the
relative guard is an argument that it should be rare, not evidence that it is.

Cost on a healthy shot is one extra tail measurement, as shot 1 shows.
