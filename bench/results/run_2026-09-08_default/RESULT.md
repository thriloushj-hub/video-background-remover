# The SHIPPED default reproduces the fix (8 Sep 2026)

Every earlier arm set `lookahead_frames` with a flag. A flag cannot confirm a
default, so the probe gained a `default` arm that leaves the config alone, and
this is that arm on `13543c7`.

On the VM, before the run: `Config().seed.lookahead_frames` is **4**,
`lookahead_require_same_cast` is **True**, **190 tests pass**. A100, 126.9 s.

    python bench/probe_lookahead_seed.py --clip bilibili_2shot.mp4 \
        --config pipe.json --arm off --arm default

## Result

    arm off      lookahead_frames 0
    arm default  lookahead_frames 4      <- read from the config, not a flag

    shot 1  identical between arms across all 56 frames
    shot 2  seed frame 0 -> 2
            off  587 598 607 627 634 638 650 661 ...
            def  866 879 885 900 916 923 965 959 ...
            gain mean +307.5 rows, min +273, max +353

Identical to the flag-driven arm in `../run_2026-09-08_la/` on every figure, so
the default ships the behaviour that was measured rather than something near it.

## Provenance

Transcribed from the run's own cell output. `final57.zip` was the **fourth**
`files.download` of the session and did not reach `~/Downloads` — the first two
landed, the third and fourth did not, so the section 6 trap is real but is not
"one per VM". `probe_default.json` and its log stayed on the VM. Re-deriving
them costs 127 s.
