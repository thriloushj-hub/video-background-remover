# The cast guard, re-swept over all 79 shots (8 Sep 2026)

Same A100, same clips, same command as `../run_2026-09-08_sweep/`, on
`27e73f0` (`189 passed` on the VM first). **188.9 s.**

    python bench/sweep_lookahead.py --clips-dir clips --config pipe.json --k 4

## Result

    clips 19   shots 79   moved 2   cast_lost 0   cast_gained 0

| clip | shot | frames | before the guard | with the guard |
|---|---|---|---|---|
| bilibili | 11 | 1159–1175 | -> frame 3 | **-> frame 3** (unchanged; this is the fix) |
| butter | 5 | 199–232 | -> frame 3, **2 subjects lost** | **stays at frame 0** |
| butter | 8 | 271–289 | -> frame 3 | **-> frame 3** (cast 7 -> 7) |

butter shot 5 now declines in its own words:

    look-ahead: frame 3 is cleaner (0.023 vs 1.000) but holds 4 of 6
    subjects; staying at frame 0

So the guard removes the one shot in the delivery where the look-ahead would
have traded a subject for a cleaner mask, and leaves both of the others alone.

## Provenance, and one thing that did not come back

Everything above is read from the run's own output and from
`sweep_guard.json` **on the VM**. The zip did not download: this was the
**third** `files.download` of the session, the first two landed, and the third
reported success in a green cell and never reached `~/Downloads` — the trap in
section 6 of the handoff, seen again. The per-shot tails are unchanged from
`../run_2026-09-08_sweep/sweep_lookahead.json`, which IS on disk, because the
guard changes decisions and not measurements; what is lost is only that file's
copy of them. Re-deriving it costs 189 s.
