# run_2026-09-05b — 18 windows, and three clips at full length

The second A100 pass of 5 Sep, and the one that made the project reviewable.
Write-up: `Video_Background_Remover/Full_Length_And_Handover.md` in the vault.

Two runs on the same VM:

1. `python -u vbgr_bench.py --product --fix --outputs` on **18 windows** —
   three more than any previous run, because dlh, es2 and bilibili finally have
   exported v1 alphas to be compared against. 0 failures. `bench18.log`.
2. `python -m vbgr.cli run -i full -o full_out --mode all` on the **full-length**
   1917, butter and ipman. 0 failures, `bad_frames` 0, and butter and ipman are
   multi-shot (9 and 7 cuts) — the first time per-shot re-seeding has run on
   real multi-shot footage. `full3.log`.

## Contents

| | |
|---|---|
| `bench_out/bench_results.json` | every arm per clip, 18 clips |
| `bench_out/<clip>/alpha`, `alpha_on`, `alpha_product` | the persisted alpha stacks, all three arms |
| `bench_out/<clip>/sheet_*.jpg`, `seed_*.png` | contact sheets per arm, and the seeds the gate kept |
| `bench_out/<clip>/product_out/` | the window's alpha mp4 and transparent webm |
| `full_out/` | full-length output, all five modes, three clips |
| `side_by_side/` | source \| v1 \| v2 over one background, built on CPU afterwards |
| `rvm_alpha/` | attempt 1 re-run on all 18 windows |
| `edge_*.json`, `disagree_*.json`, `halo_split*.json` | three arms each |
| `bench18.log`, `full3.log` | the full run logs |

`frames/` and the `*_av`/`green` duplicates of the window outputs were left on
the VM: 224 MB of decoded source frames reproducible from `bench_payload`.

## How it came off the VM

**One zip, one download, verified on arrival** — 371,894,226 bytes, byte-exact,
`unzip -t` clean. The earlier run lost its alphas because a *second*
`files.download` in one Colab session silently does nothing and the runtime was
deleted before anyone checked. The rule is now in HANDOFF §6 and it was
followed here: everything in one archive, downloaded once, existence confirmed
in `~/Downloads` before the runtime was deleted.

## Relationship to run_2026-09-05

That run answered 5.1t and 5.1q on 15 clips. This one is a superset — same
code, three more clips, plus full length and rendered output. The fifteen
shared clips reproduce exactly. Neither supersedes the other on paper: the
earlier directory is the record of what settled the edge question, this one is
the record of the handover.
