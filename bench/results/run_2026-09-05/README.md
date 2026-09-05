# run_2026-09-05 — the fix-ON alphas, and the product arm re-run

`python -u vbgr_bench.py --product --fix` on a Colab A100, 15 of 15 clips,
0 failures, ~59 min, one shot per clip, `bad_frames` 0 everywhere. It closed
5.1t (persist the fix-ON alphas) and 5.1q (re-run the product arm on the
finished per-object fix). RVM was reproduced on the same VM from a fresh clone
and a fresh `rvm_resnet50.pth`, so the edge comparison lives inside one run.

Write-up: `Video_Background_Remover/Edge_Question_Fix_On.md` in the vault.

## What is here

| | |
|---|---|
| `bench_results.json` | every arm's `score()` row per clip, plus `product_meta` and per-object areas |
| `edge_alpha.json`, `edge_alpha_on.json`, `edge_alpha_product.json` | `score_edges.py` against the RVM run, one file per v2 arm. Each records the arm it measured |
| `disagree_alpha*.json` | `score_disagreement.py`, same three arms, scored against the exported v1 window alphas |
| `halo_split.json`, `halo_split_alpha_on.json`, `halo_split_alpha_product.json` | shell / solo decomposition, same three arms |
| `product_table.txt` | the run's own two tables as printed |
| `run_log_summary.txt` | per-clip pipeline summary lines |
| `bench_product.log` | the full run log, including the `[sam2matting] seeding N object(s) (explicit)` lines that are the evidence for 5.1q |
| `bench_out/<clip>/sheet_off.jpg`, `sheet_on.jpg`, `sheet_product.jpg` | contact sheets, all three arms |
| `bench_out/<clip>/seed_*.png` | the seed masks the gate kept |
| `bench_out/<clip>/product_out/*.mp4` | the product path's alpha output, with and without audio |

`bench_out/bench_results.json` is the same file as the top-level one. The
top-level copy was reconstructed from the VM's stdout before the archive came
down; when the archive arrived the two compared **equal**, which is why the
transfer route in `Bench_Run_Cells.md` §7f is written up as trustworthy.

## What is NOT here, and why

**The alpha PNG stacks — `alpha/`, `alpha_on/`, `alpha_product/` — are gone.**
Every number above was computed from them on the VM, and all of it is on disk,
but the pixels are not.

That was not the plan. `alphas_2026-09-05.zip` (77 MB) was built and its
download was triggered, but **`files.download` had already been used once this
session and Chrome silently dropped the later calls** — the same trap as
HANDOFF §6, and this time the runtime was deleted before anyone checked
Downloads for the file. `results_small.zip` was the first download of the
session and is the only one that landed; everything in `bench_out/` above came
out of it.

**Consequence:** any future metric change that needs to be evaluated on the
fix-ON arm costs another ~60-minute GPU pass. Nothing already written up
depends on those pixels.

**The rule this earns:** confirm the file exists in `~/Downloads` *before*
deleting a runtime, and treat a second `files.download` in one session as
having failed until proven otherwise.

## Source frames

`bench_out/<clip>/frames/` was in the archive and is deliberately not committed
— 166 MB of decoded source frames, reproducible from `bench_payload/clips_*.zip`.
