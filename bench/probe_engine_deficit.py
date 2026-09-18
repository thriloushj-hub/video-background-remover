"""JOB 2 -- does the matting engine give back less than it is handed?  (17 Sep)

The claim under test, from the 5.7/5.8 work: the engine loses 1.5-2.5x on top
of whatever the seed reaches, and on bilibili's shot 11 it lost 243 rows
(`seed 825 -> erode_dilate 831 -> guided_filter 831 -> alpha 588`) with every
CPU stage clean.  Job 1 (17 Sep) could not reproduce a multiplier at the seed
frame on four windows (microsoft 15 -> 13 px, interview 15 -> 16, tryguys
6 -> 7; only ipman 13 -> 24), so this sweeps the quantity properly:

    deficit(f) = seed_short - alpha_short(f)   ...on the SEED's own columns

per shot, per edge, for the seed frame and the following frames -- not just
frame 0, and not just the bottom edge.  `bench/probe_engine_coverage.py`
measured frame 0 on the bottom edge only, which is how the question stayed
open.

Columns are anchored on the SEED (solid in the 80-100 px band off the edge),
so every frame of the alpha is compared against what the engine was actually
handed.  The subject moves, so per-frame deficits are reported for the first
DEPTH frames only and the seed-frame value is the headline number.

Arms are config-only, built on `bench/ab_edge_pad_final.run_arm`:

    base    shipped defaults (lookahead on, edge_pad 0)
    band    edge_pad 64 + edge_pad_band (5.10b) -- does a fuller seed survive?
    la0     lookahead_frames 0 -- ONLY on the bilibili_2shot control, where it
            re-creates the seed the 243-row loss was measured against.  On the
            shipped default 5.7 moves that seed and the loss must shrink; on
            la0 it must reproduce.  A control that cannot fail is not one.

Usage (GPU VM):
    python -u bench/probe_engine_deficit.py --config /content/pipe.json \
        --out /content/job2 [--control CLIP.mp4] edge/*.mp4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "bench"))

import ab_edge_pad_final as ab                             # noqa: E402

DEPTH = 12          # frames after the seed frame that are reported per frame


def seed_cols(seed: np.ndarray, edge: str) -> np.ndarray:
    return ab.ref_columns([seed], edge)


def short_px(a: np.ndarray, edge: str, cols: np.ndarray) -> float | None:
    s = ab.edge_stats(np.asarray(a, np.float32), edge, cols)
    return None if s is None else s["short_px"]


def analyse(name: str, arm: str, alphas: np.ndarray, seeds: dict, meta: dict):
    rows = []
    for shot in meta["shots"]:
        fi = shot["start"] + shot["seed_frame"]
        if fi not in seeds:
            continue
        seed = seeds[fi]
        for edge in ab.EDGES:
            cols = seed_cols(seed, edge)
            s_short = short_px(seed, edge, cols)
            if s_short is None:
                continue
            upto = min(shot["end"], fi + DEPTH)
            per = [short_px(alphas[f], edge, cols) for f in range(fi, upto)]
            per = [round(float(x), 1) for x in per if x is not None]
            if not per:
                continue
            rows.append(dict(
                clip=name, arm=arm, shot=f"{shot['start']}-{shot['end']}",
                seed_frame=int(fi), edge=edge, n=int(len(cols)),
                seed_short=round(float(s_short), 1),
                alpha_short_at_seed=per[0],
                deficit_at_seed=round(per[0] - float(s_short), 1),
                worst_deficit=round(max(p - float(s_short) for p in per), 1),
                alpha_short_per_frame=per))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", default=None,
                    help="segment that gets the extra la0 arm (bilibili_2shot)")
    a = ap.parse_args()
    base = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    all_rows = []

    def arms_for(clip):
        arms = [("base", base, 0, False), ("band", base, 64, True)]
        if a.control and os.path.basename(clip) == os.path.basename(a.control):
            la0 = copy.deepcopy(base)
            la0.setdefault("seed", {})["lookahead_frames"] = 0
            arms.append(("la0", la0, 0, False))
        return arms

    clips = list(a.clips) + ([a.control] if a.control else [])
    for clip in clips:
        name = os.path.splitext(os.path.basename(clip))[0]
        for label, b, pad, band in arms_for(clip):
            print(f"\n=== {name} arm {label} (pad={pad} band={band})", flush=True)
            alphas, seeds, meta, _ = ab.run_arm(
                copy.deepcopy(b), pad, clip,
                os.path.join(a.out, name, "render_" + label), band=band)
            print(meta["summary"], flush=True)
            rows = analyse(name, label, alphas, seeds, meta)
            all_rows.extend(rows)
            for r in rows:
                print(f"  {r['clip']:<20}{r['arm']:<6}shot {r['shot']:<10}"
                      f"{r['edge']:<7}n {r['n']:>5}  seed {r['seed_short']:>6.1f}"
                      f" -> alpha {r['alpha_short_at_seed']:>6.1f}"
                      f"  deficit {r['deficit_at_seed']:>+7.1f}"
                      f"  worst {r['worst_deficit']:>+7.1f}"
                      f"  per-frame {r['alpha_short_per_frame']}", flush=True)
            json.dump(all_rows, open(os.path.join(a.out, "job2_deficit.json"), "w"),
                      indent=1)
    print("\nJOB2_DONE", flush=True)


if __name__ == "__main__":
    main()
