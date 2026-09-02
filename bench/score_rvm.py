"""Score RVM (attempt 1) against v1 (attempt 2) on the frozen windows.

Both sides read as 8-bit <=960px PNG alphas, which is the representation
`vbgr_bench.save_alphas` persists, so nothing is compared at a resolution the
other does not have. Same caveat as everywhere else: shell/solo computed this
way are not comparable to numbers computed by resampling v1's video, only to
other arms measured the same way.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench.metrics import (area_jitter, area_stability,          # noqa: E402
                           coverage_disagreement, halo_split)

V1 = "/home/claude/score/v1_alpha"
RVM = "/home/claude/rvm_alpha"
V2 = "/home/claude/v2_alpha"          # optional: v2 benchmark-arm alphas


def load(d):
    return [cv2.imread(p, 0).astype(np.float32) / 255
            for p in sorted(glob.glob(d + "/*.png"))]


def pair(a, b):
    n = min(len(a), len(b))
    h, w = b[0].shape
    A = [cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in a[:n]]
    return A, b[:n]


rows = {}
for clip in sorted(os.listdir(V1)):
    a1 = load(f"{V1}/{clip}")
    ar = load(f"{RVM}/{clip}") if os.path.isdir(f"{RVM}/{clip}") else []
    if not (a1 and ar):
        print(f"{clip:12s} SKIP (v1 {len(a1)}, rvm {len(ar)})")
        continue
    A1, AR = pair(a1, ar)
    r = coverage_disagreement(A1, AR)
    hs = halo_split(A1, AR)
    row = dict(
        mean_area_v1=round(float(np.mean([(x > 0.5).mean() for x in A1])), 4),
        mean_area_rvm=round(float(np.mean([(x > 0.5).mean() for x in AR])), 4),
        cv_v1=round(area_stability(A1), 4), cv_rvm=round(area_stability(AR), 4),
        jit_v1=round(area_jitter(A1), 4), jit_rvm=round(area_jitter(AR), 4),
        shell=round(hs["halo_shell"], 4), solo=round(hs["halo_solo"], 4),
        hole_big=round(r["hole_big"], 4), rim_px=round(r["excess_rim_px"], 2),
    )
    if os.path.isdir(f"{V2}/{clip}"):
        a2 = load(f"{V2}/{clip}")
        if a2:
            _, A2 = pair(a1, a2)
            row["cv_v2"] = round(area_stability(A2), 4)
            row["jit_v2"] = round(area_jitter(A2), 4)
            row["mean_area_v2"] = round(
                float(np.mean([(x > 0.5).mean() for x in A2])), 4)
            # RVM against v2 directly, on the frames both produced
            AR2, A2b = pair(ar, a2)
            rr = coverage_disagreement(AR2, A2b)
            hs2 = halo_split(AR2, A2b)
            row["v2_vs_rvm_shell"] = round(hs2["halo_shell"], 4)
            row["v2_vs_rvm_solo"] = round(hs2["halo_solo"], 4)
            row["v2_vs_rvm_holebig"] = round(rr["hole_big"], 4)
    rows[clip] = row
    print(clip, json.dumps(row), flush=True)

json.dump(rows, open("/home/claude/score/rvm_scores.json", "w"), indent=1)
print("SCORE DONE", len(rows))
