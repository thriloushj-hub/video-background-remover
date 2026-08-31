"""Score the PRODUCT path's outputs against v1, the same way the bench path is scored.

Every v2-vs-v1 number this project has produced came from `vbgr_bench.py`,
which does not use `pipeline.py` -- it seeds from Mask R-CNN and calls the
engine directly, skipping shots, re-seeding, track health, decontamination and
output writing. So "v2" has always meant the benchmark path.

`vbgr2 run` writes `<clip>_alpha_av.mp4`, a greyscale video of the matte. That
is directly comparable to a v1 window, so the product path can finally be put
through the same halo/hole decomposition. Any disagreement between this and
the bench numbers is the two paths differing, which is exactly the thing worth
knowing.

    python bench/score_pipeline_outputs.py --dir bench/results/pipeline_2026-08-30

Reads the same per-window v1 baselines and reports `halo_shell` / `halo_solo`
rather than raw halo, because raw halo conflates over-inclusion with a subject
the run never saw -- see Halo_Was_Missing_Subjects.
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from bench.metrics import coverage_disagreement, halo_split  # noqa: E402
from vbgr.compose import alpha_from_greenscreen  # noqa: E402

V1_DIR = os.path.expanduser("~/mnt/output vids")
BASE = json.load(open(os.path.join(_ROOT, "bench/results/v1_window_baselines.json")))


def read_alpha_video(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255)
    return out


def v1_alphas(clip, lo, hi, key):
    cap = cv2.VideoCapture(os.path.join(V1_DIR, f"{clip}_matte.mp4"))
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if lo <= i < hi:
            out.append(alpha_from_greenscreen(f, tuple(key)))
        if i >= hi:
            break
        i += 1
    return out


def clip_name(fname):
    m = re.match(r"(.+?)_w\d+[-_]\d+_alpha", fname)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of vbgr2 run outputs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "*_alpha_av.mp4"))
                    or glob.glob(os.path.join(a.dir, "*_alpha.mp4"))):
        clip = clip_name(os.path.basename(p))
        if clip not in BASE:
            print(f"{os.path.basename(p)}: no baseline for {clip!r}, skipping")
            continue
        row = BASE[clip]
        if not row.get("trusted", True) or row.get("window") is None:
            print(f"{clip}: baseline VOIDED, skipping")
            continue
        lo, hi = row["window"]
        b = v1_alphas(clip, lo, hi, row["green_bgr"])
        v2 = read_alpha_video(p)
        n = min(len(b), len(v2))
        if n == 0:
            continue
        if v2[0].shape != b[0].shape:
            v2 = [cv2.resize(x, (b[0].shape[1], b[0].shape[0])) for x in v2]
        cd = coverage_disagreement(b[:n], v2[:n])
        hs = halo_split(b[:n], v2[:n])
        rows[clip] = {"frames": n, **{k: round(float(v), 4) for k, v in cd.items()},
                      **hs}
        print(f"{clip:12s} frames {n:3d}  halo {hs['halo_frac']:.4f} "
              f"(shell {hs['halo_shell']:.4f} solo {hs['halo_solo']:.4f})  "
              f"hole_big {cd['hole_big']:.4f}  rim {cd['excess_rim_px']:.2f}px")
    out = a.out or os.path.join(a.dir, "pipeline_vs_v1.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\n{len(rows)} clip(s) -> {out}")
    print("hole_big is the alarm. halo_solo is never quotable without a picture.")


if __name__ == "__main__":
    main()
