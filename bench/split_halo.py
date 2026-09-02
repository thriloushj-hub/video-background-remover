"""Split `halo` into the part that is over-inclusion and the part that is a
missing subject.

Why this exists
---------------
`coverage_disagreement` calls every baseline-only pixel outside v2's *closed*
silhouette a halo, and reads a high halo as "v1 was keying in background, so v2
is right to be smaller".  That reading holds only while both runs are covering
the same *people*.

On butter they are not.  The window is a camera pull-back, and by frame 66
there are seven dancers in shot.  v2 seeds once at frame 0 and holds its four;
v1 holds all seven.  The three v2 never saw are whole human figures sitting far
from any v2 foreground, so they score as halo -- and butter's 0.541, the number
carrying the entire v2-beats-v1 claim, is substantially that.

So halo is split by whether the component touches v2's foreground at all:

    halo_shell   hugging v2's silhouette  -> v1 over-includes at the boundary.
                 This is the defensible claim.
    halo_solo    standalone, far from any v2 foreground -> v1 holds something
                 v2 does not.  Never quotable on its own: it is a subject v2
                 missed until a picture says otherwise.

`solo_max_frac` is the largest standalone component as a share of the frame in
the frame where it occurs -- a quick "is this person-sized" read.

    python bench/split_halo.py --clip butter
    python bench/split_halo.py --all
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from bench.metrics import halo_split  # noqa: E402
from vbgr.compose import alpha_from_greenscreen  # noqa: E402

V1_DIR = os.path.expanduser("~/mnt/output vids")
RUN = os.path.join(_ROOT, "bench/results/run_2026-08-27")
BASE = json.load(open(os.path.join(_ROOT, "bench/results/v1_window_baselines.json")))


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


def v2_alphas(clip, shape):
    fs = sorted(glob.glob(f"{RUN}/bench_out/{clip}/alpha/*.png"))
    H, W = shape
    return [cv2.resize(cv2.imread(p, 0), (W, H),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            for p in fs]


def split(clip):
    row = BASE.get(clip)
    # mv has no windowed baseline and cannot have one (22 cuts in 588 frames),
    # so its entry is null.  Skip it loudly rather than crashing on --all.
    if row is None:
        print(f"{clip:12s} SKIPPED -- no v1 window baseline exists for this clip")
        return None
    if not row.get("trusted", True):
        print(f"{clip:12s} SKIPPED -- v1 baseline is marked untrusted")
        return None
    lo, hi = row["window"]
    a1 = v1_alphas(clip, lo, hi, row["green_bgr"])
    if not a1:
        return None
    a2 = v2_alphas(clip, a1[0].shape)
    n = min(len(a1), len(a2))
    return halo_split(a1[:n], a2[:n])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--all", action="store_true")
    # The split has to be recomputed per run, not reused across runs: any
    # change that moves the seeds moves v2's silhouette, and the shell/solo
    # boundary is drawn against that silhouette.  The 30 Aug duplicate-seed
    # fix moved interview, so reusing the 27 Aug file would have been wrong
    # on exactly the clip the fix was for.
    ap.add_argument("--run", default=RUN,
                    help="run directory containing bench_out/<clip>/alpha")
    ap.add_argument("--out")
    args = ap.parse_args()
    RUN = os.path.abspath(args.run)
    out_path = args.out or os.path.join(RUN, "halo_split.json")
    got = {}
    if os.path.exists(out_path):
        got = json.load(open(out_path))
    clips = sorted(BASE) if args.all else [args.clip]
    for c in clips:
        if c in got:
            continue
        r = split(c)
        if r is None:
            continue
        got[c] = r
        print(f"{c:12s} halo {r['halo_frac']:.4f} = shell {r['halo_shell']:.4f}"
              f" + solo {r['halo_solo']:.4f}  ({r['solo_components']} solo comps,"
              f" biggest {r['solo_max_frame_frac']:.4f} of frame)")
        json.dump(got, open(out_path, "w"), indent=1)
