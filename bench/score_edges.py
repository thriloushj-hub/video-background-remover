"""Edge quality, RVM against v2 -- the comparison this project could never make.

`edge_soft` and `sharpness` have been banned from every v2-vs-v1 claim, and
rightly: v1's alpha is *recovered* from a green composite by pushing chroma
distance through clip((d-12)/28), and that ramp stamps its own boundary profile
on everything -- measured sharpness spans only 0.654-0.673 across fifteen very
different clips. The number describes the recovery, not the matte.

RVM is different. It outputs genuine per-pixel alpha, exactly as v2 does. So
between RVM and v2 these metrics mean what they say, and attempt 1 finally
gives us a like-for-like edge comparison -- on the axis Runbo actually named
when he said attempt 1 "didn't have the best edge detection".

Caveat, stated rather than buried: both sides are read at the persisted <=960px
width, so absolute values are not the full-resolution ones. Both sides are at
the SAME width, so the comparison is fair; the absolutes are not quotable
against a full-res run.

Which v2 arm this is pointed at matters, and it is the whole reason 5.1t
existed. Until 2 Sep the only v2 alphas ever written to disk were the fix-OFF
ones, and 3.6 measures the motion fix at edge_soft +0.1098 -- the same size as
the gap this script was reporting against RVM. So the directories are arguments
now rather than constants, and the arm is printed and stored beside the result,
because "v2's boundary is the harder one" is a different claim depending on
which arm produced it.

    python bench/score_edges.py --rvm rvm_alpha --v2 bench_out --arm alpha_on

The JSON it writes is now {"arm": ..., "rows": {...}} rather than the bare row
map the 2 Sep file holds; the older file is left as it is rather than rewritten,
since it is the record of what was actually run that day.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def load(d):
    return [cv2.imread(p, 0).astype(np.float32) / 255
            for p in sorted(glob.glob(d + "/*.png"))]


def band(a, w=6):
    hard = (a > 0.5).astype(np.uint8)
    k = np.ones((w, w), np.uint8)
    return (cv2.dilate(hard, k) - cv2.erode(hard, k)) > 0


def edge_soft(a):
    """Share of the boundary band that is genuinely fractional."""
    m = band(a)
    if m.sum() < 50:
        return 0.0
    v = a[m]
    return float(((v > 0.05) & (v < 0.95)).mean())


def sharpness(a):
    """How fast alpha crosses the band. Lower = softer/blurrier."""
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    m = band(a)
    if m.sum() < 50:
        return 0.0
    return float(np.clip(g[m].mean(), 0, 1))


def soft_share(a):
    """Share of the WHOLE frame sitting strictly between 0 and 1.

    A hard mask scores ~0 here no matter how clean its outline is. This is the
    measure of 'is there real alpha at all', which is the structural difference
    between a matte and a key.
    """
    return float(((a > 0.05) & (a < 0.95)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvm", default="/home/claude/rvm_alpha",
                    help="dir of per-clip RVM alpha PNG folders")
    ap.add_argument("--v2", default="/home/claude/v2_alpha",
                    help="dir of per-clip v2 alpha folders")
    ap.add_argument("--arm", default="",
                    help="subdirectory inside each v2 clip dir holding the "
                         "arm's PNGs, e.g. 'alpha' (fix-off) or 'alpha_on' "
                         "(fix-on). Empty means the clip dir itself.")
    ap.add_argument("--out", default="/home/claude/score/edge_scores.json")
    a = ap.parse_args()

    def v2_dir(clip):
        return os.path.join(a.v2, clip, a.arm) if a.arm else os.path.join(a.v2, clip)

    rows = {}
    for clip in sorted(os.listdir(a.rvm)):
        if not os.path.isdir(v2_dir(clip)):
            continue
        ar, a2 = load(os.path.join(a.rvm, clip)), load(v2_dir(clip))
        n = min(len(ar), len(a2))
        if not n:
            continue
        h, w = a2[0].shape
        ar = [cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in ar[:n]]
        a2 = a2[:n]
        step = 2
        rows[clip] = dict(
            soft_rvm=round(float(np.mean([edge_soft(x) for x in ar[::step]])), 4),
            soft_v2=round(float(np.mean([edge_soft(x) for x in a2[::step]])), 4),
            sharp_rvm=round(float(np.mean([sharpness(x) for x in ar[::step]])), 4),
            sharp_v2=round(float(np.mean([sharpness(x) for x in a2[::step]])), 4),
            share_rvm=round(float(np.mean([soft_share(x) for x in ar[::step]])), 5),
            share_v2=round(float(np.mean([soft_share(x) for x in a2[::step]])), 5),
        )
        print(clip, json.dumps(rows[clip]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"arm": a.arm or "(clip dir)", "rows": rows},
              open(a.out, "w"), indent=1)
    n = len(rows)
    arm = a.arm or "(clip dir)"
    print(f"\nv2 arm: {arm}   clips: {n}")
    print("v2 softer band on", sum(r["soft_v2"] > r["soft_rvm"] for r in rows.values()), "of", n)
    print("v2 sharper on   ", sum(r["sharp_v2"] > r["sharp_rvm"] for r in rows.values()), "of", n)
    print("v2 more real alpha on", sum(r["share_v2"] > r["share_rvm"] for r in rows.values()), "of", n)
    print("EDGE DONE")


if __name__ == "__main__":
    main()
