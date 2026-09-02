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
"""
import glob
import json
import os

import cv2
import numpy as np

RVM = "/home/claude/rvm_alpha"
V2 = "/home/claude/v2_alpha"


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


rows = {}
for clip in sorted(os.listdir(RVM)):
    if not os.path.isdir(f"{V2}/{clip}"):
        continue
    ar, a2 = load(f"{RVM}/{clip}"), load(f"{V2}/{clip}")
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

json.dump(rows, open("/home/claude/score/edge_scores.json", "w"), indent=1)
n = len(rows)
print("\nv2 softer band on", sum(r["soft_v2"] > r["soft_rvm"] for r in rows.values()), "of", n)
print("v2 sharper on   ", sum(r["sharp_v2"] > r["sharp_rvm"] for r in rows.values()), "of", n)
print("v2 more real alpha on", sum(r["share_v2"] > r["share_rvm"] for r in rows.values()), "of", n)
print("EDGE DONE")
