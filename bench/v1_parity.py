#!/usr/bin/env python3
"""How far is v2 from v1 *as a viewer sees it*, and does a candidate close the gap.

WHY THIS EXISTS
---------------
Thriloush watched the 22 Sep render and said it is still bad, and to make it
look like v1 first.  He is right, and the earlier measurement is why I missed
it: `big_blink` deliberately counted only large on/off swings, because the
first defect found was a lap switching between held and dropped.  That fix
worked.  But the flicker actually visible on tryguys is a **thin line crawling
along every silhouette**, hair worst -- small per pixel, constant, everywhere.
Scoring it out as "boundary noise" is HANDOFF section 7 all over again: a metric
that moves while the picture does not.

So this scores the three things that decide whether it looks like v1, with v1
measured the same way as the target rather than as a baseline to beat:

  crawl_px    THE FLICKER.  Pixels on the silhouette that flip across alpha 0.5
              and flip back one frame later, divided by the silhouette's
              length.  So it reads as "how many pixels of edge wobble per
              frame", independent of how much perimeter a clip has (the trap
              `excess` and `halo` were both retracted for).  No magnitude
              threshold, because the crawl is small per pixel and that is
              exactly what was wrongly discounted.

  miss_pct    THE MISSING PARTS.  Fraction of frame where v1 is solid (>0.9)
              and v2 is absent (<0.1): the tryguys lap, the interview hand.
              Runbo's "can't be missing things like hands".

  extra_pct   the reverse, so a candidate cannot win `miss` by dilating.

  soft_pct    share of pixels at partial alpha (0.1-0.9).  v2 runs 2-3x v1
              here (the 3.2 motion-fix cost).  It is the cost column: hardening
              the edge to stop the crawl shows up here, and on hair it is the
              thing worth keeping.

PARITY, not victory: for crawl and miss, v1's own number is the target.

CANDIDATES (`--arm`), all post-processing v2's alpha, so they can be measured
here before anything is put in the pipeline or re-rendered:

  none        as delivered.
  hyst:H      Schmitt trigger at the 0.5 crossing: a pixel that was in needs to
              fall below 0.5-H to leave, one that was out needs 0.5+H to
              enter.  Rejected on 21 Sep for not moving `big_blink` -- which is
              precisely the metric that could not see the crawl.  Re-measured
              here against the right one.
  rimema:W    temporal EMA with weight W applied ONLY inside the boundary band,
              motion-compensated, interior untouched.
  hyst+rimema combines them.

Every arm is also scored for `soft_pct` and for hair retention, because the
cheapest way to stop a crawl is to make every edge hard, and that would trade
Runbo's complaint for the one he made about attempt 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vbgr import compose, motion  # noqa: E402

V1_GREEN = (151, 253, 119)
V2_GREEN = (0, 177, 64)


def read_alpha(path, key, max_frames=None, scale=1.0):
    cap = cv2.VideoCapture(path)
    out, src = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out.append(compose.alpha_from_greenscreen(f, key))
        src.append(f)
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    return out, src


# --------------------------------------------------------------------------- #
# candidate post-processes
# --------------------------------------------------------------------------- #

def apply_arm(alphas, frames, arm, flowest):
    if arm == "none":
        return alphas
    parts = arm.split("+")
    h = w = 0.0
    for p in parts:
        name, _, val = p.partition(":")
        if name == "hyst":
            h = float(val or 0.15)
        elif name == "rimema":
            w = float(val or 0.6)
        else:
            raise SystemExit(f"unknown arm part {p}")

    out = [alphas[0].copy()]
    prev_out = alphas[0]
    for i in range(1, len(alphas)):
        a = alphas[i].astype(np.float32).copy()

        if w > 0:
            fl = flowest.flow(frames[i - 1], frames[i])
            warped, trust = motion.flow_alpha_prior(prev_out, frames[i - 1],
                                                    frames[i], fl, 18.0)
            band = ((a > 0.02) & (a < 0.98)).astype(np.float32)
            band = cv2.GaussianBlur(band, (0, 0), 2.0)
            ww = np.clip(w * trust * band, 0.0, 1.0).astype(np.float32)
            a = np.clip(a * (1.0 - ww) + warped * ww, 0.0, 1.0)

        if h > 0:
            pin = prev_out > 0.5
            lo, hi = 0.5 - h, 0.5 + h
            inband = (a > lo) & (a < hi)
            a = np.where(inband & pin, np.maximum(a, hi),
                         np.where(inband & ~pin, np.minimum(a, lo), a))
            a = a.astype(np.float32)

        out.append(a)
        prev_out = a
    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def score(alphas, ref=None, rim_px=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_px + 1,) * 2)
    n_pix = float(alphas[0].size)
    crawl_num = bound = 0.0
    n_tri = 0
    prev2 = prev = None
    soft = 0.0
    for a in alphas:
        b = a > 0.5
        u = b.astype(np.uint8)
        rim = (cv2.dilate(u, k) - cv2.erode(u, k)) > 0
        bound += float(((cv2.dilate(u, np.ones((3, 3), np.uint8))
                         - cv2.erode(u, np.ones((3, 3), np.uint8))) > 0).sum())
        soft += float(((a > 0.1) & (a < 0.9)).sum()) / n_pix
        if prev2 is not None:
            crawl_num += float(((b != prev) & (b == prev2) & rim).sum())
            n_tri += 1
        prev2, prev = prev, b
    mean_bound = bound / len(alphas)
    res = {
        "frames": len(alphas),
        "crawl_px": crawl_num / max(n_tri, 1) / max(mean_bound, 1.0),
        "crawl_pct": 100.0 * crawl_num / max(n_tri, 1) / n_pix,
        "soft_pct": 100.0 * soft / len(alphas),
        "boundary_px": mean_bound,
    }
    if ref is not None:
        n = min(len(ref), len(alphas))
        miss = extra = 0.0
        for r, a in zip(ref[:n], alphas[:n]):
            if r.shape != a.shape:                       # v1 is padded to mod-16
                r = r[:a.shape[0], :a.shape[1]]
            miss += float(((r > 0.9) & (a < 0.1)).sum()) / n_pix
            extra += float(((a > 0.9) & (r < 0.1)).sum()) / n_pix
        res["miss_pct"] = 100.0 * miss / n
        res["extra_pct"] = 100.0 * extra / n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", required=True)
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--v1-suffix", default="_v1.mp4")
    ap.add_argument("--v2-suffix", default="_v2.mp4")
    ap.add_argument("--src-dir", default=None, help="source clips, needed by rimema")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--arms", nargs="*", default=["none"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    flowest = motion.FlowEstimator()
    results = {}
    for c in args.clips:
        v1a, _ = read_alpha(os.path.join(args.v1_dir, c + args.v1_suffix),
                            V1_GREEN, args.max_frames, args.scale)
        v2a, v2f = read_alpha(os.path.join(args.v2_dir, c + args.v2_suffix),
                              V2_GREEN, args.max_frames, args.scale)
        src = v2f
        if args.src_dir:
            _, src = read_alpha(os.path.join(args.src_dir, c + ".mp4"),
                                V2_GREEN, args.max_frames, args.scale)
        m1 = score(v1a)
        print(f"\n{c}  ({len(v2a)} frames)")
        print(f"  {'arm':14s}{'crawl_px':>10s}{'vs v1':>8s}{'miss%':>9s}"
              f"{'vs v1':>8s}{'extra%':>9s}{'soft%':>8s}{'vs v1':>8s}")
        print(f"  {'v1 (target)':14s}{m1['crawl_px']:10.3f}{1.00:8.2f}"
              f"{0.0:9.4f}{'-':>8s}{0.0:9.4f}{m1['soft_pct']:8.3f}{1.00:8.2f}")
        results[c] = {"v1": m1, "arms": {}}
        for arm in args.arms:
            out = apply_arm(v2a, src, arm, flowest)
            m = score(out, ref=v1a)
            results[c]["arms"][arm] = m
            print(f"  {arm:14s}{m['crawl_px']:10.3f}"
                  f"{m['crawl_px'] / max(m1['crawl_px'], 1e-9):8.2f}"
                  f"{m['miss_pct']:9.4f}{'':>8s}{m['extra_pct']:9.4f}"
                  f"{m['soft_pct']:8.3f}"
                  f"{m['soft_pct'] / max(m1['soft_pct'], 1e-9):8.2f}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
