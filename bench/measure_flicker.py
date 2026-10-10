#!/usr/bin/env python3
"""Per-clip flicker measurement, and an instrument that can tell flicker from motion.

WHY THIS EXISTS
---------------
Runbo's only verdict on the 20 Sep drop was "It seems to flicker a lot".  The
answer given that day was a single set-wide number -- per-frame flip rate,
v1 0.693% against v2 0.772% -- produced by an ad-hoc cell that was never
committed.  That is HANDOFF section 7 entry twenty-nine: a harness that cannot be
re-read cannot be re-checked.  This is the committed version.

It also fixes a real weakness in that number.  `flip_rate` counts every pixel
that crosses alpha 0.5 between consecutive frames, and a subject that MOVES
crosses plenty of them.  So a clip with a fast dancer scores high for being
correct, and "v2 flickers 11% more than v1" could just as easily read "v2's
subjects are bigger".  Three additions separate the two:

  flip_rate    fraction of frame crossing 0.5 per frame pair.  Kept unchanged so
               the 0.693 / 0.772 figures stay comparable.

  blink_rate   fraction of frame where a pixel flips at t and flips BACK at
               t+1 (an A-B-A pattern over three frames).  Monotone motion
               cannot produce this: a pixel the arm sweeps over goes out and
               stays out.  A pixel that leaves and returns one frame later is
               flicker by construction.  THIS is the number the fix has to move.

  blink_per_bpx  blink pixels divided by mean boundary length.  A clip with
               seven dancers has far more perimeter than a talking head, and
               `excess` (25 Aug) and `halo` (29 Aug) were both retracted for
               being a fraction of area quoted as a claim.  Same trap, same
               guard.

  boundary_frac  mean perimeter as a fraction of frame, so a reader can see
               how much of a rate difference is simply more edge.

CONTROL: microsoft is a static talking head.  Its blink_rate should be near
zero on both builds.  If it is not, the instrument is measuring compression
noise rather than the matte, and nothing else in the table can be read.

CAVEAT, stated rather than buried: v1 ships no alpha, only a green composite,
so v1's alpha is recovered with `compose.alpha_from_greenscreen`, which
over-softens hard edges.  Where a clip has a v2 `_alpha.mp4` this script scores
v2 BOTH ways and prints both, so the size of that instrument gap is visible
instead of assumed.  Read the v1-vs-v2 comparison off the green column only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vbgr import compose  # noqa: E402

V1_GREEN = (151, 253, 119)   # auto-detected on every delivered clip


def _frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()


def alpha_stream(path, mode, key=V1_GREEN, max_frames=None):
    """Yield float32 alpha in [0,1] from a green composite or an alpha video."""
    for f in _frames(path, max_frames):
        if mode == "green":
            yield compose.alpha_from_greenscreen(f, key)
        else:
            yield (f[..., 0].astype(np.float32) / 255.0)


def score(path, mode, key=V1_GREEN, thresh=0.5, max_frames=None):
    """Streaming: keeps three binary masks, never the whole clip."""
    k3 = np.ones((3, 3), np.uint8)
    prev2 = prev = None
    n_pix = None
    flips = 0.0          # summed fraction over frame pairs
    n_pairs = 0
    blinks = 0.0         # summed fraction over frame triples
    n_triples = 0
    bound = 0.0
    n_frames = 0
    worst_blink, worst_at = 0.0, -1

    for i, a in enumerate(alpha_stream(path, mode, key, max_frames)):
        b = (a > thresh)
        if n_pix is None:
            n_pix = float(b.size)

        # boundary length: dilate(m) - erode(m), i.e. the 1px rim
        u = b.astype(np.uint8)
        rim = cv2.dilate(u, k3) - cv2.erode(u, k3)
        bound += float(rim.sum())
        n_frames += 1

        if prev is not None:
            flips += float(np.count_nonzero(b != prev)) / n_pix
            n_pairs += 1
        if prev2 is not None:
            # A-B-A: differs from t-1 AND equals t-2  ->  it went and came back
            blink = np.count_nonzero((b != prev) & (b == prev2))
            fr = float(blink) / n_pix
            blinks += fr
            n_triples += 1
            if fr > worst_blink:
                worst_blink, worst_at = fr, i

        prev2, prev = prev, b

    mean_bound = bound / max(n_frames, 1)
    blink_rate = blinks / max(n_triples, 1)
    return {
        "frames": n_frames,
        "flip_rate_pct": 100.0 * flips / max(n_pairs, 1),
        "blink_rate_pct": 100.0 * blink_rate,
        "blink_per_bpx": (blink_rate * n_pix / mean_bound) if mean_bound else 0.0,
        "boundary_frac_pct": 100.0 * mean_bound / n_pix if n_pix else 0.0,
        "worst_blink_pct": 100.0 * worst_blink,
        "worst_blink_frame": worst_at,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", required=True,
                    help="folder of v1 green composites")
    ap.add_argument("--v2-dir", required=True,
                    help="folder of v2 green composites")
    ap.add_argument("--v1-suffix", default="_matte.mp4")
    ap.add_argument("--v2-suffix", default="_green.mp4")
    ap.add_argument("--v1-key", default="151,253,119",
                    help="v1's green key, BGR; auto-detected as this on every "
                         "delivered clip")
    ap.add_argument("--v2-key", default="0,177,64",
                    help="v2 writes compose.DEFAULT_GREEN, which is NOT v1's "
                         "key -- getting this wrong gives mean_area 1.0")
    ap.add_argument("--v2-alpha-dir", default=None,
                    help="optional: folder of v2 <clip>_alpha.mp4 for the "
                         "instrument cross-check")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.clips:
        clips = args.clips
    else:
        clips = sorted(
            f[: -len(args.v2_suffix)] for f in os.listdir(args.v2_dir)
            if f.endswith(args.v2_suffix))

    k1 = tuple(int(x) for x in args.v1_key.split(","))
    k2 = tuple(int(x) for x in args.v2_key.split(","))

    rows = {}
    for c in clips:
        v1p = os.path.join(args.v1_dir, f"{c}{args.v1_suffix}")
        v2p = os.path.join(args.v2_dir, f"{c}{args.v2_suffix}")
        if not (os.path.exists(v1p) and os.path.exists(v2p)):
            print(f"  skip {c}: missing {'v1' if not os.path.exists(v1p) else 'v2'}")
            continue
        r = {"v1": score(v1p, "green", k1, max_frames=args.max_frames),
             "v2": score(v2p, "green", k2, max_frames=args.max_frames)}
        if args.v2_alpha_dir:
            ap_ = os.path.join(args.v2_alpha_dir, f"{c}_alpha.mp4")
            if os.path.exists(ap_):
                r["v2_alpha"] = score(ap_, "alpha", max_frames=args.max_frames)
        rows[c] = r
        print(f"  {c:12s} v1 blink {r['v1']['blink_rate_pct']:.4f}%  "
              f"v2 blink {r['v2']['blink_rate_pct']:.4f}%  "
              f"(flip {r['v1']['flip_rate_pct']:.3f} -> "
              f"{r['v2']['flip_rate_pct']:.3f})", flush=True)

    print()
    hdr = (f"{'clip':12s} {'v1 flip%':>9s} {'v2 flip%':>9s} "
           f"{'v1 blink%':>10s} {'v2 blink%':>10s} {'ratio':>6s} "
           f"{'v1 b/bpx':>9s} {'v2 b/bpx':>9s} {'v2 bnd%':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for c, r in sorted(rows.items(),
                       key=lambda kv: -kv[1]["v2"]["blink_rate_pct"]):
        a, b = r["v1"], r["v2"]
        ratio = b["blink_rate_pct"] / a["blink_rate_pct"] if a["blink_rate_pct"] else float("nan")
        print(f"{c:12s} {a['flip_rate_pct']:9.3f} {b['flip_rate_pct']:9.3f} "
              f"{a['blink_rate_pct']:10.4f} {b['blink_rate_pct']:10.4f} "
              f"{ratio:6.2f} {a['blink_per_bpx']:9.3f} {b['blink_per_bpx']:9.3f} "
              f"{b['boundary_frac_pct']:8.3f}")

    if rows:
        for k in ("flip_rate_pct", "blink_rate_pct"):
            m1 = float(np.mean([r["v1"][k] for r in rows.values()]))
            m2 = float(np.mean([r["v2"][k] for r in rows.values()]))
            print(f"\nset mean {k:15s} v1 {m1:.4f}   v2 {m2:.4f}")

    if args.v2_alpha_dir:
        print("\ninstrument cross-check (green-recovered vs real alpha, v2):")
        for c, r in sorted(rows.items()):
            if "v2_alpha" in r:
                print(f"  {c:12s} blink% green {r['v2']['blink_rate_pct']:.4f}"
                      f"   alpha {r['v2_alpha']['blink_rate_pct']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
