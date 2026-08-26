"""Is each clip's v1 baseline window the same footage as the v2 window clip?

    python bench/check_window_alignment.py

Why this exists
---------------
It found a real defect on 2026-08-26, and one that had already produced a
published conclusion.

`v1_window_baselines.json` records, per clip, a window of frame indices into
that clip's v1 *matte* file. The v2 side runs on a window clip cut from the
*source*. Nothing checked that the two were the same footage, and on ipman they
were not: the baseline row used matte frames 61-133, which is a close-up of one
fighter, while the window clip is a wide two-shot of both. Every ipman v1-vs-v2
number produced between 16 and 26 Aug compared different shots.

The failure is specific to ipman for a knowable reason: its matte covers only
source frames 160-484 of 501 (docs/BUGS.md #1), so matte indices and source
indices differ by 160 on that clip and match everywhere else. The correct
window, source 318-389, is matte 158-229.

Method
------
Take the window clip's frame 0 -- which is source footage -- and NCC its luma
against every frame of the matte file, inside v1's own recovered alpha so the
green background does not dominate. An aligned clip peaks exactly at the
recorded window start.

The discriminator is sharp. The fourteen aligned clips score 0.87-1.00 at their
recorded start. ipman scores **-0.186** there: not a weak match, an
anti-correlated one.

Run this after changing any window, and before trusting a v1 column.
"""
import argparse
import json
import os
import sys
import tempfile
import zipfile

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vbgr.compose import alpha_from_greenscreen  # noqa: E402

DEFAULT_V1_DIR = r"C:\Users\Tirloosh\Downloads\output vids -20260807T044446Z-1-001\output vids"
PROBE_W = 160
MIN_MASK_PX = 40
OFFSET_TOL = 3
GOOD_NCC = 0.70


def _gray(frame, shape):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), shape).astype(np.float32)


def _ncc(a, b, mask):
    a, b = a[mask], b[mask]
    if a.size < MIN_MASK_PX:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / d) if d else -1.0


def window_clips(payload_dir, out_dir):
    """Extract the window clips out of the payload zips, so no manual unzip is needed."""
    got = {}
    for z in sorted(f for f in os.listdir(payload_dir) if f.startswith("clips_")):
        with zipfile.ZipFile(os.path.join(payload_dir, z)) as zf:
            for n in zf.namelist():
                if n.endswith(".mp4"):
                    zf.extract(n, out_dir)
                    got[n.split("_w")[0]] = os.path.join(out_dir, n)
    return got


def check(clip, spec, matte_path, clip_path, step=3):
    lo = spec["window"][0]
    key = tuple(spec["green_bgr"])

    cap = cv2.VideoCapture(clip_path)
    ok, probe_frame = cap.read()
    cap.release()
    if not ok:
        return None

    cap = cv2.VideoCapture(matte_path)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    h, w = frame.shape[:2]
    shape = (PROBE_W, max(2, int(PROBE_W * h / w)))
    probe = _gray(probe_frame, shape)

    scores, j = {}, 0
    while ok:
        # coarse global scan, but always score the frames around the claimed start
        if j % step == 0 or abs(j - lo) <= 4:
            a = cv2.resize(alpha_from_greenscreen(frame, key), shape)
            m = a > 0.5
            if m.sum() >= MIN_MASK_PX:
                scores[j] = _ncc(probe, _gray(frame, shape), m)
        j += 1
        ok, frame = cap.read()
    cap.release()
    if not scores:
        return None

    best = max(scores, key=scores.get)
    at_start = scores.get(lo)
    offset = best - lo
    aligned = abs(offset) <= OFFSET_TOL and scores[best] > GOOD_NCC
    return dict(window_start=lo, best_match=best, offset=offset,
                ncc_best=round(scores[best], 4),
                ncc_at_start=None if at_start is None else round(at_start, 4),
                verdict="aligned" if aligned else
                        ("MISALIGNED" if abs(offset) > OFFSET_TOL else "weak match -- check"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", default=os.environ.get("VBGR_V1_DIR", DEFAULT_V1_DIR))
    ap.add_argument("--payload", default=os.path.join(_ROOT, "bench_payload"))
    ap.add_argument("--baselines", default=os.path.join(_ROOT, "bench", "results",
                                                        "v1_window_baselines.json"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "bench", "results",
                                                  "window_alignment_check.json"))
    ap.add_argument("clips", nargs="*",
                    help="clip names to check (default: all). Checking a few at "
                         "a time keeps memory down on long matte files.")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing --out instead of replacing it")
    a = ap.parse_args()

    V1 = json.load(open(a.baselines))
    if not os.path.isdir(a.v1_dir):
        raise SystemExit(f"v1 delivery folder not found: {a.v1_dir}\n"
                         f"pass --v1-dir or set VBGR_V1_DIR")

    with tempfile.TemporaryDirectory() as tmp:
        clips = window_clips(a.payload, tmp)
        hdr = f"{'clip':12}{'start':>7}{'best':>7}{'off':>6}{'ncc':>8}{'@start':>9}  verdict"
        print(hdr)
        print("-" * len(hdr))
        rows, bad = {}, []
        for clip, spec in V1.items():
            if clip not in clips:
                continue
            if a.clips and clip not in a.clips:
                continue
            if not spec.get("trusted", True) or spec.get("window") is None:
                print(f"{clip:12}baseline voided -- nothing to align against")
                continue
            matte = os.path.join(a.v1_dir, f"{clip}_matte.mp4")
            if not os.path.exists(matte):
                print(f"{clip:12}no matte file")
                continue
            r = check(clip, spec, matte, clips[clip])
            if r is None:
                print(f"{clip:12}unreadable")
                continue
            rows[clip] = r
            if r["verdict"] != "aligned":
                bad.append(clip)
            at = "    n/a" if r["ncc_at_start"] is None else f"{r['ncc_at_start']:9.3f}"
            print(f"{clip:12}{r['window_start']:>7}{r['best_match']:>7}{r['offset']:>+6d}"
                  f"{r['ncc_best']:>8.3f}{at}  {r['verdict']}")

    if a.merge and os.path.exists(a.out):
        prev = json.load(open(a.out))
        prev.update(rows)
        rows = prev
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\njson -> {a.out}")
    if bad:
        print(f"\n{len(bad)} clip(s) NOT aligned: {', '.join(bad)}")
        print("A v1 column for a misaligned clip is scored on different footage. "
              "Do not compare it to anything until the window is corrected.")
        raise SystemExit(1)
    print("\nEvery clip's baseline window matches its window clip.")


if __name__ == "__main__":
    main()
