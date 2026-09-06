"""Who is each side holding that the other is not, across a whole clip?

`bench/metrics.halo_split` answers this in one direction only, because that is
the direction the halo claim needed: components of the BASELINE that the run
does not hold, far from anything the run holds.  On butter that turned a 0.541
"v1 keys in background" result into "v2 missed three dancers", which is the
single most useful thing the split has done.

The reverse direction has never been computed, and on 5 Sep a frame of the
full-length 1917 side-by-side showed why it should be: v2 appears to be holding
a distant background figure that v1 is not -- the opposite of the window
finding, where v1 holds a bystander v2 drops on purpose.  One frame is not
evidence, and the rule on this project is that a standalone component means
nothing until a picture says what it is.

So this runs the same decomposition BOTH WAYS over an entire clip, per frame,
and writes a contact sheet of the frames where either side is holding the most
that the other is not.  The thresholds are `halo_split`'s own, unchanged, so
the numbers are comparable to every shell/solo figure in the project.

Everything streams: a 724-frame 1080p clip is 4.8 GB if you hold it.  The
analysis also runs at **960px wide, not 1920** -- both because the closing
kernel is 4% of frame width and a 153x153 ellipse per frame per direction is
minutes of CPU, and because 960 is the width every other shell/solo number in
this project is measured at.  Connectivity is a property of the resolution you
measure at, so matching it is the point rather than a concession.

    python bench/check_full_clip_solos.py \
        --source "output vids/1917.mp4" --v1 "output vids/1917_matte.mp4" \
        --v2-alpha full_out/1917_alpha.mp4 --offset 0 \
        --out-json solos_1917.json --out-sheet solos_1917.jpg
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vbgr.compose import alpha_from_greenscreen  # noqa: E402

GREEN = (151, 253, 119)
CLOSE_FRAC = 0.04      # halo_split defaults, deliberately not re-tuned
TOUCH_MIN = 0.02
SOLO_MIN_FRAC = 0.002


def _alphas(f1, f2, width):
    """v1's alpha recovered from its green composite, and v2's, both at `width`."""
    h = int(round(f1.shape[0] * width / f1.shape[1]))
    a1 = alpha_from_greenscreen(
        cv2.resize(f1, (width, h), interpolation=cv2.INTER_AREA), GREEN)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if f2.ndim == 3 else f2
    a2 = cv2.resize(g2, (width, h),
                    interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    return a1, a2


def solo_components(base_hard, run_hard):
    """Components of base-only that stand clear of everything `run` holds.

    Same construction as metrics.halo_split: close the run's silhouette so a
    hole inside it is not called standalone, then keep components whose overlap
    with a dilated run mask is under TOUCH_MIN.
    """
    r = max(3, int(CLOSE_FRAC * base_hard.shape[1]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    closed = cv2.morphologyEx(run_hard.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    only = (base_hard & ~run_hard & ~closed).astype(np.uint8)
    near = cv2.dilate(run_hard.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(only, 8)
    frame_px = float(base_hard.size)
    out = []
    for j in range(1, n):
        ar = float(st[j, 4])
        if float(((lab == j) & near).sum()) / ar < TOUCH_MIN:
            if ar / frame_px >= SOLO_MIN_FRAC:
                out.append((ar / frame_px, tuple(int(v) for v in st[j, :4])))
    return out


def panel(img, alpha, boxes, title, w):
    """One panel of the sheet: the frame dimmed by `alpha`, with `boxes` drawn.

    `boxes` are in ANALYSIS coordinates (alpha's width), not the source frame's.
    The first version of this scaled them by the source width and drew every box
    at half its true position -- a picture that lies, which is worse than no
    picture, and exactly the class of error this script exists to prevent.
    """
    h = int(round(img.shape[0] * w / img.shape[1]))
    vis = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    vis = vis * (0.25 + 0.75 * a) + np.float32([40, 30, 25]) * (1 - a)
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    sx = w / alpha.shape[1]
    for _f, (x, y, bw, bh) in boxes:
        cv2.rectangle(vis, (int(x * sx), int(y * sx)),
                      (int((x + bw) * sx), int((y + bh) * sx)), (60, 220, 255), 2)
    strip = np.full((26, w, 3), 24, np.uint8)
    cv2.putText(strip, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([strip, vis])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2-alpha", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--width", type=int, default=960,
                    help="analyse at this width -- 960 matches every other "
                         "shell/solo figure in the project")
    ap.add_argument("--sheet-w", type=int, default=430)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-sheet")
    a = ap.parse_args()

    cs, c1, c2 = (cv2.VideoCapture(a.source), cv2.VideoCapture(a.v1),
                  cv2.VideoCapture(a.v2_alpha))
    lo = max(0, a.offset)
    for _ in range(lo):
        cs.read(); c2.read()
    for _ in range(lo - a.offset):
        c1.read()

    per = []
    i = lo
    while True:
        oks, fs = cs.read()
        ok1, f1 = c1.read()
        ok2, f2 = c2.read()
        if not (oks and ok1 and ok2):
            break
        a1, a2 = _alphas(f1, f2, a.width)
        h1, h2 = a1 > 0.5, a2 > 0.5
        v1_only = solo_components(h1, h2)
        v2_only = solo_components(h2, h1)
        if v1_only or v2_only:
            per.append({"frame": i,
                        "v1_only": round(sum(f for f, _ in v1_only), 5),
                        "v2_only": round(sum(f for f, _ in v2_only), 5),
                        "n_v1_only": len(v1_only), "n_v2_only": len(v2_only)})
        i += 1
    for c in (cs, c1, c2):
        c.release()

    n = i - lo
    tot1 = sum(p["v1_only"] for p in per)
    tot2 = sum(p["v2_only"] for p in per)
    res = {"clip": os.path.basename(a.source), "frames": n, "offset": a.offset,
           "frames_with_v1_only": sum(1 for p in per if p["v1_only"] > 0),
           "frames_with_v2_only": sum(1 for p in per if p["v2_only"] > 0),
           "mean_v1_only_frac": round(tot1 / max(n, 1), 6),
           "mean_v2_only_frac": round(tot2 / max(n, 1), 6),
           "per_frame": per}
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(f"{res['clip']}  {n} frames from {lo}")
    print(f"  v1 holds something v2 does not : {res['frames_with_v1_only']:4d} frames"
          f"  mean {res['mean_v1_only_frac']:.5f} of frame")
    print(f"  v2 holds something v1 does not : {res['frames_with_v2_only']:4d} frames"
          f"  mean {res['mean_v2_only_frac']:.5f} of frame")

    if not a.out_sheet or not per:
        return
    picks = sorted(per, key=lambda p: -(p["v1_only"] + p["v2_only"]))[:a.top]
    picks = sorted(p["frame"] for p in picks)
    cs, c1, c2 = (cv2.VideoCapture(a.source), cv2.VideoCapture(a.v1),
                  cv2.VideoCapture(a.v2_alpha))
    for _ in range(lo):
        cs.read(); c2.read()
    for _ in range(lo - a.offset):
        c1.read()
    rows, want, i = [], set(picks), lo
    while want:
        oks, fs = cs.read(); ok1, f1 = c1.read(); ok2, f2 = c2.read()
        if not (oks and ok1 and ok2):
            break
        if i in want:
            want.discard(i)
            a1, a2 = _alphas(f1, f2, a.width)
            h1, h2 = a1 > 0.5, a2 > 0.5
            rows.append(np.hstack([
                panel(fs, np.ones_like(a1), [], f"f{i}  source", a.sheet_w),
                panel(fs, a1, solo_components(h1, h2),
                      "v1  (boxes: held by v1, not v2)", a.sheet_w),
                panel(fs, a2, solo_components(h2, h1),
                      "v2  (boxes: held by v2, not v1)", a.sheet_w)]))
        i += 1
    for c in (cs, c1, c2):
        c.release()
    if rows:
        cv2.imwrite(a.out_sheet, np.vstack(rows),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"  sheet -> {a.out_sheet}  frames {picks}")


if __name__ == "__main__":
    main()
