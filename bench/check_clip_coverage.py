"""Per-frame coverage against v1 — the check that catches what solos cannot.

`check_full_clip_solos.py` counts whole STANDALONE regions one matte holds and
the other does not, because that is the question the halo claim needed. It is
blind by construction to a missing part of a subject that still touches the
part being held — and on bilibili that is exactly what happened: v2 dropped a
woman's boots at a cut, the boots touch her held legs, and solos saw six frames
out of fifty.

So this is the blunt companion measure: per frame, the area each matte holds,
their IoU, and the plain fraction of frame v1 holds that v2 does not. No
thresholds, no components, nothing clever. Run it beside the solos pass, not
instead of it.

    python bench/check_clip_coverage.py --v1 "<clip>_matte.mp4" \
        --v2-alpha out/<clip>_alpha.mp4 --out-json <clip>_cov.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="v1's delivered *_matte.mp4 (a green composite)")
    ap.add_argument("--v2-alpha", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    c1, c2 = cv2.VideoCapture(a.v1), cv2.VideoCapture(a.v2_alpha)
    for _ in range(max(0, a.offset)):
        c2.read()
    rows, n = [], 0
    while True:
        ok1, f1 = c1.read()
        ok2, f2 = c2.read()
        if not (ok1 and ok2):
            break
        a1 = alpha_from_greenscreen(f1, GREEN)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if f2.ndim == 3 else f2
        a2 = cv2.resize(g2, (a1.shape[1], a1.shape[0]),
                        interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        h1, h2 = a1 > 0.5, a2 > 0.5
        inter = float((h1 & h2).sum())
        union = float((h1 | h2).sum())
        rows.append([n, round(float(h1.mean()), 5), round(float(h2.mean()), 5),
                     round(inter / max(union, 1.0), 4),
                     round(float((h1 & ~h2).mean()), 5),
                     round(float((h2 & ~h1).mean()), 5)])
        n += 1
    for c in (c1, c2):
        c.release()

    json.dump({"clip": os.path.basename(a.v1), "frames": n, "offset": a.offset,
               "cols": ["frame", "v1_area", "v2_area", "iou", "v1_not_v2", "v2_not_v1"],
               "rows": rows}, open(a.out_json, "w"))
    iou = [r[3] for r in rows] or [0.0]
    gap = [r[4] for r in rows] or [0.0]
    print("%-11s %4df  mean IoU %.4f  worst v1_not_v2 %.5f  frames IoU<0.80 %3d  frames gap>1%% %3d"
          % (os.path.basename(a.v1).replace("_matte.mp4", ""), n,
             float(np.mean(iou)), max(gap),
             sum(1 for v in iou if v < 0.80), sum(1 for v in gap if v > 0.01)))


if __name__ == "__main__":
    main()
