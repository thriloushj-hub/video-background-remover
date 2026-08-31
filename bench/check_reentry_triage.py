"""3.3a -- can a correctly-matted person trip the re-entry scan on pose alone?

The product path decides "is someone uncovered?" with
``reid.uncovered_boxes``: a detection is uncovered when the current matte
covers less than ``max_overlap`` (0.35) of its *bounding box*.  Box fill is a
function of pose, not of tracking quality: an upright talking head fills ~50%
of its box, a mid-leap dancer with limbs extended fills far less.

This measures the ceiling on that effect with no detector and no GPU.  For each
scan frame (every ``scan_every`` frames, matching the pipeline), it takes each
subject-sized connected component of the persisted alpha and computes

    fill = alpha area / area of the component's own tight bounding box

A detector box is looser than a tight alpha bbox, so real coverage is always
<= this number.  Any component below 0.35 here would have been called
"uncovered" by the product path even though the matte is holding that person
perfectly.

Usage:  python bench/check_reentry_triage.py [run_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

MAX_OVERLAP = 0.35        # reid.max_overlap
MIN_AREA_FRAC = 0.0015    # reid.min_area_frac
SCAN_EVERY = 8            # reid.scan_every


def clip_rows(alpha_dir: Path):
    files = sorted(alpha_dir.glob("*.png"))
    if not files:
        return []
    rows = []
    for i in range(0, len(files), SCAN_EVERY):
        a = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        H, W = a.shape[:2]
        binm = (a > 127).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
        for c in range(1, n):
            x, y, w, h, area = stats[c]
            if area < MIN_AREA_FRAC * H * W:
                continue
            rows.append({"frame": i, "area": int(area),
                         "box_area": int(w * h),
                         "fill": float(area) / float(w * h)})
    return rows


def main(run_dir: str) -> None:
    root = Path(run_dir)
    out = {}
    print(f"{'clip':<11}{'comps':>6}{'min':>8}{'median':>8}{'max':>8}"
          f"{'<0.35':>7}{'share':>8}")
    for d in sorted(p for p in root.iterdir() if (p / "alpha").is_dir()):
        rows = clip_rows(d / "alpha")
        if not rows:
            continue
        f = np.array([r["fill"] for r in rows])
        below = int((f < MAX_OVERLAP).sum())
        out[d.name] = {"components": len(rows),
                       "fill_min": round(float(f.min()), 4),
                       "fill_median": round(float(np.median(f)), 4),
                       "fill_max": round(float(f.max()), 4),
                       "below_threshold": below,
                       "share_below": round(below / len(f), 4),
                       "rows": rows}
        print(f"{d.name:<11}{len(rows):>6}{f.min():>8.3f}{np.median(f):>8.3f}"
              f"{f.max():>8.3f}{below:>7}{below/len(f):>8.3f}")
    dest = root.parent / "reentry_triage.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "bench/results/run_2026-08-30/bench_out")
