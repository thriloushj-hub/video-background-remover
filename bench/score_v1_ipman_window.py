"""v1 baseline for the NEW ipman window (source frames 318-389, 13.25-16.25s).

Why this is possible now and was not before
-------------------------------------------
v1's ipman matte covers source frames 160-484 (docs/BUGS.md #1: constant offset
160, median NCC 0.909, 172 confident matches, verified by eye at both ends).

  old window, source frames  24- 95  ->  ENTIRELY OUTSIDE that range.
  new window, source frames 318-389  ->  entirely inside it.

That is why there has never been a v1 ipman row, and why there can be one now.
Matte frame index = source frame - 160, so the window is matte frames 158-229.

This script re-validates the offset rather than trusting it, then scores.

    python score_v1_ipman_window.py <ipman_matte.mp4> <ipman.mp4>

Metric definitions are imported from bench/metrics.py so the numbers are
identical in kind to the frozen baseline -- not a reimplementation.
"""
import sys, os
import cv2
import numpy as np

sys.path.insert(0, "/mnt/user-data/uploads/vbgr2")
sys.path.insert(0, "/mnt/user-data/uploads")
from bench import metrics                      # noqa: E402
from vbgr.compose import alpha_from_greenscreen, DEFAULT_GREEN   # noqa: E402

OFFSET = 160                 # docs/BUGS.md #1
WIN_SRC = (318, 390)         # source frames, end-exclusive


def read(path, n=None):
    cap = cv2.VideoCapture(path); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f)
        if n and len(out) >= n: break
    cap.release(); return out


def fit(img, short_side):
    if not short_side: return img
    h, w = img.shape[:2]
    if min(h, w) <= short_side: return img
    s = short_side / float(min(h, w))
    return cv2.resize(img, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)


def detect_green_key(frames, n=5):
    cols = []
    for f in frames[:n]:
        border = np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]])
        cols.append(np.median(border, axis=0))
    return tuple(int(v) for v in np.median(np.array(cols), axis=0)) if cols else DEFAULT_GREEN


def ncc(a, b):
    a = a.astype(np.float32) - a.mean(); b = b.astype(np.float32) - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def main(matte_path, source_path):
    matte = read(matte_path)
    src = read(source_path)
    print(f"matte {len(matte)} frames, source {len(src)} frames")
    assert len(matte) == 325, f"expected 325 matte frames, got {len(matte)}"
    assert len(src) == 501, f"expected 501 source frames, got {len(src)}"

    key = detect_green_key(matte)
    print(f"detected green key (BGR): {key}")

    # ---- re-validate the offset instead of trusting BUGS.md -------------- #
    print("\nre-validating offset (NCC on non-green pixels, 5 probe frames):")
    probes = [170, 250, 318, 350, 389]
    best_offsets = []
    for s in probes:
        sm = cv2.cvtColor(fit(src[s], 240), cv2.COLOR_BGR2GRAY)
        scores = {}
        for off in range(OFFSET - 3, OFFSET + 4):
            mi = s - off
            if not (0 <= mi < len(matte)): continue
            mm = cv2.cvtColor(fit(matte[mi], 240), cv2.COLOR_BGR2GRAY)
            a = fit(alpha_from_greenscreen(matte[mi], key), 240)
            m = a > 0.5
            scores[off] = ncc(sm[m], mm[m]) if m.sum() > 200 else -1
        bo = max(scores, key=scores.get)
        best_offsets.append(bo)
        print(f"  source frame {s:3d}: best offset {bo}  (ncc {scores[bo]:.3f})")
    if len(set(best_offsets)) == 1 and best_offsets[0] == OFFSET:
        print(f"  -> offset {OFFSET} confirmed on all probes")
    else:
        print(f"  !! offsets disagree: {best_offsets} -- do NOT trust the numbers below")

    # ---- score the window ------------------------------------------------ #
    a0, a1 = WIN_SRC[0] - OFFSET, WIN_SRC[1] - OFFSET
    sel = matte[a0:a1]
    print(f"\nwindow: source {WIN_SRC[0]}-{WIN_SRC[1]-1} = matte {a0}-{a1-1}, {len(sel)} frames")
    assert len(sel) == 72, f"expected 72 frames, got {len(sel)}"

    for short_side, label in ((400, "400px short side (matches the other v1 rows)"),
                              (None, "native (matches the v2 SAM2Matting numbers)")):
        alphas = [alpha_from_greenscreen(fit(f, short_side), key) for f in sel]
        frac = np.array([float((a > 0.5).mean()) for a in alphas])
        row = dict(
            area_cv=round(float(frac.std() / (frac.mean() + 1e-9)), 4),
            edge_soft=round(float(np.mean([metrics.edge_softness(a) for a in alphas[::2]])), 4),
            dropouts=metrics.dropout_events(alphas)[0],
            mean_area=round(float(frac.mean()), 4),
        )
        print(f"\nv1 baseline, ipman new window -- {label}")
        print(f"  area_cv   {row['area_cv']}")
        print(f"  edge_soft {row['edge_soft']}")
        print(f"  dropouts  {row['dropouts']}")
        print(f"  mean_area {row['mean_area']}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
