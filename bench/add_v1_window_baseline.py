"""Add a delivered clip to the benchmark: verify, cut a window, baseline it.

Why this exists
---------------
`bench/results/v1_window_baselines.json` is the file every v2-vs-v1 number is
measured against, and until now **nothing in the repo wrote it** -- it was built
by hand in a chat that is gone. That is how ipman's row came to describe a
window lying entirely outside v1's matte coverage, and it went unnoticed for
ten days.

So this does the three things that row needed and did not have, in order, and
refuses rather than guessing:

  1. **Alignment.** NCC of the source frame against the matte frame at the same
     index. ipman's offset was 160 and nobody checked. Below --min-ncc this
     stops.
  2. **A single-shot window.** Shot cuts are detected on the source and the
     window must sit strictly inside one shot. Two of the three original
     windows straddled cuts.
  3. **The baseline row**, computed from v1's own matte over exactly that
     window.

`edge_soft`, `sharpness` and `bg_share` are written as null on purpose, exactly
as ipman's restored row is. v1's alpha is recovered by pushing chroma distance
through ``clip((d - 12) / 28)``, and that ramp pins sharpness to 0.654-0.673
across fifteen very different clips -- so those columns describe the recovery,
never v1's matte, and must never appear in a claim. Writing a plausible number
into a column that may not be quoted is worse than leaving it empty.

    python bench/add_v1_window_baseline.py dlh es2 bilibili --dry-run
    python bench/add_v1_window_baseline.py dlh es2 bilibili --write
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from bench.metrics import area_stability, dropout_events            # noqa: E402
from vbgr.compose import alpha_from_greenscreen                     # noqa: E402
from vbgr.shots import compute_scene_cuts                           # noqa: E402

# The v1 delivery folder. Machine-specific, so it comes from the environment:
# set VBGR_V1_DIR, or pass --v1-dir. Nothing here should hard-code one
# person's home directory -- this repo gets read by other people.
DEFAULT_V1_DIR = os.environ.get("VBGR_V1_DIR", "")
GREEN = (151, 253, 119)
WIN = 72
SMALL_W = 640


def read_small(path, limit=None):
    cap = cv2.VideoCapture(path)
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or (limit and i >= limit):
            break
        h = int(round(f.shape[0] * SMALL_W / f.shape[1]))
        out.append(cv2.resize(f, (SMALL_W, h), interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    return out


def ncc(src, matte):
    """Correlate source against matte INSIDE v1's own foreground.

    Whole-frame correlation is the wrong comparison and quietly refuses every
    clip: the matte has its background replaced by flat green, so most of the
    frame disagrees by construction. On the first run of this script that
    scored three genuinely aligned clips at 0.47-0.65 and rejected all three.
    check_window_alignment.py already had this right -- compare only where v1
    says there is a subject.
    """
    a = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(matte, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    m = cv2.resize(alpha_from_greenscreen(matte, GREEN),
                   (a.shape[1], a.shape[0])) > 0.5
    if m.sum() < 500:
        return 0.0
    a, b = a[m], b[m]
    a = a - a.mean(); b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float((a * b).sum() / d)


def pick_window(alphas, cuts, n):
    """The 72-frame single-shot window with the most subject motion.

    Motion is the mean absolute frame-to-frame change in v1's own alpha, which
    needs no seeding stage and is not fooled by a low-texture background the
    way optical flow is (that blind spot cost a week; see
    Window_Selection_And_Flow_Blind_Spot).
    """
    bounds = list(cuts) + [n]
    d = np.array([np.abs(alphas[i + 1] - alphas[i]).mean()
                  for i in range(len(alphas) - 1)])
    best = None
    for s, e in zip(bounds, bounds[1:]):
        if e - s < WIN:
            continue
        for lo in range(s, e - WIN + 1, 4):
            score = float(d[lo:lo + WIN - 1].mean())
            if best is None or score > best[0]:
                best = (score, lo, lo + WIN)
    return best


def baseline(alphas, lo, hi, cuts):
    A = alphas[lo:hi]
    frac = np.array([float((a > 0.5).mean()) for a in A])
    inside = [c - lo for c in cuts if lo <= c < hi]
    n_drop, _ = dropout_events(list(A), cuts=inside)
    return dict(window=[lo, hi], frames=hi - lo, green_bgr=list(GREEN),
                bg_share=None,
                area_cv=round(float(area_stability(A)), 4),
                edge_soft=None, sharpness=None,
                dropouts=int(n_drop),
                mean_area=round(float(frac.mean()), 4),
                subjects_f0=None, trusted=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--v1-dir", default=os.environ.get("VBGR_V1_DIR", DEFAULT_V1_DIR))
    ap.add_argument("--out-clips", default=os.path.join(_ROOT, "bench_payload", "new_windows"))
    ap.add_argument("--baselines", default=os.path.join(
        _ROOT, "bench", "results", "v1_window_baselines.json"))
    ap.add_argument("--min-ncc", type=float, default=0.80)
    ap.add_argument("--write", action="store_true",
                    help="write the baseline rows and cut the window clips")
    a = ap.parse_args()

    base = json.load(open(a.baselines))
    rows = {}
    for clip in a.clips:
        src_p = os.path.join(a.v1_dir, f"{clip}.mp4")
        mat_p = os.path.join(a.v1_dir, f"{clip}_matte.mp4")
        if not (os.path.exists(src_p) and os.path.exists(mat_p)):
            print(f"{clip:10s} SKIP -- source or matte missing"); continue
        if clip in base and base[clip]:
            print(f"{clip:10s} SKIP -- already has a baseline row; refusing to "
                  f"overwrite one silently"); continue

        src = read_small(src_p)
        mat = read_small(mat_p)
        n = min(len(src), len(mat))
        probes = [int(n * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
        scores = [ncc(src[i], mat[i]) for i in probes]
        med = float(np.median(scores))
        if med < a.min_ncc:
            print(f"{clip:10s} REFUSED -- source and matte do not align at the "
                  f"same index (median NCC {med:.3f} over {probes}). That is "
                  f"the ipman defect; find the offset before adding this clip.")
            continue

        alphas = [alpha_from_greenscreen(f, GREEN) for f in mat[:n]]
        cuts = compute_scene_cuts(src[:n], warn_suppressed=False)
        pick = pick_window(alphas, cuts, n)
        if pick is None:
            print(f"{clip:10s} REFUSED -- no {WIN}-frame single-shot window "
                  f"exists (shots from cuts {list(cuts)} in {n} frames)")
            continue
        score, lo, hi = pick
        row = baseline(alphas, lo, hi, cuts)
        rows[clip] = row
        print(f"{clip:10s} align NCC {med:.3f} | cuts {len(cuts)} | "
              f"window [{lo}, {hi}] motion {score:.5f} | "
              f"area_cv {row['area_cv']} mean_area {row['mean_area']} "
              f"dropouts {row['dropouts']}")

        if a.write:
            os.makedirs(a.out_clips, exist_ok=True)
            outp = os.path.join(a.out_clips, f"{clip}_w{lo}_{hi - 1}.mp4")
            # frame-accurate: decode from the start and select by index, the
            # way ipman's replacement was cut. -ss would be close enough to
            # look right and wrong enough to void the row.
            if os.path.exists(outp):
                print(f"{'':10s} -> {outp} (already cut, left alone)")
            else:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", src_p,
                     "-vf", f"select='between(n\\,{lo}\\,{hi - 1})',setpts=N/FRAME_RATE/TB",
                     "-an", "-c:v", "libx264", "-crf", "16", outp], check=True)
                print(f"{'':10s} -> {outp}")

    if a.write and rows:
        base.update(rows)
        json.dump(base, open(a.baselines, "w"), indent=1)
        print(f"\nwrote {len(rows)} row(s) -> {a.baselines}")
    elif rows:
        print("\n--dry-run: nothing written. Re-run with --write.")


if __name__ == "__main__":
    main()
