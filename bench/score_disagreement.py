"""Score v2 against v1 by *where* the two mattes disagree, not by how much area each covers.

    python bench/score_disagreement.py --run bench/results/run_2026-08-23
    python bench/score_disagreement.py --run <dir> --source sheet     # no persisted alphas

Why this exists
---------------
The 2026-08-23 benchmark table read as a 15/15 loss for v2: higher ``area_cv``
on every clip, lower ``mean_area`` on every clip, no exception.  The contact
sheets said the opposite -- on ipman both fighters are held for all 72 frames
and it is v1's 0.2453 that is inflated, because two men in a wide shot do not
cover a quarter of it.

``area_cv`` and ``mean_area`` measure coverage.  Coverage only compares two
runs that are trying to cover the same thing, and v1 is not: it keys in
background.  Against an over-inclusive baseline "smaller" and "more correct"
are the same observation, so the table cannot separate v2 being right (ipman,
butter) from v2 being broken (interview's torn face) -- and it scored those
two failures in the same direction.

So this reports the disagreement split by where it sits:

    halo    v1 foreground outside v2's silhouette   -> v1 over-includes
    hole    v1 foreground inside v2's silhouette    -> v2 tore the subject
    excess  v2 foreground v1 calls background       -> read the rim WIDTH

Reading it
----------
``halo`` high and ``hole`` ~0 is v2 winning, and it is the only shape of
result that supports a v2-beats-v1 claim.

``excess_frac`` looked alarming on the dance-type clips (dance3 0.259, dance
0.219, shakira 0.177, dance2 0.173) and it is an artefact.  There is not one
standalone excess component anywhere in the set -- every bit of it is attached
to a subject both runs already hold.  Divided by boundary length it is a
**0.51-1.06 px rim on 14 of 15 clips** while the fraction spans 0.027-0.259:
one constant sub-pixel disagreement, and the spread is how much perimeter the
subjects have.  Small dancers have a lot of edge for their area.  ipman is the
single real exception at 5.98 px, where the two mattes genuinely differ in both
directions.  So read ``excess_rim_px``; ``excess_frac`` is there for
continuity.  ``hole`` is the alarm; it is the
column that would have caught interview, which every existing metric read as
clean (area_cv 0.0175, dropouts 0, subj_lost_at never).

Two sources
-----------
``--source alpha`` (default) reads the alphas the runner persists.  Trust the
hole column here.

``--source sheet`` recovers v2's alpha from the contact-sheet tiles, for runs
made before the runner persisted alphas.  Halo is reliable through the proxy
-- ipman scores 0.78, butter 0.51, every talking-head control under 0.01.
Hole is NOT: the proxy mis-recovers dark subjects against green, and it scores
a clean control (microsoft, 0.022) at the same magnitude as interview's real
tear (0.035).  A hole number off a sheet is a hint to go look, never a verdict.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from bench.metrics import area_jitter, area_stability, coverage_disagreement  # noqa: E402
from vbgr.compose import alpha_from_greenscreen  # noqa: E402

DEFAULT_V1_DIR = r"C:\Users\Tirloosh\Downloads\output vids -20260807T044446Z-1-001\output vids"


def v1_window_alphas(v1_dir, clip, lo, hi, key):
    """v1's alpha on the benchmark window, recovered from its green composite.

    The window indices are frame numbers **into the matte file**, not the
    source.  Verified by recomputing v1's own baseline from them: 1917 lands
    at mean_area 0.3184 against the recorded 0.3147, ipman 0.2475 against
    0.2453, microsoft 0.3960 against 0.3937.
    """
    path = os.path.join(v1_dir, f"{clip}_matte.mp4")
    if not os.path.exists(path):
        return []
    cap = cv2.VideoCapture(path)
    out, i = [], 0
    while i < hi:
        ok, f = cap.read()
        if not ok:
            break
        if i >= lo:
            out.append(f)
        i += 1
    cap.release()
    return [alpha_from_greenscreen(f, key) for f in out]


def alphas_from_dir(d):
    return [cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            for p in sorted(glob.glob(os.path.join(d, "*.png")))]


def alphas_from_sheet(path, cols=6, rows=4):
    """Recover v2's alpha from the contact sheet, plus the label mask to ignore.

    The sheet composites over pure green and stamps a frame label at (5, 20),
    so those pixels are excluded rather than scored as a disagreement.
    """
    img = cv2.imread(path)
    if img is None:
        return [], None
    th, tw = img.shape[0] // rows, img.shape[1] // cols
    tiles = [img[(k // cols) * th:(k // cols + 1) * th,
                 (k % cols) * tw:(k % cols + 1) * tw] for k in range(cols * rows)]
    ignore = np.zeros((th, tw), bool)
    ignore[0:30, 0:170] = True
    return [alpha_from_greenscreen(t, (0, 255, 0)) for t in tiles], ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir holding bench_out/")
    ap.add_argument("--source", choices=["alpha", "sheet"], default="alpha")
    # Which persisted arm to score. The runner writes "alpha" for the plain
    # benchmark arm, "alpha_product" for the pipeline.py arm (--product) and
    # "alpha_reseed" for the entrant arm. They are separate directories on
    # purpose: an arm must never be able to overwrite another's pixels.
    ap.add_argument("--arm", default="alpha",
                    help="alpha | alpha_product | alpha_reseed")
    # v1 read from pre-exported PNGs rather than the delivery MP4s. This is how
    # the 2 Sep product run was scored: the product alphas were 240 MB on a
    # Colab VM and could not be brought down, so v1's window alphas were
    # carried up instead. See bench/export_v1_window_alphas.py.
    # NOTE: shell values from this path differ slightly from the MP4 path,
    # because the resize happens at a different point. Do not mix them.
    ap.add_argument("--v1-alpha-dir", default=None,
                    help="directory of <clip>/NNNN.png v1 alphas "
                         "(from export_v1_window_alphas.py)")
    ap.add_argument("--v1-dir", default=os.environ.get("VBGR_V1_DIR", DEFAULT_V1_DIR))
    ap.add_argument("--baselines",
                    default=os.path.join(_ROOT, "bench", "results",
                                         "v1_window_baselines.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    V1 = json.load(open(a.baselines))
    root = a.run if os.path.basename(a.run) == "bench_out" else os.path.join(a.run, "bench_out")
    if not os.path.isdir(root):
        raise SystemExit(f"no bench_out under {a.run}")
    if not a.v1_alpha_dir and not os.path.isdir(a.v1_dir):
        raise SystemExit(f"v1 delivery folder not found: {a.v1_dir}\n"
                         f"pass --v1-dir, set VBGR_V1_DIR, or use "
                         f"--v1-alpha-dir with exported PNGs")
    if a.v1_alpha_dir and not os.path.isdir(a.v1_alpha_dir):
        raise SystemExit(f"no v1 alpha directory at {a.v1_alpha_dir}")

    rows, skipped = {}, {}
    for clip, v in V1.items():
        d = os.path.join(root, clip)
        if not os.path.isdir(d):
            continue
        # A voided baseline is not a missing one. ipman's window clip lies
        # outside v1's matte coverage entirely, so there is nothing to compare
        # against and a row here would be a comparison of different footage --
        # which is exactly what went unnoticed from 16 to 26 Aug.
        if not v.get("trusted", True) or v.get("window") is None:
            skipped[clip] = ("v1 baseline is voided -- see its note in "
                             "v1_window_baselines.json")
            continue
        ignore = None
        if a.source == "alpha":
            A2 = alphas_from_dir(os.path.join(d, a.arm))
            if not A2:
                skipped[clip] = (f"no persisted alphas in {a.arm}/ -- either "
                                 f"this run predates save_alphas() or that arm "
                                 f"was not run")
                continue
        else:
            A2, ignore = alphas_from_sheet(os.path.join(d, "sheet_off.jpg"))
            if not A2:
                skipped[clip] = "no contact sheet"
                continue

        lo, hi = v["window"]
        if a.v1_alpha_dir:
            A1full = alphas_from_dir(os.path.join(a.v1_alpha_dir, clip))
            if not A1full:
                skipped[clip] = f"no exported v1 alphas under {a.v1_alpha_dir}"
                continue
        else:
            A1full = v1_window_alphas(a.v1_dir, clip, lo, hi,
                                      tuple(v["green_bgr"]))
            if not A1full:
                skipped[clip] = "no v1 matte file"
                continue
        # the sheet samples the window; the persisted alphas are every frame
        idx = (np.linspace(0, len(A1full) - 1, len(A2)).astype(int)
               if len(A2) != len(A1full) else np.arange(len(A1full)))
        h, w = A2[0].shape
        A1 = [cv2.resize(A1full[i], (w, h), interpolation=cv2.INTER_AREA) for i in idx]

        r = coverage_disagreement(A1, A2, ignore=ignore)
        r["area_cv_v1"] = round(area_stability(A1), 4)
        r["area_cv_v2"] = round(area_stability(A2), 4)
        # v1 was never given a jitter column, so the only like-for-like
        # stability comparison in the whole project rested on ipman's area_cv.
        # area_jitter is a function of the same area series area_cv is, so it
        # survives v1's green recovery for exactly the same reason area_cv
        # does -- and unlike area_cv it does not read butter's camera
        # pull-back as flicker.
        r["area_jitter_v1"] = round(area_jitter(A1), 4)
        r["area_jitter_v2"] = round(area_jitter(A2), 4)
        r["verdict"] = verdict(r, a.source)
        rows[clip] = r

    hdr = (f"{'clip':12}{'halo':>7}{'hole':>7}{'holeBig':>9}{'excess':>8}"
           f"{'rim px':>8}{'cv_v1':>8}{'cv_v2':>8}{'jit_v1':>8}{'jit_v2':>8}  verdict")
    print(f"\nsource: {a.source}" + ("   (hole is indicative only)"
                                     if a.source == "sheet" else ""))
    print(hdr)
    print("-" * len(hdr))
    for c, r in rows.items():
        print(f"{c:12}{r['halo_frac']:7.3f}{r['hole_frac']:7.3f}{r['hole_big']:9.3f}"
              f"{r['excess_frac']:8.3f}{r['excess_rim_px']:8.2f}"
              f"{r['area_cv_v1']:8.3f}{r['area_cv_v2']:8.3f}"
              f"{r['area_jitter_v1']:8.3f}{r['area_jitter_v2']:8.3f}"
              f"  {r['verdict']}")
    for c, why in skipped.items():
        print(f"{c:12}skipped -- {why}")

    suffix = a.source if a.arm == "alpha" else f"{a.source}_{a.arm}"
    out = a.out or os.path.join(a.run, f"coverage_disagreement_{suffix}.json")
    json.dump({"source": a.source, "arm": a.arm,
               "v1_from": "png" if a.v1_alpha_dir else "mp4",
               "rows": rows, "skipped": skipped},
              open(out, "w"), indent=2)
    print(f"\njson -> {out}")
    print("halo high + hole ~0 is the only shape that supports 'v2 beats v1'. "
          "Every verdict here still wants the contact sheet beside it.")
    print("excess: read the rim WIDTH, not the fraction. A ~1px rim across the "
          "whole set is the two alphas being recovered differently at the 0.5 "
          "threshold, not v2 covering more.")


def verdict(r, source):
    """Deliberately coarse. This points at which clip to look at, it does not settle it."""
    if r["hole_big"] >= 0.02 and source == "alpha":
        return "v2 DEFECT -- look"
    if r["hole_big"] >= 0.02:
        return "possible v2 defect -- look"
    if r["halo_frac"] >= 0.15:
        return "v1 over-inclusive"
    if r["excess_rim_px"] >= 2.0:
        return "v2 covers more -- look"
    return "agree"


if __name__ == "__main__":
    main()
