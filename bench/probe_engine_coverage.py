"""Does the matting engine hold what its seed gave it? (5.7 / 5.8)

Why this exists
---------------
On `bilibili` the shot starting at f1159 loses a subject's boots for sixteen
frames, returning at f1175 -- which is exactly the next shot start.  The
earlier diagnosis blamed the mask stage and is retracted: re-running the
SHIPPED build_seed on CPU gives a seed reaching row 825 (68 px short of its
box, not 298), and every CPU stage after it is clean --

    seed 825 -> erode_dilate 831 -> guided_filter 831 -> shipped alpha 588

-- so the loss happens inside the engine, between a mask reaching 831 and an
alpha reaching 588.  That earlier number came from an ad-hoc Colab cell that
was never committed, which is why nothing could re-derive it.  Hence this file
being in the repo.

What it reports, per arm, for a clip segment cut to START on a shot boundary
(so segment frame 0 is the frame the shipped run seeded that shot from):

    seed_bottom    lowest row of the fused seed at frame 0
    alpha_bottom   lowest row of the output alpha, per frame
    deficit        seed_bottom - alpha_bottom at frame 0; >0 means the engine
                   gave back less than it was handed

Arms are config only, so a run is reproducible from the printed config:

    python bench/probe_engine_coverage.py --clip seg.mp4 --config pipe.json \
        --arm baseline --arm extend:0.15
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vbgr import refine                                    # noqa: E402
from vbgr.config import Config                             # noqa: E402
from vbgr.detect import held_props, select_person_boxes    # noqa: E402
from vbgr.pipeline import Pipeline                         # noqa: E402
from vbgr.seed import build_seed                           # noqa: E402


def bottom(a, thr=0.5):
    ys = np.nonzero(np.any(np.asarray(a) > thr, axis=1))[0]
    return int(ys.max()) if ys.size else -1


def load_frames(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", required=True, help="pipeline json (engine, checkpoint, repo_dir)")
    ap.add_argument("--arm", action="append", default=[],
                    help="baseline | extend:<max_frac>")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    frames = load_frames(a.clip)
    print(f"{os.path.basename(a.clip)}: {len(frames)} frames "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    base = json.load(open(a.config))
    results = {}

    for arm in (a.arm or ["baseline"]):
        cfg = Config.from_dict(base) if hasattr(Config, "from_dict") else Config()
        if not hasattr(Config, "from_dict"):
            for section, vals in base.items():
                for k, v in vals.items():
                    setattr(getattr(cfg, section), k, v)
        if arm.startswith("extend:"):
            cfg.seed.extend_tail_max = float(arm.split(":", 1)[1])
        else:
            cfg.seed.extend_tail_max = 0.0

        pipe = Pipeline(cfg)
        H, W = frames[0].shape[:2]
        people, props = pipe.detector.detect(frames[0])
        kept, _ = select_person_boxes(
            people, W, H, cfg.detect.person_rel_size_min, cfg.detect.box_score_ratio,
            cfg.detect.person_rel_area_min, cfg.detect.conf_abs_min)
        seed = build_seed(frames[0], pipe.seeder, kept, held_props(props, kept), cfg.seed)
        sb = bottom(seed.mask)
        ed = bottom(refine.erode_dilate(seed.mask, cfg.matting.r_erode, cfg.matting.r_dilate))
        boxes = [d.box for d in kept]
        box_bottom = int(max(b[3] for b in boxes)) if boxes else -1

        alphas, rep = pipe._run_shot(frames, 0)
        ab = [bottom(x) for x in alphas]
        results[arm] = dict(seed_bottom=sb, after_erode_dilate=ed, box_bottom=box_bottom,
                            alpha_bottom=ab, notes=list(seed.notes),
                            extend_tail_max=cfg.seed.extend_tail_max)
        print(f"\n--- arm {arm}")
        print(f"    box bottom {box_bottom}   seed {sb}   after erode_dilate {ed}")
        print(f"    alpha bottom, frames 0..{min(len(ab), 20) - 1}: {ab[:20]}")
        print(f"    DEFICIT at frame 0 (seed - alpha): {ed - ab[0]}")
        print(f"    seed notes: {seed.notes}")

    if len(results) > 1:
        arms = list(results)
        a0, a1 = results[arms[0]], results[arms[1]]
        n = min(len(a0["alpha_bottom"]), len(a1["alpha_bottom"]))
        gain = [a1["alpha_bottom"][i] - a0["alpha_bottom"][i] for i in range(n)]
        print(f"\nrows gained by {arms[1]} over {arms[0]}, per frame: {gain[:20]}")
        print(f"mean gain over the segment: {np.mean(gain):.1f} rows")

    if a.out_json:
        json.dump(results, open(a.out_json, "w"), indent=1)
        print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
