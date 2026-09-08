"""Does the look-ahead seed recover bilibili's boots? (5.7)

Why this exists
---------------
The number that has to move is the alpha bottom of bilibili's SECOND shot.
The shipped build seeds that shot from its first frame, whose mask head
happens to fail, and the shot inherits that seed for all sixteen of its
frames -- the boots are missing for exactly as long as the shot lasts.
5.7's `pick_seed_frame` seeds from the best of the shot's first few frames
instead.  Success is shot 2 frame 0 going from 587 to roughly 830.

The previous answer to this question came from an ad-hoc Colab cell that was
never committed, which is why nobody could re-derive it.  Hence this file.

Two things it does that a bare `_run_shot` call does not, and both matter:

  * it splits shots exactly as `run_clip` does, so "shot 2" here is the same
    shot 2 the shipped run produced.  A segment cut to the bad shot alone
    does NOT reproduce the defect -- the re-encode moves the mask boundary
    (see the 8 Sep entry in the handoff's section 7), so the clip must start
    on a real boundary AND include the shot before it;
  * it reports `seed_frame` per shot, so a null result can be read as either
    "the scan never fired" or "it fired and did not help", which are
    different bugs.

Arms are config only, so a run is reproducible from what it prints:

    python bench/probe_lookahead_seed.py \
        --clip "Claude outputs/gpu/bilibili_2shot.mp4" \
        --config pipe.json --arm off --arm lookahead:4 \
        --out-json bench/results/run_.../probe_lookahead.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vbgr import shots                                     # noqa: E402
from vbgr.config import Config                             # noqa: E402
from vbgr.detect import select_person_boxes                # noqa: E402
from vbgr.pipeline import Pipeline                         # noqa: E402
from vbgr.seed import worst_seed_tail                      # noqa: E402


def bottom(a, thr=0.5):
    """Lowest row holding any alpha above `thr`, or -1 for an empty frame."""
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


def load_config(path):
    if path.endswith((".yaml", ".yml")):
        return Config.load(path)
    return Config.from_dict(json.load(open(path)))


def measure_tails(pipe, cfg, frames, k):
    """Frame-0-onward seed tails, measured independently of the pipeline.

    A look-ahead arm that changes nothing has two very different explanations:
    the scan never ran, or it ran and judged frame 0 healthy.  This makes the
    difference readable from the probe's own output whatever the pipeline did.
    """
    H, W = frames[0].shape[:2]
    out = []
    for i in range(min(max(k, 2), len(frames))):
        people, _props = pipe.detector.detect(frames[i])
        kept, _ = select_person_boxes(
            people, W, H, cfg.detect.person_rel_size_min,
            cfg.detect.box_score_ratio, cfg.detect.person_rel_area_min,
            cfg.detect.conf_abs_min)
        if not kept:
            out.append(None)
            continue
        out.append(round(float(worst_seed_tail(
            seeder=pipe.seeder, frame_bgr=frames[i],
            boxes=[d.box for d in kept])), 4))
    return out


def run_arm(frames, cfg, label, tail_scan=4):
    """Split shots the way run_clip does, then run each one."""
    if cfg.shots.enabled and len(frames) > cfg.shots.min_shot_len * 2:
        starts = shots.compute_scene_cuts(
            frames, cfg.shots.cut_threshold, cfg.shots.hist_bins,
            cfg.shots.min_shot_len, cfg.shots.use_absdiff_guard,
            cfg.shots.absdiff_sigma)
    else:
        starts = [0]
    ranges = shots.shot_ranges(starts, len(frames))

    pipe = Pipeline(cfg)
    out = []
    for a, b in ranges:
        tails = measure_tails(pipe, cfg, frames[a:b], tail_scan)
        alphas, rep = pipe._run_shot(frames[a:b], a)
        out.append(dict(a=a, b=b,
                        seed_frame=int(getattr(rep, "seed_frame", 0)),
                        first_frame_tails=tails,
                        alpha_bottom=[bottom(x) for x in alphas],
                        # How MUCH the matte holds, not only how far down it
                        # reaches.  A shot where the look-ahead changes the cast
                        # moves area, not the lowest row, so alpha_bottom alone
                        # cannot tell a repaired mask from a dropped subject.
                        alpha_mean=[round(float(np.mean(x)), 5) for x in alphas],
                        notes=list(rep.notes)))
    print(f"\n--- arm {label}: lookahead_frames={cfg.seed.lookahead_frames} "
          f"min_tail={cfg.seed.lookahead_min_tail} "
          f"min_gain={cfg.seed.lookahead_min_gain}  starts {starts}")
    for i, s in enumerate(out, 1):
        print(f"    shot {i}  frames {s['a']}..{s['b']}  "
              f"seeded from frame {s['seed_frame']} of the shot")
        print(f"      seed tails, first frames: {s['first_frame_tails']}")
        print(f"      alpha bottom: {s['alpha_bottom'][:20]}")
        print(f"      alpha mean over the shot: "
              f"{sum(s['alpha_mean'])/max(len(s['alpha_mean']),1):.5f}")
        for n in s["notes"]:
            print(f"      note: {n}")
        if not s["notes"]:
            print("      note: (none) -- if lookahead_frames > 1 the scan "
                  "should have said something; suspect the arm never applied")
    return dict(starts=starts, shots=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", required=True,
                    help="pipeline config (json or yaml): engine, checkpoint, repo_dir")
    ap.add_argument("--arm", action="append", default=[],
                    help="off | lookahead:<k>   (repeatable; order is compared pairwise)")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    frames = load_frames(a.clip)
    if not frames:
        raise SystemExit(f"{a.clip}: decoded 0 frames")
    print(f"{os.path.basename(a.clip)}: {len(frames)} frames "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")

    results = {}
    for arm in (a.arm or ["off", "lookahead:4"]):
        cfg = load_config(a.config)
        cfg.seed.lookahead_frames = (
            int(arm.split(":", 1)[1]) if arm.startswith("lookahead:") else 0)
        results[arm] = run_arm(frames, cfg, arm)
        results[arm]["lookahead_frames"] = cfg.seed.lookahead_frames

    names = list(results)
    if len(names) > 1:
        base, test = results[names[0]], results[names[1]]
        print(f"\n=== {names[1]} against {names[0]} ===")
        if [s["a"] for s in base["shots"]] != [s["a"] for s in test["shots"]]:
            print("    SHOT BOUNDARIES DIFFER between arms -- not comparable.")
        else:
            for i, (b0, b1) in enumerate(zip(base["shots"], test["shots"]), 1):
                n = min(len(b0["alpha_bottom"]), len(b1["alpha_bottom"]))
                gain = [b1["alpha_bottom"][j] - b0["alpha_bottom"][j]
                        for j in range(n)]
                m0 = sum(b0['alpha_mean']) / max(len(b0['alpha_mean']), 1)
                m1 = sum(b1['alpha_mean']) / max(len(b1['alpha_mean']), 1)
                print(f"    shot {i}: seed frame {b0['seed_frame']} -> "
                      f"{b1['seed_frame']}, frame 0 bottom "
                      f"{b0['alpha_bottom'][0]} -> {b1['alpha_bottom'][0]}, "
                      f"mean gain {np.mean(gain):+.1f} rows, "
                      f"alpha mean {m0:.5f} -> {m1:.5f} ({m1-m0:+.5f})")
                print(f"      per frame: {gain[:20]}")

    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
        json.dump(results, open(a.out_json, "w"), indent=1)
        print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
