"""Would the look-ahead seed help or hurt, across every shot in the delivery? (5.7)

Why this exists
---------------
5.7 was validated on one shot of one clip (`bench/results/run_2026-09-08_la/`):
bilibili's shot 2 went 587 -> 866 and the control shot was bit-identical.  That
is not enough to turn it on by default.  The delivery is eighteen clips and
about fifty-six shots, and the failure mode to look for is the MIRROR of the one
5.7 fixes: a shot whose frame 0 is genuinely representative and whose later
frames are not -- someone entering or leaving in the first few frames -- where
moving the seed swaps the cast rather than repairing a bad mask.

The cheap screen, which is what this file is
--------------------------------------------
The look-ahead decision depends only on the DETECTOR and the SEEDER at a shot's
first few frames.  It does not depend on matting at all.  So the whole set can
be screened without propagating a single frame: about four seeded frames per
shot, against the ~13 GPU-hours a full two-arm re-render of the delivery would
cost.  Only the shots this reports as MOVED need a real matting A/B.

For every shot it records:

    tails        seed tail (short of box, as a fraction of box height) per frame
    kept         how many people cleared the seed gate on each of those frames
    decision     what pick_seed_frame returns, plus its own notes
    cast_change  for a MOVED shot: does the chosen frame hold the same people as
                 frame 0?  Boxes are matched by IoU >= 0.5.  A lost subject is
                 the mirror failure; a gained one is a different judgement call
                 and is reported separately rather than counted as a win.

Frames are decoded straight from the delivered mp4 with the pipeline's own
reader.  Nothing is re-encoded and nothing is written out and read back: on this
project a JPEG at quality 100 moved a Mask R-CNN mask boundary by 229 px, and a
segment re-cut from a clip failed to reproduce a defect that was really there.
If the model is the measuring instrument, anything that touches its input is
part of the measurement.

    python bench/sweep_lookahead.py --clips-dir clips --config pipe.json \
        --k 4 --out-json bench/results/run_.../sweep_lookahead.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vbgr import shots, video_io                            # noqa: E402
from vbgr.config import Config                              # noqa: E402
from vbgr.detect import select_person_boxes                 # noqa: E402
from vbgr.pipeline import Pipeline                          # noqa: E402
from vbgr.seed import pick_seed_frame, worst_seed_tail      # noqa: E402


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def cast_delta(boxes0, boxes1, thr=0.5):
    """(lost, gained) -- subjects at frame 0 with no match at the chosen frame,
    and vice versa.  Matching is IoU, which is enough over a few frames."""
    lost = sum(1 for b in boxes0 if max([iou(b, c) for c in boxes1] or [0]) < thr)
    gained = sum(1 for c in boxes1 if max([iou(c, b) for b in boxes0] or [0]) < thr)
    return lost, gained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=4, help="lookahead_frames to screen at")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--only", default=None, help="comma-separated clip stems")
    a = ap.parse_args()

    base = json.load(open(a.config))
    cfg = Config.from_dict(base)
    cfg.seed.lookahead_frames = a.k
    pipe = Pipeline(cfg)

    paths = sorted(glob.glob(os.path.join(a.clips_dir, "*.mp4")))
    paths = [p for p in paths if not os.path.basename(p).endswith("_matte.mp4")]
    if a.only:
        want = set(a.only.split(","))
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in want]

    results, t_start = {}, time.time()
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        frames = video_io.read_all(path, cfg.io.max_size)
        if not frames:
            print(f"{name}: decoded 0 frames -- SKIPPED", flush=True)
            continue
        H, W = frames[0].shape[:2]
        if cfg.shots.enabled and len(frames) > cfg.shots.min_shot_len * 2:
            starts = shots.compute_scene_cuts(
                frames, cfg.shots.cut_threshold, cfg.shots.hist_bins,
                cfg.shots.min_shot_len, cfg.shots.use_absdiff_guard,
                cfg.shots.absdiff_sigma)
        else:
            starts = [0]
        ranges = shots.shot_ranges(starts, len(frames))

        def sel(pe, w, h):
            return select_person_boxes(
                pe, w, h, cfg.detect.person_rel_size_min, cfg.detect.box_score_ratio,
                cfg.detect.person_rel_area_min, cfg.detect.conf_abs_min)

        clip_rows = []
        for si, (s, e) in enumerate(ranges, 1):
            shot = frames[s:e]
            tails, kept_n, boxes_at = [], [], {}
            for i in range(min(a.k, len(shot))):
                people, _ = pipe.detector.detect(shot[i])
                kept, _ = sel(people, W, H)
                kept_n.append(len(kept))
                if not kept:
                    tails.append(None)
                    boxes_at[i] = []
                    continue
                bx = [list(map(int, d.box)) for d in kept]
                boxes_at[i] = bx
                t = worst_seed_tail(seeder=pipe.seeder, frame_bgr=shot[i],
                                    boxes=[d.box for d in kept])
                tails.append(None if t is None else round(float(t), 4))
            idx, notes = pick_seed_frame(shot, pipe.detector, pipe.seeder, cfg.seed, sel)
            lost = gained = 0
            if idx:
                lost, gained = cast_delta(boxes_at.get(0, []), boxes_at.get(idx, []))
            clip_rows.append(dict(shot=si, a=s, b=e, tails=tails, kept=kept_n,
                                  seed_frame=idx, notes=notes,
                                  cast_lost=lost, cast_gained=gained))
            flag = "MOVED" if idx else "stay "
            extra = f"  cast lost {lost} gained {gained}" if idx else ""
            print(f"  {name} shot{si:>2} [{s}:{e}] {flag} -> frame {idx}  "
                  f"tails {tails}  kept {kept_n}{extra}", flush=True)
        results[name] = dict(n_frames=len(frames), starts=starts, shots=clip_rows)
        print(f"{name}: {len(frames)}f, {len(ranges)} shot(s), "
              f"{time.time()-t0:.1f}s", flush=True)
        del frames

    # -- the summary that decides the default -------------------------------- #
    all_shots = [r for c in results.values() for r in c["shots"]]
    moved = [r for r in all_shots if r["seed_frame"]]
    lost = [r for r in moved if r["cast_lost"]]
    gained = [r for r in moved if r["cast_gained"] and not r["cast_lost"]]
    print("\n=== SUMMARY ===", flush=True)
    print(f"clips {len(results)}  shots {len(all_shots)}  "
          f"moved {len(moved)}  of those, cast LOST on {len(lost)}, "
          f"cast gained on {len(gained)}", flush=True)
    for r in moved:
        who = [n for n, c in results.items() if r in c["shots"]][0]
        print(f"  MOVED {who} shot{r['shot']} [{r['a']}:{r['b']}] -> frame "
              f"{r['seed_frame']}  lost {r['cast_lost']} gained {r['cast_gained']}",
              flush=True)
    print(f"total {time.time()-t_start:.1f}s", flush=True)

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump(dict(k=a.k, results=results,
                   summary=dict(clips=len(results), shots=len(all_shots),
                                moved=len(moved), cast_lost=len(lost),
                                cast_gained=len(gained))),
              open(a.out_json, "w"), indent=1)
    print(f"wrote {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
