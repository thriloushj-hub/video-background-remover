#!/usr/bin/env python3
"""Regression harness.

Usage
-----
Score an existing set of outputs (no GPU needed -- this is how the v1 baseline
in docs/BASELINE.md was produced)::

    python bench/run_bench.py score \\
        --source "output vids" --run v1_sam3_matanyone \\
        --alpha-from green --pattern "{clip}_matte.mp4" \\
        --out bench/results/v1.json

Compare two runs::

    python bench/run_bench.py compare bench/results/v1.json bench/results/v2.json

Run the pipeline over the frozen benchmark set and score it in one go::

    python bench/run_bench.py run --config configs/default.yaml \\
        --clips bench/clips.txt --run v2_sam2matting

Notes
-----
``--alpha-from green`` recovers alpha by inverting a green composite.  That is
an approximation and slightly over-softens genuinely hard edges, so it is only
valid for comparing runs *that were both scored the same way*.  When a run
emits a real alpha pass, use ``--alpha-from grey`` (an alpha-as-luma video) or
``--alpha-from webm`` (a real alpha channel) instead.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import metrics as M                                    # noqa: E402
from vbgr import compose, video_io                                # noqa: E402


# --------------------------------------------------------------------------- #
# Loading alpha from whatever a run happened to emit
# --------------------------------------------------------------------------- #

def _fit(img: np.ndarray, short_side: Optional[int]) -> np.ndarray:
    """Downscale so min(h, w) == short_side.

    Benchmarking at full 1080p needs ~5 GB per clip just to hold the arrays,
    which is more than most machines want to spend on a metric.  The
    reference-free metrics are all *relative*, so as long as every run in a
    comparison uses the same ``--short-side`` the numbers are commensurable.
    Mixing resolutions across runs is not valid and ``compare`` warns about it.
    """
    if not short_side:
        return img
    h, w = img.shape[:2]
    if min(h, w) <= short_side:
        return img
    s = short_side / float(min(h, w))
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def load_alphas(path: str, mode: str = "green",
                key=(0, 177, 64), max_frames: Optional[int] = None,
                short_side: Optional[int] = 540) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if mode == "webm":
        cap = cv2.VideoCapture(path)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            a = (f[..., 3] if f.shape[2] == 4 else
                 cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 255.
            out.append(_fit(a, short_side))
            if max_frames and len(out) >= max_frames:
                break
        cap.release()
        return out

    for f in video_io.read_frames(path):
        f = _fit(f, short_side)
        if mode == "grey":
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.)
        elif mode == "green":
            out.append(compose.alpha_from_greenscreen(f, key))
        else:
            raise ValueError(f"unknown --alpha-from {mode!r}")
        if max_frames and len(out) >= max_frames:
            break
    return out


def load_frames(path: str, max_frames: Optional[int] = None,
                short_side: Optional[int] = 540) -> List[np.ndarray]:
    out = []
    for f in video_io.read_frames(path):
        out.append(np.ascontiguousarray(_fit(f, short_side)))
        if max_frames and len(out) >= max_frames:
            break
    return out


def detect_green_key(path: str, n: int = 5) -> tuple:
    """Read the modal border colour -- runs disagree on the exact green."""
    cols = []
    for i, f in enumerate(video_io.read_frames(path)):
        if i >= n:
            break
        border = np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]])
        cols.append(np.median(border, axis=0))
    if not cols:
        return compose.DEFAULT_GREEN
    return tuple(int(v) for v in np.median(np.array(cols), axis=0))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_score(a: argparse.Namespace) -> None:
    rows: List[Dict] = []
    pattern = a.pattern
    src_pattern = a.source_pattern

    clips = a.clips or sorted({
        os.path.basename(p).replace("_matte.mp4", "").replace(".mp4", "")
        for p in glob.glob(os.path.join(a.source, "*.mp4"))
        if p.endswith("_matte.mp4")
    })

    for clip in clips:
        out_path = os.path.join(a.source, pattern.format(clip=clip))
        src_path = os.path.join(a.source, src_pattern.format(clip=clip))
        if not os.path.exists(out_path):
            print(f"  skip {clip}: no {out_path}")
            continue

        key = detect_green_key(out_path) if a.alpha_from == "green" else None
        alphas = load_alphas(out_path, a.alpha_from,
                             key or compose.DEFAULT_GREEN, a.max_frames,
                             a.short_side)
        frames = None
        if os.path.exists(src_path) and not a.no_flow:
            frames = load_frames(src_path, a.max_frames, a.short_side)
            if len(frames) != len(alphas):
                print(f"  !! {clip}: source has {len(frames)} frames, "
                      f"output has {len(alphas)} "
                      f"({len(frames) - len(alphas)} DROPPED)")
                # Truncating does NOT realign them: if the run dropped frames
                # from the middle, alpha[i] and frame[i] are different moments
                # in time and every motion-compensated metric is meaningless.
                # Drop the flow term rather than report a fabricated number.
                print(f"     -> frames are not time-aligned; scoring "
                      f"{clip} without motion compensation")
                frames = None

            if frames and alphas and frames[0].shape[:2] != alphas[0].shape[:2]:
                # v1 outputs are padded up to a multiple of 16 by the encoder,
                # so an 804-line source comes back as an 816-line matte. The
                # matte is therefore not pixel-aligned with its own source,
                # which quietly breaks any per-pixel comparison.
                fh_, fw_ = frames[0].shape[:2]
                print(f"  !! {clip}: output is {alphas[0].shape[:2]} but source "
                      f"is {(fh_, fw_)} -- resampling alpha to match")
                alphas = [cv2.resize(x, (fw_, fh_),
                                     interpolation=cv2.INTER_LINEAR)
                          for x in alphas]

        m = M.evaluate(clip, a.run, alphas, frames, stride=a.stride)
        rows.append(m.row())
        print("  " + json.dumps(m.row()))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"run": a.run, "alpha_from": a.alpha_from,
                   "short_side": a.short_side, "rows": rows}, fh, indent=2)
    print(f"\nwrote {a.out} ({len(rows)} clips)")
    print_table(rows)


def cmd_compare(a: argparse.Namespace) -> None:
    runs = []
    for p in a.results:
        with open(p) as fh:
            runs.append(json.load(fh))

    keys = ["edge_soft", "edge_sharp", "temporal", "dropouts", "frag", "area_cv"]
    # Direction of improvement for each metric.
    better = {"edge_soft": +1, "edge_sharp": +1, "temporal": -1,
              "dropouts": -1, "frag": -1, "area_cv": -1}

    base = {r["clip"]: r for r in runs[0]["rows"]}
    print(f"\nbaseline: {runs[0]['run']}")
    ss = {r.get("short_side") for r in runs}
    if len(ss) > 1:
        print(f"WARNING: runs were scored at different resolutions {ss}; "
              f"the reference-free metrics are not comparable across them.")
    for other in runs[1:]:
        print(f"\nvs {other['run']}:")
        hdr = f"{'clip':<14}" + "".join(f"{k:>14}" for k in keys)
        print(hdr)
        print("-" * len(hdr))
        wins = losses = 0
        for r in other["rows"]:
            b = base.get(r["clip"])
            if not b:
                continue
            cells = []
            for k in keys:
                d = r[k] - b[k]
                mark = "" if d == 0 else ("+" if d * better[k] > 0 else "-")
                if d != 0:
                    wins += d * better[k] > 0
                    losses += d * better[k] < 0
                cells.append(f"{r[k]:>10.4g}{mark:>2}" if isinstance(r[k], float)
                             else f"{r[k]:>10}{mark:>2}")
            print(f"{r['clip']:<14}" + "".join(f"{c:>14}" for c in cells))
        print(f"\n  {wins} metric improvements, {losses} regressions")


def cmd_run(a: argparse.Namespace) -> None:
    from vbgr.config import Config
    from vbgr.pipeline import Pipeline

    cfg = Config.load(a.config) if a.config else Config()
    if a.engine:
        cfg.matting.engine = a.engine
    pipe = Pipeline(cfg, require_commercial=a.require_commercial)

    with open(a.clips) as fh:
        paths = [l.strip() for l in fh if l.strip() and not l.startswith("#")]

    rows = []
    for p in paths:
        rep = pipe.run_clip(p, a.out_dir)
        alpha_path = rep.outputs.get("alpha")
        if not alpha_path:
            continue
        alphas = load_alphas(alpha_path, "grey", short_side=a.short_side,
                             max_frames=a.max_frames)
        frames = load_frames(p, a.max_frames, a.short_side)
        n = min(len(frames), len(alphas))
        m = M.evaluate(os.path.splitext(os.path.basename(p))[0], a.run,
                       alphas[:n], frames[:n], stride=a.stride)
        rows.append(m.row())
        print("  " + json.dumps(m.row()))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"run": a.run, "alpha_from": "grey",
                   "short_side": a.short_side, "rows": rows}, fh, indent=2)
    print_table(rows)


def print_table(rows: Sequence[Dict]) -> None:
    if not rows:
        return
    keys = ["clip", "frames", "edge_soft", "edge_sharp", "temporal",
            "dropouts", "frag", "area_cv"]
    keys = [k for k in keys if k in rows[0]]
    w = {k: max(len(k), max(len(f"{r[k]}") for r in rows)) + 2 for k in keys}
    print()
    print("".join(k.rjust(w[k]) for k in keys))
    print("".join("-" * w[k] for k in keys))
    for r in rows:
        print("".join(f"{r[k]}".rjust(w[k]) for k in keys))
    num = [k for k in keys if k not in ("clip",)]
    print("".join("-" * w[k] for k in keys))
    print("mean".rjust(w["clip"]) + "".join(
        f"{np.mean([r[k] for r in rows]):.4g}".rjust(w[k]) for k in num))


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score an existing directory of outputs")
    s.add_argument("--source", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--pattern", default="{clip}_matte.mp4")
    s.add_argument("--source-pattern", default="{clip}.mp4")
    s.add_argument("--alpha-from", default="green",
                   choices=["green", "grey", "webm"])
    s.add_argument("--clips", nargs="*")
    s.add_argument("--stride", type=int, default=2)
    s.add_argument("--max-frames", type=int)
    s.add_argument("--short-side", type=int, default=540,
                   help="downscale to this short side before scoring; every "
                        "run in a comparison must use the same value")
    s.add_argument("--no-flow", action="store_true",
                   help="skip motion compensation (faster, less accurate)")
    s.add_argument("--out", default="bench/results/run.json")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="diff two or more result files")
    c.add_argument("results", nargs="+")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("run", help="run the pipeline then score it")
    r.add_argument("--config")
    r.add_argument("--clips", required=True)
    r.add_argument("--run", required=True)
    r.add_argument("--engine")
    r.add_argument("--out-dir", default="results")
    r.add_argument("--stride", type=int, default=2)
    r.add_argument("--short-side", type=int, default=540)
    r.add_argument("--max-frames", type=int)
    r.add_argument("--require-commercial", action="store_true")
    r.add_argument("--out", default="bench/results/run.json")
    r.set_defaults(func=cmd_run)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
