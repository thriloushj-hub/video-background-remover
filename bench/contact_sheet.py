#!/usr/bin/env python3
"""Visual diffing: alpha over a checkerboard, and failure-frame contact sheets.

Numbers tell you *that* a run regressed; these tell you *where*.  Two outputs:

``sheet``
    A grid of frames sampled around the frames the metrics flagged -- dropout
    events, high-motion frames, low-coverage frames -- rather than uniformly.
    Uniform sampling almost never lands on the two seconds that are broken,
    which is why eyeballing a 30-second clip so rarely finds the bug.

``strip``
    Side-by-side source | alpha-on-checkerboard | composite for one frame, at
    full resolution, for looking at hair and blur edges properly.

Usage::

    python bench/contact_sheet.py sheet --source /path/to/outvids --clip 1917 \\
        --out sheets/1917.png
    python bench/contact_sheet.py strip --source /path/to/outvids --clip ipman \\
        --frame 180 --out sheets/ipman_180.png
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import metrics as M                                   # noqa: E402
from bench.run_bench import detect_green_key, load_alphas, load_frames  # noqa: E402
from vbgr import compose, motion                                 # noqa: E402


def over_checker(frame: np.ndarray, alpha: np.ndarray,
                 size: int = 16) -> np.ndarray:
    h, w = alpha.shape[:2]
    bg = compose.checkerboard(h, w, size)
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h))
    a = alpha[..., None].astype(np.float32)
    return np.clip(frame.astype(np.float32) * a +
                   bg.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Caption below the image, never over it.

    Drawing the caption inside the frame hides the top ~26 px, which is exactly
    where a standing subject's head is. It reads as a decapitation defect that
    is not in the data.
    """
    bar = np.zeros((26, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([img, bar], axis=0)


def grid(tiles: Sequence[np.ndarray], cols: int = 4) -> np.ndarray:
    if not tiles:
        return np.zeros((10, 10, 3), np.uint8)
    h, w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / cols))
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        if t.shape[:2] != (h, w):
            t = cv2.resize(t, (w, h))
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return out


def interesting_frames(alphas: Sequence[np.ndarray],
                       frames: Optional[Sequence[np.ndarray]],
                       n: int = 12) -> List[int]:
    """Pick the frames most likely to show a failure.

    Ranked by a blend of: proximity to a detected dropout event, mask-area
    deviation from the local median, fragmentation, and (when frames are
    available) flow magnitude.  Uniform sampling is used only to fill out the
    remainder.
    """
    T = len(alphas)
    score = np.zeros(T, np.float64)

    areas = np.array([float((a > 0.5).mean()) for a in alphas])
    med = np.median(areas) or 1e-9
    score += np.abs(areas - med) / med

    _, events = M.dropout_events(alphas)
    for e in events:
        lo, hi = max(0, e - 3), min(T, e + 4)
        score[lo:hi] += 3.0

    for i in range(0, T, max(1, T // 60)):
        score[i] += 0.4 * M.significant_components(alphas[i])

    if frames is not None and len(frames) == T:
        fe = motion.FlowEstimator(short_side=240)
        step = max(1, T // 40)
        for i in range(step, T, step):
            mag = fe.magnitude(fe.flow(frames[i - step], frames[i]), 5.0)
            score[i] += float(np.percentile(mag, 95)) / 10.0

    picked: List[int] = []
    order = np.argsort(-score)
    for i in order:
        if len(picked) >= n:
            break
        if all(abs(int(i) - p) > max(4, T // (n * 3)) for p in picked):
            picked.append(int(i))
    picked += [i for i in np.linspace(0, T - 1, n).astype(int)
               if i not in picked][:max(0, n - len(picked))]
    return sorted(picked[:n])


def cmd_sheet(a: argparse.Namespace) -> None:
    out_path = os.path.join(a.source, a.pattern.format(clip=a.clip))
    src_path = os.path.join(a.source, a.source_pattern.format(clip=a.clip))
    key = detect_green_key(out_path) if a.alpha_from == "green" else None
    alphas = load_alphas(out_path, a.alpha_from, key or compose.DEFAULT_GREEN,
                         short_side=a.short_side)
    frames = load_frames(src_path, short_side=a.short_side) \
        if os.path.exists(src_path) else None
    if frames is not None and len(frames) != len(alphas):
        frames = None

    idx = interesting_frames(alphas, frames, a.n)
    tiles = []
    for i in idx:
        base = frames[i] if frames is not None else \
            np.zeros((*alphas[i].shape, 3), np.uint8)
        t = over_checker(base, alphas[i], 12)
        tiles.append(label(t, f"f{i}  area={float((alphas[i]>0.5).mean()):.3f}"))

    sheet = grid(tiles, a.cols)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out}  frames={idx}")


def cmd_strip(a: argparse.Namespace) -> None:
    out_path = os.path.join(a.source, a.pattern.format(clip=a.clip))
    src_path = os.path.join(a.source, a.source_pattern.format(clip=a.clip))
    key = detect_green_key(out_path) if a.alpha_from == "green" else None
    alphas = load_alphas(out_path, a.alpha_from, key or compose.DEFAULT_GREEN,
                         max_frames=a.frame + 1, short_side=a.short_side)
    frames = load_frames(src_path, max_frames=a.frame + 1,
                         short_side=a.short_side)
    i = min(a.frame, len(alphas) - 1)
    al = alphas[i]
    fr = frames[min(i, len(frames) - 1)]
    if fr.shape[:2] != al.shape[:2]:
        al = cv2.resize(al, fr.shape[1::-1])

    panes = [label(fr, "source"),
             label(cv2.cvtColor((al * 255).astype(np.uint8),
                                cv2.COLOR_GRAY2BGR), "alpha"),
             label(over_checker(fr, al, 20), "over checkerboard")]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    cv2.imwrite(a.out, np.concatenate(panes, axis=1))
    print(f"wrote {a.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sheet", cmd_sheet), ("strip", cmd_strip)):
        s = sub.add_parser(name)
        s.add_argument("--source", required=True)
        s.add_argument("--clip", required=True)
        s.add_argument("--pattern", default="{clip}_matte.mp4")
        s.add_argument("--source-pattern", default="{clip}.mp4")
        s.add_argument("--alpha-from", default="green",
                       choices=["green", "grey", "webm"])
        s.add_argument("--short-side", type=int, default=400)
        s.add_argument("--out", required=True)
        if name == "sheet":
            s.add_argument("--n", type=int, default=12)
            s.add_argument("--cols", type=int, default=4)
        else:
            s.add_argument("--frame", type=int, default=0)
        s.set_defaults(func=fn)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
