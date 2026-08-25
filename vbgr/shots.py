"""Shot detection.

A propagator's memory must never cross a cut, so this gates everything
downstream.  v1 used HSV-histogram Bhattacharyya distance against a fixed
threshold, which is a reasonable primary signal but has two known failure
modes that we patch here:

* **Fixed threshold.** A film-graded clip and a phone clip have wildly
  different inter-frame histogram distances.  We add an adaptive guard based on
  the running median absolute frame difference, so a hard cut is caught even
  when the absolute Bhattacharyya value is below threshold.
* **Flash frames.** A muzzle flash, a strobe, or a single blown-out frame trips
  the histogram test twice in two frames, producing a 1-frame "shot" that then
  gets its own (garbage) seed.  We enforce a minimum shot length.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def hsv_hist(frame_bgr: np.ndarray, bins: int = 64) -> np.ndarray:
    """HSV histogram as a single probability distribution summing to 1.

    BUG FIXED 2026-08-15.  This used to normalise each of the three channel
    histograms to sum 1 and then concatenate them, returning a vector summing
    to **3**.  ``bhattacharyya`` computes ``sqrt(1 - BC)`` and clamps at zero,
    and BC on a 3-summing vector is always >= 1, so the histogram distance was
    **identically 0.000 for every pair of frames ever compared** -- identical
    frames and hard cuts alike.  ``cut_threshold`` has therefore never fired
    once, in v1 or v2: every cut this project has found came from the absdiff
    guard alone.  Caught by printing the top histogram distances on a clip with
    a known cut and getting 0.000.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hs = []
    for c, rng in zip(range(3), ((0, 180), (0, 256), (0, 256))):
        h = cv2.calcHist([hsv], [c], None, [bins], rng).ravel()
        h /= (h.sum() + 1e-9)
        hs.append(h)
    v = np.concatenate(hs).astype(np.float32)
    return v / (v.sum() + 1e-9)          # <- one distribution, not three


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya distance between two probability distributions.

    Both inputs must sum to 1.  They are renormalised here rather than trusted,
    because the silent failure mode when they do not -- a constant 0.0 -- looks
    exactly like "no cuts in this clip".
    """
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    bc = float(np.sqrt(np.clip(p * q, 0, None)).sum())
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def frame_distances(frames: Sequence[np.ndarray], bins: int = 64
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (histogram distance, mean abs pixel difference) per frame pair."""
    small = [cv2.resize(f, (256, 144), interpolation=cv2.INTER_AREA) for f in frames]
    hists = [hsv_hist(f, bins) for f in small]
    hd = np.zeros(len(frames), np.float32)
    ad = np.zeros(len(frames), np.float32)
    for i in range(1, len(frames)):
        hd[i] = bhattacharyya(hists[i - 1], hists[i])
        ad[i] = float(np.abs(small[i].astype(np.int16)
                             - small[i - 1].astype(np.int16)).mean())
    return hd, ad


def compute_scene_cuts(frames: Sequence[np.ndarray],
                       cut_threshold: float = 0.20,
                       bins: int = 64,
                       min_shot_len: int = 12,
                       use_absdiff_guard: bool = True,
                       absdiff_sigma: float = 6.0,
                       warn_suppressed: bool = True) -> List[int]:
    """Return sorted shot start indices (always includes 0).

    ``cut_threshold`` was 0.35 and is now 0.20.  That number was never
    validated, because until the ``hsv_hist`` normalisation bug was fixed the
    histogram distance was constant 0.0 and no threshold could ever fire.
    Measured on the four benchmark windows after the fix:

        clip           median   highest non-cut   at a real cut
        ipman (old)     0.018        0.035        0.404 (f7), 0.457 (f33)
        ipman (new)     0.044        0.087        -- no cuts
        butter          0.031        0.105        0.264 (f6), 0.291 (f43)
        1917            0.018        0.026        -- no cuts

    Highest non-cut value seen is 0.105, lowest real cut 0.264, so 0.20 sits in
    a wide empty gap.  That is four cuts across two clips -- thin.  Revisit
    with more footage.

    Keep the absdiff guard: it is not redundant.  A cut between two shots of
    the same people in the same room -- `tryguys` 148->149, close-up to wide;
    `dance` 321->322, medium to wide -- barely moves the histogram at all, and
    only the absdiff arm catches it.  Both were confirmed by eye.  The two arms
    are OR-ed, and the absdiff arm is gated by the isolated-spike rule below.
    """
    n = len(frames)
    if n == 0:
        return []
    if n < 2:
        return [0]

    hd, ad = frame_distances(frames, bins)
    cuts = set(np.nonzero(hd > cut_threshold)[0].tolist())

    if use_absdiff_guard and n > 8:
        med = float(np.median(ad[1:]))
        mad = float(np.median(np.abs(ad[1:] - med))) + 1e-6
        # 1.4826 * MAD is a robust sigma estimate for a normal distribution.
        thr = med + absdiff_sigma * 1.4826 * mad
        hit = ad > thr
        # Isolated-spike rule.  A hard cut puts exactly ONE frame over the
        # threshold.  A lighting ramp, a dissolve or a camera flare puts a RUN
        # of frames over it.  Without this, `interview` reported 13 "cuts" at
        # frames 32-40 that are a gradual exposure change -- mean luma climbing
        # 109.5 -> 119.7 with the histogram distance never leaving 0.02-0.03.
        # The rule takes that to 3 and keeps every real cut on ipman, butter,
        # shakira, microsoft, eddie, tryguys and dance.
        for i in np.nonzero(hit)[0].tolist():
            if i == 0:
                continue
            if hit[max(i - 1, 0)] or hit[min(i + 1, n - 1)]:
                continue
            cuts.add(int(i))

    cuts.discard(0)
    starts, suppressed = [0], []
    for c in sorted(cuts):
        if c - starts[-1] >= min_shot_len and n - c >= min_shot_len:
            starts.append(int(c))
        else:
            suppressed.append(int(c))
    if suppressed and warn_suppressed:
        # A cut nearer than min_shot_len to the previous one is still a cut.
        # Swallowing it silently is how butter's frame-6 boundary went unnoticed
        # while its seed was propagated straight through it.
        print(f"[shots] WARNING: {len(suppressed)} detected cut(s) at "
              f"{suppressed} suppressed by min_shot_len={min_shot_len}. The "
              f"shot boundary is real; the seed will be propagated across it. "
              f"Do not treat this clip as a single shot.")
    return starts


def shot_ranges(starts: Sequence[int], n_frames: int) -> List[Tuple[int, int]]:
    """Convert start indices to half-open [start, end) ranges covering ALL frames.

    The v1 audit found one clip (ipman) with 176 of 501 frames missing from the
    output.  Whatever the original cause, guaranteeing here that the ranges
    tile [0, n) exactly makes that class of bug impossible: assert coverage.
    """
    s = sorted(set(list(starts) + [0]))
    ranges = [(s[i], s[i + 1] if i + 1 < len(s) else n_frames)
              for i in range(len(s))]
    ranges = [(a, b) for a, b in ranges if b > a]
    covered = sum(b - a for a, b in ranges)
    assert covered == n_frames, (
        f"shot ranges cover {covered}/{n_frames} frames -- would drop "
        f"{n_frames - covered}")
    return ranges
