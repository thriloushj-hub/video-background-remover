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
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hs = []
    for c, rng in zip(range(3), ((0, 180), (0, 256), (0, 256))):
        h = cv2.calcHist([hsv], [c], None, [bins], rng).ravel()
        h /= (h.sum() + 1e-9)
        hs.append(h)
    return np.concatenate(hs).astype(np.float32)


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
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
                       cut_threshold: float = 0.35,
                       bins: int = 64,
                       min_shot_len: int = 12,
                       use_absdiff_guard: bool = True,
                       absdiff_sigma: float = 6.0) -> List[int]:
    """Return sorted shot start indices (always includes 0)."""
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
        cuts |= set(np.nonzero(ad > thr)[0].tolist())

    cuts.discard(0)
    starts = [0]
    for c in sorted(cuts):
        if c - starts[-1] >= min_shot_len and n - c >= min_shot_len:
            starts.append(int(c))
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
