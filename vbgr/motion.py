"""Optical-flow utilities: the motion-blur half of the fix.

Two independent uses of flow in this pipeline:

1.  **Flow-adaptive trimap widening.**  The unknown band of the trimap is
    widened in proportion to local flow magnitude.  A limb moving 30 px/frame
    smears its alpha over ~30 px; a 4 px band cannot represent that, so the
    matting head is never even asked about the blurred region and the limb gets
    keyed out.  Widening the band where (and only where) motion is fast gives
    the head room to work without softening static edges.

2.  **Flow-warped alpha prior.**  A memory propagator is a propagator, not a
    discoverer: it cannot invent a fast, low-contrast limb that was never in
    the seed.  Warping the previous alpha forward gives that limb a plausible
    starting alpha, which the matting head can then correct.  We only trust the
    warp where it is photometrically consistent, so it does not smear.

Everything here is pure OpenCV/NumPy and runs on CPU.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #

class FlowEstimator:
    """Dense optical flow at a reduced resolution, upsampled to full size.

    Flow is only ever used as a *band-width* and *prior* signal here, so
    computing it at a 480 px short side and upsampling costs nothing in quality
    and is roughly 5-8x faster than full resolution.
    """

    def __init__(self, method: str = "dis", short_side: int = 480):
        self.method = method
        self.short_side = short_side
        self._dis = None
        if method == "dis":
            # MEDIUM preset: good accuracy/speed balance, handles large motion.
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self._dis.setUseSpatialPropagation(True)

    # -- internal ---------------------------------------------------------- #

    def _scale(self, h: int, w: int) -> float:
        s = self.short_side / float(min(h, w))
        return min(1.0, s)

    def _prep(self, img: np.ndarray, s: float) -> np.ndarray:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if s < 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return g

    # -- public ------------------------------------------------------------ #

    def flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Backward flow from `curr` to `prev`, at full resolution.

        Returned as HxWx2 float32 in *full-resolution pixels*, i.e. suitable for
        directly warping `prev` into `curr`'s frame via ``warp``.
        """
        h, w = curr.shape[:2]
        s = self._scale(h, w)
        p, c = self._prep(prev, s), self._prep(curr, s)

        if self.method == "dis":
            f = self._dis.calc(c, p, None)          # curr -> prev
        else:
            f = cv2.calcOpticalFlowFarneback(
                c, p, None, 0.5, 4, 21, 3, 7, 1.5, 0
            )

        if s < 1.0:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            f /= s                                   # rescale vector lengths
        return f.astype(np.float32)

    def magnitude(self, flow: np.ndarray, smooth_sigma: float = 9.0) -> np.ndarray:
        """Smoothed per-pixel flow magnitude in px/frame."""
        m = np.linalg.norm(flow, axis=2).astype(np.float32)
        if smooth_sigma > 0:
            k = int(smooth_sigma * 3) | 1
            m = cv2.GaussianBlur(m, (k, k), smooth_sigma)
        return m


def warp(img: np.ndarray, flow: np.ndarray,
         interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Warp `img` (in the previous frame's coordinates) into the current frame.

    `flow` is backward flow (current -> previous) as produced by
    ``FlowEstimator.flow``, so this is a straight ``remap``.
    """
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    return cv2.remap(img, map_x, map_y, interp,
                     borderMode=cv2.BORDER_REPLICATE)


def warp_consistency(prev_bgr: np.ndarray, curr_bgr: np.ndarray,
                     flow: np.ndarray) -> np.ndarray:
    """Per-pixel photometric residual of the warp, in 0-255 units.

    Small residual => the warp is trustworthy at that pixel.  This is what
    stops the flow-warped alpha prior from smearing across occlusion
    boundaries, where flow is by definition wrong.
    """
    warped = warp(prev_bgr, flow)
    d = np.abs(warped.astype(np.float32) - curr_bgr.astype(np.float32))
    return d.mean(axis=2) if d.ndim == 3 else d


# --------------------------------------------------------------------------- #
# Flow-adaptive trimap
# --------------------------------------------------------------------------- #

def boundary_distance(hard_mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) from every pixel to the mask boundary."""
    m = (hard_mask > 0).astype(np.uint8)
    if m.max() == 0 or m.min() == 1:
        # Degenerate: no boundary exists.  Return a large distance everywhere.
        return np.full(m.shape, 1e6, np.float32)
    # distanceTransform measures distance to the nearest ZERO pixel, so d_in is
    # only meaningful inside the mask and d_out only outside it.  Taking
    # min(d_in, d_out) is wrong -- it is identically 0 everywhere, because one
    # of the two is 0 at every pixel.  Select by side instead.
    d_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    return np.where(m > 0, d_in, d_out).astype(np.float32)


def flow_adaptive_trimap(alpha: np.ndarray,
                         flow_mag: Optional[np.ndarray] = None,
                         base: float = 4.0,
                         gain: float = 0.9,
                         w_max: float = 40.0,
                         fg_thresh: float = 0.95,
                         bg_thresh: float = 0.05) -> np.ndarray:
    """Build a trimap whose unknown band widens with local motion.

    Parameters
    ----------
    alpha : HxW float32 in [0, 1]
    flow_mag : HxW float32 flow magnitude in px/frame, or None for a fixed band
    base, gain, w_max : band width is ``clip(base + gain * flow_mag, base, w_max)``

    Returns
    -------
    HxW uint8 trimap: 0 = background, 128 = unknown, 255 = foreground.

    Notes
    -----
    Using a distance transform rather than iterated dilation is what makes the
    *spatially varying* band width possible at all -- you cannot vary a
    structuring element per pixel, but you can threshold a distance field
    against a per-pixel width map.
    """
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    hard = (a > 0.5).astype(np.uint8)

    trimap = np.zeros(a.shape, np.uint8)
    trimap[hard > 0] = 255

    dist = boundary_distance(hard)

    if flow_mag is None:
        width = np.full(a.shape, base, np.float32)
    else:
        width = np.clip(base + gain * flow_mag.astype(np.float32), base, w_max)

    unknown = dist <= width
    # Anything already fractional in the input alpha is unknown by definition.
    unknown |= (a > bg_thresh) & (a < fg_thresh)

    trimap[unknown] = 128
    return trimap


def high_motion_frames(flow_mags, percentile: float = 95.0,
                       threshold_px: float = 6.0):
    """Tag frames whose Pth-percentile flow magnitude exceeds a threshold.

    Used to gate the expensive per-frame refinement pass: on a talking-head
    clip essentially no frames qualify, so the motion machinery is free.
    """
    out = []
    for m in flow_mags:
        out.append(float(np.percentile(m, percentile)) > threshold_px)
    return out


# --------------------------------------------------------------------------- #
# Flow-warped alpha prior
# --------------------------------------------------------------------------- #

def flow_alpha_prior(prev_alpha: np.ndarray,
                     prev_bgr: np.ndarray,
                     curr_bgr: np.ndarray,
                     flow: np.ndarray,
                     consistency_thresh: float = 18.0) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the previous alpha into the current frame, with a trust map.

    Returns ``(warped_alpha, trust)`` where ``trust`` is in [0, 1] and falls to
    zero where the warp is photometrically inconsistent (occlusion boundaries,
    disocclusions, flow failure).
    """
    warped = warp(prev_alpha, flow, cv2.INTER_LINEAR)
    resid = warp_consistency(prev_bgr, curr_bgr, flow)
    # Soft, not hard: a step function here produces visible seams.
    trust = np.clip(1.0 - resid / max(consistency_thresh, 1e-6), 0.0, 1.0)
    trust = cv2.GaussianBlur(trust, (0, 0), 2.0)
    return warped.astype(np.float32), trust.astype(np.float32)


def blend_with_prior(alpha: np.ndarray,
                     warped_alpha: np.ndarray,
                     trust: np.ndarray,
                     flow_mag: np.ndarray,
                     weight: float = 0.5,
                     high_motion_px: float = 6.0) -> np.ndarray:
    """Pull `alpha` toward the flow-warped prior, but only where it helps.

    The prior is applied with weight proportional to (a) how much we trust the
    warp and (b) how fast the region is moving.  In static regions the weight
    is zero, so a good static edge is never softened.
    """
    motion_w = np.clip(flow_mag / max(high_motion_px, 1e-6), 0.0, 1.0)
    w = (weight * trust * motion_w).astype(np.float32)
    return np.clip(alpha * (1.0 - w) + warped_alpha * w, 0.0, 1.0)
