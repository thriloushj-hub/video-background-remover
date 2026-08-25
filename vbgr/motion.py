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


# --------------------------------------------------------------------------- #
# Camera motion vs subject motion
# --------------------------------------------------------------------------- #

def global_affine(flow: np.ndarray, step: int = 8,
                  trim_rounds: int = 2, trim_frac: float = 0.35
                  ) -> np.ndarray:
    """Fit the 2x3 affine that best explains `flow` as a camera move.

    A dolly, pan, tilt or zoom moves *every* pixel in a way an affine describes
    well.  A person moving does not.  So the affine fit is the camera, and what
    is left over is the subjects.

    Fitted on a subsampled grid and then re-fitted twice after discarding the
    worst ``trim_frac`` of residuals, so a large fast subject cannot drag the
    model onto itself.  Plain least squares is not enough here: on the butter
    window four dancers fill the frame at the start.
    """
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    xs = xs.ravel().astype(np.float32)
    ys = ys.ravel().astype(np.float32)
    u = flow[::step, ::step, 0].ravel().astype(np.float32)
    v = flow[::step, ::step, 1].ravel().astype(np.float32)

    A = np.stack([xs, ys, np.ones_like(xs)], 1)
    keep = np.ones(len(xs), bool)
    M = np.zeros((2, 3), np.float32)
    for _ in range(trim_rounds + 1):
        if keep.sum() < 12:
            break
        sol_u, *_ = np.linalg.lstsq(A[keep], u[keep], rcond=None)
        sol_v, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        M = np.stack([sol_u, sol_v]).astype(np.float32)
        r = np.hypot(A @ sol_u - u, A @ sol_v - v)
        cut = np.quantile(r, 1.0 - trim_frac)
        keep = r <= max(cut, 1e-6)
    return M


def decompose_motion(flow: np.ndarray, step: int = 8,
                     percentile: float = 95.0):
    """Split a flow field into camera and subject motion, in px/frame.

    Returns ``(camera_px, subject_px, residual_map)``:

    * ``camera_px``  -- median magnitude of the fitted global affine, i.e. how
      much the whole frame is moving.
    * ``subject_px`` -- ``percentile``-th percentile of the residual after the
      camera model is removed, i.e. how much the *content* is moving on top of
      the camera.
    * ``residual_map`` -- the per-pixel residual magnitude.

    Why this exists: candidate benchmark windows were being ranked on raw mean
    flow, and on butter that picked a **dolly-back** -- 37 px/frame of camera,
    not of dancers.  Ranking on ``subject_px`` picks the window where the
    subjects are actually hard.
    """
    h, w = flow.shape[:2]
    M = global_affine(flow, step=step)
    ys, xs = np.mgrid[0:h, 0:w]
    gx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
    gy = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
    camera = float(np.median(np.hypot(gx, gy)))
    resid = np.hypot(flow[:, :, 0] - gx, flow[:, :, 1] - gy).astype(np.float32)
    return camera, float(np.percentile(resid, percentile)), resid


def texture_coverage(frame_bgr: np.ndarray, min_grad: float = 8.0,
                     sigma: float = 3.0) -> float:
    """Fraction of the frame with enough gradient for flow to mean anything.

    Optical flow is undefined on a featureless surface: there is nothing to
    match, so DIS returns something near zero no matter how the camera moves.
    On the butter window the background is a smooth colour-gradient wall, only
    25% of the frame carries usable texture, and the flow field consequently
    reports a nearly static camera while the framing is visibly pulling back.

    Anything derived from flow -- ``decompose_motion`` here, but also the
    flow-warped alpha prior and the flow-adaptive trimap in the motion fix --
    is only as trustworthy as this number.  Below about 0.35 treat flow-derived
    quantities as indicative, not evidence.
    """
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    tex = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigma)
    return float((tex > min_grad).mean())


def subject_motion(frames, masks, percentile: float = 95.0,
                   min_pixels: int = 500, flow=None):
    """Flow magnitude measured *inside the subject*, in px/frame.

    ``decompose_motion`` works by fitting a global affine to the whole flow
    field and calling the residual "subject motion".  That fails on footage
    with a flat background: optical flow is undefined where there is nothing
    to match, so on butter three quarters of the frame reports a static scene
    while the framing is visibly pulling back, and the affine ends up fitted to
    the dancers -- the very thing it is meant to exclude.

    Restricting the measurement to the subject sidesteps the problem entirely,
    because people are textured even when their background is not.  Measured
    over each clip's benchmark window:

        clip       texture, whole frame    texture, inside subject
        butter              22%                     61%
        shakira             24%                     72%
        dance               31%                     87%
        1917                93%                     84%
        ipman               51%                     52%

    The three clips that fall below the 0.35 usability floor frame-wide are
    comfortably above it inside the subject, and the clips that were already
    fine do not regress.

    ``masks`` is one boolean/0-1 mask per frame.  v1's matte works as a source
    for these, which is what makes this runnable on CPU with no seeding stage.

    Returns ``(subject_px, texture_in_subject)``.  Read ``texture_in_subject``
    first: below ~0.35 the number is still indicative only -- ``microsoft`` is
    such a case at 0.30, a smoothly lit talking head on a plain backdrop.
    """
    fe = flow or FlowEstimator()
    mags, texs = [], []
    for i in range(1, len(frames)):
        m = np.asarray(masks[i], bool)
        if m.shape != frames[i].shape[:2]:
            m = cv2.resize(m.astype(np.uint8), frames[i].shape[1::-1],
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        if int(m.sum()) < min_pixels:
            continue
        fl = fe.flow(frames[i - 1], frames[i])
        mags.append(float(np.percentile(np.linalg.norm(fl, axis=2)[m], percentile)))
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        t = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3),
                                      cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)),
                             (0, 0), 3.0) > 8.0
        texs.append(float(t[m].mean()))
    if not mags:
        return 0.0, 0.0
    return float(np.mean(mags)), float(np.mean(texs))
