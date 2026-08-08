"""Alpha boundary refinement and temporal stabilisation.

These run *after* the matting engine and are engine-agnostic, so they apply
equally to MatAnyone 2, SAM2Matting or RVM output.  Keeping them here rather
than inside an engine wrapper is deliberate: it means a fair A/B of engines can
hold the post-processing constant.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import motion as _motion


# --------------------------------------------------------------------------- #
# Edge-aware smoothing
# --------------------------------------------------------------------------- #

def guided_filter(alpha: np.ndarray, guide_bgr: np.ndarray,
                  radius: int = 4, eps: float = 1e-4,
                  band_only: bool = True) -> np.ndarray:
    """Edge-aware refinement of alpha using the image as a guide.

    Snaps the alpha boundary onto real image edges.  Restricted to the
    fractional band so a clean solid interior is never touched.
    """
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    guide = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    d = 2 * radius + 1
    mean_g = cv2.boxFilter(guide, -1, (d, d))
    mean_a = cv2.boxFilter(a, -1, (d, d))
    corr_gg = cv2.boxFilter(guide * guide, -1, (d, d))
    corr_ga = cv2.boxFilter(guide * a, -1, (d, d))

    var_g = corr_gg - mean_g * mean_g
    cov_ga = corr_ga - mean_g * mean_a

    A = cov_ga / (var_g + eps)
    b = mean_a - A * mean_g

    mean_A = cv2.boxFilter(A, -1, (d, d))
    mean_b = cv2.boxFilter(b, -1, (d, d))
    out = np.clip(mean_A * guide + mean_b, 0.0, 1.0).astype(np.float32)

    if band_only:
        band = ((a > 0.02) & (a < 0.98)).astype(np.float32)
        band = cv2.GaussianBlur(band, (0, 0), 1.5)
        out = out * band + a * (1.0 - band)
    return out


# --------------------------------------------------------------------------- #
# Temporal stabilisation
# --------------------------------------------------------------------------- #

class TemporalStabilizer:
    """Flow-guided temporal filter for alpha.

    Naive temporal smoothing (an EMA over alpha) is the wrong thing: it removes
    flicker but also smears every fast edge into a ghost.  Motion-compensating
    first, and only blending where the warp is photometrically consistent, kills
    the flicker while leaving genuinely moving edges alone.
    """

    def __init__(self, weight: float = 0.35,
                 consistency_thresh: float = 18.0,
                 flow: Optional[_motion.FlowEstimator] = None):
        self.weight = weight
        self.consistency_thresh = consistency_thresh
        self.flow = flow or _motion.FlowEstimator()
        self._prev_alpha: Optional[np.ndarray] = None
        self._prev_bgr: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_alpha = None
        self._prev_bgr = None

    def step(self, frame_bgr: np.ndarray, alpha: np.ndarray,
             flow: Optional[np.ndarray] = None) -> np.ndarray:
        a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        if self._prev_alpha is None:
            self._prev_alpha, self._prev_bgr = a, frame_bgr.copy()
            return a

        if flow is None:
            flow = self.flow.flow(self._prev_bgr, frame_bgr)

        warped, trust = _motion.flow_alpha_prior(
            self._prev_alpha, self._prev_bgr, frame_bgr, flow,
            self.consistency_thresh)

        w = (self.weight * trust).astype(np.float32)
        out = np.clip(a * (1.0 - w) + warped * w, 0.0, 1.0)

        self._prev_alpha, self._prev_bgr = out, frame_bgr.copy()
        return out


# --------------------------------------------------------------------------- #
# Band refinement using a per-frame matting head
# --------------------------------------------------------------------------- #

def refine_band(alpha: np.ndarray,
                frame_bgr: np.ndarray,
                trimap: np.ndarray,
                head=None) -> np.ndarray:
    """Recompute alpha inside the trimap's unknown band.

    `head` is any callable ``(frame_bgr, trimap) -> alpha`` -- in practice the
    SAM2Matting progressive matting head or ViTMatte.  When no head is
    available we fall back to a guided-filter refinement, which is much weaker
    but still better than leaving a hard boundary inside a wide band.

    Only the unknown band is written back, so the solid interior that the
    propagator is confident about is preserved verbatim.
    """
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    unknown = (trimap == 128)
    if not unknown.any():
        return a

    if head is not None:
        refined = np.clip(np.asarray(head(frame_bgr, trimap), np.float32), 0.0, 1.0)
    else:
        refined = guided_filter(a, frame_bgr, radius=6, eps=1e-4, band_only=False)

    out = a.copy()
    out[unknown] = refined[unknown]
    # Hard constraints from the trimap must be respected exactly.
    out[trimap == 255] = 1.0
    out[trimap == 0] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Small morphological helpers
# --------------------------------------------------------------------------- #

def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def erode_dilate(mask: np.ndarray, r_erode: int = 0, r_dilate: int = 0) -> np.ndarray:
    m = mask
    if r_erode > 0:
        m = cv2.erode(m, _disk(r_erode))
    if r_dilate > 0:
        m = cv2.dilate(m, _disk(r_dilate))
    return m


def remove_specks(alpha: np.ndarray, min_area: int = 48) -> np.ndarray:
    """Drop tiny disconnected alpha blobs -- almost always keying noise."""
    hard = (alpha > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    if n <= 1:
        return alpha
    out = alpha.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[lab == i] = 0.0
    return out
