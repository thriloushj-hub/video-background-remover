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
                 flow: Optional[_motion.FlowEstimator] = None,
                 still_px: Optional[float] = None,
                 still_sigma: float = 9.0):
        """`still_px` (21 Sep): if set, the blend weight is additionally scaled
        per pixel by ``clip(1 - flow_mag / still_px, 0, 1)``, so the filter
        acts only where the picture is not moving.  Measured offline on the
        delivered alphas (bench/sweep_flicker_fix.py, arm `still`): the
        photometric `trust` gate alone still leaks smear into fast motion at
        partial trust -- at weight 0.6 it cost ipman 3.6% and dance 4.3% of
        their soft band for no flicker gain -- while the stillness gate cut
        that cost 5-7x and kept the win on the static clip Runbo flagged.
        None keeps the original behaviour exactly."""
        self.weight = weight
        self.consistency_thresh = consistency_thresh
        self.flow = flow or _motion.FlowEstimator()
        self.still_px = still_px
        self.still_sigma = still_sigma
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

        w = self.weight * trust
        if self.still_px is not None:
            mag = self.flow.magnitude(flow, self.still_sigma)
            w = w * np.clip(1.0 - mag / max(self.still_px, 1e-6), 0.0, 1.0)
        w = w.astype(np.float32)
        out = np.clip(a * (1.0 - w) + warped * w, 0.0, 1.0)

        self._prev_alpha, self._prev_bgr = out, frame_bgr.copy()
        return out


# --------------------------------------------------------------------------- #
# Band refinement using a per-frame matting head
# --------------------------------------------------------------------------- #

_WARNED = {"head_none": False}


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

    refined = None
    if head is not None:
        got = head(frame_bgr, trimap)
        # A head that declines returns None.  np.asarray(None) is a 0-d NaN
        # array, so the old code got all the way to `refined[unknown]` and died
        # with "IndexError: invalid index to scalar variable" -- a message that
        # names neither the head nor the engine.  SAM2Matting's matte_frame
        # returns None by design; see its docstring.
        if got is None:
            if not _WARNED["head_none"]:
                print("[refine] matting head returned None; falling back to "
                      "the guided filter. The band is being smoothed, not "
                      "re-matted -- say so in any write-up.")
                _WARNED["head_none"] = True
        else:
            got = np.asarray(got, np.float32)
            if got.shape != a.shape:
                raise ValueError(
                    f"[refine] matting head returned {got.shape}, expected "
                    f"{a.shape}")
            refined = np.clip(got, 0.0, 1.0)
    if refined is None:
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


# --------------------------------------------------------------------------- #
# Core solidification
# --------------------------------------------------------------------------- #

def core_solidify(alpha: np.ndarray, thresh: float = 0.25,
                  radius_frac: float = 0.00625,
                  floor: float = 0.05) -> np.ndarray:
    """Force the subject's interior to full opacity, leaving true edges alone.

    Thriloush's 20 Sep full-length review found interiors and limbs dropping
    to alpha 0.3-0.7 during fast motion and occlusion (butter f281-283 lost
    30% of its solid area against v1; the interview hand, the occluded dance3
    dancer).  IoU at a 0.5 threshold is blind to a pixel falling from 1.0 to
    0.6, which is why every column stayed green while the composites visibly
    faded.

    Construction, validated offline on the six clips he named (sweep in
    bench/results/run_2026-09-18_rerender/core_sweep*.json):

        hard = alpha > thresh
        core = erode( close(hard, r), r )  &  (alpha > floor)
        out[core] = 1.0

    - close() first, so an interior region that dips BELOW `thresh` (the
      faded areas sit at median 0.40, p10 0.14) is bridged before eroding;
    - erode() by the same radius, so nothing within r of a true boundary is
      touched -- hair and real motion-blur edges keep their fractional alpha;
    - the `floor` guard keeps close() from webbing across true gaps (between
      legs, between two subjects): a bridged background pixel has alpha ~0
      and is excluded, a faded subject pixel does not.

    Limitation, on purpose: a thin faded extremity (a gesturing hand) is
    removed entirely by the erode and stays faded.  Interiors are what this
    fixes; thin structures need the engine, not post-processing.

    `radius_frac` is a fraction of frame width (0.00625 = 6 px at 960, the
    sweep's unit; 12 px at 1920).
    """
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    r = max(3, int(round(radius_frac * a.shape[1])))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    hard = (a > thresh).astype(np.uint8)
    closed = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, k)
    core = (cv2.erode(closed, k) > 0) & (a > floor)
    out = a.copy()
    out[core] = 1.0
    return out


def edge_finish(alpha: np.ndarray, mode: str = "soft", thr: float = 0.5,
                round_sigma: float = 2.0, aa_sigma: float = 1.5,
                morph_px: int = 8) -> np.ndarray:
    """Output edge finish -- see IOConfig.edge_finish (2 Oct 2026).

    Works on a float alpha in [0, 1] and returns the same.  "soft" is the
    identity.  All hard modes cut at `thr` first, so nothing here can add or
    remove a region; they only restyle the boundary within a few pixels.
    """
    if mode == "soft":
        return alpha
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    hard = (a > thr).astype(np.float32)
    if mode == "hard":
        pass
    elif mode == "hard_round":
        if round_sigma > 0:
            hard = (cv2.GaussianBlur(hard, (0, 0), round_sigma) > 0.5).astype(np.float32)
    elif mode == "hard_v1":
        r = int(morph_px)
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            hard = cv2.dilate(cv2.erode(hard, k), k)
    else:
        raise ValueError(f"unknown edge_finish {mode!r}")
    if aa_sigma > 0:
        hard = cv2.GaussianBlur(hard, (0, 0), aa_sigma)
    return np.clip(hard, 0.0, 1.0)
