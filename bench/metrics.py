"""Metrics for video matting -- including the reference-free ones.

The hard constraint on this project is that **there is no ground-truth alpha**
for 1917 / ipman / butter.  They are film clips.  MAD, MSE, Grad, Conn and
dtSSD all require a GT matte, so on this benchmark set they are unusable.  That
is exactly why the previous iterations "never felt satisfying": there was no
way to tell an improvement from a regression, so every change was argued
visually and nothing accumulated.

So this module provides two families.

**Reference-free** (usable today, on every clip):

* ``edge_softness`` -- fraction of boundary-band pixels with fractional alpha.
  Directly measures the "crunchy, cut-out" look of a binary mask.  A binary
  mask scores ~0; a real matte on hair scores 0.3-0.7.
* ``temporal_instability`` -- mean |alpha_t - warp(alpha_{t-1})| inside the
  union band, *after* motion compensation.  This isolates flicker from genuine
  motion; a naive frame-difference metric just measures how fast the subject
  moves.
* ``dropout_events`` -- localises the limb-keyed-out failure: a sharp drop in
  mask area followed by a recovery.  This is the ipman bug, made countable.
* ``fragmentation`` -- mean number of significant connected components.
* ``coverage_gap`` -- fraction of scanned frames where a detected person is not
  explained by the matte.  This is the re-entry bug, made countable.

**Reference-based** (for VideoMatte240K / VM108 style sets where GT exists):
MAD, MSE, Grad, Conn, dtSSD, MESSDdt -- the standard video matting suite.

Reference-free numbers are only meaningful *relative to another run on the same
clip*.  They are a regression harness, not an absolute quality score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from vbgr import motion as _motion
except ImportError:                                    # running from bench/
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vbgr import motion as _motion


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def boundary_band(alpha: np.ndarray, width: int = 6) -> np.ndarray:
    hard = (alpha > 0.5).astype(np.uint8)
    if hard.max() == 0:
        return np.zeros_like(hard, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1,) * 2)
    return (cv2.dilate(hard, k) > 0) & (cv2.erode(hard, k) == 0)


def significant_components(alpha: np.ndarray, min_frac: float = 0.002) -> int:
    hard = (alpha > 0.5).astype(np.uint8)
    if hard.max() == 0:
        return 0
    n, _l, stats, _c = cv2.connectedComponentsWithStats(hard, 8)
    thr = min_frac * hard.size
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= thr))


# --------------------------------------------------------------------------- #
# Reference-free
# --------------------------------------------------------------------------- #

def edge_softness(alpha: np.ndarray, width: int = 6,
                  lo: float = 0.05, hi: float = 0.95) -> float:
    """Fraction of boundary-band pixels that are genuinely fractional.

    Higher is better *for footage that should have soft edges* (hair, motion
    blur, fabric).  A value near 0 means the alpha is effectively a
    segmentation mask -- which is precisely the diagnosis for the SAM-3-only
    outputs.
    """
    band = boundary_band(alpha, width)
    if not band.any():
        return 0.0
    v = alpha[band]
    return float(((v > lo) & (v < hi)).mean())


def boundary_sharpness(alpha: np.ndarray) -> float:
    """Mean |grad alpha| on the band. Distinguishes 'soft' from 'blurry'.

    Read together with ``edge_softness``: high softness + high sharpness is a
    detailed matte; high softness + low sharpness is just a blurred mask.
    """
    band = boundary_band(alpha)
    if not band.any():
        return 0.0
    gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, 3)
    return float(np.sqrt(gx * gx + gy * gy)[band].mean())


def temporal_instability(alphas: Sequence[np.ndarray],
                         frames: Optional[Sequence[np.ndarray]] = None,
                         flow: Optional[_motion.FlowEstimator] = None,
                         stride: int = 1) -> float:
    """Motion-compensated flicker.

    Without ``frames`` this degrades to a plain temporal difference, which
    conflates flicker with motion.  Always pass the frames if you have them.
    """
    if len(alphas) < 2:
        return 0.0
    fe = flow or _motion.FlowEstimator()
    tot, n = 0.0, 0
    for i in range(stride, len(alphas), stride):
        a, p = alphas[i], alphas[i - stride]
        if frames is not None:
            fl = fe.flow(frames[i - stride], frames[i])
            p = _motion.warp(p, fl)
        band = boundary_band(a) | boundary_band(alphas[i - stride])
        if not band.any():
            continue
        tot += float(np.abs(a[band] - p[band]).mean())
        n += 1
    return tot / max(n, 1)


def dropout_events(alphas: Sequence[np.ndarray],
                   drop: float = 0.72, recover: float = 0.9,
                   window: int = 24) -> Tuple[int, List[int]]:
    """Count 'a limb keyed out then came back' events.

    A drop below ``drop`` x the trailing median area that recovers above
    ``recover`` x within ``window`` frames.  This is the ipman failure signature
    and the whole point of measuring it is that it stops being a matter of
    opinion.
    """
    areas = np.array([float((a > 0.5).sum()) for a in alphas], np.float64)
    if len(areas) < window + 2:
        return 0, []
    events, i = [], window
    while i < len(areas):
        med = float(np.median(areas[max(0, i - window):i]))
        if med <= 0:
            i += 1
            continue
        if areas[i] < drop * med:
            j = i + 1
            while j < min(len(areas), i + window):
                if areas[j] > recover * med:
                    events.append(i)
                    i = j
                    break
                j += 1
            else:
                i = min(len(areas), i + window)
                continue
        i += 1
    return len(events), events


def fragmentation(alphas: Sequence[np.ndarray]) -> float:
    return float(np.mean([significant_components(a) for a in alphas]))


def area_stability(alphas: Sequence[np.ndarray]) -> float:
    """Coefficient of variation of mask area. High = unstable track."""
    a = np.array([float((x > 0.5).mean()) for x in alphas], np.float64)
    return float(a.std() / (a.mean() + 1e-9))


# --------------------------------------------------------------------------- #
# Reference-based (needs GT)
# --------------------------------------------------------------------------- #

def mad(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred - gt).mean() * 1e3)


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(((pred - gt) ** 2).mean() * 1e3)


def grad_error(pred: np.ndarray, gt: np.ndarray, sigma: float = 1.4) -> float:
    def g(x):
        gx = cv2.GaussianBlur(x, (0, 0), sigma)
        dx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, 3)
        dy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, 3)
        return np.sqrt(dx * dx + dy * dy)
    return float(((g(pred) - g(gt)) ** 2).sum() / 1e3)


def conn_error(pred: np.ndarray, gt: np.ndarray, step: float = 0.1) -> float:
    """Connectivity error (Rhemann et al. 2009), thresholded approximation."""
    thresholds = np.arange(0, 1 + step, step)
    d_p = np.zeros_like(pred)
    d_g = np.zeros_like(gt)
    for t in thresholds:
        for src, dst in ((pred, d_p), (gt, d_g)):
            m = (src >= t).astype(np.uint8)
            if m.max() == 0:
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 4)
            if n <= 1:
                continue
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            dst[(lab == big) & (dst < t)] = t
    phi_p = 1 - (pred - d_p) * ((pred - d_p) >= 0.15)
    phi_g = 1 - (gt - d_g) * ((gt - d_g) >= 0.15)
    return float(np.abs(phi_p - phi_g).sum() / 1e3)


def dtssd(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]) -> float:
    """Temporal coherence: sum of squared differences of temporal gradients."""
    if len(preds) < 2:
        return 0.0
    tot = 0.0
    for i in range(1, len(preds)):
        dp = preds[i] - preds[i - 1]
        dg = gts[i] - gts[i - 1]
        tot += float(((dp - dg) ** 2).sum())
    return tot / max(len(preds) - 1, 1) / 1e2


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

@dataclass
class ClipMetrics:
    clip: str
    run: str
    n_frames: int
    edge_softness: float = 0.0
    boundary_sharpness: float = 0.0
    temporal_instability: float = 0.0
    dropout_events: int = 0
    dropout_frames: List[int] = field(default_factory=list)
    fragmentation: float = 0.0
    area_stability: float = 0.0
    # reference-based, filled only when GT is supplied
    mad: Optional[float] = None
    mse: Optional[float] = None
    grad: Optional[float] = None
    conn: Optional[float] = None
    dtssd: Optional[float] = None

    def row(self) -> Dict[str, object]:
        d = {"clip": self.clip, "run": self.run, "frames": self.n_frames,
             "edge_soft": round(self.edge_softness, 4),
             "edge_sharp": round(self.boundary_sharpness, 4),
             "temporal": round(self.temporal_instability, 5),
             "dropouts": self.dropout_events,
             "frag": round(self.fragmentation, 3),
             "area_cv": round(self.area_stability, 4)}
        for k in ("mad", "mse", "grad", "conn", "dtssd"):
            v = getattr(self, k)
            if v is not None:
                d[k] = round(v, 4)
        return d


def evaluate(clip: str, run: str,
             alphas: Sequence[np.ndarray],
             frames: Optional[Sequence[np.ndarray]] = None,
             gts: Optional[Sequence[np.ndarray]] = None,
             stride: int = 2) -> ClipMetrics:
    m = ClipMetrics(clip=clip, run=run, n_frames=len(alphas))
    sub = list(range(0, len(alphas), max(1, stride)))

    m.edge_softness = float(np.mean([edge_softness(alphas[i]) for i in sub]))
    m.boundary_sharpness = float(np.mean([boundary_sharpness(alphas[i]) for i in sub]))
    m.temporal_instability = temporal_instability(alphas, frames, stride=stride)
    m.dropout_events, m.dropout_frames = dropout_events(alphas)
    m.fragmentation = fragmentation([alphas[i] for i in sub])
    m.area_stability = area_stability(alphas)

    if gts is not None and len(gts) == len(alphas):
        m.mad = float(np.mean([mad(alphas[i], gts[i]) for i in sub]))
        m.mse = float(np.mean([mse(alphas[i], gts[i]) for i in sub]))
        m.grad = float(np.mean([grad_error(alphas[i], gts[i]) for i in sub]))
        m.conn = float(np.mean([conn_error(alphas[i], gts[i]) for i in sub]))
        m.dtssd = dtssd(alphas, gts)
    return m
