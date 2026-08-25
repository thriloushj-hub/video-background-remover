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


def track_dominant(alphas, min_frac: float = 0.002,
                   min_overlap: float = 0.30):
    """Follow one subject across frames and return its per-frame area.

    Returns ``(areas, present)``.

    Two design decisions, both forced by getting it wrong first:

    1. **Association is by overlap with the previous tracked mask, not by
       centroid distance.**  Nearest-largest-component flips between people on
       a multi-subject clip, which manufactured five spurious events on
       ``dance`` and three on ``shakira``.

    2. **All overlapping fragments are followed, not just the best one.**  A
       matte's connectivity breaks constantly -- on ``interview`` a single blob
       splits in two at frame 39 while the *total* area does not move at all
       (29518 -> 29126), and tracking only the larger fragment reported a 66%
       "dropout" that never happened.  Same on ``dance`` at 33.  So the tracked
       subject is the union of every component covering at least
       ``min_overlap`` of itself with the previous mask.  A stranger walking in
       overlaps nothing and is excluded; a limb that genuinely keys out has no
       component to contribute and the area really does fall.
    """
    n = len(alphas)
    areas = np.zeros(n, np.float64)
    present = np.zeros(n, bool)
    prev = None
    for i, a in enumerate(alphas):
        hard = (a > 0.5).astype(np.uint8)
        nl, lab, stats, _c = cv2.connectedComponentsWithStats(hard, 8)
        keep = [k for k in range(1, nl)
                if stats[k, cv2.CC_STAT_AREA] >= min_frac * hard.size]
        if not keep:
            continue
        if prev is None:
            sel = [max(keep, key=lambda k: stats[k, cv2.CC_STAT_AREA])]
        else:
            sel = []
            for kk in keep:
                m = (lab == kk)
                inter = float(np.count_nonzero(m & prev))
                if inter / max(float(stats[kk, cv2.CC_STAT_AREA]), 1.0) >= min_overlap:
                    sel.append(kk)
            if not sel:
                continue             # lost this frame; do not adopt a stranger
        mask = np.zeros_like(hard, bool)
        for kk in sel:
            mask |= (lab == kk)
        areas[i] = float(np.count_nonzero(mask))
        present[i] = True
        prev = mask
    return areas, present


def dropout_events(alphas: Sequence[np.ndarray],
                   drop: float = 0.72, recover: float = 0.9,
                   window: int = 24, max_onset: int = 3,
                   cuts: Optional[Sequence[int]] = None
                   ) -> Tuple[int, List[int]]:
    """Count 'a limb keyed out then came back' events.

    REWRITTEN 2026-08-16.  The old version compared **total** foreground area
    against its own trailing median, and on real footage that turned out to
    count three things that are not dropouts:

    * **A different subject arriving.**  On 1917 the tracked soldier runs away
      from camera and shrinks; the area only "recovers" when a *second* soldier
      walks into frame.  The component count never falls at any point.  All four
      of 1917's supposed dropouts are this.
    * **A subject receding.**  Gradual shrink over many frames reads the same as
      an abrupt key-out.
    * **A shot cut.**  butter's only event runs 271->289, and both of those are
      cut frames -- it is one whole shot with less area in it.

    Three changes, each aimed at one of those:

    1. Measure the **tracked dominant component**, not total area, so a new
       arrival cannot supply the recovery (``track_dominant``).
    2. Require the onset to be **abrupt** -- at most ``max_onset`` frames from
       healthy to below threshold.  A limb keys out in one or two frames; a
       subject walks away over dozens.
    3. Skip any event whose span touches a shot boundary in ``cuts``.

    Deliberately NOT requiring the component count to fall: a keyed-out limb
    usually shrinks the body component rather than removing one, so a
    count-based test would miss the very failure this exists to catch.
    """
    if len(alphas) < window + 2:
        return 0, []
    areas, present = track_dominant(alphas)
    cutset = set(int(c) for c in (cuts or []))

    def near_cut(lo, hi):
        return any(c in cutset for c in range(lo - 2, hi + 3))

    events, i = [], window
    while i < len(areas):
        past = areas[max(0, i - window):i]
        past = past[past > 0]
        med = float(np.median(past)) if len(past) else 0.0
        if med <= 0:
            i += 1
            continue
        if present[i] and areas[i] < drop * med:
            # abrupt? walk back and find when it was last healthy
            onset = 0
            k = i - 1
            while k >= 0 and onset <= max_onset and areas[k] < recover * med:
                onset += 1
                k -= 1
            j = i + 1
            while j < min(len(areas), i + window):
                if present[j] and areas[j] > recover * med:
                    if onset <= max_onset and not near_cut(i, j):
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
# Coverage disagreement -- what to do when the baseline over-includes
# --------------------------------------------------------------------------- #
#
# The 2026-08-23 benchmark read as a 15/15 loss for v2: higher ``area_cv`` on
# every clip, lower ``mean_area`` on every clip, no exception.  The contact
# sheets said the opposite.  On ipman both fighters are held for all 72 frames
# and it is *v1's* 0.2453 that is wrong -- two men in a wide shot do not cover
# a quarter of it.
#
# The reason is that ``mean_area`` and ``area_cv`` measure **coverage**, and
# coverage is only a quality signal when both runs are trying to cover the same
# thing.  Against an over-inclusive baseline, "less area" and "more correct"
# are the same observation, so a coverage comparison cannot separate v2 being
# right (ipman, butter) from v2 being broken (interview's torn face).
#
# The fix is to stop comparing two scalars and to look at *where* the two
# mattes disagree.  Every pixel v1 calls subject and v2 calls background is one
# of two things:
#
#   halo  -- outside v2's silhouette entirely.  v1 was keying in background.
#            This is v2 being more correct, and it should not be a penalty.
#   hole  -- inside v2's own silhouette: a gap v2 tore in a subject it is
#            otherwise holding.  This is a real v2 defect.
#
# The two are told apart by morphology, not by area: close v2's hard mask with
# a disk and anything that fills in was a gap narrower than the disk, i.e.
# enclosed by v2's own foreground.  ``close_frac`` is half the widest gap that
# counts as a tear -- 0.04 bridges a gap up to 8% of frame width, which covers
# interview's band (~6%) and stays under the space between two people in a
# two-shot.  A clip where subjects legitimately stand closer than that needs it
# lowered, and the picture checked.
#
# Validated on the 15 contact sheets of the 2026-08-23 run (see
# ``vbgr_coverage_disagreement.md``): ipman 81% halo, butter 51% halo, and
# every talking-head control under 0.5%, which is the picture verdict expressed
# as a number for the first time.  ``hole_frac`` is NOT yet validated on real
# data -- the contact-sheet proxy has a noise floor around 2% and interview's
# real tear scores 2.0% against a clean control's 2.1%.  It needs the runner's
# persisted alphas, not a JPEG.


def coverage_disagreement(base_alphas: Sequence[np.ndarray],
                          alphas: Sequence[np.ndarray],
                          close_frac: float = 0.04,
                          min_comp_frac: float = 0.005,
                          enclosure_min: float = 0.8,
                          ignore: Optional[np.ndarray] = None
                          ) -> Dict[str, float]:
    """Decompose ``base``-minus-``run`` coverage into halo, hole and excess.

    ``base_alphas`` is the baseline (v1), ``alphas`` the run under test (v2),
    same length, same resolution, frame-aligned.

    Returns fractions of the **baseline's** foreground area:

    ``halo_frac``    baseline foreground that lies outside the run's silhouette
                     -- background the baseline was keying in.  High = the
                     baseline over-includes, and the run is right to be smaller.
    ``hole_frac``    baseline foreground the run dropped from *inside* its own
                     silhouette -- a tear.  High = a run defect.
    ``hole_big``     ``hole_frac`` counting only components that are at least
                     ``min_comp_frac`` of the baseline area **and** have at
                     least ``enclosure_min`` of their perimeter against run
                     foreground.  Speckle fails the first test; a shell of
                     baseline halo hugging the run's silhouette fails the
                     second -- that shell is bounded by foreground on one side
                     only, a tear is bounded on both.  On the 23 Aug sheets
                     this is what separates ipman (0.032 raw, 0.007 enclosed,
                     and the sheet shows no tear) from interview (0.035 raw,
                     0.035 enclosed, and the sheet shows the tear).
    ``excess_frac``  run foreground the baseline calls background, as a
                     fraction of baseline area.  **Read ``excess_rim_px``
                     first** -- this number is dominated by how much perimeter
                     the subjects have, not by how much extra is covered.
    ``excess_rim_px``  the same excess divided by the baseline's boundary
                     length, i.e. its mean width in pixels.  Measured on the
                     23 Aug set this is 0.51-1.06 px on 14 of 15 clips while
                     ``excess_frac`` spans 0.027-0.259, so a tenfold spread in
                     the fraction is one constant sub-pixel rim: the two
                     alphas are recovered differently and disagree by one pixel
                     at the 0.5 threshold.  dance3's 0.259 is two small dancers
                     with a lot of edge, not v2 covering more.  ipman is the
                     one real exception at 5.98 px, where the two mattes
                     genuinely differ in both directions.

    ``halo_frac`` and ``hole_frac`` are deliberately not netted against each
    other.  Summing them is how the old table lost the distinction in the
    first place.
    """
    if len(base_alphas) != len(alphas):
        raise ValueError("base and run must have the same frame count")
    tot = halo = hole = big = excess = perim = 0.0
    for b, a in zip(base_alphas, alphas):
        if b.shape != a.shape:
            raise ValueError(f"shape mismatch {b.shape} vs {a.shape}")
        h1 = b > 0.5
        h2 = a > 0.5
        if ignore is not None:
            h1 = h1 & ~ignore
            h2 = h2 & ~ignore
        area = float(h1.sum())
        if area == 0:
            continue
        r = max(3, int(close_frac * b.shape[1]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
        closed = cv2.morphologyEx(h2.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        only_base = h1 & ~h2
        hm = (only_base & closed).astype(np.uint8)
        tot += area
        halo += float((only_base & ~closed).sum())
        hole += float(hm.sum())
        excess += float(((~h1) & h2).sum())
        base_u = h1.astype(np.uint8)
        perim += float(((cv2.dilate(base_u, np.ones((3, 3), np.uint8)) > 0)
                        & (base_u == 0)).sum())
        n, lab, st, _c = cv2.connectedComponentsWithStats(hm, 8)
        cross = np.ones((3, 3), np.uint8)
        fg_near = cv2.dilate(h2.astype(np.uint8), cross) > 0
        for j in range(1, n):
            if st[j, 4] < min_comp_frac * area:
                continue
            comp = (lab == j).astype(np.uint8)
            rim = (cv2.dilate(comp, cross) > 0) & (comp == 0)
            if not rim.any():
                continue
            if float((rim & fg_near).sum()) / float(rim.sum()) >= enclosure_min:
                big += float(st[j, 4])
    if tot == 0:
        return dict(halo_frac=0.0, hole_frac=0.0, hole_big=0.0,
                    excess_frac=0.0, excess_rim_px=0.0)
    return dict(halo_frac=round(halo / tot, 4),
                hole_frac=round(hole / tot, 4),
                hole_big=round(big / tot, 4),
                excess_frac=round(excess / tot, 4),
                excess_rim_px=round(excess / perim, 2) if perim else 0.0)


def area_jitter(alphas: Sequence[np.ndarray]) -> float:
    """Frame-to-frame instability, immune to the camera moving.

    ``area_stability`` (``area_cv``) is the coefficient of variation of area
    over the whole window, so it cannot separate a matte that flickers from a
    subject that legitimately changes size.  On butter the camera dollies back
    and everyone shrinks about 9x, which scores ``area_cv`` 0.74 -- and this
    project has now read that number as matte instability three separate times.

    This measures the *step*, not the spread: the standard deviation of
    ``log(area_t / area_{t-1})``.  A smooth dolly is a small, near-constant
    step and scores low; a matte that drops and recovers scores high.  Report
    both -- ``area_cv`` still catches slow drift, which this cannot see.
    """
    a = np.array([float((x > 0.5).mean()) for x in alphas], np.float64)
    a = a[a > 0]
    if a.size < 3:
        return 0.0
    return float(np.diff(np.log(a)).std())


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
