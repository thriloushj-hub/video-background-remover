"""Memory quality gate.

The single most under-appreciated cause of progressive drift in VOS-based
matting is that the tracker writes *every* prediction into memory, including
the ones made while the subject is occluded, motion-blurred into the
background, or briefly out of frame.  Once a bad mask is in memory it is
matched against on every subsequent frame, so a two-second occlusion can
degrade the remaining thirty seconds of the shot.

This module scores each prediction *before* it is committed.  Four independent
signals, because no single one is reliable:

* **Area ratio** vs. the running median.  A track that collapses (limb lost,
  subject keyed out) or explodes (background leaked in) shows up here first.
* **Flow IoU** against the motion-compensated previous alpha.  During smooth
  motion the warped previous mask is an excellent predictor; a large
  disagreement means the tracker jumped, which is always an error rather than
  real motion.
* **Binariness** of the boundary band.  A healthy matte has a soft band; an
  alpha that has gone fully binary usually means the matting head has given up
  and is passing the segmentation through.
* **Fragmentation.**  A subject that suddenly becomes twelve disconnected blobs
  is a failure, not a person.

The gate deliberately does *not* try to fix the frame.  It answers one
question -- "is this safe to remember?" -- and hands the decision to the
pipeline, which either withholds the frame from memory (if the engine supports
that) or, after enough consecutive rejects, declares the track lost and calls
the re-prompting layer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from . import motion as _motion


@dataclass
class GateVerdict:
    ok: bool
    score: float
    reason: str
    area_frac: float = 0.0
    flow_iou: float = 1.0
    binariness: float = 0.0
    n_components: int = 1

    def __str__(self) -> str:
        return (f"{'PASS' if self.ok else 'REJECT'} "
                f"score={self.score:.3f} area={self.area_frac:.4f} "
                f"flowIoU={self.flow_iou:.3f} bin={self.binariness:.3f} "
                f"cc={self.n_components} ({self.reason})")


class QualityGate:
    def __init__(self,
                 max_area_ratio: float = 1.6,
                 min_area_ratio: float = 0.55,
                 min_flow_iou: float = 0.55,
                 min_area_frac: float = 0.0008,
                 max_components: int = 8,
                 history: int = 24,
                 lost_after: int = 8):
        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio
        self.min_flow_iou = min_flow_iou
        self.min_area_frac = min_area_frac
        self.max_components = max_components
        self.lost_after = lost_after

        self._areas: Deque[float] = deque(maxlen=history)
        self._prev_alpha: Optional[np.ndarray] = None
        self._prev_bgr: Optional[np.ndarray] = None
        self._consec_bad = 0

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self) -> None:
        self._areas.clear()
        self._prev_alpha = None
        self._prev_bgr = None
        self._consec_bad = 0

    @property
    def track_lost(self) -> bool:
        return self._consec_bad >= self.lost_after

    @property
    def consecutive_rejects(self) -> int:
        return self._consec_bad

    # -- scoring ------------------------------------------------------------ #

    def evaluate(self, alpha: np.ndarray,
                 frame_bgr: Optional[np.ndarray] = None,
                 flow: Optional[np.ndarray] = None,
                 accept: bool = True) -> GateVerdict:
        """Score `alpha`.  ``accept=False`` scores without updating history."""
        a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
        H, W = a.shape[:2]
        hard = a > 0.5
        area = float(hard.sum())
        area_frac = area / float(H * W)

        # --- area stability ------------------------------------------------ #
        ratio = 1.0
        if self._areas:
            med = float(np.median(self._areas))
            if med > 0:
                ratio = area / med

        # --- flow-compensated agreement ------------------------------------ #
        flow_iou = 1.0
        if self._prev_alpha is not None and frame_bgr is not None:
            if flow is None:
                flow = _FLOW.flow(self._prev_bgr, frame_bgr)
            warped = _motion.warp(self._prev_alpha, flow) > 0.5
            inter = np.logical_and(warped, hard).sum()
            union = np.logical_or(warped, hard).sum()
            flow_iou = float(inter / union) if union > 0 else 1.0

        # --- boundary softness --------------------------------------------- #
        band = _boundary_band(hard)
        if band.any():
            frac = a[band]
            binariness = float(((frac < 0.02) | (frac > 0.98)).mean())
        else:
            binariness = 1.0

        # --- fragmentation -------------------------------------------------- #
        n_cc = _significant_components(hard, min_frac=0.002)

        # --- verdict -------------------------------------------------------- #
        reason = "ok"
        ok = True
        if area_frac < self.min_area_frac:
            ok, reason = False, "track collapsed (near-empty alpha)"
        elif self._areas and ratio > self.max_area_ratio:
            ok, reason = False, f"area exploded ({ratio:.2f}x median)"
        elif self._areas and ratio < self.min_area_ratio:
            ok, reason = False, f"area collapsed ({ratio:.2f}x median)"
        elif flow_iou < self.min_flow_iou:
            ok, reason = False, f"disagrees with flow-warped prior (IoU {flow_iou:.2f})"
        elif n_cc > self.max_components:
            ok, reason = False, f"fragmented into {n_cc} components"

        # A single continuous score, useful for ranking frames as ReID anchors:
        # we want to bank embeddings from the *best* frames, not merely passable
        # ones.
        score = float(np.clip(
            min(1.0, flow_iou / max(self.min_flow_iou, 1e-6)) *
            min(1.0, 1.0 / max(ratio, 1e-6) if ratio > 1 else ratio /
                max(self.min_area_ratio, 1e-6)) *
            (1.0 - 0.5 * binariness), 0.0, 1.0))

        v = GateVerdict(ok, score, reason, area_frac, flow_iou, binariness, n_cc)

        if accept:
            self._consec_bad = 0 if ok else self._consec_bad + 1
            if ok:
                self._areas.append(area)
                self._prev_alpha = a
                if frame_bgr is not None:
                    self._prev_bgr = frame_bgr.copy()
            elif self._prev_alpha is None and frame_bgr is not None:
                # Bootstrap: we need *some* reference even if frame 0 scored badly.
                self._prev_alpha, self._prev_bgr = a, frame_bgr.copy()
        return v

    def prime(self, alpha: np.ndarray, frame_bgr: np.ndarray) -> None:
        """Seed the gate from a known-good first frame."""
        a = np.clip(np.asarray(alpha, np.float32), 0, 1)
        self._areas.append(float((a > 0.5).sum()))
        self._prev_alpha = a
        self._prev_bgr = frame_bgr.copy()
        self._consec_bad = 0


_FLOW = _motion.FlowEstimator()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _boundary_band(hard: np.ndarray, width: int = 6) -> np.ndarray:
    m = hard.astype(np.uint8)
    if m.max() == 0:
        return np.zeros_like(m, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1,) * 2)
    return (cv2.dilate(m, k) > 0) & (cv2.erode(m, k) == 0)


def _significant_components(hard: np.ndarray, min_frac: float = 0.002) -> int:
    m = hard.astype(np.uint8)
    if m.max() == 0:
        return 0
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    thr = min_frac * m.size
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= thr))
