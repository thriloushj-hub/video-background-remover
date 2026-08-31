"""Object-level re-identification -- the fix for re-entry after occlusion.

The problem, precisely
----------------------
v1 decided "is this person new?" by measuring overlap between a fresh YOLO box
and the current matte.  That is not identity, it is presence.  Consequences:

* A subject who is fully occluded and re-enters is a stranger to the pipeline.
  It may re-seed them (fine), double-count them, or -- worst -- silently fail to
  re-seed because they re-entered *behind* another subject's matte, so the
  overlap test says "already covered".
* There is no way to distinguish "the tracker lost this person" from "this
  person left the shot", so a lost track is never recovered.

The tracker itself will not fix this.  When an object disappears and re-enters,
a VOS memory has no reliable mechanism to re-associate it with the original
instance -- its memory of that object is stale by exactly the duration of the
occlusion.

The fix
-------
Maintain a small feature pool per identity and match detections against it:

1.  Every N frames, run the cheap detector.  (Cheap first: the expensive
    promptable segmenter only runs on boxes that fail the coverage test.)
2.  Embed each detection as a masked crop and score it against every known
    identity's pool by cosine similarity, taking the mean of the top-k pool
    matches -- more robust than a single centroid when a subject changes pose
    or lighting mid-shot.
3.  Solve the assignment globally (Hungarian), not greedily.  Greedy matching
    swaps identities between two similar-looking people standing next to each
    other, which is the classic ReID failure.
4.  Only *high-quality* observations are banked.  Banking a frame from the
    middle of an occlusion is how a feature pool slowly becomes a picture of
    the occluder.

Backends
--------
DINOv3 when available; otherwise a hand-rolled colour-histogram + gradient
descriptor.  The fallback is much weaker but it is dependency-free, which means
the matching, banking and assignment logic is unit-testable on CPU with no
model downloads -- and that logic is where the bugs live.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]          # x1, y1, x2, y2


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

class FeatureExtractor:
    """Embed a masked person crop into a unit-norm vector."""

    def __init__(self, model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
                 device: str = "auto", size: int = 224):
        self.model_name = model
        self.size = size
        self._impl = None
        self.backend = "uninitialised"
        self._device = device

    # -- init ---------------------------------------------------------------- #

    def _ensure(self):
        if self._impl is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            from .engines.base import resolve_device
            dev = resolve_device(self._device)
            proc = AutoImageProcessor.from_pretrained(self.model_name)
            net = AutoModel.from_pretrained(self.model_name).to(dev).eval()
            self._impl = ("dino", torch, proc, net, dev)
            self.backend = "dinov3"
        except Exception as e:                       # noqa: BLE001
            print(f"[reid] DINOv3 unavailable ({type(e).__name__}); "
                  f"falling back to the classical descriptor. Re-entry "
                  f"matching will be less reliable.")
            self._impl = ("classic",)
            self.backend = "classic"

    # -- public -------------------------------------------------------------- #

    def embed(self, frame_bgr: np.ndarray, box: Box,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._ensure()
        crop, cmask = _crop(frame_bgr, box, mask, self.size)
        if self._impl[0] == "dino":
            v = self._embed_dino(crop, cmask)
        else:
            v = _classic_descriptor(crop, cmask)
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype(np.float32)

    def _embed_dino(self, crop_bgr: np.ndarray, cmask: np.ndarray) -> np.ndarray:
        _, torch, proc, net, dev = self._impl
        # Grey out the background so the embedding describes the *person*, not
        # the room they happen to be standing in.  Without this, two people in
        # the same corridor embed almost identically.
        m = cmask[..., None].astype(np.float32)
        blended = crop_bgr.astype(np.float32) * m + 128.0 * (1.0 - m)
        rgb = np.ascontiguousarray(blended.astype(np.uint8)[:, :, ::-1])
        with torch.inference_mode():
            inp = proc(images=rgb, return_tensors="pt").to(dev)
            out = net(**inp)
        if getattr(out, "pooler_output", None) is not None:
            v = out.pooler_output[0]
        else:
            v = out.last_hidden_state[0].mean(0)
        return v.float().cpu().numpy()


def _crop(frame: np.ndarray, box: Box, mask: Optional[np.ndarray],
          size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                      min(w, int(box[2])), min(h, int(box[3])))
    if x2 <= x1 or y2 <= y1:
        return (np.zeros((size, size, 3), np.uint8),
                np.zeros((size, size), np.float32))
    crop = cv2.resize(frame[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
    if mask is None:
        cm = np.ones((size, size), np.float32)
    else:
        cm = cv2.resize(np.asarray(mask, np.float32)[y1:y2, x1:x2],
                        (size, size), interpolation=cv2.INTER_AREA)
        cm = np.clip(cm, 0, 1)
    return crop, cm


def _classic_descriptor(crop_bgr: np.ndarray, cmask: np.ndarray) -> np.ndarray:
    """Colour + gradient descriptor over a 3x1 body-part grid.

    The vertical split matters: head / torso / legs have distinct colour
    statistics, and a whole-crop histogram throws that away, which is what makes
    naive colour ReID confuse any two people in similar lighting.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gx = cv2.Sobel(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1, 3)
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    mag = np.sqrt(gx * gx + gy * gy)

    h = crop_bgr.shape[0]
    parts: List[np.ndarray] = []
    w8 = (cmask > 0.5).astype(np.uint8)
    for i in range(3):
        sl = slice(i * h // 3, (i + 1) * h // 3)
        m = w8[sl]
        for c, bins, rng in ((0, 24, 180), (1, 12, 256), (2, 12, 256)):
            hh = cv2.calcHist([hsv[sl]], [c], m, [bins], [0, rng]).ravel()
            parts.append(hh / (hh.sum() + 1e-9))
        oh, _ = np.histogram(ang[sl][m > 0] if m.any() else ang[sl].ravel(),
                             bins=9, range=(0, 1),
                             weights=(mag[sl][m > 0] if m.any() else mag[sl].ravel()))
        parts.append(oh / (oh.sum() + 1e-9))
    return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #

@dataclass
class Identity:
    ident: int
    pool: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=12))
    last_seen: int = -1
    last_box: Optional[Box] = None
    last_area: float = 0.0
    # Frames on which the tracker was judged to have lost this identity.
    lost_since: Optional[int] = None
    active: bool = True

    def similarity(self, v: np.ndarray, top_k: int = 3) -> float:
        if not self.pool:
            return 0.0
        s = np.array([float(np.dot(v, p)) for p in self.pool], np.float32)
        k = min(top_k, len(s))
        return float(np.sort(s)[-k:].mean())

    def bank(self, v: np.ndarray) -> None:
        self.pool.append(v)


class IdentityBank:
    """Tracks identities across a shot and decides new vs. returning."""

    def __init__(self,
                 extractor: Optional[FeatureExtractor] = None,
                 pool_size: int = 12,
                 match_threshold: float = 0.62,
                 new_identity_threshold: float = 0.45):
        self.extractor = extractor or FeatureExtractor()
        self.pool_size = pool_size
        self.match_threshold = match_threshold
        self.new_identity_threshold = new_identity_threshold
        self.identities: Dict[int, Identity] = {}
        self._next = 1

    # -- lifecycle ----------------------------------------------------------- #

    def reset(self) -> None:
        self.identities.clear()
        self._next = 1

    def new_identity(self) -> Identity:
        i = Identity(self._next)
        i.pool = deque(maxlen=self.pool_size)
        self.identities[self._next] = i
        self._next += 1
        return i

    # -- matching ------------------------------------------------------------ #

    def match(self, frame_bgr: np.ndarray,
              boxes: Sequence[Box],
              masks: Optional[Sequence[np.ndarray]] = None,
              frame_idx: int = 0) -> List[Tuple[int, Optional[int], float]]:
        """Assign detections to identities.

        Returns ``[(det_index, identity_or_None, score), ...]``.  ``None`` means
        the detection is far enough from every known identity to be a genuinely
        new person.  Scores between the two thresholds are *ambiguous* and are
        returned with ``None`` too -- when in doubt we would rather start a new
        track than merge two people, because a merge is unrecoverable while a
        split just costs one extra matting pass.
        """
        if not boxes:
            return []

        vecs = [self.extractor.embed(frame_bgr, b,
                                     masks[i] if masks is not None else None)
                for i, b in enumerate(boxes)]

        ids = [i for i in self.identities.values() if i.pool]
        if not ids:
            return [(i, None, 0.0) for i in range(len(boxes))]

        sim = np.zeros((len(boxes), len(ids)), np.float32)
        for r, v in enumerate(vecs):
            for c, ident in enumerate(ids):
                sim[r, c] = ident.similarity(v)

        pairs = _hungarian(sim)

        out: List[Tuple[int, Optional[int], float]] = []
        assigned = {r: (c, s) for r, c, s in pairs}
        for r in range(len(boxes)):
            if r in assigned:
                c, s = assigned[r]
                out.append((r, ids[c].ident if s >= self.match_threshold else None, s))
            else:
                out.append((r, None, 0.0))
        return out

    def observe(self, ident: int, frame_bgr: np.ndarray, box: Box,
                mask: Optional[np.ndarray], frame_idx: int,
                quality_ok: bool = True) -> None:
        """Record a sighting.  Only bank the embedding if quality is good."""
        it = self.identities.setdefault(ident, Identity(ident))
        it.last_seen = frame_idx
        it.last_box = box
        it.lost_since = None
        it.active = True
        if mask is not None:
            it.last_area = float((np.asarray(mask) > 0.5).sum())
        if quality_ok:
            it.bank(self.extractor.embed(frame_bgr, box, mask))

    def mark_lost(self, ident: int, frame_idx: int) -> None:
        it = self.identities.get(ident)
        if it and it.lost_since is None:
            it.lost_since = frame_idx
            it.active = False

    def lost_identities(self) -> List[Identity]:
        return [i for i in self.identities.values() if not i.active]


def _hungarian(sim: np.ndarray) -> List[Tuple[int, int, float]]:
    """Maximise total similarity. Falls back to greedy if scipy is missing."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-sim)
        return [(int(a), int(b), float(sim[a, b])) for a, b in zip(r, c)]
    except Exception:                                # noqa: BLE001
        out, used_r, used_c = [], set(), set()
        for a, b in sorted(np.ndindex(*sim.shape),
                           key=lambda ab: -sim[ab]):
            if a in used_r or b in used_c:
                continue
            used_r.add(a)
            used_c.add(b)
            out.append((int(a), int(b), float(sim[a, b])))
        return out


# --------------------------------------------------------------------------- #
# Coverage test (cheap gate before any expensive call)
# --------------------------------------------------------------------------- #

def box_coverage(alpha: np.ndarray, box: Box) -> float:
    """Fraction of a detection box already explained by the current matte."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(alpha.shape[1], x2), min(alpha.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    sub = alpha[y1:y2, x1:x2]
    return float((sub > 0.5).mean())


def mask_coverage(alpha: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of a detection's own MASK already explained by the matte.

    The mask-shaped counterpart to :func:`box_coverage`, and the reason it
    exists: box fill is a function of pose.  Measured over the 30 persisted
    seed masks of correctly-held subjects, *box* coverage runs 0.302..0.738 --
    three of them already below the 0.35 threshold at frame 0, with nothing
    wrong -- while *mask* coverage runs 0.802..0.975.  A subject the matte is
    genuinely not holding scores ~0 either way.  Only one of those two numbers
    can carry an absolute threshold.
    """
    m = np.asarray(mask)
    if m.shape[:2] != alpha.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (alpha.shape[1], alpha.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    m = m > 0.5
    if not m.any():
        return 1.0          # nothing to explain; never call it uncovered
    return float((alpha > 0.5)[m].mean())


def confirm_uncovered(alpha: np.ndarray,
                      masks: Sequence[Optional[np.ndarray]],
                      mask_max_overlap: float = 0.50) -> List[int]:
    """Indices of ``masks`` the matte still does not explain.

    Second stage of a two-stage test.  :func:`uncovered_boxes` is the cheap
    pre-filter -- one detector pass, no segmentation -- and this confirms its
    hits against the detection's actual silhouette.  A mask that could not be
    produced (``None``) is treated as unconfirmed rather than uncovered: a
    failure to segment is not evidence of a missing subject.
    """
    out = []
    for i, mk in enumerate(masks):
        if mk is None:
            continue
        if mask_coverage(alpha, mk) < mask_max_overlap:
            out.append(i)
    return out


def uncovered_boxes(alpha: np.ndarray, boxes: Sequence[Box],
                    max_overlap: float = 0.35,
                    min_area_frac: float = 0.0015) -> List[int]:
    """Indices of boxes the current matte does not explain.

    This is the cheap gate: on a frame where everyone is already matted it
    returns [] and the expensive promptable segmenter is never called, so the
    whole re-entry system costs one detector pass every N frames on a fixed-cast
    clip.

    **It is a pre-filter, not a verdict.**  Box fill depends on pose, so this
    fires on correctly-held subjects -- dance3's median component fills 0.345
    of its own box against a 0.35 threshold.  Confirm every hit with
    :func:`confirm_uncovered` before acting on it.
    """
    H, W = alpha.shape[:2]
    out = []
    for i, b in enumerate(boxes):
        area = max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))
        if area < min_area_frac * H * W:
            continue
        if box_coverage(alpha, b) < max_overlap:
            out.append(i)
    return out
