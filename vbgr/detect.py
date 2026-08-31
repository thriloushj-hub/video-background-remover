"""Person / prop detection.

Carried over from v1 with its hard-won lessons intact, because they were
learned from real failures and should not be re-learned:

* **yolov8x, not yolov8n.**  Nano missed people in white against a bright set
  entirely, so they were never seeded.  The large model is ~10x slower per call
  but YOLO is a rounding error next to matting and segmentation, so this costs
  nothing at the pipeline level.
* **No CLAHE.**  Contrast-boosting the frame before detection was tried and
  *hurt* recall: it pushes the image out of the natural-image distribution the
  detector was trained on, especially on strongly lit and saturated footage.
* **Size gate on box height, not area, and independent of position.**  Two
  failures forced this.  Using area, a full-size person entering from the frame
  edge got dropped because the crop collapsed their width.  Rescuing anything
  central kept a small background person standing dead centre between the
  leads.  Height survives horizontal cropping, and dropping the position term
  fixes the centred-bystander case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]

# COCO classes a person plausibly holds. Used only to *keep* held props in the
# seed; background furniture is excluded by the same list.
HOLDABLE = {
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    32: "sports ball", 38: "tennis racket", 39: "bottle", 41: "cup",
    43: "knife", 63: "laptop", 64: "mouse", 66: "keyboard", 67: "cell phone",
    73: "book", 76: "scissors",
}


@dataclass
class Detection:
    box: Box
    conf: float
    cls: int
    label: str = ""

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return float(max(0, self.box[2] - self.box[0]) *
                     max(0, self.box[3] - self.box[1]))


class PersonDetector:
    def __init__(self, model: str = "yolov8x.pt", conf: float = 0.25,
                 iou: float = 0.5, device: str = "auto"):
        self.model_name = model
        self.conf = conf
        self.iou = iou
        self._device = device
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        from ultralytics import YOLO
        from .engines.base import resolve_device
        self._model = YOLO(self.model_name)
        self.device = resolve_device(self._device)

    # -- raw detection ------------------------------------------------------ #

    def detect(self, frame_bgr: np.ndarray,
               conf: Optional[float] = None) -> Tuple[List[Detection], List[Detection]]:
        """Return ``(people, props)``."""
        self._ensure()
        res = self._model.predict(frame_bgr, conf=conf or self.conf,
                                  iou=self.iou, device=self.device,
                                  verbose=False)[0]
        people: List[Detection] = []
        props: List[Detection] = []
        if res.boxes is None:
            return people, props
        for b in res.boxes:
            c = int(b.cls.item())
            xyxy = tuple(int(v) for v in b.xyxy[0].tolist())
            d = Detection(xyxy, float(b.conf.item()), c,
                          res.names.get(c, str(c)))
            if c == 0:
                people.append(d)
            elif c in HOLDABLE:
                props.append(d)
        return people, props


# --------------------------------------------------------------------------- #
# Subject selection
# --------------------------------------------------------------------------- #

def prominence(d: Detection, W: int, H: int) -> float:
    """conf * centrality * relative area -- for ranking and a floor only."""
    cx = (d.box[0] + d.box[2]) / 2.0 / max(W, 1)
    cy = (d.box[1] + d.box[3]) / 2.0 / max(H, 1)
    centrality = 1.0 - float(np.hypot(cx - 0.5, cy - 0.5)) / 0.7071
    return float(d.conf * max(centrality, 0.0) * (d.area / max(W * H, 1)))


def _touches_edge(d: Detection, W: int, H: int, margin: int = 4) -> bool:
    """True if the box runs into any frame edge, i.e. the person is cropped."""
    return (d.box[0] <= margin or d.box[1] <= margin or
            d.box[2] >= W - margin or d.box[3] >= H - margin)


def select_person_boxes(people: Sequence[Detection], W: int, H: int,
                        rel_size_min: float = 0.60,
                        score_ratio: float = 0.15,
                        rel_area_min: float = 0.20,
                        conf_abs_min: float = 0.50,
                        ) -> Tuple[List[Detection], List[Detection]]:
    """Split detections into subjects and background bystanders.

    Two gates, deliberately:

    * **Height**, as before, ignoring position.  This is what lets a full-height
      person entering from the frame edge survive -- their *area* is small
      because they are horizontally cropped, so an area-only gate drops them.
      That failure is why this gate was height-based in the first place.

    * **Area, but only for boxes that do not touch a frame edge.**  Height alone
      cannot tell "close" from "far": measured on 1917, the background soldier
      is 0.658 of the foreground soldier's height -- clearing a 0.60 height gate
      -- but only 0.149 of his box area.  Under perspective, apparent area falls
      off as roughly the square of apparent height, so area separates near from
      far far more sharply.  Restricting the area gate to boxes that are *not*
      cropped by the frame keeps the edge-entrant guarantee intact, because a
      cropped box is exactly the case where small area does not mean far away.

    Measured on the three test clips, first frame (height / box-area / edge):

        ipman   1.000 1.000 edge   0.413 0.098 -      (5 more, all < 0.35)
        butter  1.000 1.000 edge   0.943 0.312 -      0.900 0.500 -
                0.799 0.281 -      0.793 0.191 -      0.772 0.150 -
                0.159 0.014 -
        1917    1.000 1.000 edge   0.658 0.149 -      0.501 0.084 edge

    ``rel_area_min=0.20`` drops 1917's background soldier (0.149) and leaves
    ipman at one subject.  It also drops butter's two smallest standing figures
    (0.191, 0.150), taking butter from 6 kept to 4 -- see docs, that one is a
    judgement call about the shot rather than a defect.

    * **Confidence, absolutely as well as relatively.**  ``score_ratio`` floors
      confidence at a fraction of the best detection in the frame, which on a
      clip with one very confident subject is barely a floor at all: on eddie
      it is ``0.15 * 0.94 = 0.141``.  A 0.277 detection of a blurred portrait
      hanging on the wall behind the subject cleared it, cleared the height
      gate at 0.952 of the tallest, and was 14 px short of the frame edge so
      the area gate did not apply either -- three gates, none of which is about
      *whether this is a person at all*.  Over 284 kept detections across the
      15-clip set that phantom is the only one below **0.755**, so an absolute
      floor anywhere in 0.28..0.75 removes it and nothing else.

    Returns ``(kept, dropped)``.
    """
    if not people:
        return [], []

    tallest = max(d.height for d in people)
    largest = max(d.area for d in people) or 1e-9
    top_conf = max(d.conf for d in people) or 1e-9

    kept, dropped = [], []
    for d in people:
        if d.height < rel_size_min * tallest:
            dropped.append(d)
        elif (rel_area_min > 0
              and not _touches_edge(d, W, H)
              and d.area < rel_area_min * largest):
            # Passed the height gate but is small in area and is not cropped by
            # the frame: a distant bystander, not a near subject.
            dropped.append(d)
        elif d.conf < conf_abs_min:
            # Not a person, as opposed to not a *subject*: the three gates
            # above all ask about relative size or prominence, and a confident
            # detector on a real person does not score 0.28.
            dropped.append(d)
        elif d.conf < score_ratio * top_conf:
            # Confidence floor only.  It is tempting to floor on prominence
            # (conf * centrality * area) instead, but that quietly reintroduces
            # the exact bug the height gate exists to fix: a full-height person
            # entering from the frame edge has a small area and low centrality,
            # so a prominence floor drops them again.  Confidence is the only
            # term here that is independent of size and position.
            dropped.append(d)
        else:
            kept.append(d)

    if not kept:                       # never return nothing
        kept = [max(people, key=lambda d: d.height)]
        dropped = [d for d in people if d is not kept[0]]
    return kept, dropped


def box_containment(a: Box, b: Box) -> float:
    """Fraction of box ``a`` that lies inside box ``b``.

    Not IoU.  Two people standing shoulder to shoulder can have a large IoU
    while neither is inside the other; a duplicate detection of one person is
    almost entirely inside the other.  Containment sees that difference, IoU
    does not.
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    return float(inter / aa) if aa > 0 else 0.0


def mask_containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of mask ``a``'s pixels that also lie in mask ``b``."""
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    aa = int(a.sum())
    if aa == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / aa)


def drop_nested_duplicates(kept: Sequence[Detection],
                           masks: Optional[Sequence[np.ndarray]] = None,
                           box_contain_max: float = 0.80,
                           mask_contain_max: float = 0.70,
                           ) -> Tuple[List[Detection], List[Detection]]:
    """Drop a detection that is really a *part of* another kept detection.

    Why this exists
    ---------------
    Since one ``obj_id`` is seeded per kept person, a person detected twice --
    once whole, once as a nested sub-region -- becomes two objects tracked
    against each other.  SAM2 propagates objects with a non-overlap
    constraint, so both retreat from their shared boundary and the union
    develops a low-alpha **seam down the middle of one person**.

    That is exactly the ``interview`` defect: three seeds, and ``seed_2`` (the
    woman's front) is **97.8% inside** ``seed_1`` (the whole woman).  The
    result was ``hole_big`` 0.055 -- a band torn through her face and hair --
    while ``area_cv`` 0.0175, ``dropouts`` 0 and ``subj_lost_at never`` all
    read that clip clean.

    Measured across the 15-clip benchmark set (27 Aug run seeds), this is the
    only clip with any nesting at all, and the gap is enormous:

        containment   interview   next highest
        box mask-bbox     0.968   0.283 (butter)
        mask pixels       0.978   0.001 (tryguys)

    So ``0.80`` / ``0.70`` sit in empty space in both directions.  Nothing
    else in the set moves.

    Masks are the evidence that matters
    -----------------------------------
    A box test alone is not safe: a small person genuinely standing in front
    of a large one can be box-contained.  Their *masks* are not -- occlusion
    means the near person's pixels are precisely the ones the far person
    does not have.  So when ``masks`` are supplied both tests must fire; with
    boxes only, the box test is used alone and is deliberately the weaker
    claim.  Callers that have masks should pass them.

    Returns ``(kept, dropped)``.
    """
    kept = list(kept)
    if len(kept) < 2:
        return kept, []

    order = sorted(range(len(kept)), key=lambda i: -kept[i].area)
    survivors: List[int] = []
    dropped: List[Detection] = []
    for i in order:
        nested = False
        for j in survivors:
            if box_containment(kept[i].box, kept[j].box) < box_contain_max:
                continue
            if masks is not None:
                if mask_containment(masks[i], masks[j]) < mask_contain_max:
                    continue
            nested = True
            break
        if nested:
            dropped.append(kept[i])
        else:
            survivors.append(i)

    survivors.sort()
    return [kept[i] for i in survivors], dropped


def new_subjects(masks: Sequence[np.ndarray],
                 held_alpha: np.ndarray,
                 overlap_max: float = 0.30,
                 min_frame_frac: float = 0.002,
                 alpha_thresh: float = 0.5) -> List[int]:
    """Indices of detections the matte is **not** already holding.

    This is the decision behind re-seeding.  The benchmark seeds once at frame
    0, so anyone who walks into shot afterwards is invisible to it forever --
    measured on the 27 Aug run, butter loses three dancers to the camera
    pull-back and dance loses one, and all four scored as v1 "halo" because
    v1 holds them and we do not (see Halo_Was_Missing_Subjects).

    A detection is new when little of it overlaps what the tracker already has.
    The test is deliberately on **coverage of the detection**, not IoU: a
    person standing partly behind someone we are holding still has most of
    their own pixels outside the matte, while a re-detection of a subject we
    already hold is almost entirely inside it.

    ``min_frame_frac`` drops specks, so a re-seed scan cannot inject noise as
    a subject.

    .. warning::

       **This is the per-scan test only, and it is not enough on its own.**
       Measured on the GPU, 30 Aug: butter found **9** new subjects across
       four scans where the truth is about three. The caller marks an accepted
       person "claimed" with their mask at that frame, and by the next scan
       they have *moved*, so the overlap against the stale mask is low and
       they are accepted again -- roughly three people times three scans.
       Suppressing a repeat needs the entrant actually tracked between scans,
       not a frozen mask. Do not read a re-seed count as a headcount.

    Returns the indices into ``masks``, in the order given.
    """
    held = np.asarray(held_alpha, np.float32) > alpha_thresh
    out: List[int] = []
    for i, m in enumerate(masks):
        mk = np.asarray(m) > (127 if np.asarray(m).dtype == np.uint8 else 0.5)
        ar = float(mk.sum())
        if ar == 0 or ar / mk.size < min_frame_frac:
            continue
        if float((mk & held).sum()) / ar <= overlap_max:
            out.append(i)
    return out


def held_props(props: Sequence[Detection], people: Sequence[Detection],
               max_frac: float = 0.25) -> List[Detection]:
    """Props that overlap a kept person and are small relative to them.

    Gating on overlap-with-person is what stops the seed swallowing background
    furniture that happens to be a recognised class.
    """
    out = []
    for p in props:
        for q in people:
            if _iou(p.box, q.box) > 0.0 and p.area <= max_frac * max(q.area, 1):
                out.append(p)
                break
    return out


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


iou = _iou
