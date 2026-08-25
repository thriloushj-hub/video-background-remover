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
