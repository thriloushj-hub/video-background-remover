"""First-frame seeding with SAM 3.

Everything downstream depends on this mask: a propagator locks the seed into
permanent memory, so a seed that includes a slice of wall carries that wall
through the entire shot.  The v1 design is kept largely intact -- it was
carefully tuned -- with the fusion logic reorganised so each signal is
separately testable.

Three signals, unioned then gated:

1. **Concept mask** -- SAM 3's promptable-concept head, prompted both with the
   YOLO box and with the text "person", matched back to the box by IoU.  Clean
   silhouettes, but it can miss a limb.
2. **Interactive head** -- the forceful box->mask head, which always fires and
   recovers missed limbs and held objects.  Its *additional* regions are the
   risky ones, so they are gated by appearance before being kept.
3. **Hole logic** -- fill every fully-enclosed hole, then re-open only those
   whose colour matches the scene background.  A dark held prop stays filled;
   a genuine background gap that the body wraps around stays open.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detect import Box, Detection, iou as box_iou

# --------------------------------------------------------------------------- #


@dataclass
class SeedResult:
    mask: np.ndarray                      # HxW uint8 {0,1}
    kept: List[Detection] = field(default_factory=list)
    dropped: List[Detection] = field(default_factory=list)
    per_person: List[np.ndarray] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class SAM3Seeder:
    """Wraps whichever SAM 3 distribution is installed.

    Both the Ultralytics packaging and the Meta reference implementation are
    supported because they diverge on entry point but not on capability, and
    which one is available depends on how the environment was set up.
    """

    def __init__(self, model: str = "sam3.pt",
                 mask_threshold: float = 0.0,
                 detect_threshold: float = 0.35,
                 interactive_mask_threshold: float = 0.0,
                 device: str = "auto"):
        self.model_name = model
        self.mask_threshold = mask_threshold
        self.detect_threshold = detect_threshold
        self.interactive_mask_threshold = interactive_mask_threshold
        self._device = device
        self._impl = None
        self.backend = "uninitialised"

    def _ensure(self):
        """Load a SAM 3 image backend, and say precisely what failed if not.

        This used to swallow the ultralytics exception with a bare
        ``except Exception: pass`` and then report one generic sentence, which
        is why "the full pipeline is unrun" sat in the tracker for three weeks
        with nobody knowing whether ultralytics was missing, the weights were
        missing, or the import path was wrong.  Run on an A100 on 30 Aug, the
        answer was **all three are different problems and neither backend
        exists as coded**::

            from ultralytics import SAM        -> OK
            SAM("sam3.pt")                     -> FileNotFoundError: 'sam3.pt'
            from sam3.build_sam import ...     -> ModuleNotFoundError

        The weights ``sam3.pt`` are not in ultralytics' auto-download set, and
        the bundled fork lays the package out as ``sam3.model_builder`` and
        ``sam3.model.build_sam3matting`` -- there is no ``sam3.build_sam``.
        That import was written against Meta's reference layout, not the
        SAM2Matting fork that is actually installed.

        Both errors are now reported.  A diagnostic that hides the real
        exception costs more than the one line it saves.
        """
        if self._impl is not None:
            return
        from .engines.base import resolve_device
        self.device = resolve_device(self._device)
        why = []
        try:
            from ultralytics import SAM
            self._impl = ("ultralytics", SAM(self.model_name))
            self.backend = "ultralytics"
            return
        except Exception as e:                               # noqa: BLE001
            why.append(f"ultralytics: {type(e).__name__}: {e}")
        try:
            from sam3.build_sam import build_sam3_image_predictor
            self._impl = ("meta", build_sam3_image_predictor(
                self.model_name, device=self.device))
            self.backend = "meta"
            return
        except Exception as e:                               # noqa: BLE001
            why.append(f"sam3 (meta layout): {type(e).__name__}: {e}")
        raise RuntimeError(
            "No SAM 3 image backend could be loaded. Both were tried:\n  "
            + "\n  ".join(why)
            + f"\n(model requested: {self.model_name!r})\n"
            "The benchmark does not hit this path -- it seeds from torchvision "
            "Mask R-CNN, which is what every validated seed in this project "
            "came from. See the seeding-backend note in HANDOFF.")

    # -- prompting ---------------------------------------------------------- #

    def masks_from_boxes(self, frame_bgr: np.ndarray,
                         boxes: Sequence[Box]) -> List[np.ndarray]:
        self._ensure()
        kind, m = self._impl
        if not boxes:
            return []
        if kind == "ultralytics":
            res = m.predict(frame_bgr, bboxes=[list(b) for b in boxes],
                            device=self.device, verbose=False)[0]
            return _ultra_masks(res, frame_bgr.shape[:2])
        m.set_image(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        out = []
        for b in boxes:
            mk, _, _ = m.predict(box=np.array(b, np.float32),
                                 multimask_output=False)
            out.append((np.squeeze(mk) > self.interactive_mask_threshold
                        ).astype(np.uint8))
        return out

    def masks_from_text(self, frame_bgr: np.ndarray,
                        text: str = "person") -> List[Tuple[np.ndarray, Box, float]]:
        """Exhaustive concept detection.  Returns ``[(mask, box, score), ...]``."""
        self._ensure()
        kind, m = self._impl
        if kind == "ultralytics":
            try:
                res = m.predict(frame_bgr, prompt=text, device=self.device,
                                verbose=False)[0]
            except TypeError:
                res = m.predict(frame_bgr, texts=[text], device=self.device,
                                verbose=False)[0]
            masks = _ultra_masks(res, frame_bgr.shape[:2])
            out = []
            for i, mk in enumerate(masks):
                score = float(res.boxes.conf[i]) if res.boxes is not None else 1.0
                if score < self.detect_threshold:
                    continue
                out.append((mk, _bbox_of(mk), score))
            return out
        m.set_image(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        res = m.predict_concept(text=text)
        return [(mk.astype(np.uint8), _bbox_of(mk), float(s))
                for mk, s in zip(res["masks"], res["scores"])
                if s >= self.detect_threshold]

    def masks_from_points(self, frame_bgr: np.ndarray,
                          points: Sequence[Tuple[int, int]],
                          labels: Optional[Sequence[int]] = None) -> np.ndarray:
        self._ensure()
        kind, m = self._impl
        labels = list(labels or [1] * len(points))
        if kind == "ultralytics":
            res = m.predict(frame_bgr, points=[list(p) for p in points],
                            labels=labels, device=self.device, verbose=False)[0]
            ms = _ultra_masks(res, frame_bgr.shape[:2])
            return _union(ms, frame_bgr.shape[:2])
        m.set_image(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        mk, _, _ = m.predict(point_coords=np.array(points, np.float32),
                             point_labels=np.array(labels, np.int32),
                             multimask_output=False)
        return (np.squeeze(mk) > self.mask_threshold).astype(np.uint8)


def _ultra_masks(res, shape) -> List[np.ndarray]:
    if getattr(res, "masks", None) is None or res.masks is None:
        return []
    out = []
    for d in res.masks.data:
        a = d.detach().cpu().numpy().astype(np.uint8)
        if a.shape != shape:
            a = cv2.resize(a, shape[::-1], interpolation=cv2.INTER_NEAREST)
        out.append(a)
    return out


def _bbox_of(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _union(masks: Sequence[np.ndarray], shape) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    for m in masks:
        out |= (m > 0).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# Appearance gating
# --------------------------------------------------------------------------- #

def _hsv_hist(img: np.ndarray, mask: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], (mask > 0).astype(np.uint8),
                     [bins, bins], [0, 180, 0, 256])
    return (h / (h.sum() + 1e-9)).astype(np.float32)


def drop_background_regions(frame_bgr: np.ndarray,
                            base_mask: np.ndarray,
                            extra_mask: np.ndarray,
                            margin: float = 0.10) -> Tuple[np.ndarray, int]:
    """Keep only the added regions that look like the subject.

    Back-projection appearance model: build a colour histogram of the known
    subject (``base_mask``) and of the known background, then score each added
    connected component by which it resembles more.  This is what lets the
    forceful interactive head be used at all -- it recovers missed limbs, but
    without this gate it also grabs the chair the subject is sitting next to.
    """
    added = ((extra_mask > 0) & (base_mask == 0)).astype(np.uint8)
    if not added.any():
        return np.zeros_like(base_mask), 0

    bg = ((base_mask == 0) & (added == 0)).astype(np.uint8)
    h_fg = _hsv_hist(frame_bgr, base_mask)
    h_bg = _hsv_hist(frame_bgr, bg)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(added, 8)
    keep = np.zeros_like(added)
    dropped = 0
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        if stats[i, cv2.CC_STAT_AREA] < 32:
            continue
        h_c = _hsv_hist(frame_bgr, comp)
        # Bhattacharyya coefficient: higher = more similar.
        s_fg = float(np.sqrt(np.clip(h_c * h_fg, 0, None)).sum())
        s_bg = float(np.sqrt(np.clip(h_c * h_bg, 0, None)).sum())
        if s_fg > s_bg + margin:
            keep |= comp
        else:
            dropped += 1
    return keep, dropped


# --------------------------------------------------------------------------- #
# Holes
# --------------------------------------------------------------------------- #

def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every region not reachable by a flood from the image border."""
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    # Anything the flood could not reach from (0,0) through background is a hole.
    holes = ((ff == 0) & (m == 0)).astype(np.uint8)
    return (m | holes).astype(np.uint8)


def reopen_background_gaps(frame_bgr: np.ndarray,
                           filled: np.ndarray,
                           original: np.ndarray,
                           margin: float = 0.12,
                           min_area: int = 64,
                           max_frac: float = 0.05) -> np.ndarray:
    """Re-open filled holes whose colour matches the scene background.

    The distinction that matters: a dark phone held against a torso and the gap
    between an arm on a hip and the torso are both "enclosed holes".  One should
    stay filled, one should not.  Colour is the only cheap signal that
    separates them.
    """
    gaps = ((filled > 0) & (original == 0)).astype(np.uint8)
    if not gaps.any():
        return filled

    bg = ((filled == 0)).astype(np.uint8)
    if not bg.any():
        return filled
    h_bg = _hsv_hist(frame_bgr, bg)
    h_fg = _hsv_hist(frame_bgr, (original > 0).astype(np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(gaps, 8)
    out = filled.copy()
    total = filled.size
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp = (lab == i).astype(np.uint8)
        if area > max_frac * total:
            out[comp > 0] = 0          # a huge "hole" is never a held object
            continue
        h_c = _hsv_hist(frame_bgr, comp)
        s_bg = float(np.sqrt(np.clip(h_c * h_bg, 0, None)).sum())
        s_fg = float(np.sqrt(np.clip(h_c * h_fg, 0, None)).sum())
        if s_bg > s_fg + margin:
            out[comp > 0] = 0
    return out


class MaskRCNNSeeder:
    """Seed masks from torchvision Mask R-CNN. The default seeder.

    Why this and not SAM 3
    ----------------------
    Two reasons, and the second is the real one.

    First, **neither SAM 3 backend is loadable**: ``SAM("sam3.pt")`` raises
    ``FileNotFoundError`` because those weights are not in ultralytics'
    auto-download set and nothing in this project has ever shipped them, and
    ``sam3.build_sam`` does not exist in the fork we install.  See
    ``SeedSAM._ensure`` and Pipeline_Never_Ran.

    Second, and more important: **this is where every validated seed in this
    project came from.**  ``vbgr_bench.py`` has always seeded from Mask R-CNN
    and never touched ``pipeline.py``, so the seed area gate (3.0), the
    one-object-id-per-person fix (3.2a) and the nested-duplicate suppression
    (3.2u) were all measured on these masks.  Using anything else in the
    product would mean shipping a seeder nothing has ever been measured on.
    Choosing this collapses the measured path and the shipped path into one.

    Boxes still come from the pipeline's own detector (YOLO), so the gates keep
    working exactly as tested; this class only answers "what is the mask inside
    this box".  A requested box that no instance matches is **skipped with a
    note, never filled with its rectangle** -- a rectangle seed quietly feeds
    the tracker a slab of background, and this project has enough silent
    failures.
    """

    supports_recover = True

    def __init__(self, detector=None, score_min: float = 0.80,
                 match_iou: float = 0.30, device: str = "auto",
                 recover: bool = True, recover_score_min: float = 0.35):
        self._det = detector
        self.score_min = score_min
        self.match_iou = match_iou
        self.recover = recover
        self.recover_score_min = recover_score_min
        self._device = device
        self.backend = "maskrcnn"
        self.notes: List[str] = []
        # Aligned 1:1 with the last ``boxes`` argument, None where nothing
        # matched.  ``masks_from_boxes`` returns the compact list it always
        # has, so anything that needs to know WHICH box was skipped reads this.
        self.last_aligned: List[Optional[np.ndarray]] = []

    def _ensure(self):
        if self._det is not None:
            return
        import torch
        import torchvision
        from .engines.base import resolve_device
        self.device = resolve_device(self._device)
        m = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT").eval()
        if self.device == "cuda":
            m = m.cuda()
        self._torch = torch
        self._det = m

    def _instances(self, frame_bgr: np.ndarray, score_min=None):
        """(masks, boxes) for people above ``score_min``, at frame resolution.

        A detector exposing ``instances(frame_bgr) -> (masks, boxes)`` is used
        directly.  That hook exists so the matching logic below -- which is
        where the real risk is -- can be tested without torch installed.
        """
        thr = self.score_min if score_min is None else score_min
        if callable(getattr(self._det, "instances", None)):
            try:
                return self._det.instances(frame_bgr, thr)
            except TypeError:
                return self._det.instances(frame_bgr)
        self._ensure()
        import torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        if getattr(self, "device", "cpu") == "cuda":
            t = t.cuda()
        with torch.no_grad():
            o = self._det([t])[0]
        sel = (o["labels"] == 1) & (o["scores"] > thr)
        masks = o["masks"][sel, 0].detach().cpu().numpy() > 0.5
        boxes = o["boxes"][sel].detach().cpu().numpy()
        return masks, boxes

    def _match(self, frame_bgr, boxes, score_min):
        """Aligned list of masks for ``boxes``, None where nothing matched."""
        inst_masks, inst_boxes = self._instances(frame_bgr, score_min)
        out: List[Optional[np.ndarray]] = []
        used = set()
        for b in boxes:
            best, best_iou = None, 0.0
            for i, ib in enumerate(inst_boxes):
                if i in used:
                    continue
                v = box_iou(tuple(map(float, ib)), tuple(map(float, b)))
                if v > best_iou:
                    best, best_iou = i, v
            if best is None or best_iou < self.match_iou:
                out.append(None)
                continue
            used.add(best)
            out.append(inst_masks[best].astype(np.uint8))
        return out

    def masks_from_boxes(self, frame_bgr: np.ndarray,
                         boxes: Sequence[Box],
                         recover_boxes: Sequence[Box] = ()) -> List[np.ndarray]:
        """Masks for ``boxes``, compact -- a box nothing matched is skipped.

        ``recover_boxes`` is the 5.5 escape hatch and it is deliberately
        narrow.  A box listed there has already been confirmed by the re-entry
        scan as a person the matte is not holding, so when the strict pass
        finds nothing for it the match is retried **once**, for that box only,
        at ``recover_score_min``.  Everything else is untouched: ``match_iou``
        still decides whether an instance belongs to the box, a box that still
        matches nothing is still skipped rather than filled with its rectangle,
        and no box outside this list is ever matched at the lower threshold.
        """
        if not boxes:
            self.last_aligned = []
            return []
        self.notes = []
        aligned = self._match(frame_bgr, boxes, None)
        want = {tuple(map(float, b)) for b in recover_boxes}
        missing = [k for k, m in enumerate(aligned)
                   if m is None and tuple(map(float, boxes[k])) in want]
        if missing and self.recover and self.recover_score_min < self.score_min:
            second = self._match(frame_bgr, [boxes[k] for k in missing],
                                 self.recover_score_min)
            for k, m in zip(missing, second):
                if m is None:
                    continue
                aligned[k] = m
                self.notes.append(
                    f"recovered box {tuple(int(x) for x in boxes[k])} at "
                    f"score>={self.recover_score_min} after the strict pass "
                    f"({self.score_min}) matched nothing -- 5.5, confirmed "
                    f"uncovered by the re-entry scan")
        for k, m in enumerate(aligned):
            if m is None:
                self.notes.append(
                    f"no mask instance matched box "
                    f"{tuple(int(x) for x in boxes[k])}; subject skipped")
        self.last_aligned = aligned
        return [m for m in aligned if m is not None]

    def masks_from_text(self, frame_bgr: np.ndarray, text: str = "person"):
        raise NotImplementedError(
            "Mask R-CNN has no text prompt; it is COCO-class only. "
            "build_seed treats this as unavailable and notes it.")

    def masks_from_points(self, frame_bgr: np.ndarray, *a, **k):
        raise NotImplementedError(
            "Mask R-CNN has no point prompt. build_seed notes it and carries on.")


def build_seeder(cfg=None):
    """Construct the configured seeder. Mask R-CNN by default -- see 5.1k."""
    from .config import SeedConfig
    cfg = cfg or SeedConfig()
    backend = getattr(cfg, "backend", "maskrcnn")
    if backend == "maskrcnn":
        return MaskRCNNSeeder(
            score_min=getattr(cfg, "maskrcnn_score_min", 0.80),
            match_iou=getattr(cfg, "maskrcnn_match_iou", 0.30),
            recover=getattr(cfg, "maskrcnn_recover", True),
            recover_score_min=getattr(cfg, "maskrcnn_recover_score_min", 0.35))
    if backend == "sam3":
        return SAM3Seeder(mask_threshold=cfg.mask_threshold,
                          detect_threshold=cfg.detect_threshold,
                          interactive_mask_threshold=cfg.interactive_mask_threshold)
    raise ValueError(f"unknown seeder backend {backend!r}")


# --------------------------------------------------------------------------- #
# Seed quality: is the mask we are about to propagate actually the whole person?
# --------------------------------------------------------------------------- #

def seed_tail_fraction(mask: np.ndarray, box: Box) -> float:
    """Fraction of a detection box's HEIGHT the instance mask fails to reach.

    5.7.  A shot is seeded from one frame and every later frame inherits that
    seed until the next cut, so a mask that stops short of its own box is not a
    small error -- it is the whole shot.  On ``bilibili`` the shot at f1159 is
    seeded from a mask reaching row 596 inside a box reaching row 894, and the
    subject's boots are missing for the next sixteen frames, returning at f1175
    which is exactly the next shot start.

    Height and one end at a time, deliberately, for the reason 3.3a cost us:
    mask AREA over box area is a function of pose (dance3's median component
    fills 0.345 of its own box while being perfectly held), so an area gate
    fires on crouching and sitting people.  A mask that covers the top of its
    box and stops is a different shape of thing from a mask that is simply
    narrow, and this measures only that.

    Returns the larger of the bottom and top tails, 0.0 for a mask that spans
    its box and 1.0 for an empty mask.
    """
    if mask is None:
        return 1.0
    ys = np.nonzero(np.any(np.asarray(mask) > 0, axis=1))[0]
    if ys.size == 0:
        return 1.0
    x0, y0, x1, y1 = (float(v) for v in box)
    bh = max(y1 - y0, 1.0)
    bottom = max(0.0, (y1 - (float(ys.max()) + 1.0))) / bh
    top = max(0.0, (float(ys.min()) - y0)) / bh
    return float(max(bottom, top))


def worst_seed_tail(seeder, frame_bgr: np.ndarray, boxes: Sequence[Box]) -> float:
    """The worst `seed_tail_fraction` over the boxes kept for this frame.

    A box that matches no instance at all counts as 1.0: `masks_from_boxes`
    skips it, so that subject would not be seeded on this frame either.
    """
    if not boxes:
        return 0.0
    aligned = seeder._match(frame_bgr, list(boxes), None)
    return max(seed_tail_fraction(m, b) for m, b in zip(aligned, boxes))


def extend_short_tails(concept: Sequence[np.ndarray], seeder,
                       boxes: Sequence[Box], max_frac: float):
    """Fill the gap between a mask's bottom and its own box bottom.

    5.7.  A shot is seeded once and every frame inherits it, and the matting
    engine does not merely copy the seed -- on ``bilibili`` f1159 it is handed a
    mask reaching row 831 and returns alpha reaching 588, so a seed that is a
    little short at the feet comes out a lot short.  This closes the seed's own
    gap before the engine ever sees it.

    Deliberately narrow, because "extend the mask to its box" as a general rule
    is the shape of five findings this project has already retracted:

    * bottom only -- a mask short at the *top* is hair or a raised arm, where
      the box is loose for reasons that are not occlusion;
    * capped at ``max_frac`` of box height, so a genuinely half-occluded person
      (behind a desk, cut by another subject) is left alone rather than having
      the desk keyed in;
    * horizontally limited to the columns the mask already occupies in its
      lowest rows, so it extends the legs that are there rather than filling
      the box's full width with floor.

    Returns (masks, n_extended).  ``seeder.last_aligned`` says which box each
    compact mask came from; without it nothing is extended.
    """
    aligned = list(getattr(seeder, "last_aligned", []) or [])
    if len(aligned) != len(boxes):
        return list(concept), 0
    out, n = [], 0
    it = iter(concept)
    for b, am in zip(boxes, aligned):
        if am is None:
            continue
        m = next(it, None)
        if m is None:
            break
        m = np.asarray(m)
        ys = np.nonzero(np.any(m > 0, axis=1))[0]
        x0, y0, x1, y1 = (float(v) for v in b)
        bh = max(y1 - y0, 1.0)
        if ys.size:
            bot = int(ys.max())
            gap = (y1 - 1.0) - bot
            if 0 < gap <= max_frac * bh:
                lo = max(bot - max(2, int(0.02 * bh)), 0)
                cols = np.nonzero(np.any(m[lo:bot + 1] > 0, axis=0))[0]
                if cols.size:
                    m = m.copy()
                    m[bot + 1:int(y1), cols.min():cols.max() + 1] = 1
                    n += 1
        out.append(m)
    return out, n


def pick_seed_frame(frames, detector, seeder, cfg, select_boxes):
    """Which frame of this shot should the seed be built from?

    5.7.  Returns ``(index, notes)``.  Index 0 -- the current behaviour -- unless
    a later frame in the first ``cfg.lookahead_frames`` is materially better.

    The test is deliberately RELATIVE.  A global threshold on the seed frame is
    what 3.3a proved cannot work: box fill is a function of pose, and every
    absolute cut put a keep between two drops.  Here the comparison is the same
    subject a few frames apart, so pose and framing are nearly constant and the
    only thing that changes is whether the mask head succeeded.  A person who is
    genuinely half-occluded is equally occluded in all of them, gains nothing,
    and does not move the seed.

    Cheap by construction: the scan only runs when frame 0 already looks bad,
    so a healthy shot pays one extra tail measurement and nothing else.
    """
    notes = []
    k = int(getattr(cfg, "lookahead_frames", 0) or 0)
    if k <= 1 or len(frames) < 2:
        return 0, notes
    H, W = frames[0].shape[:2]

    def tail_at(i):
        people, props = detector.detect(frames[i])
        kept, _ = select_boxes(people, W, H)
        if not kept:
            return None, None
        return worst_seed_tail(seeder, frames[i], [d.box for d in kept]), kept

    min_tail = getattr(cfg, "lookahead_min_tail", 0.15)
    t0, kept0 = tail_at(0)
    # Say so when the scan declines.  A look-ahead arm that changes nothing has
    # two completely different explanations -- the scan never ran, or it ran and
    # found frame 0 healthy -- and a silent return makes them indistinguishable
    # in a run log.  That ambiguity is what made the 8 Sep GPU arm unreadable.
    if t0 is None:
        notes.append("look-ahead: no boxes at frame 0; seeding from it anyway")
        return 0, notes
    if t0 <= min_tail:
        notes.append(f"look-ahead: frame 0 mask tail {t0:.3f} <= "
                     f"{min_tail:.3f}; not scanning")
        return 0, notes

    best_i, best_t, best_kept = 0, t0, kept0
    for i in range(1, min(k, len(frames))):
        ti, ki = tail_at(i)
        if ti is not None and ti < best_t:
            best_i, best_t, best_kept = i, ti, ki
    gain = t0 - best_t
    # Repairing the mask is not worth losing a subject.  A later frame that
    # keeps fewer people than frame 0 is a different cast, not a better seed --
    # the mirror of the failure this exists to fix.
    if (best_i and getattr(cfg, "lookahead_require_same_cast", True)
            and len(best_kept or []) < len(kept0 or [])):
        notes.append(f"look-ahead: frame {best_i} is cleaner "
                     f"({best_t:.3f} vs {t0:.3f}) but holds "
                     f"{len(best_kept or [])} of {len(kept0 or [])} subjects; "
                     f"staying at frame 0")
        return 0, notes
    if best_i and gain >= getattr(cfg, "lookahead_min_gain", 0.10):
        notes.append(f"seeded from frame {best_i} of this shot: frame 0 mask "
                     f"tail {t0:.3f}, frame {best_i} {best_t:.3f}")
        return best_i, notes
    notes.append(f"frame 0 mask tail {t0:.3f}; no better frame in the first {k}")
    return 0, notes


# --------------------------------------------------------------------------- #
# The full seed
# --------------------------------------------------------------------------- #

def build_seed(frame_bgr: np.ndarray,
               seeder: SAM3Seeder,
               kept: Sequence[Detection],
               props: Sequence[Detection] = (),
               cfg=None,
               recover_boxes: Sequence[Box] = ()) -> SeedResult:
    """Fuse concept + text + interactive masks into one first-frame seed.

    ``recover_boxes`` (5.5) names boxes the re-entry scan has already confirmed
    as a person the matte is not holding.  For those, and only those, a seeder
    that supports it retries the mask match at a lower score once the strict
    pass has come back empty.

    Without it, a heavily motion-blurred subject is detected, flagged, handed
    to a local re-matte pass -- and then dropped right here, because
    ``masks_from_boxes`` returned nothing for his box.  The pass then re-mattes
    the cast it already had, which is why ten local passes on the full-length
    1917 changed the matte by nothing at all.
    """
    from .config import SeedConfig
    cfg = cfg or SeedConfig()
    H, W = frame_bgr.shape[:2]
    notes: List[str] = []

    boxes = [d.box for d in kept]
    kw = ({"recover_boxes": recover_boxes}
          if recover_boxes and getattr(seeder, "supports_recover", False)
          else {})
    concept = seeder.masks_from_boxes(frame_bgr, boxes, **kw)
    notes += [n for n in getattr(seeder, "notes", ()) if "recovered box" in n]
    if getattr(cfg, "extend_tail_max", 0.0) > 0.0:
        concept, n_ext = extend_short_tails(
            concept, seeder, boxes, cfg.extend_tail_max)
        if n_ext:
            notes.append(f"extended {n_ext} short mask tail(s) to the box")
    base = _union(concept, (H, W))

    # Text-prompted exhaustive detection, matched back to the kept boxes.
    if cfg.concept_text_detect:
        try:
            for mk, mb, _s in seeder.masks_from_text(frame_bgr, cfg.concept_text):
                if any(box_iou(mb, b) >= cfg.concept_match_iou for b in boxes):
                    base |= (mk > 0).astype(np.uint8)
        except Exception as e:                                # noqa: BLE001
            notes.append(f"text prompt unavailable: {type(e).__name__}")

    # Interactive head: always fires, then its additions are appearance-gated.
    if cfg.force_interactive:
        inter = _union(seeder.masks_from_boxes(frame_bgr, boxes, **kw), (H, W))
        if cfg.gate_background_regions:
            keep_extra, n_drop = drop_background_regions(
                frame_bgr, base, inter, cfg.gate_bg_margin)
            base |= keep_extra
            if n_drop:
                notes.append(f"gated out {n_drop} interactive region(s)")
        else:
            base |= inter

    # Held props present in the first frame.
    if props:
        base |= _union(seeder.masks_from_boxes(frame_bgr, [p.box for p in props]),
                       (H, W))

    original = base.copy()
    if cfg.fill_mask_holes:
        base = fill_holes(base)
        if cfg.gap_fill:
            base = reopen_background_gaps(
                frame_bgr, base, original, cfg.gap_bg_margin,
                cfg.gap_min_area, cfg.gap_max_frac)

    per_person = [(m > 0).astype(np.uint8) for m in concept]
    return SeedResult(mask=base.astype(np.uint8), kept=list(kept),
                      per_person=per_person, notes=notes)


def split_by_person(fused: np.ndarray,
                    per_person: Sequence[np.ndarray],
                    min_frac: float = 0.002) -> List[np.ndarray]:
    """Partition a fused seed into one mask per person, losing no pixels.

    Why this is not ``SAM2MattingEngine.split_objects``
    --------------------------------------------------
    That function splits the union by *connected component*, and its docstring
    argues two people who touch can share an object because "the tracker sees
    one blob anyway".  That is exactly the claim 3.2a disproved: given one
    identity holding two blobs, the tracker converged onto one ipman fighter at
    frame 7 and never recovered the other.  And touching is not rare -- 3.2s
    found ``codylexi`` is two subjects and **one** connected component for the
    entire window, with shakira, dance and tryguys close behind.

    Why not just pass ``SeedResult.per_person``
    -------------------------------------------
    Because it is only the concept masks.  ``build_seed`` then ORs in the text
    prompt, the interactive head, held props and hole filling, so
    ``union(per_person)`` is a strict subset of ``SeedResult.mask``.  Handing
    the engine ``per_person`` directly would silently drop every one of those
    additions.

    So this assigns each pixel of the fused mask to the nearest person seed --
    a nearest-label partition, not a re-segmentation.  The union of the result
    is the fused mask exactly, and each person keeps their own ``obj_id``
    whether or not they touch a neighbour.

    Returns ``[]`` when there is nothing to split (no seeds, or one), which the
    caller should read as "use the fused mask as-is".
    """
    f = (np.asarray(fused) > (127 if np.max(fused) > 1.5 else 0.5))
    people = [np.asarray(m) > 0 for m in per_person]
    people = [m for m in people if m.any()]
    if len(people) < 2 or not f.any():
        return []

    # distance from every pixel to each person's own seed
    dists = np.stack([
        cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
        for m in people])
    owner = np.argmin(dists, axis=0)

    out, thr = [], min_frac * f.size
    for i in range(len(people)):
        m = f & (owner == i)
        if m.sum() >= thr:
            out.append(m.astype(np.uint8) * 255)
    # A partition that collapsed to one object is not a split; say so by
    # returning nothing rather than handing back a relabelled union.
    return out if len(out) >= 2 else []
