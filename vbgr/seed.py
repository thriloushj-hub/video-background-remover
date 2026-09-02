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

    def __init__(self, detector=None, score_min: float = 0.80,
                 match_iou: float = 0.30, device: str = "auto"):
        self._det = detector
        self.score_min = score_min
        self.match_iou = match_iou
        self._device = device
        self.backend = "maskrcnn"
        self.notes: List[str] = []

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

    def _instances(self, frame_bgr: np.ndarray):
        """(masks, boxes) for people above ``score_min``, at frame resolution.

        A detector exposing ``instances(frame_bgr) -> (masks, boxes)`` is used
        directly.  That hook exists so the matching logic below -- which is
        where the real risk is -- can be tested without torch installed.
        """
        if callable(getattr(self._det, "instances", None)):
            return self._det.instances(frame_bgr)
        self._ensure()
        import torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        if getattr(self, "device", "cpu") == "cuda":
            t = t.cuda()
        with torch.no_grad():
            o = self._det([t])[0]
        sel = (o["labels"] == 1) & (o["scores"] > self.score_min)
        masks = o["masks"][sel, 0].detach().cpu().numpy() > 0.5
        boxes = o["boxes"][sel].detach().cpu().numpy()
        return masks, boxes

    def masks_from_boxes(self, frame_bgr: np.ndarray,
                         boxes: Sequence[Box]) -> List[np.ndarray]:
        if not boxes:
            return []
        self.notes = []
        inst_masks, inst_boxes = self._instances(frame_bgr)
        out = []
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
                self.notes.append(
                    f"no mask instance matched box {tuple(int(x) for x in b)} "
                    f"(best IoU {best_iou:.2f} < {self.match_iou}); subject skipped")
                continue
            used.add(best)
            out.append(inst_masks[best].astype(np.uint8))
        return out

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
        return MaskRCNNSeeder(score_min=getattr(cfg, "maskrcnn_score_min", 0.80),
                              match_iou=getattr(cfg, "maskrcnn_match_iou", 0.30))
    if backend == "sam3":
        return SAM3Seeder(mask_threshold=cfg.mask_threshold,
                          detect_threshold=cfg.detect_threshold,
                          interactive_mask_threshold=cfg.interactive_mask_threshold)
    raise ValueError(f"unknown seeder backend {backend!r}")


# --------------------------------------------------------------------------- #
# The full seed
# --------------------------------------------------------------------------- #

def build_seed(frame_bgr: np.ndarray,
               seeder: SAM3Seeder,
               kept: Sequence[Detection],
               props: Sequence[Detection] = (),
               cfg=None) -> SeedResult:
    """Fuse concept + text + interactive masks into one first-frame seed."""
    from .config import SeedConfig
    cfg = cfg or SeedConfig()
    H, W = frame_bgr.shape[:2]
    notes: List[str] = []

    boxes = [d.box for d in kept]
    concept = seeder.masks_from_boxes(frame_bgr, boxes)
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
        inter = _union(seeder.masks_from_boxes(frame_bgr, boxes), (H, W))
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
