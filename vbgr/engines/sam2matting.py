"""SAM2Matting adapter (sequence-mode).

SAM2Matting: Generalized Image and Video Matting
  ECCV 2026, Fudan CVL
  repo    https://github.com/FudanCVL/SAM2Matting
  paper   https://arxiv.org/abs/2606.27339
  weights https://huggingface.co/FudanCVL/SAM2Matting
  licence CC BY-NC-SA 4.0 -- NON-COMMERCIAL

Why it is the primary candidate
-------------------------------
It is the explicit decoupling of the two halves the previous attempts each got
right separately: a VOS tracker (SAM 2.1 or SAM 3) supplies temporal identity,
and a separate ROI-detection + progressive-matting head supplies the soft
boundary.  The authors call out rapid-motion and open-world targets
specifically -- the ipman / butter / 1917 failure profile.

Written against the real upstream API (``inference_video_sam3.py``):

    from sam3.model.sam3matting_video_predictor import build_sam3matting_video_predictor
    predictor = build_sam3matting_video_predictor(checkpoint=None, device=device)
    predictor.load_state_dict(load_tracker_state_dict(ckpt), strict=False)
    state = predictor.init_state(video_path=frame_dir)
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=logits_288)
    for idx, obj_ids, _, alpha, _ in predictor.propagate_in_video(state): ...

Two upstream details that are easy to get wrong:

* ``init_state`` wants a **directory of JPEG/PNG frames**, not an array.  We
  materialise one in a temp dir; it is deleted on ``reset``.
* The seed mask must be handed over as **logits at 288x288**, built as
  ``(mask > 0.005) * 20 - 10``.  Passing a 0/1 float mask at native resolution
  silently produces a near-empty matte.

Consequence for the pipeline: ``propagate_in_video`` is a generator over the
whole clip with no supported interruption point, so this engine declares no
``memory_gate`` capability.  The pipeline compensates by chunking the timeline
and re-seeding at chunk boundaries when ReID reports a lost track.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .base import EngineInfo, MattingEngine, register, resolve_device, to_alpha_2d

INFO = EngineInfo(
    name="SAM2Matting",
    license="CC BY-NC-SA 4.0",
    commercial_ok=False,
    mode="sequence",
    # NOTE 2026-08-15: "per_frame_head" was here and it was WRONG -- written
    # from the paper, not from the object.  The real SAM3MattingVideoPredictor
    # exposes no standalone (image, trimap) -> alpha callable; see matte_frame
    # below.  Declaring it made the pipeline route the flow-adaptive trimap
    # into refine.refine_band with a head that returns None, which crashed.
    # NOTE 2026-08-16: "text_prompt" was here and it was WRONG too.  SAM 3 the
    # model family does concept/text prompting, but the class this adapter
    # actually drives -- SAM3MattingVideoPredictor -- has no text entry point.
    # Its only prompt methods are add_new_mask, add_new_points and
    # add_new_points_or_box; the sole attribute matching /text|concept|phrase/
    # is `bf16_context`, which matches on "context".  Third paper-vs-object
    # mismatch on this project.
    capabilities={"mask_prompt", "point_prompt", "box_prompt",
                  "multi_object"},
    url="https://github.com/FudanCVL/SAM2Matting",
    notes="Tracker-to-matting decoupling. SAM3 variant handles rapid motion "
          "and non-human targets. No per-frame memory control.",
)

_CKPTS = {
    "sam3": "checkpoints/SAM2Matting-SAM3.pt",
    "sam2.1b+": "checkpoints/SAM2Matting-SAM2.1Base+.pt",
    "sam2.1t": "checkpoints/SAM2Matting-SAM2.1Tiny.pt",
}


@register("sam2matting")
class SAM2MattingEngine(MattingEngine):
    INFO = INFO
    info = INFO

    def __init__(self, checkpoint: Optional[str] = None,
                 backbone: str = "sam3",
                 device: str = "auto",
                 half: bool = True,
                 compiled: bool = False,
                 repo_dir: Optional[str] = None,
                 split_seed_components: bool = True):
        self.device = resolve_device(device)
        self.half = half and self.device == "cuda"
        self.backbone = backbone.lower()
        self.checkpoint = checkpoint or _CKPTS[self.backbone]
        self.compiled = compiled
        self.split_seed_components = split_seed_components
        self.repo_dir = repo_dir or os.environ.get(
            "SAM2MATTING_DIR", "third_party/SAM2Matting")
        self._predictor = None
        self._tmp: Optional[str] = None
        # (T, K) fraction-of-frame per seeded subject, filled by matte()
        self.object_areas: Optional[np.ndarray] = None
        self._obj_area_frame: list = []

    # -- model -------------------------------------------------------------- #

    def _ensure(self):
        if self._predictor is not None:
            return
        import sys
        import torch
        from iopath.common.file_io import g_pathmgr

        if self.repo_dir and os.path.isdir(self.repo_dir) and self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        self._torch = torch

        if self.backbone.startswith("sam3"):
            from sam3.model.sam3matting_video_predictor import (
                build_sam3matting_video_predictor as build)
        else:
            from sam2.model.sam2matting_video_predictor import (
                build_sam2matting_video_predictor as build)

        # Upstream splits the checkpoint: only detector.backbone.vision_backbone.*
        # and tracker.* belong to the video predictor.
        with g_pathmgr.open(self.checkpoint, "rb") as fh:
            sd = torch.load(fh, map_location="cpu", weights_only=True)["model"]
        keep = {}
        for k, v in sd.items():
            if k.startswith("detector.backbone.vision_backbone."):
                keep[k.removeprefix("detector.")] = v
            elif k.startswith("tracker."):
                keep[k.removeprefix("tracker.")] = v

        p = build(checkpoint=None, device=self.device)
        missing, unexpected = p.load_state_dict(keep, strict=False)
        if missing:
            print(f"[sam2matting] {len(missing)} missing keys (expected for "
                  f"the detector half)")
        if self.compiled:
            trunk = p.backbone.vision_backbone.trunk
            trunk.forward = torch.compile(trunk.forward, mode="max-autotune",
                                          fullgraph=True, dynamic=False)
        self._predictor = p

    # -- seed encoding ------------------------------------------------------ #

    def _seed_logits(self, mask: np.ndarray):
        """0/1 mask -> 288x288 logit tensor, exactly as upstream does it."""
        t = self._torch
        m = np.asarray(mask, np.float32)
        if m.max() > 1.5:
            m = m / 255.0
        m = (m > 0.005).astype(np.float32) * 20.0 - 10.0
        m = t.from_numpy(m)[None, None]
        m = t.nn.functional.interpolate(m, size=(288, 288), mode="bilinear",
                                        align_corners=False)
        return m.to(self.device)

    # -- frame directory ---------------------------------------------------- #

    def _materialise(self, frames: Sequence[np.ndarray]) -> str:
        self._cleanup()
        self._tmp = tempfile.mkdtemp(prefix="vbgr2_sam2matting_")
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(self._tmp, f"{i:06d}.jpg"), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 98])
        return self._tmp

    def _cleanup(self):
        if self._tmp and os.path.isdir(self._tmp):
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = None

    # -- API ---------------------------------------------------------------- #

    # -- multi-object seeding ----------------------------------------------- #

    @staticmethod
    def split_objects(seed_mask: np.ndarray,
                      min_frac: float = 0.002) -> list:
        """Split a union seed into one mask per significant connected component.

        The seed stage builds its mask by OR-ing one mask per kept person, so a
        component in that union *is* a person (two people who physically touch
        merge into one component, but then the tracker sees one blob anyway and
        nothing is lost by treating it as one object).
        """
        hard = (np.asarray(seed_mask) > (127 if seed_mask.max() > 1.5 else 0.5))
        n, lab, stats, _c = cv2.connectedComponentsWithStats(
            hard.astype(np.uint8), 8)
        thr = min_frac * hard.size
        return [(lab == i).astype(np.uint8) * 255
                for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= thr]

    def _alpha_union(self, alpha, n_obj: int, h: int, w: int) -> np.ndarray:
        """Collapse a per-object alpha stack into one HxW matte.

        Side effect: the per-object coverage of this frame is recorded in
        ``self._obj_area_frame`` as one fraction-of-frame per object, BEFORE
        the union throws that information away.  ``matte`` collects those rows
        into ``self.object_areas``.

        This exists because ``subj_lost_at`` used to be derived from the number
        of connected components in the unioned matte, which is not the number
        of subjects: two people standing shoulder to shoulder are one
        component, so the metric reported a lost subject on every clip where
        anybody touched.  Per-object area is the honest signal -- object k is
        still tracked if object k still has area, whether or not it is fused
        to its neighbour in the union.

        ``to_alpha_2d`` cannot be used here: it walks leading dimensions with
        ``a = a[0]``, which silently keeps only the FIRST object and discards
        every other person.  Unknown layouts raise rather than guess -- a wrong
        union would look exactly like a tracking failure in the metrics.
        """
        t = self._torch
        x = alpha
        if isinstance(x, t.Tensor):
            x = x.detach().float().cpu().numpy()
        if isinstance(x, (tuple, list)):
            x = x[-1]
            if isinstance(x, t.Tensor):
                x = x.detach().float().cpu().numpy()
        x = np.squeeze(np.asarray(x, np.float32))

        if x.ndim == 2:
            a = x
            stack = x[None]
        elif x.ndim == 3 and x.shape[0] == n_obj:
            a = x.max(axis=0)
            stack = x
        elif x.ndim == 3 and x.shape[-1] == n_obj:
            a = x.max(axis=-1)
            stack = np.moveaxis(x, -1, 0)
        else:
            raise RuntimeError(
                f"[sam2matting] cannot union {n_obj} objects from an alpha of "
                f"shape {x.shape}. Refusing to guess -- a wrong union is "
                f"indistinguishable from a tracking failure in the metrics.")
        scale = 255.0 if a.max() > 1.5 else 1.0
        a = np.clip(a / scale, 0.0, 1.0)

        # per-object coverage, measured on the stack's own grid (the ratio is
        # resolution independent, so no resize is needed just to count area)
        self._obj_area_frame = [
            float((np.clip(o / scale, 0.0, 1.0) > 0.5).mean()) for o in stack]

        if a.shape != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        return a

    # -- API ---------------------------------------------------------------- #

    def matte(self, frames: Sequence[np.ndarray],
              seed_mask: Optional[np.ndarray],
              n_warmup: int = 10,
              progress: Optional[Callable[[int], None]] = None,
              seed_masks: Optional[Sequence[np.ndarray]] = None,
              extra_seeds: Optional[Sequence[Tuple[int, np.ndarray]]] = None
              ) -> np.ndarray:
        """Matte a shot.

        ``seed_masks`` -- one 0/255 mask per subject -- is the honest way to
        prompt this model when the frame holds more than one person.  Until
        2026-08-15 this method OR-ed every kept person into a single
        ``obj_id=1``, and on the ipman fast-motion window the tracker converged
        onto one of the two fighters at frame 7 and never recovered the other.
        A VOS tracker given one identity with two separate blobs will do that.

        If ``seed_masks`` is not given, the union in ``seed_mask`` is split into
        connected components (see ``split_objects``) so existing call sites get
        the fix too.  The split is always announced, never silent.

        ``extra_seeds`` -- ``(frame_idx, mask)`` pairs -- prompts additional
        objects part-way through the shot, for people who walk in after frame
        0.  Each gets its own ``obj_id`` and contributes nothing before its
        own start frame, which is the correct reading: they were not there.
        SAM2 accepts prompts on any frame before propagation starts, so this
        needs no second pass.

        Off unless the caller asks: every run before 2026-08-29 seeded only at
        frame 0, and the frozen benchmark arm still does, so the numbers stay
        comparable.  See ``vbgr_bench.py --reseed``.

        **THIS DOES NOT WORK ON THE CHOSEN MODEL.**  Run on an A100 on
        29 Aug it raises::

            TypeError: Sam3TrackerBase.track_step() got an unexpected
            keyword argument 'gt_masks'

        SAM2's documented API accepts ``add_new_mask`` on any frame before
        propagation; the SAM3 tracker underneath this predictor does not --
        it takes the prompt but its ``track_step`` signature has no slot for
        the mask on a non-initial frame.  Read from the upstream signature it
        looked fine, which is the fourth capability on this project that was
        declared from the paper rather than checked against the object (see
        Capability_Audit).

        The path that is likely to work is a **second propagation pass**
        seeded at the entrant's frame, unioned with the first -- which is a
        rework, not a keyword fix.  Until then the caller catches the error
        and keeps the off arm.
        """
        self._ensure()
        t = self._torch
        if seed_mask is None and not seed_masks:
            raise ValueError("SAM2Matting needs a prompt (mask/box/point/text)")

        if seed_masks:
            objs = list(seed_masks)
            src = "explicit"
        elif self.split_seed_components:
            objs = self.split_objects(seed_mask) or [seed_mask]
            src = "auto-split"
        else:
            objs = [seed_mask]
            src = "union"
        extra = list(extra_seeds or [])
        print(f"[sam2matting] seeding {len(objs)} object(s) ({src})"
              + (f" + {len(extra)} mid-shot at frames "
                 f"{sorted({int(f) for f, _ in extra})}" if extra else ""))

        d = self._materialise(frames)
        h, w = frames[0].shape[:2]
        out = np.zeros((len(frames), h, w), np.float32)

        # object_areas[t][k] = fraction of frame covered by seeded subject k at
        # frame t.  Filled by _alpha_union.  NaN marks a frame the predictor
        # never returned (see the missing-frame handling below), so a hold is
        # never mistaken for a measurement.
        n_obj = len(objs) + len(extra)
        obj_areas = np.full((len(frames), n_obj), np.nan, np.float32)
        self._obj_area_frame = []

        try:
            state = self._predictor.init_state(video_path=d)
            self._predictor.reset_state(state)
            for k, m in enumerate(objs):
                self._predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=k + 1,
                    mask=self._seed_logits(m))
            for j, (fidx, m) in enumerate(extra):
                self._predictor.add_new_mask(
                    inference_state=state, frame_idx=int(fidx),
                    obj_id=len(objs) + j + 1, mask=self._seed_logits(m))

            autocast = t.autocast("cuda", dtype=t.bfloat16) if self.device == "cuda" \
                else _NullCtx()
            seen = set()
            with t.inference_mode(), autocast:
                for idx, _objs, _lg, alpha, *_ in self._predictor.propagate_in_video(state):
                    a = self._alpha_union(alpha, n_obj, h, w)
                    out[idx] = a
                    row = self._obj_area_frame
                    if len(row) == len(objs):
                        obj_areas[int(idx)] = row
                    seen.add(int(idx))
                    if progress:
                        progress(int(idx))

            missing = sorted(set(range(len(frames))) - seen)
            if missing:
                # Never silently return a short/zeroed result -- that is exactly
                # the class of bug that produced a 325-frame output from a
                # 501-frame input in v1.  Hold the last good alpha instead and
                # say so loudly.
                print(f"[sam2matting] WARNING: predictor skipped "
                      f"{len(missing)} frame(s); holding last alpha")
                last = np.zeros((h, w), np.float32)
                for i in range(len(frames)):
                    if i in seen:
                        last = out[i]
                    else:
                        out[i] = last
        finally:
            self._cleanup()

        self.object_areas = obj_areas
        assert len(out) == len(frames)
        return out

    def matte_frame(self, frame_bgr: np.ndarray,
                    trimap: np.ndarray) -> Optional[np.ndarray]:
        """Always ``None`` for SAM2Matting.  Upstream has no such head.

        This method used to look for ``matting_head`` / ``predict_alpha`` on the
        predictor.  Introspection of the real object on an A100, 2026-08-15::

            <class 'sam3.model.sam3matting_video_predictor.SAM3MattingVideoPredictor'>
            matting_head: False | predict_alpha: False
            attrs: ['alpha_pred1', 'alpha_pred2', 'alpha_pred3',
                    'matting_step', 'sam_mask_decoder', ...]

        Neither candidate is usable standalone.  ``matting_step`` takes ten
        arguments -- ``frame_idx, input, is_init_cond_frame, mask_inputs,
        current_vision_feats, current_vision_pos_embeds, feat_sizes,
        output_dict, matting_output_dict, num_frames`` -- and is wired into the
        predictor's per-frame memory state, so it only runs inside
        ``propagate_in_video``.  ``alpha_pred1..3`` are ``nn.Sequential``
        decoders over internal feature maps, not over an image plus a trimap.

        Consequence for [[Motion_Blur_Fix]]: the flow-adaptive trimap cannot
        hand its widened band to a real matting head on this engine.  It falls
        back to the guided filter in ``refine.refine_band``, which is weaker,
        and any write-up must say so.  Bringing in a standalone head such as
        ViTMatte is tracked separately -- it is a new dependency and a new
        licence question on top of CC BY-NC-SA.
        """
        return None

    def reset(self) -> None:
        self._cleanup()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
