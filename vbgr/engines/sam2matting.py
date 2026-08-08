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
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .base import EngineInfo, MattingEngine, register, resolve_device, to_alpha_2d

INFO = EngineInfo(
    name="SAM2Matting",
    license="CC BY-NC-SA 4.0",
    commercial_ok=False,
    mode="sequence",
    capabilities={"mask_prompt", "point_prompt", "box_prompt", "text_prompt",
                  "per_frame_head"},
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
                 repo_dir: Optional[str] = None):
        self.device = resolve_device(device)
        self.half = half and self.device == "cuda"
        self.backbone = backbone.lower()
        self.checkpoint = checkpoint or _CKPTS[self.backbone]
        self.compiled = compiled
        self.repo_dir = repo_dir or os.environ.get(
            "SAM2MATTING_DIR", "third_party/SAM2Matting")
        self._predictor = None
        self._tmp: Optional[str] = None

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

    def matte(self, frames: Sequence[np.ndarray],
              seed_mask: Optional[np.ndarray],
              n_warmup: int = 10,
              progress: Optional[Callable[[int], None]] = None) -> np.ndarray:
        self._ensure()
        t = self._torch
        if seed_mask is None:
            raise ValueError("SAM2Matting needs a prompt (mask/box/point/text)")

        d = self._materialise(frames)
        h, w = frames[0].shape[:2]
        out = np.zeros((len(frames), h, w), np.float32)

        try:
            state = self._predictor.init_state(video_path=d)
            self._predictor.reset_state(state)
            self._predictor.add_new_mask(
                inference_state=state, frame_idx=0, obj_id=1,
                mask=self._seed_logits(seed_mask))

            autocast = t.autocast("cuda", dtype=t.bfloat16) if self.device == "cuda" \
                else _NullCtx()
            seen = set()
            with t.inference_mode(), autocast:
                for idx, _objs, _lg, alpha, *_ in self._predictor.propagate_in_video(state):
                    a = to_alpha_2d(alpha)
                    if a.shape != (h, w):
                        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                    out[idx] = a
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

        assert len(out) == len(frames)
        return out

    def matte_frame(self, frame_bgr: np.ndarray,
                    trimap: np.ndarray) -> Optional[np.ndarray]:
        """Progressive matting head on a single frame + trimap.

        This is the piece that makes flow-adaptive trimap widening pay off: the
        widened unknown band is handed to a real matting head rather than to a
        blur.
        """
        self._ensure()
        head = getattr(self._predictor, "matting_head", None) or \
            getattr(self._predictor, "predict_alpha", None)
        if head is None:
            return None
        t = self._torch
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1]).astype(np.float32) / 255.0
        with t.inference_mode():
            a = head(t.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device),
                     t.from_numpy(trimap.astype(np.float32) / 255.0)[None, None].to(self.device))
        a = to_alpha_2d(a)
        if a.shape != frame_bgr.shape[:2]:
            a = cv2.resize(a, frame_bgr.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        return a

    def reset(self) -> None:
        self._cleanup()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
