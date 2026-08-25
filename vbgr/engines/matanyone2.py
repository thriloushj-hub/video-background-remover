"""MatAnyone 2 adapter (streaming).

MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator
  CVPR 2026 Highlight, S-Lab NTU + SenseTime
  repo    https://github.com/pq-yang/MatAnyone2
  paper   https://arxiv.org/abs/2512.11782
  licence NTU S-Lab License 1.0 -- NON-COMMERCIAL

Written against the real upstream API (``inference_matanyone2.py``):

    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2.utils.get_default_model import get_matanyone2_model

    processor = InferenceCore(model, cfg=model.cfg)
    processor.step(image, mask, objects=[1])        # encode the seed
    processor.step(image, first_frame_pred=True)    # warmup frames
    processor.step(image)                           # normal frames
    alpha = processor.output_prob_to_mask(prob)

Two things the upstream script does that are easy to miss and that we
replicate exactly:

* ``image`` is **RGB, CHW, float in [0, 1]** -- not BGR HWC uint8.
* Warmup works by *prepending* n copies of the first frame and running them
  with ``first_frame_pred=True``, then discarding those outputs.  Getting the
  discard wrong is precisely how you end up with an output that is n frames
  short of the input.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .base import (EngineInfo, MattingEngine, filter_kwargs, kwarg_support,
                   register, require_kwarg, resolve_device, supports_kwarg,
                   to_alpha_2d)

INFO = EngineInfo(
    name="MatAnyone 2",
    license="NTU S-Lab License 1.0",
    commercial_ok=False,
    mode="streaming",
    # No memory_gate. It was declared here from the paper until the 3.2b
    # capability audit checked the object: InferenceCore.step() takes no
    # **kwargs, so update_memory cannot be passed at all. Task 3.4 then
    # dropped the feature outright.
    capabilities={"mask_prompt", "permanent_memory"},
    url="https://github.com/pq-yang/MatAnyone2",
    notes="Best-in-class human hair/fabric edges. Human-specific. Needs a "
          "first-frame mask, so SAM 3 feeds it.",
)

_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"


@register("matanyone2")
class MatAnyone2Engine(MattingEngine):
    INFO = INFO
    info = INFO

    def __init__(self, checkpoint: Optional[str] = None,
                 device: str = "auto", half: bool = True,
                 max_permanent_anchors: int = 24,
                 strict_capabilities: bool = False):
        self.device = resolve_device(device)
        self.half = half and self.device == "cuda"
        self.checkpoint = checkpoint
        self.max_permanent_anchors = max_permanent_anchors
        self.strict_capabilities = strict_capabilities
        self._model = None
        self._proc = None
        self._n_permanent = 0
        self._degraded: dict = {}

    # -- model -------------------------------------------------------------- #

    def _ensure(self):
        if self._proc is not None:
            return
        import torch
        from matanyone2.inference.inference_core import InferenceCore
        from matanyone2.utils.get_default_model import get_matanyone2_model
        from matanyone2.utils.download_util import load_file_from_url

        self._torch = torch
        ckpt = self.checkpoint or load_file_from_url(_URL, "pretrained_models")
        self._model = get_matanyone2_model(ckpt, self.device)
        self._proc = InferenceCore(self._model, cfg=self._model.cfg)

        # Check the declared capabilities against the API we actually got, ONCE,
        # at load time. Doing this here rather than on first use means a broken
        # capability surfaces before a three-hour batch, not during it.
        self._supports_permanent = require_kwarg(
            self._proc.step, "force_permanent", self,
            strict=self.strict_capabilities, where="load")
        # update_memory is deliberately no longer probed. It was always
        # absent -- InferenceCore.step() takes no **kwargs -- and task 3.4
        # dropped the feature that used it, so checking would only print a
        # verdict on something nothing calls.
        print(f"[MatAnyone 2] capability check: "
              f"permanent_memory="
              f"{'ok' if self._supports_permanent else 'UNAVAILABLE'}")

    def _img(self, frame_bgr: np.ndarray):
        """BGR HWC uint8 -> RGB CHW float tensor in [0, 1] on device."""
        t = self._torch
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1]).astype(np.float32) / 255.0
        return t.from_numpy(rgb).permute(2, 0, 1).to(self.device)

    # -- streaming API ------------------------------------------------------ #

    def start(self, frame_bgr: np.ndarray, seed_mask: np.ndarray,
              n_warmup: int = 10) -> np.ndarray:
        self._ensure()
        t = self._torch
        self.reset()

        img = self._img(frame_bgr)
        m = t.from_numpy(np.asarray(seed_mask, np.float32)).to(self.device)
        if m.max() > 1.5:
            m = m / 255.0

        prob = self._proc.step(img, m, objects=[1])
        prob = self._proc.step(img, first_frame_pred=True)

        # Static warmup: settle the memory on the first frame before any real
        # output frame depends on it.
        for _ in range(max(0, n_warmup - 1)):
            prob = self._proc.step(img, first_frame_pred=True)

        return to_alpha_2d(self._proc.output_prob_to_mask(prob))

    def step(self, frame_bgr: np.ndarray, *,
             commit_to_memory: bool = True,
             permanent: bool = False,
             reseed_mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._ensure()
        t = self._torch
        img = self._img(frame_bgr)

        if reseed_mask is not None:
            m = t.from_numpy(np.asarray(reseed_mask, np.float32)).to(self.device)
            if m.max() > 1.5:
                m = m / 255.0
            prob = self._proc.step(img, m, objects=[1])
            return to_alpha_2d(self._proc.output_prob_to_mask(prob))

        kw = {}
        if permanent and self._supports_permanent and \
                self._n_permanent < self.max_permanent_anchors:
            kw["force_permanent"] = True
            self._n_permanent += 1
        if not commit_to_memory:
            # Cutie-derived cores expose this as "do not write to the memory
            # bank". If the installed version does not, filter_kwargs drops it
            # LOUDLY and records the engine as degraded, so the pipeline stops
            # reporting the memory gate as active and falls back to re-seeding.
            kw["update_memory"] = False

        prob = self._proc.step(
            img, **filter_kwargs(self._proc.step, kw, engine=self, where="step"))
        return to_alpha_2d(self._proc.output_prob_to_mask(prob))

    def reset(self) -> None:
        self._n_permanent = 0
        if self._proc is not None:
            for name in ("clear_memory", "reset", "clear_non_permanent_memory"):
                fn = getattr(self._proc, name, None)
                if callable(fn):
                    fn()
                    return
            # Last resort: rebuild the core.  Cheap; the model stays loaded.
            from matanyone2.inference.inference_core import InferenceCore
            self._proc = InferenceCore(self._model, cfg=self._model.cfg)

    # -- sequence API (built on the streaming one) -------------------------- #

    def matte(self, frames: Sequence[np.ndarray],
              seed_mask: Optional[np.ndarray],
              n_warmup: int = 10,
              progress: Optional[Callable[[int], None]] = None) -> np.ndarray:
        if seed_mask is None:
            raise ValueError("MatAnyone 2 requires a first-frame mask")
        self.start(frames[0], seed_mask, n_warmup=n_warmup)

        out = np.empty((len(frames), *frames[0].shape[:2]), np.float32)
        # Motion warmup: run the first few real frames, discard their output,
        # and bank them as permanent anchors (the drift fix).
        for i in range(min(n_warmup, len(frames))):
            self.step(frames[i], permanent=True)

        for i, f in enumerate(frames):
            out[i] = self.step(f)
            if progress:
                progress(i)

        assert len(out) == len(frames)      # frame parity is non-negotiable
        return out
