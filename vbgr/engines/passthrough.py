"""Dependency-free engines.

``passthrough`` propagates the seed mask by optical flow only.  It needs no
torch, no checkpoints and no GPU, which makes it the engine used by the test
suite and by ``vbgr2 selftest``: the whole pipeline -- shot splitting, memory
gating, flow-adaptive trimap, refinement, decontamination, compositing,
frame-parity assertions -- can be exercised end to end on CPU without a single
model download.

It is not a matting engine in any useful sense and will not produce a usable
matte.  It exists so that a broken pipeline fails in CI rather than three hours
into a GPU batch.

``existing_alpha`` replays a pre-computed alpha sequence.  This is how v1
outputs get scored by the benchmark harness on equal footing with new runs.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .base import EngineInfo, MattingEngine, register
from .. import motion as _motion


@register("passthrough")
class PassthroughEngine(MattingEngine):
    INFO = EngineInfo(
        name="passthrough (flow propagation)",
        license="MIT (this repo)",
        commercial_ok=True,
        mode="streaming",
        # memory_gate removed with task 3.4. This engine's step() still
        # accepts commit_to_memory so the signature-check tests have something
        # honest to bind to, but nothing in the pipeline passes it any more.
        capabilities={"mask_prompt"},
        notes="CPU-only smoke-test engine. Not a real matter.",
    )
    info = INFO

    def __init__(self, blur: float = 1.5, **_):
        self.blur = blur
        self._flow = _motion.FlowEstimator()
        self._prev_bgr: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def start(self, frame_bgr, seed_mask, n_warmup: int = 0):
        a = np.asarray(seed_mask, np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        self._alpha = cv2.GaussianBlur(np.clip(a, 0, 1), (0, 0), self.blur)
        self._prev_bgr = frame_bgr.copy()
        return self._alpha

    def step(self, frame_bgr, *, commit_to_memory=True, permanent=False,
             reseed_mask=None):
        if reseed_mask is not None:
            return self.start(frame_bgr, reseed_mask)
        if self._alpha is None:
            raise RuntimeError("start() first")
        fl = self._flow.flow(self._prev_bgr, frame_bgr)
        a = np.clip(_motion.warp(self._alpha, fl), 0, 1)
        if commit_to_memory:
            self._alpha, self._prev_bgr = a, frame_bgr.copy()
        return a

    def reset(self):
        self._prev_bgr = self._alpha = None

    def matte(self, frames, seed_mask, n_warmup=10, progress=None):
        self.start(frames[0], seed_mask)
        out = np.empty((len(frames), *frames[0].shape[:2]), np.float32)
        for i, f in enumerate(frames):
            out[i] = self._alpha if i == 0 else self.step(f)
            if progress:
                progress(i)
        assert len(out) == len(frames)
        return out


@register("existing_alpha")
class ExistingAlphaEngine(MattingEngine):
    INFO = EngineInfo(
        name="existing alpha (replay)",
        license="n/a",
        commercial_ok=True,
        mode="sequence",
        capabilities=set(),
        notes="Replays a pre-computed alpha sequence so previous runs can be "
              "scored by the same harness.",
    )
    info = INFO

    def __init__(self, alphas: Optional[np.ndarray] = None, **_):
        self.alphas = alphas

    def matte(self, frames, seed_mask=None, n_warmup=10, progress=None):
        if self.alphas is None:
            raise ValueError("ExistingAlphaEngine needs alphas=")
        a = np.asarray(self.alphas, np.float32)
        if len(a) != len(frames):
            raise ValueError(f"alpha has {len(a)} frames, video has {len(frames)}")
        return np.clip(a, 0, 1)

    def reset(self):
        pass
