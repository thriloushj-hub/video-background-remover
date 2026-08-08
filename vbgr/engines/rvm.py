"""Robust Video Matting adapter -- the baseline, and the commercial fallback.

RVM (WACV 2022, Peter Lin)
  repo    https://github.com/PeterL1n/RobustVideoMatting
  licence GPL-3.0  -- commercial use IS permitted under GPL terms (copyleft),
          unlike SAM2Matting (CC BY-NC-SA) and MatAnyone 2 (NTU S-Lab).

Two roles here:

1.  **Regression baseline.**  This is what the first intern used, so it anchors
    the benchmark table.  "Better than the old thing" has to be measurable.
2.  **The only currently-wired engine that can ship.**  If the licence question
    lands on "commercial", this is the starting point -- and the honest read is
    that RVM alone will not hit the quality bar (weak edges, no control over
    which person is kept, unstable on cluttered footage).  See docs/LICENSING.md
    for what a commercial path actually looks like.

RVM is auto-matting: it takes no prompt.  It therefore ignores ``seed_mask``,
which means it cannot select a subject.  The pipeline still applies the seed as
a *post-hoc* selection mask so the comparison against the prompted engines is
apples-to-apples on subject choice.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .base import EngineInfo, MattingEngine, register, resolve_device, to_alpha_2d

INFO = EngineInfo(
    name="RobustVideoMatting",
    license="GPL-3.0",
    commercial_ok=True,
    mode="streaming",
    capabilities={"auto_human"},
    url="https://github.com/PeterL1n/RobustVideoMatting",
    notes="Real-time, no prompt. Mattes every human in view. Weak edges and "
          "less stable on cluttered/harmonised footage. Baseline only.",
)


@register("rvm")
class RVMEngine(MattingEngine):
    INFO = INFO
    info = INFO

    def __init__(self, variant: str = "resnet50", device: str = "auto",
                 half: bool = True, downsample_ratio: Optional[float] = None):
        self.device = resolve_device(device)
        self.half = half and self.device == "cuda"
        self.variant = variant
        self.downsample_ratio = downsample_ratio
        self._model = None
        self._rec = [None] * 4

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        self._torch = torch
        m = torch.hub.load("PeterL1n/RobustVideoMatting", self.variant)
        m = m.to(self.device).eval()
        if self.half:
            m = m.half()
        self._model = m

    def _auto_ratio(self, h: int, w: int) -> float:
        if self.downsample_ratio is not None:
            return self.downsample_ratio
        # Upstream guidance: ~0.25 for 1080p, ~0.125 for 4K.
        return 0.25 if max(h, w) <= 2048 else 0.125

    def start(self, frame_bgr, seed_mask=None, n_warmup: int = 0):
        self.reset()
        return self.step(frame_bgr)

    def step(self, frame_bgr: np.ndarray, **_) -> np.ndarray:
        self._ensure()
        t = self._torch
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1]).astype(np.float32) / 255.0
        x = t.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device)
        if self.half:
            x = x.half()
        with t.inference_mode():
            fgr, pha, *self._rec = self._model(
                x, *self._rec, self._auto_ratio(h, w))
        return to_alpha_2d(pha)

    def reset(self) -> None:
        self._rec = [None] * 4

    def matte(self, frames: Sequence[np.ndarray],
              seed_mask: Optional[np.ndarray] = None,
              n_warmup: int = 10,
              progress: Optional[Callable[[int], None]] = None) -> np.ndarray:
        self.reset()
        # RVM is recurrent with no prompt, so warmup means "run some frames and
        # throw them away" to let the recurrent state settle.
        for i in range(min(n_warmup, len(frames))):
            self.step(frames[i])
        out = np.empty((len(frames), *frames[0].shape[:2]), np.float32)
        for i, f in enumerate(frames):
            out[i] = self.step(f)
            if progress:
                progress(i)
        if seed_mask is not None:
            # Post-hoc subject selection so the A/B is fair on "which person".
            keep = (np.asarray(seed_mask, np.float32) > 0.5)
            if keep.any():
                out = _restrict_to_component_of(out, keep)
        assert len(out) == len(frames)
        return out


def _restrict_to_component_of(alphas: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Zero alpha blobs that never overlap the seed, per frame."""
    import cv2
    out = alphas.copy()
    for i, a in enumerate(alphas):
        hard = (a > 0.5).astype(np.uint8)
        n, lab = cv2.connectedComponents(hard, 8)
        if n <= 1:
            continue
        good = set(np.unique(lab[keep & (lab > 0)]).tolist())
        if not good:
            continue
        mask = np.isin(lab, list(good)) | (lab == 0)
        out[i] = np.where(mask, a, 0.0)
        keep = out[i] > 0.5           # carry selection forward
    return out
