"""Compositing outputs, and recovering alpha from a green-screen render.

``alpha_from_greenscreen`` exists so the benchmark harness can score the v1
outputs: those were delivered as green composites with no alpha channel, which
means the only way to measure them is to invert the composite.  Keeping this
next to the compositor makes the inverse relationship obvious.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


DEFAULT_GREEN = (0, 177, 64)          # BGR


def composite_over_color(frame_bgr: np.ndarray, alpha: np.ndarray,
                         color: Tuple[int, int, int] = DEFAULT_GREEN,
                         foreground: Optional[np.ndarray] = None) -> np.ndarray:
    """``a*F + (1-a)*color``.

    ``foreground`` should be the *decontaminated* F, not the observed frame --
    see decontaminate.py for why using the observed pixel produces a fringe.
    """
    a = np.clip(alpha, 0, 1).astype(np.float32)[..., None]
    F = (foreground if foreground is not None else frame_bgr).astype(np.float32)
    bg = np.array(color, np.float32).reshape(1, 1, 3)
    return np.clip(a * F + (1 - a) * bg, 0, 255).astype(np.uint8)


def composite_over_image(frame_bgr: np.ndarray, alpha: np.ndarray,
                         bg_bgr: np.ndarray,
                         foreground: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = alpha.shape[:2]
    if bg_bgr.shape[:2] != (h, w):
        bg_bgr = _cover_resize(bg_bgr, w, h)
    a = np.clip(alpha, 0, 1).astype(np.float32)[..., None]
    F = (foreground if foreground is not None else frame_bgr).astype(np.float32)
    return np.clip(a * F + (1 - a) * bg_bgr.astype(np.float32), 0, 255).astype(np.uint8)


def _cover_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (int(np.ceil(iw * s)), int(np.ceil(ih * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0 = (r.shape[0] - h) // 2
    x0 = (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def to_bgra(frame_bgr: np.ndarray, alpha: np.ndarray,
            foreground: Optional[np.ndarray] = None,
            premultiplied: bool = False) -> np.ndarray:
    """Straight (default) or premultiplied BGRA.

    Straight alpha is what After Effects / Resolve / Premiere expect on import.
    Premultiplied is what web canvas compositing wants.  Getting this wrong is
    the other classic source of edge fringing, independent of decontamination.
    """
    a = np.clip(alpha, 0, 1).astype(np.float32)
    F = (foreground if foreground is not None else frame_bgr).astype(np.float32)
    if premultiplied:
        F = F * a[..., None]
    out = np.empty((*a.shape, 4), np.uint8)
    out[..., :3] = np.clip(F, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def alpha_to_bgr(alpha: np.ndarray) -> np.ndarray:
    """Alpha as a grey video (for an 'alpha' output mode / matte pass)."""
    g = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
# Inverse: recover alpha from a green-screen composite
# --------------------------------------------------------------------------- #

def alpha_from_greenscreen(comp_bgr: np.ndarray,
                           key: Tuple[int, int, int] = DEFAULT_GREEN,
                           softness: float = 26.0) -> np.ndarray:
    """Approximate the alpha that produced a green composite.

    Works in YCrCb, where the key colour occupies a compact chroma region, so a
    subject wearing a *dark* colour (the ipman failure case) is not confused
    with the key the way a naive RGB distance would.

    This is an approximation -- the recovered alpha is soft-thresholded chroma
    distance, so it slightly over-softens genuinely hard edges.  It is used only
    for *relative* comparison between runs, never as ground truth.
    """
    img = comp_bgr.astype(np.float32)
    ycc = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    key_ycc = cv2.cvtColor(
        np.array(key, np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2YCrCb
    ).astype(np.float32).reshape(3)

    d = np.sqrt((ycc[..., 1] - key_ycc[1]) ** 2 + (ycc[..., 2] - key_ycc[2]) ** 2)
    a = np.clip(d / max(softness, 1e-6), 0.0, 1.0)

    # Luma guard: pure black subject pixels have near-key chroma noise but very
    # different luma, so rescue them.
    luma_gap = np.abs(ycc[..., 0] - key_ycc[0]) / 128.0
    a = np.clip(np.maximum(a, luma_gap * (d > softness * 0.35)), 0.0, 1.0)
    return a.astype(np.float32)


def checkerboard(h: int, w: int, size: int = 16,
                 a: int = 200, b: int = 155) -> np.ndarray:
    """Classic alpha-inspection backdrop."""
    yy, xx = np.mgrid[0:h, 0:w]
    m = (((yy // size) + (xx // size)) % 2).astype(np.uint8)
    img = np.where(m[..., None] == 0, a, b).astype(np.uint8)
    return np.repeat(img, 3, axis=2) if img.shape[2] == 1 else img
