"""Foreground colour decontamination.

Why this matters even when the alpha is perfect
-----------------------------------------------
Compositing is ``out = a*F + (1-a)*newBG``.  If you use the *observed* pixel as
F you are actually compositing ``a*(a*F + (1-a)*oldBG) + (1-a)*newBG``, so the
old background bleeds through the whole boundary band.  On a green-screen output
this shows up as a dark or coloured halo; on a transparent PNG sequence dropped
into an editor it shows up as a fringe that no amount of alpha tuning fixes.
Every professional keyer has a "spill suppression / decontaminate colors" stage
for exactly this reason.

Method
------
Multi-level fast foreground estimation (Germer, Uelwer, Conrad, Harmeling 2020,
"Fast Multi-Level Foreground Estimation").  We solve, per pixel, a 2x2
least-squares system for (F, B) that trades off

  * the compositing equation  I = a*F + (1-a)*B, and
  * smoothness of F and B against their 4-neighbours, weighted by how similar
    the neighbour's alpha is (so F and B are allowed to jump exactly where
    alpha jumps).

Solved coarse-to-fine on an image pyramid, which is what makes it fast enough
to run per frame.  Pure NumPy, no torch, no solver.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Core solver
# --------------------------------------------------------------------------- #

def _neighbours(x: np.ndarray):
    """4-neighbourhood with replicated borders. Yields shifted copies."""
    yield np.pad(x, ((1, 0), (0, 0)) + ((0, 0),) * (x.ndim - 2), mode="edge")[:-1]
    yield np.pad(x, ((0, 1), (0, 0)) + ((0, 0),) * (x.ndim - 2), mode="edge")[1:]
    yield np.pad(x, ((0, 0), (1, 0)) + ((0, 0),) * (x.ndim - 2), mode="edge")[:, :-1]
    yield np.pad(x, ((0, 0), (0, 1)) + ((0, 0),) * (x.ndim - 2), mode="edge")[:, 1:]


def _iterate(image: np.ndarray, alpha: np.ndarray,
             F: np.ndarray, B: np.ndarray,
             n_iter: int, reg: float) -> Tuple[np.ndarray, np.ndarray]:
    a0 = alpha[..., None]                       # H,W,1
    a1 = 1.0 - a0

    for _ in range(n_iter):
        a00 = np.full(alpha.shape, reg, np.float32)
        a11 = np.full(alpha.shape, reg, np.float32)
        b0 = reg * F
        b1 = reg * B

        nF = list(_neighbours(F))
        nB = list(_neighbours(B))
        for na, nf, nb in zip(_neighbours(alpha), nF, nB):
            da = reg + np.abs(alpha - na)       # H,W
            a00 += da
            a11 += da
            d = da[..., None]
            b0 = b0 + d * nf
            b1 = b1 + d * nb

        a00 = a00 + alpha * alpha
        a01 = alpha * (1.0 - alpha)
        a11 = a11 + (1.0 - alpha) * (1.0 - alpha)
        b0 = b0 + a0 * image
        b1 = b1 + a1 * image

        det = a00 * a11 - a01 * a01
        inv = (1.0 / np.maximum(det, 1e-12))[..., None]
        A00, A01, A11 = a00[..., None], a01[..., None], a11[..., None]

        F = np.clip((A11 * b0 - A01 * b1) * inv, 0.0, 1.0).astype(np.float32)
        B = np.clip((A00 * b1 - A01 * b0) * inv, 0.0, 1.0).astype(np.float32)

    return F, B


def estimate_foreground_background(image: np.ndarray,
                                   alpha: np.ndarray,
                                   regularization: float = 1e-5,
                                   n_small_iterations: int = 20,
                                   n_big_iterations: int = 3,
                                   small_size: int = 32
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate true foreground and background colour.

    Parameters
    ----------
    image : HxWx3 float32 in [0, 1] (RGB or BGR, it is channel agnostic)
    alpha : HxW   float32 in [0, 1]

    Returns
    -------
    (F, B) each HxWx3 float32 in [0, 1].
    """
    image = np.ascontiguousarray(image, np.float32)
    alpha = np.ascontiguousarray(alpha, np.float32)
    h, w = alpha.shape

    if min(h, w) > small_size:
        sh, sw = max(1, h // 2), max(1, w // 2)
        img_s = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        a_s = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
        F, B = estimate_foreground_background(
            img_s, a_s, regularization,
            n_small_iterations, n_big_iterations, small_size)
        F = cv2.resize(F, (w, h), interpolation=cv2.INTER_LINEAR)
        B = cv2.resize(B, (w, h), interpolation=cv2.INTER_LINEAR)
        n_iter = n_big_iterations
    else:
        F = image.copy()
        B = image.copy()
        n_iter = n_small_iterations

    return _iterate(image, alpha, F, B, n_iter, regularization)


# --------------------------------------------------------------------------- #
# Frame-level convenience wrapper
# --------------------------------------------------------------------------- #

def decontaminate(frame_bgr: np.ndarray,
                  alpha: np.ndarray,
                  regularization: float = 1e-5,
                  n_small_iterations: int = 20,
                  n_big_iterations: int = 3,
                  small_size: int = 32,
                  band_only: bool = True,
                  band_lo: float = 0.02,
                  band_hi: float = 0.98) -> np.ndarray:
    """Return a decontaminated foreground image (uint8 BGR).

    With ``band_only`` (the default) the solved F is only substituted where
    alpha is genuinely fractional.  In the solid interior the observed pixel is
    already the true foreground, and substituting an estimate there would only
    add error.
    """
    img = frame_bgr.astype(np.float32) / 255.0
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    F, _ = estimate_foreground_background(
        img, a, regularization, n_small_iterations, n_big_iterations, small_size)

    if band_only:
        band = ((a > band_lo) & (a < band_hi)).astype(np.float32)
        band = cv2.GaussianBlur(band, (0, 0), 1.0)[..., None]
        F = F * band + img * (1.0 - band)

    return np.clip(F * 255.0 + 0.5, 0, 255).astype(np.uint8)


def spill_suppress(frame_bgr: np.ndarray, alpha: np.ndarray,
                   key_bgr: Tuple[int, int, int] = (0, 177, 64),
                   strength: float = 1.0) -> np.ndarray:
    """Optional extra pass for footage shot on an actual green screen.

    Not needed for AI keying of normal footage -- included because the batch
    tool is also pointed at green-screen source sometimes and the fringe there
    is chromatic rather than luminance-based.
    """
    img = frame_bgr.astype(np.float32)
    key = np.array(key_bgr, np.float32)
    key = key / (np.linalg.norm(key) + 1e-6)
    proj = (img * key).sum(axis=2, keepdims=True)
    excess = np.clip(proj - np.median(proj), 0, None)
    band = ((alpha > 0.02) & (alpha < 0.98)).astype(np.float32)[..., None]
    out = img - strength * excess * key * band
    return np.clip(out, 0, 255).astype(np.uint8)
