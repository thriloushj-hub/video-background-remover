"""3.5 -- foreground decontamination actually removes the fringe.

The point of decontamination is that in the boundary band the observed pixel is
already a mix of subject and old background, so compositing it over a new
background carries a rim of the old one. This builds a composite from a KNOWN
foreground and background, which is the only way to measure the error rather
than eyeball it.

The DecontamConfig docstring used to quote "10/2 leaves ~62% of the fringe
error; 20/3 leaves ~40%" from a synthetic that was never committed, so nobody
could reproduce or re-check it. These numbers come from the fixture below.
"""
import numpy as np
import pytest

from vbgr.decontaminate import decontaminate


def _composite(seed=0):
    """I = a*F + (1-a)*B with a wide fractional band and a hostile B.

    B is saturated blue against a warm F, which is the worst case: any
    surviving fringe is unmistakable rather than a subtle shift.
    """
    h = w = 256
    F = np.zeros((h, w, 3), np.float32)
    F[..., 0], F[..., 1], F[..., 2] = 120, 150, 205          # BGR, warm
    F += np.random.RandomState(seed).normal(0, 6, F.shape).astype(np.float32)
    B = np.zeros((h, w, 3), np.float32)
    B[..., 0], B[..., 1], B[..., 2] = 230, 60, 30            # BGR, cold blue

    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(yy - h / 2, xx - w / 2)
    a = np.clip((90 - d) / 18.0, 0, 1).astype(np.float32)

    obs = np.clip(a[..., None] * F + (1 - a[..., None]) * B, 0, 255).astype(np.uint8)
    return obs, a, F


def _fringe_error(est, F, a):
    band = (a > 0.02) & (a < 0.98)
    return float(np.abs(est.astype(np.float32) - F)[band].mean())


def test_decontamination_removes_most_of_the_fringe():
    obs, a, F = _composite()
    before = _fringe_error(obs, F, a)
    after = _fringe_error(decontaminate(obs, a), F, a)
    assert before > 50, "fixture should present a large fringe to remove"
    assert after < 0.35 * before, (
        f"decontamination left {100*after/before:.0f}% of the fringe error "
        f"(before {before:.1f}, after {after:.1f})")


def test_more_iterations_never_make_it_worse():
    """Guards the speed/accuracy dial against being turned the wrong way."""
    obs, a, F = _composite()
    errs = [_fringe_error(
        decontaminate(obs, a, n_small_iterations=ns, n_big_iterations=nb), F, a)
        for ns, nb in [(10, 2), (20, 3), (40, 5)]]
    assert errs == sorted(errs, reverse=True), f"not monotonic: {errs}"


def test_band_only_leaves_the_solid_interior_alone():
    """In the interior the observed pixel IS the foreground. Substituting an
    estimate there can only add error."""
    obs, a, _ = _composite()
    out = decontaminate(obs, a, band_only=True)
    interior = a > 0.99
    delta = np.abs(out.astype(np.float32) - obs.astype(np.float32))[interior].mean()
    assert delta < 0.5, f"interior moved by {delta:.3f}, band_only is not holding"


def test_output_is_uint8_and_same_shape():
    obs, a, _ = _composite()
    out = decontaminate(obs, a)
    assert out.dtype == np.uint8
    assert out.shape == obs.shape
