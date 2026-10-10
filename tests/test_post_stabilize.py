"""Post-stabiliser (21 Sep): the stillness-gated second temporal pass.

What it is for: Runbo's "it seems to flicker a lot" on tryguys.  Measured on
the delivered alpha, that flicker is large 0<->1 strobing of a region in a
STATIC part of the shot (the lap between two seated people), where the shipped
smoother's trust gate is open but its 0.35 authority cannot hold it.

What each test pins, and why:
  - default ON at 0.8 (flipped 21 Sep); set 0.0 to reproduce the 20 Sep
    delivery -- a default is which code exists (HANDOFF section 7 entry thirty);
  - a static strobe is damped -- the thing it exists to do;
  - a moving subject does not ghost, and with the still gate it ghosts LESS than
    the ungated filter at the same weight -- the reason the gate exists;
  - fast-moving pixels are left alone entirely -- ipman/dance, where v2 already
    beats v1 and the ungated filter only smeared the soft band;
  - still_px=None reproduces the original TemporalStabilizer exactly, so the
    shipped `stab` is untouched by this change.
"""
import cv2
import numpy as np

from vbgr import refine
from vbgr.config import Config


def scene(h=180, w=320, cx=100, bg=40):
    f = np.full((h, w, 3), bg, np.uint8)
    f[:, :, 1] = (bg + 30) % 255
    cv2.circle(f, (cx, h // 2), 30, (250, 250, 250), -1)
    # texture so the flow estimator has something to lock onto
    rng = np.random.default_rng(0)
    f = np.clip(f.astype(np.int16) + rng.integers(-12, 12, f.shape), 0, 255)
    return f.astype(np.uint8)


def disc(h=180, w=320, cx=100, cy=90, r=30):
    m = np.zeros((h, w), np.float32)
    cv2.circle(m, (cx, cy), r, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 1.0)


def test_default_is_on_at_0_8():
    """Pinned: flipped on 21 Sep after the GPU check.  A stale checkout that
    reverts this must fail here, not on the VM (HANDOFF section 6, 1fc4936)."""
    cfg = Config()
    assert cfg.refine.post_stabilize == 0.8
    assert cfg.refine.post_stabilize_still_px == 6.0


def test_still_px_none_is_the_original_filter():
    f0, f1 = scene(cx=100), scene(cx=104)
    a0, a1 = disc(cx=100), disc(cx=104)
    old = refine.TemporalStabilizer(weight=0.35)
    new = refine.TemporalStabilizer(weight=0.35, still_px=None)
    old.step(f0, a0); new.step(f0, a0)
    assert np.array_equal(old.step(f1, a1), new.step(f1, a1))


def test_static_strobe_is_damped():
    """A static scene; a patch of the matte toggles 0,1,0,1 frame to frame.
    That is the tryguys lap.  The gated filter must shrink the swing."""
    f = scene(cx=100)
    base = disc(cx=100)
    patch = np.zeros_like(base)
    patch[120:150, 200:240] = 1.0
    st = refine.TemporalStabilizer(weight=0.8, still_px=6.0)
    outs = []
    for i in range(8):
        a = np.clip(base + patch * (i % 2), 0, 1)
        outs.append(st.step(f, a)[135, 220])
    swing_in = 1.0
    swing_out = max(abs(outs[i] - outs[i - 1]) for i in range(4, 8))
    assert swing_out < 0.5 * swing_in, outs


def test_moving_subject_does_not_ghost():
    f0, f1 = scene(cx=60), scene(cx=160)
    a0, a1 = disc(cx=60), disc(cx=160)
    gated = refine.TemporalStabilizer(weight=0.8, still_px=6.0)
    ungated = refine.TemporalStabilizer(weight=0.8)
    gated.step(f0, a0); ungated.step(f0, a0)
    g = gated.step(f1, a1)
    u = ungated.step(f1, a1)
    assert g[90, 60] < 0.35, g[90, 60]
    assert g[90, 60] <= u[90, 60] + 1e-6


def test_fast_pixels_are_untouched():
    """Where the flow is far above still_px the weight is zero, so the output
    is the input -- confident alpha on a fast limb is never pulled."""
    f0, f1 = scene(cx=60), scene(cx=160)
    a0, a1 = disc(cx=60), disc(cx=160)
    st = refine.TemporalStabilizer(weight=0.8, still_px=6.0)
    st.step(f0, a0)
    out = st.step(f1, a1)
    # centre of the moved disc: it moved 100 px, far past still_px
    assert abs(out[90, 160] - a1[90, 160]) < 0.02, (out[90, 160], a1[90, 160])


def test_confident_static_alpha_is_unchanged():
    f = scene(cx=100)
    a = disc(cx=100)
    st = refine.TemporalStabilizer(weight=0.8, still_px=6.0)
    for _ in range(5):
        out = st.step(f, a)
    inside = a > 0.98
    outside = a < 0.02
    assert np.abs(out[inside] - a[inside]).max() < 1e-3
    assert np.abs(out[outside] - a[outside]).max() < 1e-3
