"""refine.fill_dropouts (9 Oct): add-only, both-neighbours-agree, still-gated,
blob-gated repair of 1-2 frame dropouts and shot warm-up."""
import numpy as np, cv2
from vbgr import refine, motion
from vbgr.config import Config


def _scene(n=8, hole_at=(), warm=0, moving=False):
    H, W = 120, 160
    frames, alphas = [], []
    for i in range(n):
        f = np.full((H, W, 3), 40, np.uint8)
        # textured still background so flow is well defined
        rng = np.random.RandomState(0)
        f += rng.randint(0, 30, (H, W, 3)).astype(np.uint8)
        x = 60 + (4 * i if moving else 0)
        cv2.rectangle(f, (x, 30), (x + 40, 90), (200, 180, 160), -1)
        a = np.zeros((H, W), np.float32)
        a[30:90, x:x + 40] = 1.0
        if i in hole_at:
            a[60:90, x:x + 40] = 0.0            # bottom half dropped
        if i < warm:
            a[:] = 0.0                            # engine not warmed up
        frames.append(f); alphas.append(a)
    return frames, alphas


def test_single_frame_dropout_is_filled():
    frames, alphas = _scene(hole_at=(4,))
    out, nch, npx = refine.fill_dropouts(frames, alphas, motion.FlowEstimator(), min_blob=300)
    assert nch == 1 and npx > 500
    assert out[4][70:85, 65:95].mean() > 0.9


def test_two_frame_dropout_is_filled():
    frames, alphas = _scene(hole_at=(4, 5))
    out, nch, npx = refine.fill_dropouts(frames, alphas, motion.FlowEstimator(), min_blob=300)
    assert nch == 2
    assert out[4][70:85, 65:95].mean() > 0.9 and out[5][70:85, 65:95].mean() > 0.9


def test_warmup_backfills_first_frames():
    frames, alphas = _scene(warm=3)
    out, nch, npx = refine.fill_dropouts(frames, alphas, motion.FlowEstimator(), warmup=4, min_blob=300)
    assert nch == 3
    for k in range(3):
        assert out[k][40:80, 65:95].mean() > 0.9


def test_add_only_and_untouched_frames_identical():
    frames, alphas = _scene(hole_at=(4,))
    out, _, _ = refine.fill_dropouts(frames, alphas, motion.FlowEstimator(), min_blob=300)
    for i in range(len(alphas)):
        assert np.all(out[i] >= alphas[i] - 1e-6)
        if i != 4:
            assert np.array_equal(out[i], alphas[i])


def test_fast_motion_is_left_alone():
    # subject moving 4 px/frame with no dropout: the still gate must block any
    # "fill" that the warped neighbours would otherwise suggest
    frames, alphas = _scene(moving=True)
    out, nch, npx = refine.fill_dropouts(frames, alphas, motion.FlowEstimator(), still_px=2.0, min_blob=300)
    assert nch == 0 and npx == 0


def test_small_blobs_ignored():
    frames, alphas = _scene(hole_at=(4,))
    out, nch, npx = refine.fill_dropouts(frames, alphas, motion.FlowEstimator(), min_blob=100000)
    assert nch == 0


def test_default_is_on():
    c = Config()
    assert c.refine.dropout_fill is True
    assert c.refine.dropout_warmup_frames == 4
