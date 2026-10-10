import numpy as np
from vbgr.seed import bottom_pockets, pocket_fill
from vbgr.config import MattingConfig


def _legs(H=120, W=200):
    """Two legs running off the bottom edge, joined above: a closed pocket."""
    m = np.zeros((H, W), np.float32)
    m[40:, 40:80] = 1; m[40:, 120:160] = 1; m[40:60, 40:160] = 1
    return m


def test_pocket_found_between_legs():
    pk = bottom_pockets(_legs() > 0.5, gap=100, hmax=100)
    assert len(pk) == 1 and pk[0][100, 100] and not pk[0][100, 20]


def test_open_gap_is_not_a_pocket():
    m = _legs(); m[40:60, 80:120] = 0          # no join above: open to background
    assert bottom_pockets(m > 0.5, gap=100, hmax=100) == []


def test_wide_gap_skipped():
    assert bottom_pockets(_legs() > 0.5, gap=20, hmax=100) == []


def test_gate_accepts_person_rejects_wall():
    a = _legs(); fr = np.zeros(a.shape + (3,), np.uint8)
    out, n, k = pocket_fill(fr, a, gap=100, hmax=100, min_px=10, segment=lambda f, pt: _pocket_and_legs(a))
    assert k == 1 and out[100, 100] == 1.0 and n > 0
    wall = np.ones(a.shape, bool)               # SAM says: huge wall segment
    out2, n2, k2 = pocket_fill(fr, a, gap=100, hmax=100, min_px=10, segment=lambda f, pt: wall)
    assert k2 == 0 and out2[100, 100] == 0.0


def _pocket_and_legs(a):
    m = a > 0.5
    return m | bottom_pockets(m, 100, 100)[0]


def test_default_off():
    assert MattingConfig().bottom_pocket is False


def test_bottom_median_removes_one_frame_toggle_only_in_band():
    from vbgr.seed import bottom_temporal_median
    a = np.zeros((7, 10, 4), np.float32)
    a[3, 8:, :] = 1          # one-frame blip in the bottom band
    a[3, 0, :] = 1           # one-frame blip above the band
    o = bottom_temporal_median(a, rows_from=5, w=2)
    assert o[3, 8:].max() == 0 and o[3, 0].min() == 1


def test_pocket_fill_leaves_no_seam():
    a = _legs(); a[60:, 79] = 0.3; a[60:, 120] = 0.3   # soft edge next to the pocket
    fr = np.zeros(a.shape + (3,), np.uint8)
    out, n, k = pocket_fill(fr, a, gap=100, hmax=100, min_px=10,
                            segment=lambda f, pt: _pocket_and_legs(a))
    assert k == 1 and out[100, 79] == 1.0 and out[100, 120] == 1.0
