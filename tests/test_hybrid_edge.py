"""hybrid_edge: crisp tracker mask + strong alpha outside it (25 Sep)."""
import numpy as np
from vbgr.config import Config
from vbgr.pipeline import hybrid_edge


def test_default_off():
    assert Config().matting.hybrid_edge is False


def test_faint_rim_dropped_strong_hair_kept_far_solid_kept():
    m = np.zeros((1, 60, 60), np.uint8); m[0, 20:40, 20:40] = 1
    a = np.zeros((1, 60, 60), np.float32)
    a[0, 18:42, 18:42] = 0.4          # faint rim around the body
    a[0, 20:40, 20:40] = 0.7          # soft interior (mask says solid)
    a[0, 15:20, 30] = 0.95            # strong hair strand just outside
    a[0, 50:55, 50:55] = 1.0          # solid part the mask missed, far away
    o = hybrid_edge(a, m, band_px=6, strong=0.6)[0]
    assert o[30, 30] > 0.99           # interior solid
    assert o[18, 25] < 0.05           # faint rim gone
    assert o[16, 30] > 0.8            # hair kept
    assert o[52, 52] > 0.99           # far solid kept
