"""_alpha_union must record per-object coverage before it unions it away.

That recording is the whole basis of the rewritten subj_lost_at: the union
cannot tell two touching people from one, but the per-object stack can.
"""
import types

import numpy as np
import pytest

from vbgr.engines.sam2matting import SAM2MattingEngine


class _FakeTensor:                     # stands in for torch.Tensor
    pass


def _eng():
    e = SAM2MattingEngine.__new__(SAM2MattingEngine)
    e._torch = types.SimpleNamespace(Tensor=_FakeTensor)
    e._obj_area_frame = []
    return e


def _stack(fracs, h=100, w=100):
    """One (K,H,W) alpha stack where object k covers `fracs[k]` of the frame.

    Coverage is laid down cell-exactly rather than by whole rows, so a coarse
    grid cannot introduce rounding the assertions would have to absorb.
    """
    out = np.zeros((len(fracs), h, w), np.float32)
    for k, f in enumerate(fracs):
        n = int(round(f * h * w))
        flat = out[k].reshape(-1)
        flat[:n] = 1.0
    return out


def test_records_each_object_area():
    e = _eng()
    a = e._alpha_union(_stack([0.10, 0.25]), 2, 100, 100)
    assert a.shape == (100, 100)
    assert e._obj_area_frame == pytest.approx([0.10, 0.25], abs=1e-3)


def test_union_is_the_max_not_the_first():
    """The old to_alpha_2d bug kept only object 0. Guard against a relapse."""
    e = _eng()
    s = np.zeros((2, 100, 100), np.float32)
    s[0, :10] = 1.0
    s[1, 50:60] = 1.0
    a = e._alpha_union(s, 2, 100, 100)
    assert a[:10].mean() == pytest.approx(1.0)
    assert a[50:60].mean() == pytest.approx(1.0)


def test_channels_last_layout_also_recorded():
    e = _eng()
    s = np.moveaxis(_stack([0.10, 0.25]), 0, -1)
    e._alpha_union(s, 2, 100, 100)
    assert e._obj_area_frame == pytest.approx([0.10, 0.25], abs=1e-3)


def test_0_255_input_is_scaled_before_measuring():
    e = _eng()
    e._alpha_union(_stack([0.10, 0.25]) * 255.0, 2, 100, 100)
    assert e._obj_area_frame == pytest.approx([0.10, 0.25], abs=1e-3)


def test_single_object_2d_input():
    e = _eng()
    s = np.zeros((100, 100), np.float32)
    s[:20] = 1.0
    e._alpha_union(s, 1, 100, 100)
    assert e._obj_area_frame == pytest.approx([0.20], abs=1e-3)


def test_area_is_measured_before_resize():
    """A resize must not change the ratio the metric reads."""
    e = _eng()
    e._alpha_union(_stack([0.10, 0.25], h=50, w=50), 2, 200, 200)
    assert e._obj_area_frame == pytest.approx([0.10, 0.25], abs=1e-3)


def test_unknown_layout_still_refuses_to_guess():
    e = _eng()
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        e._alpha_union(np.zeros((3, 100, 100), np.float32), 2, 100, 100)
