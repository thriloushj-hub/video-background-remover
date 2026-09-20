"""core_solidify: fix faded interiors, never touch true edges or true gaps.

Born from the 20 Sep full-length review: interiors fading to alpha 0.3-0.7
under fast motion while every IoU column stayed green.  Each test pins one
clause of the construction, because each clause exists to prevent a specific
failure seen in the offline sweep:

- interior recovery is the point;
- the erode clause is what keeps hair and real motion-blur edges fractional;
- the floor clause is what stopped variant A webbing solid alpha across the
  gap between two subjects (false_solid 0.28 on butter before it, 0.03 after);
- thin faded extremities are documented as out of scope (the erode removes
  them), so a test pins that too -- if someone later "fixes" it by accident,
  the gap guard has probably been broken.
"""
import numpy as np
import pytest

from vbgr.refine import core_solidify


W, H = 960, 540
R = max(3, int(round(0.00625 * W)))  # the default kernel radius at this width


def canvas():
    return np.zeros((H, W), np.float32)


def test_faded_interior_is_recovered():
    a = canvas()
    a[100:300, 100:400] = 1.0          # solid body
    a[180:220, 200:300] = 0.4          # faded patch deep inside
    out = core_solidify(a)
    assert float(out[190:210, 220:280].min()) == 1.0


def test_interior_dip_below_thresh_is_bridged_and_recovered():
    # close() bridges holes narrower than the kernel (2R+1); wider ones stay.
    a = canvas()
    a[100:300, 100:400] = 1.0
    a[195:205, 245:255] = 0.10         # 10px dip < 13px kernel -> bridged
    out = core_solidify(a)
    assert float(out[197:203, 247:253].min()) == 1.0


def test_interior_dip_wider_than_kernel_is_not_bridged():
    a = canvas()
    a[100:300, 100:400] = 1.0
    a[180:240, 220:340] = 0.10         # 60x120 dip >> kernel -> centre stays
    out = core_solidify(a)
    assert float(out[205:215, 275:285].max()) == pytest.approx(0.10)


def test_soft_edge_keeps_its_outer_band():
    # A fractional band above `thresh` counts as part of the body, so its
    # INNER portion may solidify; the outermost R pixels never do.  That is
    # the contract: nothing within R of the true boundary is touched.
    a = canvas()
    a[100:300, 100:400] = 1.0
    a[100:300, 400:420] = 0.5
    out = core_solidify(a)
    assert np.array_equal(out[:, 420 - R:420], a[:, 420 - R:420])
    assert float(out[150, 405]) == 1.0  # inner band is interior by contract


def test_gap_between_two_subjects_stays_empty():
    a = canvas()
    a[100:300, 100:250] = 1.0
    a[100:300, 250 + R:400 + R] = 1.0  # second body, gap narrower than close kernel
    gap = a[150:250, 251:250 + R - 1]
    assert float(gap.max()) == 0.0
    out = core_solidify(a)
    assert float(out[150:250, 251:250 + R - 1].max()) == 0.0


def test_thin_faded_extremity_is_out_of_scope():
    a = canvas()
    a[100:300, 100:400] = 1.0
    a[195:198, 400:500] = 0.4          # 3px-thick faded "finger"
    out = core_solidify(a)
    assert float(out[195:198, 420:500].max()) == pytest.approx(0.4)


def test_solid_input_is_identity():
    a = canvas()
    a[100:300, 100:400] = 1.0
    out = core_solidify(a)
    assert np.array_equal(out, a)


def test_default_config_is_off():
    from vbgr.config import Config
    assert Config().refine.core_solidify is False
