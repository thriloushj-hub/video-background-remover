"""halo_split regressions.

`halo` was being read as "the baseline is keying in background, so v2 is right
to be smaller".  On butter that reading was wrong: the window is a camera
pull-back, seven dancers end up in shot, v2 seeds four at frame 0 and v1 holds
all seven, so three whole people scored as halo.  Measured on the 27 Aug run,
butter's 0.5407 is 0.5222 standalone and 0.0186 shell.

The cases below are the three shapes that matters: a shell hugging v2's
silhouette, a whole subject v2 never had, and the two together.
"""
import numpy as np

from bench.metrics import halo_split

H, W = 200, 400


def _blob(x1, y1, x2, y2):
    a = np.zeros((H, W), np.float32)
    a[y1:y2, x1:x2] = 1.0
    return a


def test_a_shell_around_the_subject_is_shell_not_solo():
    v2 = _blob(100, 60, 160, 160)
    v1 = _blob(96, 56, 164, 164)          # same subject, 4px fatter all round

    r = halo_split([v1], [v2])

    assert r["halo_solo"] == 0.0
    assert r["halo_shell"] > 0.0
    assert r["solo_components"] == 0


def test_a_subject_v2_never_had_is_solo_not_shell():
    """butter and dance: v1 holds a person v2 was never seeded on."""
    v2 = _blob(40, 60, 100, 160)
    v1 = np.maximum(v2, _blob(300, 60, 360, 160))   # a second person, far away

    r = halo_split([v1], [v2])

    assert r["halo_shell"] == 0.0
    assert r["halo_solo"] > 0.4                     # the whole second person
    assert r["solo_components"] == 1
    assert r["solo_max_frame_frac"] > 0.05          # person-sized


def test_both_at_once_are_reported_separately():
    v2 = _blob(40, 60, 100, 160)
    v1 = np.maximum(_blob(36, 56, 104, 164), _blob(300, 60, 360, 160))

    r = halo_split([v1], [v2])

    assert r["halo_shell"] > 0.0
    assert r["halo_solo"] > 0.0
    assert abs(r["halo_frac"] - (r["halo_shell"] + r["halo_solo"])) < 0.002
    assert r["solo_components"] == 1


def test_agreement_scores_zero_halo():
    v2 = _blob(100, 60, 160, 160)
    r = halo_split([v2], [v2])
    assert r["halo_frac"] == 0.0


def test_speckle_does_not_count_as_a_missing_subject():
    """Solo pixels are still solo, but nothing person-sized is claimed."""
    v2 = _blob(40, 60, 100, 160)
    v1 = v2.copy()
    v1[10, 300] = 1.0
    v1[12, 320] = 1.0

    r = halo_split([v1], [v2])

    assert r["solo_components"] == 0
    assert r["solo_max_frame_frac"] == 0.0
