"""5.7: the seed-tail measure, and the bounded extension built on it."""
import numpy as np
import pytest

from vbgr.seed import extend_short_tails, seed_tail_fraction, worst_seed_tail


class _Seeder:
    """Stands in for MaskRCNNSeeder; only last_aligned and _match are used."""
    def __init__(self, aligned):
        self.last_aligned = aligned

    def _match(self, frame, boxes, score_min):
        return self.last_aligned


def _mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_tail_fraction_zero_when_mask_spans_its_box():
    m = _mask(200, 200, 20, 120, 40, 90)
    assert seed_tail_fraction(m, (40, 20, 90, 120)) == pytest.approx(0.0, abs=0.02)


def test_tail_fraction_measures_the_bottom_gap():
    m = _mask(200, 200, 20, 70, 40, 90)          # mask ends at 69
    # box runs to 120, so 50 of 100 rows are unmasked at the bottom
    assert seed_tail_fraction(m, (40, 20, 90, 120)) == pytest.approx(0.5, abs=0.02)


def test_tail_fraction_is_one_for_an_unmatched_box():
    assert seed_tail_fraction(None, (0, 0, 10, 10)) == 1.0


def test_worst_seed_tail_takes_the_worst_person():
    good = _mask(200, 200, 20, 120, 40, 90)
    bad = _mask(200, 200, 20, 70, 110, 160)
    s = _Seeder([good, bad])
    got = worst_seed_tail(s, np.zeros((200, 200, 3), np.uint8),
                          [(40, 20, 90, 120), (110, 20, 160, 120)])
    assert got == pytest.approx(0.5, abs=0.02)


def test_extension_closes_a_small_gap_to_the_box_bottom():
    m = _mask(300, 300, 40, 240, 100, 180)       # 60 rows short of a 260 box
    s = _Seeder([object()])
    out, n = extend_short_tails([m], s, [(90., 40., 190., 260.)], 0.30)
    assert n == 1
    ys = np.nonzero(np.any(out[0] > 0, axis=1))[0]
    assert ys.max() == 259          # box y1 is 260, so its last row is 259


def test_extension_leaves_a_half_occluded_person_alone():
    """The desk case. Extending here would key in the desk, which is the
    failure mode that made 'extend the mask to its box' unshippable."""
    m = _mask(300, 300, 40, 140, 100, 180)       # 120 rows short of a 220 box
    s = _Seeder([object()])
    out, n = extend_short_tails([m], s, [(90., 40., 190., 260.)], 0.15)
    assert n == 0
    assert np.array_equal(out[0], m)


def test_extension_does_not_reach_past_the_masks_own_columns():
    """It extends the legs that are there, not the full width of the box."""
    m = _mask(300, 300, 40, 240, 100, 130)
    s = _Seeder([object()])
    out, n = extend_short_tails([m], s, [(50., 40., 250., 260.)], 0.30)
    assert n == 1
    xs = np.nonzero(np.any(out[0][241:259] > 0, axis=0))[0]
    assert xs.min() >= 100 and xs.max() <= 129


def test_extension_is_a_no_op_when_disabled():
    m = _mask(300, 300, 40, 240, 100, 180)
    s = _Seeder([object()])
    out, n = extend_short_tails([m], s, [(90., 40., 190., 260.)], 0.0)
    assert n == 0


def test_extension_skips_boxes_that_matched_nothing():
    m = _mask(300, 300, 40, 240, 100, 180)
    s = _Seeder([None, object()])                # first box unmatched
    out, n = extend_short_tails([m], s, [(0., 0., 10., 10.), (90., 40., 190., 260.)], 0.30)
    assert n == 1 and len(out) == 1
