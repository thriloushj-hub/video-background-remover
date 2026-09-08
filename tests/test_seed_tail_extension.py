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


# --------------------------------------------------------------------------- #
# 5.7: choosing which frame of the shot to seed from
# --------------------------------------------------------------------------- #

class _Det:
    """Detector stub: one person, same box on every frame."""
    def __init__(self, box=(90., 40., 190., 260.)):
        self.box = box

    def detect(self, frame):
        class D:
            pass
        d = D(); d.box = self.box
        return [d], []


class _TailSeeder:
    """Seeder stub whose mask quality is scripted per frame index."""
    def __init__(self, tails):
        self.tails = tails

    def _match(self, frame, boxes, score_min):
        i = int(frame[0, 0, 0])                 # frame index smuggled in pixel 0
        t = self.tails[i]
        m = np.zeros((300, 300), np.uint8)
        y1 = int(260 - t * 220)
        m[40:y1, 100:180] = 1
        return [m]


def _frames(n):
    out = []
    for i in range(n):
        f = np.zeros((300, 300, 3), np.uint8)
        f[0, 0, 0] = i
        out.append(f)
    return out


class _Cfg:
    lookahead_frames = 6
    lookahead_min_tail = 0.15
    lookahead_min_gain = 0.10


def _select(people, w, h):
    return list(people), []


def test_lookahead_moves_to_a_better_frame():
    """bilibili's shape: frame 0 fails, the next frame is fine."""
    from vbgr.seed import pick_seed_frame
    idx, notes = pick_seed_frame(_frames(6), _Det(), _TailSeeder([0.33, 0.08, 0.07, 0.05, 0.33, 0.05]),
                                 _Cfg(), _select)
    assert idx != 0
    assert any("seeded from frame" in n for n in notes)


def test_lookahead_stays_when_frame_zero_is_healthy():
    from vbgr.seed import pick_seed_frame
    idx, notes = pick_seed_frame(_frames(6), _Det(), _TailSeeder([0.05, 0.02, 0.02, 0.02, 0.02, 0.02]),
                                 _Cfg(), _select)
    assert idx == 0 and notes == []


def test_lookahead_stays_for_a_persistently_occluded_subject():
    """The desk case: equally occluded on every frame, so no gain, no move."""
    from vbgr.seed import pick_seed_frame
    idx, _ = pick_seed_frame(_frames(6), _Det(), _TailSeeder([0.40, 0.39, 0.41, 0.40, 0.38, 0.40]),
                             _Cfg(), _select)
    assert idx == 0


def test_lookahead_is_off_by_default():
    from vbgr.config import Config
    assert Config().seed.lookahead_frames == 0


def test_lookahead_disabled_returns_frame_zero():
    from vbgr.seed import pick_seed_frame

    class Off(_Cfg):
        lookahead_frames = 0
    idx, notes = pick_seed_frame(_frames(6), _Det(), _TailSeeder([0.33] * 6), Off(), _select)
    assert idx == 0 and notes == []
