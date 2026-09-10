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
    """Seeder stub whose mask quality is scripted per frame index.

    One mask per box, filling that box from its top down to ``1 - tail`` of its
    height -- so tail is the scripted number and fill is ``1 - tail``.  The two
    agree here on purpose: these cases are about the look-ahead moving or not,
    not about the two measures disagreeing.  `_SplitSeeder` below is the case
    where they disagree, which is the one that cost us bilibili's trailing boot.
    """
    def __init__(self, tails):
        self.tails = tails

    def _match(self, frame, boxes, score_min):
        i = int(frame[0, 0, 0])                 # frame index smuggled in pixel 0
        t = self.tails[i]
        out = []
        for b in boxes:
            x0, y0, x1, y1 = (int(v) for v in b)
            m = np.zeros((300, 300), np.uint8)
            bh = max(y1 - y0, 1)
            m[y0:max(y0, int(y1 - t * bh)), x0:x1] = 1
            out.append(m)
        return out


class _SplitSeeder:
    """bilibili's shape: a mask that REACHES the floor holding half the subject.

    Per frame, ``(tail, width_fraction)``.  A frame can have a perfect tail --
    the mask reaches the bottom of the box -- while covering only half the box's
    width, which is what "the leading boot and not the trailing one" looks like
    to a metric.  Ranking by tail picks that frame; ranking by fill does not.
    """
    def __init__(self, spec):
        self.spec = spec

    def _match(self, frame, boxes, score_min):
        i = int(frame[0, 0, 0])
        t, wfrac = self.spec[i]
        out = []
        for b in boxes:
            x0, y0, x1, y1 = (int(v) for v in b)
            m = np.zeros((300, 300), np.uint8)
            bh, bw = max(y1 - y0, 1), max(x1 - x0, 1)
            m[y0:max(y0, int(y1 - t * bh)), x0:x0 + max(1, int(bw * wfrac))] = 1
            out.append(m)
        return out


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
    """Stays put -- and SAYS it stayed put.

    A silent return here is what made the 8 Sep GPU arm unreadable: an arm that
    changed nothing could equally mean the scan never ran or the scan ran and
    declined, and the run log could not tell the two apart.
    """
    from vbgr.seed import pick_seed_frame
    idx, notes = pick_seed_frame(_frames(6), _Det(), _TailSeeder([0.05, 0.02, 0.02, 0.02, 0.02, 0.02]),
                                 _Cfg(), _select)
    assert idx == 0
    assert any("not scanning" in n for n in notes), notes


def test_lookahead_declining_is_never_silent():
    """Every enabled outcome leaves a note; only the disabled path is silent."""
    from vbgr.seed import pick_seed_frame
    for tails in ([0.05] * 6,                       # frame 0 healthy
                  [0.40, 0.39, 0.41, 0.40, 0.38, 0.40],   # persistently occluded
                  [0.33, 0.08, 0.07, 0.05, 0.33, 0.05]):  # moves
        _idx, notes = pick_seed_frame(_frames(6), _Det(), _TailSeeder(tails),
                                      _Cfg(), _select)
        assert notes, tails


def test_lookahead_stays_for_a_persistently_occluded_subject():
    """The desk case: equally occluded on every frame, so no gain, no move."""
    from vbgr.seed import pick_seed_frame
    idx, _ = pick_seed_frame(_frames(6), _Det(), _TailSeeder([0.40, 0.39, 0.41, 0.40, 0.38, 0.40]),
                             _Cfg(), _select)
    assert idx == 0


def test_lookahead_is_on_by_default():
    """Turned on 8 Sep after the 79-shot sweep. See config.py for the numbers."""
    from vbgr.config import Config
    assert Config().seed.lookahead_frames == 4


def test_a_seeder_that_cannot_measure_tails_does_not_crash():
    """The default is on, so a backend without `_match` must decline, not raise.

    Only MaskRCNNSeeder implements it; SAM3Seeder does not.
    """
    from vbgr.seed import pick_seed_frame

    class Bare:                                  # no _match
        pass
    idx, notes = pick_seed_frame(_frames(6), _Det(), Bare(), _Cfg(), _select)
    assert idx == 0
    assert any("cannot measure mask tails" in n for n in notes), notes


def test_lookahead_disabled_returns_frame_zero():
    from vbgr.seed import pick_seed_frame

    class Off(_Cfg):
        lookahead_frames = 0
    idx, notes = pick_seed_frame(_frames(6), _Det(), _TailSeeder([0.33] * 6), Off(), _select)
    assert idx == 0 and notes == []


class _CastDet:
    """Detector stub whose kept-cast SHRINKS on later frames (butter shot 5)."""
    def __init__(self, counts):
        self.counts = counts

    def detect(self, frame):
        i = int(frame[0, 0, 0])
        out = []
        for n in range(self.counts[i]):
            class D:
                pass
            d = D()
            d.box = (10 + 20 * n, 40, 60 + 20 * n, 260)
            out.append(d)
        return out, []


def test_lookahead_will_not_trade_a_subject_for_a_cleaner_mask():
    """butter shot 5: frame 0 fails completely but holds six; frame 3 is clean
    and holds four. Repairing the mask there costs two dancers, so it stays."""
    from vbgr.seed import pick_seed_frame
    idx, notes = pick_seed_frame(
        _frames(6), _CastDet([6, 5, 5, 4, 4, 4]),
        _TailSeeder([1.00, 0.030, 0.025, 0.023, 0.023, 0.023]),
        _Cfg(), _select)
    assert idx == 0, notes
    assert any("subjects" in n for n in notes), notes


def test_the_cast_guard_can_be_turned_off():
    from vbgr.seed import pick_seed_frame

    class NoGuard(_Cfg):
        lookahead_require_same_cast = False
    idx, _ = pick_seed_frame(
        _frames(6), _CastDet([6, 5, 5, 4, 4, 4]),
        _TailSeeder([1.00, 0.030, 0.025, 0.023, 0.023, 0.023]),
        NoGuard(), _select)
    assert idx != 0


def test_the_cast_guard_leaves_a_steady_cast_alone():
    """bilibili shot 11: one subject throughout, so the guard never fires."""
    from vbgr.seed import pick_seed_frame
    idx, notes = pick_seed_frame(
        _frames(6), _CastDet([1, 1, 1, 1, 1, 1]),
        _TailSeeder([0.333, 0.081, 0.071, 0.054, 0.054, 0.054]),
        _Cfg(), _select)
    assert idx == 3, notes


def test_the_guard_defaults_on():
    from vbgr.config import Config
    assert Config().seed.lookahead_require_same_cast is True


def test_fill_does_not_discriminate_on_bilibilis_shot_and_the_tail_does():
    """The 10 Sep negative result, pinned so nobody re-tries it blind.

    bilibili shot 11's first four frames, measured on an A100: fills 0.3325,
    0.3759, 0.3908, 0.4085 against tails 0.3326, 0.0805, 0.0707, 0.0536.  Fill
    ranks the SAME frame best and by only 0.076, which does not clear
    lookahead_min_gain -- so ranking by fill stops the shot moving at all.  A
    missing boot is a small share of a standing person's box.
    """
    from vbgr.seed import pick_seed_frame
    spec = [(0.3326, 0.82), (0.0805, 0.93), (0.0707, 0.96), (0.0536, 1.00)]
    idx, notes = pick_seed_frame(_frames(4), _Det(), _SplitSeeder(spec),
                                 _Cfg(), _select)
    assert idx == 3, notes          # the tail still moves it, and must


def test_seed_box_fill_clips_to_the_box():
    from vbgr.seed import seed_box_fill
    m = np.zeros((300, 300), np.uint8)
    m[0:300, 0:300] = 1
    assert seed_box_fill(m, (10, 40, 60, 260)) == 1.0
    assert seed_box_fill(np.zeros((300, 300), np.uint8), (10, 40, 60, 260)) == 0.0
    assert seed_box_fill(None, (10, 40, 60, 260)) == 0.0
