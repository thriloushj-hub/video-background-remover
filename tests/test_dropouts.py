"""Regression tests for the rewritten dropout metric.

Written after the old total-area rule was found to be counting three things
that are not dropouts -- a different subject arriving, a subject receding, and
a shot cut -- across the whole benchmark set.  Each negative below is a
synthetic stand-in for one of those, so the failure modes cannot come back
silently.

The positive case is synthetic on purpose and that is a real limitation: there
is no confirmed instance of a genuine limb key-out-and-return anywhere in the
frozen clip set.  See Dropouts_And_The_ReEntry_Premise.md.
"""
import numpy as np
import cv2

from bench.metrics import dropout_events, track_dominant

H, W = 200, 320


def _blob(cx, cy, rx, ry):
    a = np.zeros((H, W), np.float32)
    cv2.ellipse(a, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    return a


def _subject(cx, limb=True, s=1.0):
    """A body plus a limb worth ~40% of the silhouette."""
    a = _blob(cx, 110, int(22 * s), int(40 * s))
    if limb:
        cv2.ellipse(a, (cx + int(46 * s), 122),
                    (int(48 * s), int(15 * s)), 0, 0, 360, 1.0, -1)
    return a


def test_limb_keys_out_and_returns_is_counted():
    """The failure the metric exists for."""
    seq = [_subject(120, limb=not (40 <= i < 46)) for i in range(90)]
    n, events = dropout_events(seq)
    assert n == 1, f"expected 1 dropout, got {n} at {events}"
    assert 38 <= events[0] <= 42


def test_limb_must_be_big_enough_to_matter():
    """A small limb is below drop=0.72 by construction, and that is intended."""
    seq = []
    for i in range(90):
        a = _blob(120, 110, 22, 40)
        if not (40 <= i < 46):
            cv2.ellipse(a, (150, 122), (18, 6), 0, 0, 360, 1.0, -1)
        seq.append(a)
    assert dropout_events(seq)[0] == 0


def test_receding_subject_plus_newcomer_is_not_a_dropout():
    """1917's four false positives in one picture.

    The tracked subject shrinks steadily and a *second* subject walks in, which
    under the old total-area rule supplied the 'recovery'.
    """
    seq = []
    for i in range(90):
        s = 1.0 - 0.011 * i
        f = _subject(90, True, s)
        if i >= 45:
            f = np.maximum(f, _blob(255, 110, 22, 40))
        seq.append(f)
    assert dropout_events(seq)[0] == 0


def test_component_split_alone_is_not_a_dropout():
    """interview f39: one blob becomes two, total area barely moves.

    Tracking only the larger fragment reported a 66% drop that never happened.
    """
    seq = []
    for i in range(90):
        f = _subject(120, True, 1.0)
        if 40 <= i < 46:
            f[:, 138:142] = 0.0        # sever the link, keep the pixels
        seq.append(f)
    assert dropout_events(seq)[0] == 0


def test_shot_cut_is_excluded_when_declared():
    """butter's only event ran 271->289 and both frames are cuts."""
    seq = [_subject(120, True, 1.0 if i < 45 else 0.5) for i in range(95)]
    seq = [f if i < 70 else _subject(120, True, 1.0) for i, f in enumerate(seq)]
    assert dropout_events(seq, cuts=[45, 70])[0] == 0


def test_track_follows_the_subject_not_the_stranger():
    """A newcomer must never become the tracked subject."""
    seq = []
    for i in range(40):
        f = _subject(80, True, 1.0)
        if i >= 20:
            f = np.maximum(f, _blob(270, 110, 30, 46))   # bigger stranger
        seq.append(f)
    areas, present = track_dominant(seq)
    assert present.all()
    # the tracked area must not jump when the larger stranger appears
    assert abs(areas[25] - areas[15]) / areas[15] < 0.10


def test_empty_and_short_inputs_do_not_crash():
    assert dropout_events([np.zeros((H, W), np.float32)] * 5) == (0, [])
    assert dropout_events([np.zeros((H, W), np.float32)] * 40)[0] == 0
