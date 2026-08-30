"""Re-seed scan: which detections are people we are not already holding.

The benchmark seeds once at frame 0, so anyone who walks into shot afterwards
is invisible to it forever. On the 27 Aug run butter loses three dancers to the
camera pull-back and dance loses one, and all four scored as v1 "halo" -- v1
holds them, we don't. See Halo_Was_Missing_Subjects.

The test that matters is the third one: a re-detection of somebody we already
hold must NOT come back as new, or every scan doubles the object count.
"""
import numpy as np

from vbgr.detect import new_subjects

H, W = 200, 400


def _mask(x1, y1, x2, y2):
    m = np.zeros((H, W), np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def _alpha(*boxes):
    a = np.zeros((H, W), np.float32)
    for x1, y1, x2, y2 in boxes:
        a[y1:y2, x1:x2] = 1.0
    return a


def test_a_dancer_who_walks_into_shot_is_new():
    held = _alpha((40, 60, 100, 160))
    entrant = _mask(300, 60, 360, 160)

    assert new_subjects([entrant], held) == [0]


def test_someone_we_already_hold_is_not_new():
    """The one that stops every scan doubling the object count."""
    held = _alpha((40, 60, 100, 160))
    same_person_again = _mask(42, 62, 98, 158)

    assert new_subjects([same_person_again], held) == []


def test_a_person_standing_partly_behind_one_we_hold_is_still_new():
    """Coverage of the detection, not IoU: most of their pixels are outside."""
    held = _alpha((40, 60, 200, 160))
    behind = _mask(180, 60, 300, 160)          # ~17% of them is inside

    assert new_subjects([behind], held) == [0]


def test_specks_are_never_seeded():
    held = _alpha((40, 60, 100, 160))
    speck = _mask(300, 60, 303, 63)

    assert new_subjects([speck], held) == []


def test_mixed_batch_returns_only_the_new_ones_in_order():
    held = _alpha((40, 60, 100, 160))
    ms = [_mask(42, 62, 98, 158),      # already held
          _mask(300, 60, 360, 160),    # new
          _mask(150, 60, 210, 160)]    # new

    assert new_subjects(ms, held) == [1, 2]


def test_float_alpha_and_uint8_masks_both_work():
    held = _alpha((40, 60, 100, 160))
    as_float = (_mask(300, 60, 360, 160) > 127).astype(np.float32)

    assert new_subjects([as_float], held) == [0]


def test_an_empty_matte_makes_everyone_new():
    held = np.zeros((H, W), np.float32)
    ms = [_mask(40, 60, 100, 160), _mask(300, 60, 360, 160)]

    assert new_subjects(ms, held) == [0, 1]
