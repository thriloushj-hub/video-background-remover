"""Nested-duplicate seed suppression.

The `interview` clip was the only clip in the 15-clip benchmark set to score a
real `hole_big` (0.055): a band torn down through the woman's face and hair,
while `area_cv` 0.0175, `dropouts` 0 and `subj_lost_at never` all read it
clean.  The cause is in the seed, not the matting.  Its three seeds are the
man, the woman, and *the woman's front again* -- `seed_2` is 97.8% inside
`seed_1`.  One obj_id is seeded per kept detection, so SAM2 propagated the
woman against herself under its non-overlap constraint and both objects
retreated from the shared boundary.

Every case below is built from the geometry of a real clip in the set.
"""
import numpy as np

from vbgr.detect import (Detection, box_containment, drop_nested_duplicates,
                         mask_containment)

H, W = 1080, 1920


def _d(x1, y1, x2, y2, conf=0.9):
    return Detection(box=(x1, y1, x2, y2), conf=conf, cls=0, label="person")


def _mask(x1, y1, x2, y2):
    m = np.zeros((H, W), np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_interview_nested_duplicate_is_dropped():
    """seed_1 (whole woman) and seed_2 (her front) -- keep the whole one."""
    whole = _d(197, 271, 879, 1066)
    front = _d(515, 322, 891, 1047)
    man = _d(898, 11, 1914, 1071)
    dets = [man, whole, front]
    masks = [_mask(*man.box), _mask(*whole.box), _mask(515, 322, 879, 1047)]

    kept, dropped = drop_nested_duplicates(dets, masks)

    assert dropped == [front]
    assert kept == [man, whole]


def test_two_people_side_by_side_both_survive():
    """microsoft / codylexi / ipman: neither box is inside the other."""
    left = _d(200, 200, 700, 1000)
    right = _d(650, 210, 1150, 1010)
    masks = [_mask(*left.box), _mask(*right.box)]

    kept, dropped = drop_nested_duplicates([left, right], masks)

    assert dropped == []
    assert kept == [left, right]


def test_occluded_near_person_inside_a_far_persons_box_survives():
    """The case a box-only test would get wrong.

    A small person standing in front of a large one can be box-contained.
    Their masks are not: occlusion means the near person owns exactly the
    pixels the far person does not.
    """
    big = _d(200, 100, 1400, 1070)
    near = _d(600, 500, 900, 1050)
    masks = [_mask(200, 100, 1400, 1070) & ~_mask(600, 500, 900, 1050),
             _mask(600, 500, 900, 1050)]

    assert box_containment(near.box, big.box) > 0.95      # box test alone fires
    assert mask_containment(masks[1], masks[0]) == 0.0    # mask test does not

    kept, dropped = drop_nested_duplicates([big, near], masks)

    assert dropped == []
    assert kept == [big, near]


def test_containment_is_not_iou():
    """butter's closest pair (0.283) must stay far below the 0.80 threshold."""
    a = _d(300, 200, 800, 1000)
    b = _d(700, 220, 1200, 1020)
    assert box_containment(a.box, b.box) < 0.30
    assert box_containment(b.box, a.box) < 0.30


def test_masks_optional_falls_back_to_boxes():
    whole = _d(197, 271, 879, 1066)
    front = _d(515, 322, 891, 1047)

    kept, dropped = drop_nested_duplicates([whole, front])

    assert dropped == [front]
    assert kept == [whole]


def test_single_detection_is_untouched():
    only = _d(100, 100, 500, 900)
    kept, dropped = drop_nested_duplicates([only], [_mask(*only.box)])
    assert kept == [only] and dropped == []


def test_the_larger_survives_regardless_of_input_order():
    whole = _d(197, 271, 879, 1066)
    front = _d(515, 322, 891, 1047)
    masks_fw = [_mask(515, 322, 879, 1047), _mask(*whole.box)]

    kept, dropped = drop_nested_duplicates([front, whole], masks_fw)

    assert kept == [whole] and dropped == [front]
