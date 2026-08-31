"""The two gate changes that came out of 3.3a.

Both exist because the same failure happened twice in one clip set: a test that
was really asking "is this box the right SHAPE" was being used to answer "is
this a person the matte is missing".

1. An **absolute** confidence floor beside the relative one.  eddie's phantom
   was a blurred portrait on a wall at conf 0.277.  It cleared the relative
   floor (0.15 * 0.94 = 0.141), the height gate (0.952 of the tallest) and the
   area gate (which exempts boxes near a frame edge).
2. **Mask** coverage confirming **box** coverage.  Box fill is pose-dependent;
   over the 30 persisted seeds of correctly-held subjects it runs 0.302..0.738
   against a 0.35 threshold, while mask coverage runs 0.802..0.975.
"""
from __future__ import annotations

import numpy as np

from vbgr.detect import Detection, select_person_boxes
from vbgr.reid import box_coverage, mask_coverage, confirm_uncovered


def _d(x1, y1, x2, y2, c=0.9):
    return Detection((x1, y1, x2, y2), c, 0, "person")


# --------------------------------------------------------------------------- #
# 1. absolute confidence floor
# --------------------------------------------------------------------------- #

def test_eddie_phantom_is_dropped_by_the_absolute_floor():
    """The real case, at the real numbers.

    A full-height, low-confidence detection that every relative gate passes.
    """
    subject = _d(148, 4, 1403, 711, 0.94)          # h 707
    portrait = _d(1497, 42, 1906, 715, 0.277)      # h 673 -> 0.952 of tallest
    kept, dropped = select_person_boxes([subject, portrait], 1920, 720,
                                        0.60, 0.15, 0.20, 0.50)
    assert subject in kept
    assert portrait in dropped, "a 0.277 detection cleared every gate"


def test_the_relative_floor_alone_would_have_kept_it():
    """Guards the premise: without an absolute floor this is a false negative."""
    subject = _d(148, 4, 1403, 711, 0.94)
    portrait = _d(1497, 42, 1906, 715, 0.277)
    kept, _ = select_person_boxes([subject, portrait], 1920, 720,
                                  0.60, 0.15, 0.20, 0.0)
    assert portrait in kept


def test_lowest_genuine_detection_in_the_set_survives():
    """butter f32 at 0.7555 is the lowest-confidence genuine detection over all
    284 kept detections in the 15-clip set.  The floor must not touch it."""
    lead = _d(100, 100, 400, 1000, 0.95)
    genuine = _d(900, 110, 1150, 990, 0.7555)
    kept, _ = select_person_boxes([lead, genuine], 1920, 1080,
                                  0.60, 0.15, 0.20, 0.50)
    assert genuine in kept


def test_floor_does_not_break_the_never_empty_guarantee():
    kept, _ = select_person_boxes([_d(0, 0, 300, 900, 0.31)], 1920, 1080,
                                  0.60, 0.15, 0.20, 0.50)
    assert len(kept) == 1


# --------------------------------------------------------------------------- #
# 2. mask coverage vs box coverage
# --------------------------------------------------------------------------- #

def _spread_pose(H=400, W=960):
    """A held subject whose alpha fills little of its own bounding box.

    A torso with two outstretched arms: the bounding box is wide and mostly
    empty, which is exactly the geometry that produced 19 phantom events.
    """
    alpha = np.zeros((H, W), np.float32)
    alpha[120:300, 460:500] = 1.0        # torso
    alpha[150:170, 300:660] = 1.0        # arms
    mask = alpha > 0.5
    return alpha, mask


def test_box_coverage_fires_on_a_correctly_held_spread_pose():
    """The premise of the fix: the cheap test calls this subject uncovered."""
    alpha, mask = _spread_pose()
    ys, xs = np.where(mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    assert box_coverage(alpha, box) < 0.35


def test_mask_coverage_does_not():
    """Same frame, same subject, same matte -- the mask says it is held."""
    alpha, mask = _spread_pose()
    assert mask_coverage(alpha, mask) > 0.99


def test_confirmation_rejects_the_pose_false_positive():
    alpha, mask = _spread_pose()
    assert confirm_uncovered(alpha, [mask], 0.50) == []


def test_confirmation_keeps_a_genuinely_missing_subject():
    """Someone the matte is not holding at all still fires, as they must."""
    alpha, _ = _spread_pose()
    missing = np.zeros_like(alpha, bool)
    missing[100:350, 800:900] = True          # nowhere near any alpha
    assert confirm_uncovered(alpha, [missing], 0.50) == [0]


def test_mostly_unheld_subject_still_fires():
    alpha, _ = _spread_pose()
    half = np.zeros_like(alpha, bool)
    half[120:300, 490:590] = True             # only its left edge is held
    cov = mask_coverage(alpha, half)
    assert 0.0 < cov < 0.50, cov
    assert confirm_uncovered(alpha, [half], 0.50) == [0]


def test_a_mask_that_could_not_be_produced_is_not_treated_as_uncovered():
    """A failure to segment is not evidence of a missing subject."""
    alpha, _ = _spread_pose()
    assert confirm_uncovered(alpha, [None], 0.50) == []


def test_mask_coverage_resizes_a_mask_at_source_resolution():
    """Seeder masks come back at frame resolution; alphas are scored smaller."""
    alpha, mask = _spread_pose(400, 960)
    big = np.kron(mask, np.ones((2, 2), bool))     # 800x1920
    assert big.shape == (800, 1920)
    assert mask_coverage(alpha, big) > 0.95


def test_empty_mask_is_never_uncovered():
    alpha, _ = _spread_pose()
    assert mask_coverage(alpha, np.zeros_like(alpha, bool)) == 1.0
