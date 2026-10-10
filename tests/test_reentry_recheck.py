"""The dance wall: a repair pass that its own defect switches off.

1 Oct 2026.  `_recover_entrants` scans for subjects the matte is not holding
(step 1, mask-confirmed), then re-matte a window around each (step 3).  Step 3
used to re-test its pending frames with BOX coverage.  When a cast member is
dropped, the NEIGHBOURING object's matte spreads across the gap between them
and fills her bounding box with background -- so the box test reads her as
covered and the repair never fires.

Measured on `bench/results/run_2026-09-29_hard/dance`: 11 of 11 boxes flagged
at 0.004..0.071 box coverage during the scan read 0.386..0.616 in the final
alpha, every one above `reid.max_overlap` (0.35).
"""
import numpy as np

from vbgr.config import ReIDConfig
from vbgr.reid import box_coverage, uncovered_boxes, confirm_uncovered


def _scene():
    """A 200x200 frame: a held subject whose matte spills across the gap into
    a dropped neighbour's box, and the dropped neighbour's own silhouette."""
    alpha = np.zeros((200, 200), np.float32)
    alpha[40:170, 20:130] = 1.0          # the neighbour, matted, spreading right
    her_box = (100, 40, 160, 170)        # the dropped subject's detection box
    her_mask = np.zeros((200, 200), bool)
    her_mask[40:170, 135:160] = True     # her actual silhouette: NOT matted
    return alpha, her_box, her_mask


def test_box_test_is_fooled_by_the_neighbours_spill():
    alpha, her_box, _ = _scene()
    cfg = ReIDConfig()
    cov = box_coverage(alpha, her_box)
    assert cov > cfg.max_overlap, cov          # reads as "covered"
    assert uncovered_boxes(alpha, [her_box],
                           cfg.max_overlap, cfg.min_area_frac) == []


def test_mask_recheck_still_sees_her():
    alpha, _, her_mask = _scene()
    cfg = ReIDConfig()
    # none of her own silhouette is matted, so the mask test must flag her
    assert confirm_uncovered(alpha, [her_mask], cfg.mask_max_overlap) == [0]


def test_recheck_with_mask_defaults_on():
    assert ReIDConfig().recheck_with_mask is True


def test_a_genuinely_held_subject_is_not_flagged():
    """The fix must not re-open the phantom re-entry events of 3.3a."""
    alpha = np.zeros((200, 200), np.float32)
    alpha[40:170, 135:160] = 1.0
    held = np.zeros((200, 200), bool)
    held[40:170, 135:160] = True
    assert confirm_uncovered(alpha, [held], ReIDConfig().mask_max_overlap) == []
