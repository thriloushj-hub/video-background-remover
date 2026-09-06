"""5.5 -- the seeder must not silently drop a subject the scan already found.

The failure this pins down, from the full-length 1917 run: the re-entry scan
finds the blurred runner every eight frames and prints `cov=0.000`, ten local
re-matte passes fire -- and the matte never changes, because `build_seed` asks
Mask R-CNN for a mask inside his box, his 0.80 score head says nothing is
there, and a subject with no mask is skipped. The pass then re-mattes exactly
the cast it already had.

So the tests here are about scope as much as behaviour. A relaxed threshold
that applied to every box would be a different product; these check that it
fires only for boxes the scan confirmed, only after the strict pass failed on
that box, and never in a way that attaches an instance to a box it does not
overlap.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vbgr.config import SeedConfig                      # noqa: E402
from vbgr.detect import Detection                       # noqa: E402
from vbgr.pipeline import _with_must_boxes              # noqa: E402
from vbgr.seed import MaskRCNNSeeder, build_seed        # noqa: E402

H, W = 80, 120
SHARP = (10, 10, 40, 70)      # clears 0.80
BLUR = (70, 10, 110, 70)      # the motion-blurred entrant, 0.55
FAR = (0, 0, 6, 6)            # a low-score instance nowhere near BLUR


def _mask(box):
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


class FakeInstances:
    """Stands in for Mask R-CNN. Honours score_min, which is the whole point."""

    def __init__(self, items):
        self.items = items                # [(score, box)]
        self.thresholds = []

    def instances(self, frame_bgr, score_min=0.80):
        self.thresholds.append(score_min)
        keep = [(s, b) for s, b in self.items if s > score_min]
        masks = np.array([_mask(b) for s, b in keep]) if keep else \
            np.zeros((0, H, W), bool)
        boxes = np.array([list(map(float, b)) for s, b in keep]) if keep else \
            np.zeros((0, 4), float)
        return masks, boxes


def _seeder(items=((0.95, SHARP), (0.55, BLUR)), **kw):
    return MaskRCNNSeeder(detector=FakeInstances(list(items)), **kw)


def _frame():
    return np.zeros((H, W, 3), np.uint8)


# --------------------------------------------------------------------- #
# the failure, reproduced
# --------------------------------------------------------------------- #

def test_blurred_subject_is_dropped_without_recovery():
    s = _seeder()
    out = s.masks_from_boxes(_frame(), [SHARP, BLUR])
    assert len(out) == 1
    assert s.last_aligned[0] is not None and s.last_aligned[1] is None
    assert any("no mask instance matched" in n for n in s.notes)
    assert s._det.thresholds == [None] or s._det.thresholds == [0.80]


def test_recovery_puts_him_back():
    s = _seeder()
    out = s.masks_from_boxes(_frame(), [SHARP, BLUR], recover_boxes=[BLUR])
    assert len(out) == 2
    assert s.last_aligned[1] is not None
    assert any("recovered box" in n for n in s.notes)
    # and it said so: the second pass ran at the recovery threshold
    assert 0.35 in s._det.thresholds


# --------------------------------------------------------------------- #
# scope -- the part that matters more than the fix
# --------------------------------------------------------------------- #

def test_recovery_only_touches_the_boxes_it_was_given():
    s = _seeder()
    out = s.masks_from_boxes(_frame(), [SHARP, BLUR], recover_boxes=[SHARP])
    assert len(out) == 1, "a box outside recover_boxes must stay skipped"
    assert s.last_aligned[1] is None


def test_recovery_still_obeys_match_iou():
    """A low-score instance somewhere else is not a mask for this box."""
    s = _seeder(items=((0.95, SHARP), (0.55, FAR)))
    out = s.masks_from_boxes(_frame(), [SHARP, BLUR], recover_boxes=[BLUR])
    assert len(out) == 1
    assert s.last_aligned[1] is None


def test_recovery_can_be_switched_off():
    s = _seeder(recover=False)
    out = s.masks_from_boxes(_frame(), [SHARP, BLUR], recover_boxes=[BLUR])
    assert len(out) == 1


def test_a_box_matching_nothing_is_never_filled_with_its_rectangle():
    """The class docstring's promise, still true with recovery on."""
    s = _seeder(items=((0.95, SHARP),))
    s.masks_from_boxes(_frame(), [SHARP, BLUR], recover_boxes=[BLUR])
    assert s.last_aligned[1] is None


# --------------------------------------------------------------------- #
# build_seed carries it through -- this is where the subject was lost
# --------------------------------------------------------------------- #

def _cfg():
    c = SeedConfig()
    c.concept_text_detect = False      # Mask R-CNN has no text head
    c.gate_background_regions = False  # needs real pixels
    c.fill_mask_holes = False
    return c


def _dets():
    return [Detection(box=SHARP, conf=0.9, cls=0),
            Detection(box=BLUR, conf=0.9, cls=0)]


def test_build_seed_drops_the_entrant_without_recover_boxes():
    seed = build_seed(_frame(), _seeder(), _dets(), (), _cfg())
    assert len(seed.per_person) == 1
    assert seed.mask[40, 90] == 0, "the entrant is not in the fused seed"


def test_build_seed_keeps_him_when_the_scan_confirmed_him():
    seed = build_seed(_frame(), _seeder(), _dets(), (), _cfg(),
                      recover_boxes=[BLUR])
    assert len(seed.per_person) == 2
    assert seed.mask[40, 90] == 1
    assert any("recovered box" in n for n in seed.notes)


def test_a_seeder_without_the_hook_is_not_handed_the_argument():
    """SAM3Seeder has no recover support; build_seed must not crash on it."""
    class Old:
        supports_recover = False
        notes = ()

        def masks_from_boxes(self, frame, boxes):
            return [_mask(b).astype(np.uint8) for b in boxes[:1]]

        def masks_from_text(self, frame, text):
            raise NotImplementedError

    seed = build_seed(_frame(), Old(), _dets(), (), _cfg(), recover_boxes=[BLUR])
    assert len(seed.per_person) == 1


# --------------------------------------------------------------------- #
# the other half: the local pass must not re-gate the scan's own finding
# --------------------------------------------------------------------- #

def test_must_box_is_added_back_when_the_fresh_gate_drops_it():
    kept = [Detection(box=SHARP, conf=0.9, cls=0)]
    people = kept + [Detection(box=BLUR, conf=0.4, cls=0)]
    out = _with_must_boxes(kept, people, [BLUR])
    assert [d.box for d in out] == [SHARP, BLUR]
    assert out[1].conf == pytest.approx(0.4), "keep the detector's own confidence"


def test_must_box_already_covered_is_not_duplicated():
    kept = [Detection(box=SHARP, conf=0.9, cls=0)]
    assert len(_with_must_boxes(kept, kept, [SHARP])) == 1


def test_must_box_the_detector_never_saw_is_still_added():
    kept = [Detection(box=SHARP, conf=0.9, cls=0)]
    out = _with_must_boxes(kept, kept, [BLUR])
    assert len(out) == 2 and out[1].label == "person"
