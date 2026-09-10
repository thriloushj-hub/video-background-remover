"""5.9 -- re-binarising a truncated instance mask at a lower cutoff.

The failure this exists for: on `bilibili` the DETECTION is right (a box on the
whole figure at 0.93 confidence) and the instance mask inside it stops at the
ankle, because a dark boot on a dark floor scores under the 0.5 cutoff that
turns Mask R-CNN's soft output into a binary mask.  The shot is seeded from one
frame, so the subject has one foot for all sixteen of its frames.

5.7's look-ahead cannot help: measured on a GPU, no frame in the window has a
seed holding both boots.  This changes the cutoff instead of the frame.
"""
import numpy as np

from vbgr.seed import MaskRCNNSeeder, seed_tail_fraction

BOX = (100.0, 100.0, 200.0, 400.0)


class _SoftDet:
    """Mask R-CNN stand-in whose boot only exists below a lower mask cutoff.

    `spill` is the dangerous case: the lower cutoff also drags in a slab of
    background OUTSIDE the box, which is what a leak looks like and what the
    guard has to refuse.
    """

    def __init__(self, spill=False):
        self.spill = spill
        self.calls = []

    def instances(self, frame, score_thr=0.8, mask_thr=0.5):
        self.calls.append((score_thr, mask_thr))
        m = np.zeros((500, 500), bool)
        m[100:300, 100:200] = True                  # body, at any cutoff
        if mask_thr <= 0.30:
            m[300:400, 100:200] = True              # the boot
            if self.spill:
                m[300:400, 260:460] = True          # ...and background with it
        return np.array([m]), np.array([BOX], float)


def _frame():
    return np.zeros((500, 500, 3), np.uint8)


def test_off_by_default_the_mask_still_stops_at_the_ankle():
    s = MaskRCNNSeeder(detector=_SoftDet())
    out = s.masks_from_boxes(_frame(), [BOX])
    assert seed_tail_fraction(out[0], BOX) > 0.30
    assert not any("re-binarised" in n for n in s.notes), s.notes


def test_the_lower_cutoff_recovers_the_boot():
    s = MaskRCNNSeeder(detector=_SoftDet(), mask_recover=True)
    out = s.masks_from_boxes(_frame(), [BOX])
    assert seed_tail_fraction(out[0], BOX) == 0.0
    assert any("re-binarised" in n for n in s.notes), s.notes


def test_a_mask_that_spills_outside_the_box_is_refused():
    """The whole safety argument: reaching further is not enough on its own."""
    s = MaskRCNNSeeder(detector=_SoftDet(spill=True), mask_recover=True)
    out = s.masks_from_boxes(_frame(), [BOX])
    assert seed_tail_fraction(out[0], BOX) > 0.30      # the short mask is kept
    assert any("declined" in n for n in s.notes), s.notes


def test_a_healthy_mask_is_never_retried():
    class _Healthy(_SoftDet):
        def instances(self, frame, score_thr=0.8, mask_thr=0.5):
            self.calls.append((score_thr, mask_thr))
            m = np.zeros((500, 500), bool)
            m[100:400, 100:200] = True
            return np.array([m]), np.array([BOX], float)

    d = _Healthy()
    s = MaskRCNNSeeder(detector=d, mask_recover=True)
    s.masks_from_boxes(_frame(), [BOX])
    assert len(d.calls) == 1, d.calls          # one pass, no second forward
    assert not any("5.9" in n for n in s.notes), s.notes


def test_the_cutoff_reaches_the_detector():
    d = _SoftDet()
    s = MaskRCNNSeeder(detector=d, mask_recover=True,
                       mask_binarise=0.5, mask_recover_binarise=0.25)
    s.masks_from_boxes(_frame(), [BOX])
    assert [c[1] for c in d.calls] == [0.5, 0.25], d.calls


def test_the_default_config_leaves_it_off():
    from vbgr.config import SeedConfig
    c = SeedConfig()
    assert c.maskrcnn_mask_recover is False
    assert c.maskrcnn_mask_binarise == 0.5
