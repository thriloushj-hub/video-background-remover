"""The pipeline seeds from Mask R-CNN (5.1k, decided 30 Aug).

Two reasons, and the second is the real one. Neither SAM 3 backend is
loadable -- `SAM("sam3.pt")` raises FileNotFoundError because those weights
are not in ultralytics' auto-download set, and `sam3.build_sam` does not exist
in the fork we install. And Mask R-CNN is where **every validated seed in this
project came from**: `vbgr_bench.py` has always seeded from it and never
touched `pipeline.py`, so the area gate, one-object-id-per-person and
nested-duplicate suppression were all measured on these masks.

The case that matters most is the last one: a box no instance matches is
skipped with a note, never filled with its rectangle. A rectangle seed hands
the tracker a slab of background and looks like a working seed.
"""
import numpy as np
import pytest

from vbgr.config import SeedConfig
from vbgr.seed import MaskRCNNSeeder, build_seeder

H, W = 120, 200


class _StubDetector:
    """Stands in for torchvision maskrcnn via the ``instances`` hook.

    That hook is why these run without torch: the part worth testing is the
    box-to-instance matching, not the tensor plumbing.
    """

    def __init__(self, boxes, masks):
        self.boxes = np.array(boxes, np.float32).reshape(-1, 4)
        self.masks = (np.stack(masks) > 0.5) if masks else np.zeros((0, H, W), bool)

    def instances(self, _frame):
        return self.masks, self.boxes


def _mask(x1, y1, x2, y2):
    m = np.zeros((H, W), np.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


def _frame():
    return np.zeros((H, W, 3), np.uint8)


def test_a_box_gets_the_mask_of_the_instance_it_overlaps():
    det = _StubDetector([[10, 10, 60, 100], [120, 10, 180, 100]],
                        [_mask(10, 10, 60, 100), _mask(120, 10, 180, 100)])
    s = MaskRCNNSeeder(detector=det)

    out = s.masks_from_boxes(_frame(), [(120, 10, 180, 100)])

    assert len(out) == 1
    ys, xs = np.nonzero(out[0])
    assert xs.min() >= 120 and xs.max() < 180


def test_two_boxes_never_take_the_same_instance():
    det = _StubDetector([[10, 10, 60, 100], [12, 12, 58, 98]],
                        [_mask(10, 10, 60, 100), _mask(12, 12, 58, 98)])
    s = MaskRCNNSeeder(detector=det)

    out = s.masks_from_boxes(_frame(), [(10, 10, 60, 100), (12, 12, 58, 98)])

    assert len(out) == 2
    assert not np.array_equal(out[0], out[1])


def test_an_unmatched_box_is_skipped_with_a_note_not_a_rectangle():
    """The one that matters: a rectangle seed is a slab of background."""
    det = _StubDetector([[10, 10, 60, 100]], [_mask(10, 10, 60, 100)])
    s = MaskRCNNSeeder(detector=det)

    out = s.masks_from_boxes(_frame(), [(150, 10, 190, 100)])

    assert out == []
    assert len(s.notes) == 1 and "skipped" in s.notes[0]


def test_no_boxes_is_not_an_error():
    s = MaskRCNNSeeder(detector=_StubDetector([], []))
    assert s.masks_from_boxes(_frame(), []) == []


def test_text_and_point_prompts_report_unavailable_rather_than_lying():
    s = MaskRCNNSeeder(detector=_StubDetector([], []))
    with pytest.raises(NotImplementedError):
        s.masks_from_text(_frame())
    with pytest.raises(NotImplementedError):
        s.masks_from_points(_frame())


def test_the_default_backend_is_maskrcnn():
    assert SeedConfig().backend == "maskrcnn"
    assert isinstance(build_seeder(SeedConfig()), MaskRCNNSeeder)


def test_sam3_is_still_selectable_even_though_it_does_not_load():
    from vbgr.seed import SAM3Seeder
    cfg = SeedConfig()
    cfg.backend = "sam3"
    assert isinstance(build_seeder(cfg), SAM3Seeder)


def test_an_unknown_backend_is_rejected_loudly():
    cfg = SeedConfig()
    cfg.backend = "nope"
    with pytest.raises(ValueError):
        build_seeder(cfg)
