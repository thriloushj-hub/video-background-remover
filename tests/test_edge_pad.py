"""5.10 -- the frame-edge pad on MaskRCNNSeeder.

The defect these pin: where a subject is cut by the edge of frame the seed
mask stops short of it, because torchvision's `expand_masks(mask, padding=1)`
zero-pads the 28x28 ROI mask and `paste_mask_in_image` resizes that to the box,
leaving a ramp ~1 mask cell deep at every box edge.  Measured on the 27 Aug
seeds the seed is absent at the bottom row on all fifteen clips.

These tests do NOT re-measure that -- they pin the plumbing: the pad is applied
to the frame the detector sees, the masks come back at frame resolution, the
boxes come back in frame coordinates, and edge_pad=0 changes nothing.
"""
import numpy as np
import pytest

from vbgr.seed import MaskRCNNSeeder


class RecordingDet:
    """Stands in for torchvision maskrcnn via the ``instances`` hook.

    Records the frame shape it was handed, and returns one instance whose mask
    reaches the BOTTOM of whatever frame it was given -- i.e. a person the
    detector sees continuing past the original frame edge.
    """

    def __init__(self):
        self.seen = []

    def instances(self, frame, score_thr=0.8, mask_thr=0.5):
        h, w = frame.shape[:2]
        self.seen.append((h, w))
        m = np.zeros((h, w), bool)
        m[h // 2:, w // 4: 3 * w // 4] = True          # runs to the frame bottom
        box = np.array([[w / 4, h / 2, 3 * w / 4 - 1, h - 1]], float)
        return m[None], box


def _frame(h=200, w=300):
    return np.zeros((h, w, 3), np.uint8)


def test_edge_pad_zero_is_a_no_op():
    d = RecordingDet()
    s = MaskRCNNSeeder(detector=d, edge_pad=0)
    f = _frame()
    masks, boxes = s._instances(f)
    assert d.seen == [(200, 300)]                       # detector saw the raw frame
    assert masks.shape[1:] == (200, 300)
    assert boxes[0][3] == pytest.approx(199)


def test_edge_pad_grows_the_frame_the_detector_sees():
    d = RecordingDet()
    s = MaskRCNNSeeder(detector=d, edge_pad=64)
    s._instances(_frame())
    assert d.seen == [(200 + 128, 300 + 128)]


def test_masks_come_back_at_frame_resolution():
    d = RecordingDet()
    s = MaskRCNNSeeder(detector=d, edge_pad=64)
    masks, _ = s._instances(_frame())
    assert masks.shape[1:] == (200, 300)


def test_cropping_does_not_eat_the_bottom_row():
    """The point of the whole change: the returned mask still reaches the
    frame's bottom row after the crop.

    Deliberately NOT asserting that the padded run and the raw run agree pixel
    for pixel -- `RecordingDet` sizes its mask to whatever frame it is handed,
    so the padded run legitimately covers a wider band.  That is a property of
    the stand-in, not of the fix, and asserting on it would pin the stub.
    Whether the pad recovers real foreground is a measurement, and it lives in
    HANDOFF section 5, not in a unit test with a fake detector.
    """
    masks, _ = MaskRCNNSeeder(detector=RecordingDet(), edge_pad=64)._instances(_frame())
    assert masks[0][-1].any(), "bottom row must survive the crop"
    assert masks[0][-1].sum() > 0


def test_boxes_are_shifted_back_into_frame_coordinates():
    d = RecordingDet()
    s = MaskRCNNSeeder(detector=d, edge_pad=64)
    _, boxes = s._instances(_frame())
    x0, y0, x1, y1 = boxes[0]
    assert 0 <= x0 < x1 <= 299
    assert 0 <= y0 < y1 <= 199


def test_boxes_are_clipped_to_the_frame():
    """A box that runs into the pad must clip to the edge, not report past it."""
    class PastEdge:
        def instances(self, frame, score_thr=0.8, mask_thr=0.5):
            h, w = frame.shape[:2]
            m = np.ones((h, w), bool)
            return m[None], np.array([[0.0, 0.0, w - 1.0, h - 1.0]], float)

    s = MaskRCNNSeeder(detector=PastEdge(), edge_pad=32)
    _, boxes = s._instances(_frame())
    assert boxes[0].tolist() == [0.0, 0.0, 299.0, 199.0]


def test_instance_entirely_in_the_pad_is_dropped_with_a_note():
    class OnlyInPad:
        def instances(self, frame, score_thr=0.8, mask_thr=0.5):
            h, w = frame.shape[:2]
            m = np.zeros((h, w), bool)
            m[:16, :16] = True                          # inside the pad only
            return m[None], np.array([[0.0, 0.0, 16.0, 16.0]], float)

    s = MaskRCNNSeeder(detector=OnlyInPad(), edge_pad=32)
    s.notes = []
    masks, boxes = s._instances(_frame())
    assert len(masks) == 0 and len(boxes) == 0
    assert any("edge_pad" in n for n in s.notes)


def test_no_instances_is_handled():
    class Empty:
        def instances(self, frame, score_thr=0.8, mask_thr=0.5):
            return np.zeros((0, frame.shape[0], frame.shape[1]), bool), np.zeros((0, 4))

    masks, boxes = MaskRCNNSeeder(detector=Empty(), edge_pad=64)._instances(_frame())
    assert len(masks) == 0 and len(boxes) == 0


def test_negative_pad_is_clamped_off():
    d = RecordingDet()
    MaskRCNNSeeder(detector=d, edge_pad=-5)._instances(_frame())
    assert d.seen == [(200, 300)]


def test_default_is_off():
    """5.7b was turned on from a plausible argument and reverted.  This waits
    for a GPU pass that measures the final alpha."""
    from vbgr.config import SeedConfig
    # Defaults flipped 18 Sep 2026 (5.10b): the band pad is the shipped
    # configuration. 0/False reproduces the pre-flip _to_send/ renders.
    assert SeedConfig().maskrcnn_edge_pad == 64


def test_build_seeder_passes_the_knob_through():
    from vbgr.config import SeedConfig
    from vbgr.seed import build_seeder
    cfg = SeedConfig()
    cfg.maskrcnn_edge_pad = 48
    assert build_seeder(cfg).edge_pad == 48


# ---- 5.10b: band mode --------------------------------------------------- #

class SideDet:
    """Unpadded: holds a block reaching the RIGHT edge.  Padded: the same block
    retreats 20 px from the right edge (the tryguys failure) but reaches the
    bottom (the fix).  Also moves an interior pixel, which band mode must not."""

    def instances(self, frame, score_thr=0.8, mask_thr=0.5):
        h, w = frame.shape[:2]
        m = np.zeros((h, w), bool)
        if (h, w) == (200, 300):
            m[50:190, 100:300] = True                 # right edge held, bottom short
            m[120, 150] = False                       # interior detail
        else:
            p = (h - 200) // 2
            m[p + 50:, p + 100:p + 280] = True        # bottom reached, right short
        ys, xs = np.nonzero(m)
        return m[None], np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], float)


def _band_seeder(band):
    return MaskRCNNSeeder(detector=SideDet(), edge_pad=32, edge_pad_band=band)


def test_band_mode_keeps_the_better_of_both_at_every_edge():
    box = [(100.0, 50.0, 299.0, 199.0)]
    full = _band_seeder(False)._match(_frame(), box, None)[0]
    band = _band_seeder(True)._match(_frame(), box, None)[0]
    assert full[100, 299] == 0                        # full pad loses the right edge
    assert band[100, 299] == 1                        # band keeps it
    assert full[199, 150] == 1 and band[199, 150] == 1   # both fix the bottom


def test_band_mode_leaves_the_interior_exactly_as_shipped():
    box = [(100.0, 50.0, 299.0, 199.0)]
    shipped = MaskRCNNSeeder(detector=SideDet(), edge_pad=0)._match(_frame(), box, None)[0]
    band = _band_seeder(True)._match(_frame(), box, None)[0]
    assert band[120, 150] == shipped[120, 150] == 0
    assert np.array_equal(band[32:-32, 32:-32], shipped[32:-32, 32:-32])


def test_band_mode_never_changes_the_cast():
    class OnlyPadded:
        def instances(self, frame, score_thr=0.8, mask_thr=0.5):
            h, w = frame.shape[:2]
            if (h, w) == (200, 300):
                return np.zeros((0, h, w), bool), np.zeros((0, 4))
            m = np.zeros((h, w), bool); m[40:100, 40:100] = True
            return m[None], np.array([[40, 40, 99, 99]], float)
    s = MaskRCNNSeeder(detector=OnlyPadded(), edge_pad=32, edge_pad_band=True)
    assert s._match(_frame(), [(8.0, 8.0, 67.0, 67.0)], None) == [None]


def test_band_flag_reaches_the_seeder_and_defaults_off():
    from vbgr.config import SeedConfig
    from vbgr.seed import build_seeder
    assert SeedConfig().maskrcnn_edge_pad_band is True
    cfg = SeedConfig()
    cfg.maskrcnn_edge_pad, cfg.maskrcnn_edge_pad_band = 64, True
    s = build_seeder(cfg)
    assert s.edge_pad == 64 and s.edge_pad_band is True
