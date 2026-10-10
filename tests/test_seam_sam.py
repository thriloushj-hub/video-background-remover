"""seed.seam_fill_sam: close inside box overlap, gated by SAM (24 Sep)."""
import numpy as np
from vbgr.config import Config
from vbgr.seed import seam_candidates, seam_fill_sam


def _two():
    m = np.zeros((60, 100), np.uint8)
    m[10:50, 10:45] = 255
    m[10:50, 55:90] = 255
    return m, [(5, 5, 60, 55), (40, 5, 95, 55)]


def test_default_on():
    assert Config().seed.seam_sam is True


def test_candidates_only_inside_box_overlap():
    m, boxes = _two()
    c = seam_candidates(m, boxes, 21)
    assert c[20:40, 46:54].any()
    assert not c[:, :40].any() and not c[:, 61:].any()


def test_sam_gate_keeps_only_person_pixels():
    m, boxes = _two()
    sam = np.zeros((1, 60, 100), bool); sam[0, 30:50, 44:56] = True
    out, n = seam_fill_sam(None, m, boxes, 21, person_masks=sam)
    added = (out > 0) & ~(m > 0)
    assert n == int(added.sum()) > 0
    assert not added[10:30].any()


def test_no_overlap_is_a_no_op():
    m, _ = _two()
    out, n = seam_fill_sam(None, m, [(0, 0, 40, 60), (60, 0, 100, 60)], 21,
                           person_masks=np.ones((1, 60, 100), bool))
    assert n == 0 and (out == m).all()
