"""Silhouette-gated recovery (2 Oct 2026).

Recovery passes merge into the matte with np.maximum, so a recovery matte
that fuses two adjacent people paints the gap between them as body, for
good.  `_gate_recovery` restricts what recovery may ADD to a dilated union of
the detector's per-person silhouettes at the nearest scan frames.  The wall
between two people lies in no silhouette and is dropped; the people survive.
"""
import numpy as np
import pytest

from vbgr.config import Config
from vbgr.pipeline import Pipeline


class _Rep:
    def __init__(self):
        self.notes = []


def _pipe(on: bool, dilate: int = 10):
    cfg = Config()
    cfg.reid.gate_recovery_to_silhouette = on
    cfg.reid.gate_dilate_px = dilate
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg
    return p


def _scene():
    H, W = 120, 200
    # two people, 40 px apart, each a solid rectangle
    a = np.zeros((H, W), bool); a[20:110, 30:70] = True
    b = np.zeros((H, W), bool); b[20:110, 110:150] = True
    # a fused recovery matte: both people AND the floor between them
    rec = np.zeros((3, H, W), np.float32)
    rec[:, 20:110, 30:150] = 1.0
    return a, b, rec


def test_wall_between_people_is_dropped_people_survive():
    a, b, rec = _scene()
    p = _pipe(True, dilate=10)
    out = p._gate_recovery(rec, 0, {0: a | b, 8: a | b}, _Rep())
    mid = out[1]
    assert mid[60, 50] == 1.0 and mid[60, 130] == 1.0      # people kept
    assert mid[60, 90] == 0.0                              # wall gone
    # dilation keeps a 10 px margin around each person
    assert mid[60, 75] == 1.0 and mid[60, 85] == 0.0


def test_off_by_default_and_passthrough():
    a, b, rec = _scene()
    assert Config().reid.gate_recovery_to_silhouette is False
    p = _pipe(False)
    out = p._gate_recovery(rec, 0, {0: a | b}, _Rep())
    assert np.array_equal(out, rec)


def test_no_silhouette_means_ungated_never_blanked():
    a, b, rec = _scene()
    p = _pipe(True)
    out = p._gate_recovery(rec, 0, {}, _Rep())
    assert np.array_equal(out, rec)


def test_uses_both_neighbouring_scans():
    """A limb present only at the NEXT scan frame is still inside the gate."""
    a, b, rec = _scene()
    limb = np.zeros_like(a); limb[20:110, 160:190] = True
    rec2 = rec.copy(); rec2[:, 20:110, 160:190] = 1.0
    p = _pipe(True, dilate=2)
    out = p._gate_recovery(rec2, 0, {0: a | b, 8: a | b | limb}, _Rep())
    assert out[1][60, 175] == 1.0


def test_logs_kept_fraction():
    a, b, rec = _scene()
    p = _pipe(True, dilate=0)
    rep = _Rep()
    p._gate_recovery(rec, 0, {0: a | b}, rep)
    assert rep.notes and "recovery-gate" in rep.notes[0]
