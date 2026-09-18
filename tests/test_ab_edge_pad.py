"""Metric helpers of bench/ab_edge_pad_final.py (5.10 Job 1)."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
ab = pytest.importorskip("ab_edge_pad_final")


def _person(H=300, W=400, stop=0):
    """A solid block reaching the bottom edge, truncated `stop` rows short."""
    a = np.zeros((H, W), np.float32)
    a[50:H - stop, 100:300] = 1.0
    return a


def test_orient_puts_the_named_edge_last():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(ab.orient(a, "bottom")[-1], a[-1])
    assert np.array_equal(ab.orient(a, "top")[-1], a[0])
    assert np.array_equal(ab.orient(a, "left")[-1], a[:, 0])
    assert np.array_equal(ab.orient(a, "right")[-1], a[:, -1])


def test_truncation_is_measured_on_shared_columns():
    full, cut = _person(), _person(stop=20)
    cols = ab.ref_columns([full, cut], "bottom")
    assert len(cols) == 200
    s_full = ab.edge_stats(full, "bottom", cols)
    s_cut = ab.edge_stats(cut, "bottom", cols)
    assert s_full["edge_alpha"] == 1.0 and s_full["short_px"] == 0
    assert s_cut["edge_alpha"] == 0.0 and s_cut["trunc"] == 1.0
    assert s_cut["short_px"] == 20
    # the dilation control does not move: same body 90 px up
    assert s_full["w90"] == s_cut["w90"] == 200


def test_a_soft_edge_is_not_called_truncated():
    a = _person()
    a[-1, 100:300] = 0.4
    s = ab.edge_stats(a, "bottom", ab.ref_columns([a], "bottom"))
    assert s["absent"] == 1.0 and s["trunc"] == 0.0
    assert abs(s["edge_alpha"] - 0.4) < 1e-6


def test_too_few_columns_returns_none():
    a = np.zeros((300, 400), np.float32)
    a[50:, 10:15] = 1
    assert ab.edge_stats(a, "bottom", ab.ref_columns([a], "bottom")) is None


def test_side_edges():
    a = np.zeros((300, 400), np.float32)
    a[50:250, 0:150] = 1          # touches the left edge
    cols = ab.ref_columns([a], "left")
    assert len(cols) == 200
    assert ab.edge_stats(a, "left", cols)["edge_alpha"] == 1.0
    assert ab.edge_stats(a, "right", ab.ref_columns([a], "right")) is None


def test_integrity_sees_a_vanished_subject_and_ignores_a_nudge():
    base = _person()
    base[60:120, 330:380] = 1     # a second, separate subject (1% of frame)
    gone = _person()
    assert ab.standalone_only(base, gone) and ab.standalone_only(gone, base) == []
    nudged = np.roll(base, 3, axis=1)
    assert ab.standalone_only(base, nudged) == []


def test_interior_churn_ignores_the_edge_band():
    a, b = _person(), _person(stop=20)
    assert ab.interior_churn(a, b)["mean_abs"] == 0.0


def test_magenta_uses_soft_alpha_with_no_floor():
    f = np.full((4, 4, 3), 100, np.uint8)
    a = np.zeros((4, 4), np.float32)
    assert (ab.over_magenta(f, a)[0, 0] == [255, 0, 255]).all()
    a[:] = 0.5
    assert (ab.over_magenta(f, a)[0, 0] == [177, 50, 177]).all()


def test_sheet_renders(tmp_path):
    H, W, T = 360, 640, 3
    frame = np.random.RandomState(0).randint(0, 255, (H, W, 3)).astype(np.uint8)
    arms = {"pad0": np.stack([_person(H, W, 20)] * T), "pad64": np.stack([_person(H, W)] * T)}
    for e in ab.EDGES:
        p = tmp_path / f"s_{e}.jpg"
        ab.build_sheet(frame, arms, 1, e, "test", str(p), panel_w=320)
        assert p.stat().st_size > 1000


def test_summarise_end_to_end(tmp_path):
    H, W, T = 360, 640, 6
    frames = [np.full((H, W, 3), 128, np.uint8)] * T
    arms = {"pad0": np.stack([_person(H, W, 20)] * T),
            "pad64": np.stack([_person(H, W)] * T),
            "pad0_repeat": np.stack([_person(H, W, 20)] * T)}
    r = ab.summarise_clip("synthetic", frames, arms, {}, {}, str(tmp_path))
    e = r["edges"]["bottom"]
    assert e["delta_edge_alpha"]["mean"] == 1.0
    assert e["control_edge_alpha"]["max_abs"] == 0.0
    assert r["churn"]["pad0_repeat"]["mean_abs"] == 0.0
    assert r["integrity"]["pad64"]["frames_pad0_only"] == 0
    ab.print_table([r])
