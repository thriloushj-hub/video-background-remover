"""CPU-only tests for the parts of the pipeline that do not need a model.

Deliberately covers the logic where bugs actually hide -- geometry, indexing,
frame accounting, gate decisions -- rather than the model wrappers, which
cannot be tested without checkpoints.

Run: ``python -m pytest tests/ -q``   (or ``python tests/test_core.py``)
"""
from __future__ import annotations

import os
import sys
import tempfile
from fractions import Fraction

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vbgr import compose, decontaminate, motion, refine, shots, video_io
from vbgr.detect import Detection, select_person_boxes
from vbgr.engines.base import build_engine, list_engines
from vbgr.gate import QualityGate
from vbgr.reid import IdentityBank, FeatureExtractor, box_coverage, uncovered_boxes
from vbgr.seed import fill_holes, reopen_background_gaps


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def disc(h=180, w=320, cx=100, cy=90, r=30):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m


def scene(h=180, w=320, cx=100, bg=40):
    f = np.full((h, w, 3), bg, np.uint8)
    f[:, :, 1] = (bg + 30) % 255
    cv2.circle(f, (cx, h // 2), 30, (250, 250, 250), -1)
    return f


# --------------------------------------------------------------------------- #
# motion
# --------------------------------------------------------------------------- #

def test_boundary_distance_is_zero_only_on_the_boundary():
    """Regression: min(d_in, d_out) is identically 0 and made every pixel
    'unknown'. Distance must be selected by side."""
    m = disc()
    d = motion.boundary_distance(m)
    assert d.max() > 20, "interior/exterior distances collapsed to zero"
    edge = cv2.Canny(m * 255, 50, 150) > 0
    assert d[edge].mean() < 2.0
    assert d[disc(cx=100, cy=90, r=5) > 0].mean() > 20


def test_trimap_band_widens_with_flow():
    a = cv2.GaussianBlur(disc().astype(np.float32), (0, 0), 1.5)
    slow = motion.flow_adaptive_trimap(a, np.zeros_like(a), 4.0, 0.9, 40.0)
    fast = motion.flow_adaptive_trimap(a, np.full_like(a, 20.0), 4.0, 0.9, 40.0)
    assert (fast == 128).sum() > 3 * (slow == 128).sum()
    # Hard labels must stay consistent with the alpha.
    assert (fast[a > 0.999] != 0).all()


def test_trimap_respects_max_width():
    a = cv2.GaussianBlur(disc().astype(np.float32), (0, 0), 1.5)
    huge = motion.flow_adaptive_trimap(a, np.full_like(a, 1e4), 4.0, 0.9, 12.0)
    d = motion.boundary_distance((a > 0.5).astype(np.uint8))
    assert (huge[d > 13] == 128).sum() == 0


def test_warp_moves_content_the_right_way():
    f0, f1 = scene(cx=100), scene(cx=130)
    fl = motion.FlowEstimator().flow(f0, f1)
    a0 = cv2.GaussianBlur(disc(cx=100).astype(np.float32), (0, 0), 1.0)
    w = motion.warp(a0, fl)
    cx_before = float(np.average(np.arange(a0.shape[1]), weights=a0.sum(0)))
    cx_after = float(np.average(np.arange(w.shape[1]), weights=w.sum(0) + 1e-9))
    assert cx_after > cx_before + 10, (cx_before, cx_after)


def test_warp_prior_is_distrusted_across_a_cut():
    f0 = scene(cx=100, bg=40)
    f1 = scene(cx=100, bg=220)          # hard cut: same geometry, new lighting
    fl = motion.FlowEstimator().flow(f0, f1)
    _, trust = motion.flow_alpha_prior(disc().astype(np.float32), f0, f1, fl)
    assert trust.mean() < 0.5, trust.mean()


# --------------------------------------------------------------------------- #
# shots
# --------------------------------------------------------------------------- #

def test_shot_ranges_always_tile_every_frame():
    for starts in ([0], [0, 30], [0, 5, 60], [12, 40]):
        r = shots.shot_ranges(starts, 100)
        assert sum(b - a for a, b in r) == 100
        assert r[0][0] == 0 and r[-1][1] == 100


def test_flash_frame_does_not_create_a_one_frame_shot():
    frames = [scene(bg=40) for _ in range(40)]
    frames[20] = scene(bg=255)                       # single blown-out frame
    starts = shots.compute_scene_cuts(frames, 0.35, 64, min_shot_len=12)
    assert all(b - a >= 12 for a, b in shots.shot_ranges(starts, len(frames)))


def test_real_cut_is_found():
    frames = [scene(bg=30) for _ in range(25)] + [scene(bg=230) for _ in range(25)]
    starts = shots.compute_scene_cuts(frames, 0.35, 64, min_shot_len=8)
    assert any(abs(s - 25) <= 2 for s in starts), starts


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #

def test_gate_rejects_collapse_explosion_and_fragmentation():
    g = QualityGate()
    a = cv2.GaussianBlur(disc().astype(np.float32), (0, 0), 1.0)
    f = scene()
    g.prime(a, f)
    for _ in range(6):
        g.evaluate(a, f)

    assert not g.evaluate(np.zeros_like(a), f, accept=False).ok
    big = cv2.GaussianBlur(disc(r=90).astype(np.float32), (0, 0), 1.0)
    assert not g.evaluate(big, f, accept=False).ok

    frag = np.zeros_like(a)
    for i in range(12):
        cv2.circle(frag, (20 + 24 * i, 40), 11, 1.0, -1)
    assert not g.evaluate(frag, f, accept=False).ok


def test_gate_declares_the_track_lost_after_enough_rejects():
    g = QualityGate(lost_after=3)
    a = cv2.GaussianBlur(disc().astype(np.float32), (0, 0), 1.0)
    g.prime(a, scene())
    assert not g.track_lost
    for _ in range(3):
        g.evaluate(np.zeros_like(a), scene())
    assert g.track_lost


def test_gate_accepts_a_moving_subject():
    """Real motion must not be mistaken for a tracking failure."""
    g = QualityGate()
    g.prime(cv2.GaussianBlur(disc(cx=60).astype(np.float32), (0, 0), 1.0),
            scene(cx=60))
    ok = []
    for cx in range(70, 140, 10):
        v = g.evaluate(cv2.GaussianBlur(disc(cx=cx).astype(np.float32), (0, 0), 1.0),
                       scene(cx=cx))
        ok.append(v.ok)
    assert sum(ok) >= len(ok) - 1, ok


# --------------------------------------------------------------------------- #
# detection selection
# --------------------------------------------------------------------------- #

def _d(x1, y1, x2, y2, c=0.9):
    return Detection((x1, y1, x2, y2), c, 0, "person")


def test_short_background_person_is_dropped_even_when_centred():
    """The centred-bystander failure: position must not rescue a small box."""
    lead_l = _d(100, 100, 300, 900)
    lead_r = _d(1400, 100, 1600, 900)
    centre_bystander = _d(900, 400, 980, 700)
    kept, dropped = select_person_boxes([lead_l, lead_r, centre_bystander],
                                        1920, 1080, 0.6)
    assert centre_bystander in dropped
    assert lead_l in kept and lead_r in kept


def test_edge_entrant_is_kept_despite_being_cropped():
    """The side-entrant failure: horizontal cropping collapses area but not
    height, so the gate must use height."""
    lead = _d(700, 100, 950, 1000)
    entrant = _d(0, 110, 60, 990)        # very narrow: cropped by frame edge
    kept, dropped = select_person_boxes([lead, entrant], 1920, 1080, 0.6)
    assert entrant in kept, "a full-height entrant was dropped"


def test_selection_never_returns_empty():
    kept, _ = select_person_boxes([_d(0, 0, 10, 10, 0.3)], 1920, 1080, 0.6)
    assert len(kept) == 1


# --------------------------------------------------------------------------- #
# reid
# --------------------------------------------------------------------------- #

def test_coverage_and_uncovered_boxes():
    a = disc(cx=100, cy=90, r=40).astype(np.float32)
    assert box_coverage(a, (70, 60, 130, 120)) > 0.8
    assert box_coverage(a, (250, 20, 310, 80)) < 0.1
    unc = uncovered_boxes(a, [(70, 60, 130, 120), (250, 20, 310, 80)],
                          max_overlap=0.35, min_area_frac=0.0)
    assert unc == [1]


def test_identity_bank_separates_two_different_people():
    """Uses the classical descriptor, so this runs without any model."""
    ex = FeatureExtractor()
    ex._impl = ("classic",)
    ex.backend = "classic"
    bank = IdentityBank(ex, match_threshold=0.7)

    red = np.zeros((200, 100, 3), np.uint8); red[:, :, 2] = 220
    blue = np.zeros((200, 100, 3), np.uint8); blue[:, :, 0] = 220
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[:, :100] = red
    frame[:, 100:] = blue

    i1 = bank.new_identity()
    bank.observe(i1.ident, frame, (0, 0, 100, 200), None, 0)

    res = bank.match(frame, [(0, 0, 100, 200), (100, 0, 200, 200)])
    same = [r for r in res if r[0] == 0][0]
    other = [r for r in res if r[0] == 1][0]
    assert same[2] > other[2], (same, other)


def test_identity_matching_is_global_not_greedy():
    ex = FeatureExtractor(); ex._impl = ("classic",); ex.backend = "classic"
    bank = IdentityBank(ex)
    frame = np.random.RandomState(0).randint(0, 255, (200, 400, 3), np.uint8)
    a = bank.new_identity(); b = bank.new_identity()
    bank.observe(a.ident, frame, (0, 0, 200, 200), None, 0)
    bank.observe(b.ident, frame, (200, 0, 400, 200), None, 0)
    res = bank.match(frame, [(0, 0, 200, 200), (200, 0, 400, 200)])
    assert len({r[1] for r in res if r[1] is not None}) == len(
        [r for r in res if r[1] is not None]), "an identity was assigned twice"


# --------------------------------------------------------------------------- #
# seed morphology
# --------------------------------------------------------------------------- #

def test_fill_holes_closes_an_enclosed_gap():
    m = np.zeros((100, 100), np.uint8)
    cv2.rectangle(m, (20, 20), (80, 80), 1, -1)
    cv2.rectangle(m, (40, 40), (60, 60), 0, -1)
    f = fill_holes(m)
    assert f[50, 50] == 1
    assert f[5, 5] == 0


def test_background_coloured_gap_is_reopened():
    frame = np.full((100, 100, 3), (20, 200, 20), np.uint8)     # green bg
    body = np.zeros((100, 100), np.uint8)
    cv2.rectangle(body, (20, 20), (80, 80), 1, -1)
    cv2.rectangle(body, (40, 40), (60, 60), 0, -1)
    frame[body > 0] = (200, 30, 30)                             # blue subject
    filled = fill_holes(body)
    out = reopen_background_gaps(frame, filled, body, margin=0.02,
                                 min_area=10, max_frac=0.5)
    assert out[50, 50] == 0, "a background-coloured gap stayed filled"


# --------------------------------------------------------------------------- #
# decontamination / compose
# --------------------------------------------------------------------------- #

def test_decontamination_recovers_a_known_foreground():
    """Composite a known F over a known B, then check F is recovered."""
    h = w = 96
    F = np.zeros((h, w, 3), np.float32); F[:, :, 2] = 0.9
    B = np.zeros((h, w, 3), np.float32); B[:, :, 1] = 0.9
    a = np.zeros((h, w), np.float32)
    cv2.circle(a, (48, 48), 30, 1.0, -1)
    a = cv2.GaussianBlur(a, (0, 0), 5.0)
    I = a[..., None] * F + (1 - a[..., None]) * B

    band = (a > 0.15) & (a < 0.85)
    err_raw = float(np.abs(I - F).mean(2)[band].mean())

    # Correctness: given enough iterations the solver converges on the true F.
    Fh, Bh = decontaminate.estimate_foreground_background(
        I, a, n_small_iterations=120, n_big_iterations=30)
    assert float(np.abs(Fh - F).mean(2)[band].mean()) < 0.01
    assert float(np.abs(Bh - B).mean(2)[band].mean()) < 0.01

    # Shipping defaults: a large improvement, traded against ~0.7s/1080p frame.
    Fh, _ = decontaminate.estimate_foreground_background(I, a)
    err_est = float(np.abs(Fh - F).mean(2)[band].mean())
    assert err_est < err_raw * 0.55, (err_est, err_raw)


def test_decontamination_leaves_the_solid_interior_alone():
    f = scene()
    a = cv2.GaussianBlur(disc().astype(np.float32), (0, 0), 3.0)
    F = decontaminate.decontaminate(f, a)
    interior = a > 0.995
    assert np.abs(F.astype(int) - f.astype(int))[interior].mean() < 2.0


def test_green_composite_round_trip():
    f = scene()
    a = cv2.GaussianBlur(disc().astype(np.float32), (0, 0), 2.0)
    g = compose.composite_over_color(f, a, (0, 177, 64))
    rec = compose.alpha_from_greenscreen(g, (0, 177, 64))
    assert float(np.corrcoef(rec.ravel(), a.ravel())[0, 1]) > 0.9


def test_bgra_straight_vs_premultiplied():
    f = scene()
    a = np.full(f.shape[:2], 0.5, np.float32)
    s = compose.to_bgra(f, a, premultiplied=False)
    p = compose.to_bgra(f, a, premultiplied=True)
    assert np.allclose(s[..., :3], f, atol=1)
    assert np.allclose(p[..., :3], (f * 0.5).astype(np.uint8), atol=2)


# --------------------------------------------------------------------------- #
# refine
# --------------------------------------------------------------------------- #

def test_refine_band_respects_hard_trimap_labels():
    a = cv2.GaussianBlur(disc().astype(np.float32), (0, 0), 2.0)
    tri = motion.flow_adaptive_trimap(a, None, 4.0)
    out = refine.refine_band(a, scene(), tri)
    assert (out[tri == 255] == 1.0).all()
    assert (out[tri == 0] == 0.0).all()


def test_remove_specks_drops_small_blobs_only():
    a = disc(cx=100, r=40).astype(np.float32)
    cv2.circle(a, (280, 30), 2, 1.0, -1)
    out = refine.remove_specks(a, min_area=48)
    assert out[30, 280] == 0
    assert out[90, 100] == 1


def test_temporal_stabiliser_does_not_ghost_fast_motion():
    st = refine.TemporalStabilizer(weight=0.35)
    f0, f1 = scene(cx=60), scene(cx=160)
    a0 = cv2.GaussianBlur(disc(cx=60).astype(np.float32), (0, 0), 1.0)
    a1 = cv2.GaussianBlur(disc(cx=160).astype(np.float32), (0, 0), 1.0)
    st.step(f0, a0)
    out = st.step(f1, a1)
    # No significant alpha should survive back at the old position.
    assert out[90, 60] < 0.35, out[90, 60]


# --------------------------------------------------------------------------- #
# engines / io
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# capability / kwarg-drop reporting
#
# The failure this guards against: an engine declares memory_gate, the kwarg
# backing it does not exist on the installed API, filter_kwargs drops it, the
# run completes, gate_rejects reports 47, and the gate did nothing. Every
# ablation built on that is noise. So the drop must be loud and recorded.
# --------------------------------------------------------------------------- #

def _fresh_engine():
    """A passthrough engine with clean degraded state and warning cache."""
    from vbgr.engines import base as B
    B._WARNED.clear()
    e = build_engine("passthrough")
    e._degraded = {}
    return e


def test_kwarg_support_distinguishes_missing_from_unverifiable():
    from vbgr.engines.base import kwarg_support
    def exact(a, b=1): pass
    def anykw(a, **kw): pass
    assert kwarg_support(exact, "b") == "yes"
    assert kwarg_support(exact, "nope") == "no"
    # **kwargs ACCEPTS the name but may ignore it inside. That is not support,
    # and conflating the two is what hides a dead feature.
    assert kwarg_support(anykw, "anything") == "unverifiable"


def test_dropping_a_critical_kwarg_warns_and_records():
    import warnings as W
    from vbgr.engines.base import filter_kwargs
    eng = _fresh_engine()
    def step_without_gate(image): pass

    with W.catch_warnings(record=True) as caught:
        W.simplefilter("always")
        kept = filter_kwargs(step_without_gate, {"update_memory": False},
                             engine=eng, where="test")

    assert kept == {}, "the unsupported kwarg should not be passed on"
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
        "dropping a critical kwarg must raise a RuntimeWarning"
    assert "memory_gate" in str(caught[0].message)
    assert "update_memory" in eng.degraded
    assert not eng.feature_active("update_memory")


def test_supported_kwarg_is_passed_through_silently():
    import warnings as W
    from vbgr.engines.base import filter_kwargs
    eng = _fresh_engine()
    def step_with_gate(image, update_memory=True): pass

    with W.catch_warnings(record=True) as caught:
        W.simplefilter("always")
        kept = filter_kwargs(step_with_gate, {"update_memory": False},
                             engine=eng, where="test")
    assert kept == {"update_memory": False}
    assert not caught, "a supported kwarg must not warn"
    assert eng.degraded == {}
    assert eng.feature_active("update_memory")


def test_require_kwarg_raises_in_strict_mode():
    from vbgr.engines.base import require_kwarg, CapabilityError
    eng = _fresh_engine()
    def step_without_gate(image): pass
    try:
        require_kwarg(step_without_gate, "update_memory", eng, strict=True)
    except CapabilityError as e:
        assert "memory_gate" in str(e)
        return
    raise AssertionError("strict mode must raise CapabilityError")


def test_require_kwarg_warns_but_continues_when_not_strict():
    import warnings as W
    from vbgr.engines.base import require_kwarg
    eng = _fresh_engine()
    def step_without_gate(image): pass
    with W.catch_warnings(record=True) as caught:
        W.simplefilter("always")
        ok = require_kwarg(step_without_gate, "update_memory", eng, strict=False)
    assert ok is False
    assert caught, "non-strict mode must still warn"
    assert not eng.feature_active("update_memory")


def test_pipeline_reports_gate_as_ineffective_when_kwarg_was_dropped():
    """The whole point: gate_is_effective must be False even though the engine
    still *declares* memory_gate in its capability set."""
    from vbgr.config import Config
    from vbgr.pipeline import Pipeline
    cfg = Config()
    cfg.matting.engine = "passthrough"
    cfg.memory_gate.enabled = True
    pipe = Pipeline(cfg)
    eng = pipe.engine
    eng._degraded = {}

    assert eng.has("memory_gate"), "passthrough declares the capability"
    assert pipe.gate_is_effective(), "should be effective before any drop"

    eng.note_degraded("update_memory", "memory_gate")
    assert not pipe.gate_is_effective(), \
        "a declared-but-dead capability must not count as effective"


def test_summary_flags_unenforced_rejections():
    from vbgr.pipeline import ClipReport, ShotReport
    r = ClipReport(name="x", n_frames=10, fps=24.0, engine="e")
    s = ShotReport(start=0, end=10); s.gate_rejects = 47
    r.shots.append(s)
    assert "gate_rejects=47" in r.summary()
    assert "NOT ENFORCED" not in r.summary()

    s.gate_rejects_unenforced = 47
    r.degraded = {"update_memory": "memory_gate"}
    out = r.summary()
    assert "NOT ENFORCED" in out, "a dead gate must not print a bare count"
    assert "DEGRADED" in out


def test_every_engine_declares_its_licence():
    for key, info in list_engines().items():
        assert info.license, key
        assert isinstance(info.commercial_ok, bool), key
        assert info.mode in ("streaming", "sequence"), key


def test_require_commercial_blocks_non_commercial_engines():
    for key in ("sam2matting", "matanyone2"):
        try:
            build_engine(key, require_commercial=True)
        except PermissionError:
            continue
        except Exception:
            raise AssertionError(f"{key} was not blocked by require_commercial")
        raise AssertionError(f"{key} was not blocked by require_commercial")


def test_engine_returns_one_alpha_per_frame():
    frames = [scene(cx=60 + 3 * i) for i in range(24)]
    a = build_engine("passthrough").matte(frames, disc(cx=60))
    assert a.shape == (24, 180, 320)


def test_video_round_trip_preserves_count_and_exact_fps():
    frames = [scene(cx=60 + i) for i in range(30)]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.mp4")
        w = video_io.VideoWriter(p, 320, 180, Fraction(24000, 1001))
        for f in frames:
            w.write(f)
        assert w.close(expect=30) == 30
        assert len(video_io.read_all(p)) == 30
        assert video_io.probe(p).fps == Fraction(24000, 1001)


def test_writer_raises_on_a_dropped_frame():
    """The ipman defect, as a test."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.mp4")
        w = video_io.VideoWriter(p, 320, 180, Fraction(24, 1))
        for f in [scene() for _ in range(10)]:
            w.write(f)
        try:
            w.close(expect=20, strict=True)
        except video_io.FrameCountMismatch:
            return
        raise AssertionError("a 50% frame drop was not detected")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                                   # noqa: BLE001
            fails += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
