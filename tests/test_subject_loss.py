"""subj_lost_at regressions.

Every case here comes from the 2026-08-23 full benchmark, where the metric
counted connected components and therefore reported a lost subject on all four
clips in which two people happened to touch.  The contact sheets show every
subject present in every frame of all four.
"""
import importlib.util
import os

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "vbgr_bench", os.path.join(_ROOT, "vbgr_bench.py"))


def _load():
    m = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(m)
    except Exception as e:                      # needs torch etc. at import
        pytest.skip(f"vbgr_bench not importable here: {e}",
                    allow_module_level=True)
    return m


bench = _load()
T, K = 72, 2


def _frames_two_blobs(gap):
    """72 frames, two 40x40 blobs whose separation is `gap` px."""
    out = np.zeros((T, 120, 200), np.float32)
    for t in range(T):
        g = gap(t)
        out[t, 40:80, 60:100] = 1.0
        out[t, 40:80, 100 + g:140 + g] = 1.0
    return out


def test_merged_subjects_are_not_lost():
    """codylexi: two subjects, ONE component all window, nobody lost."""
    A = _frames_two_blobs(lambda t: 0)          # touching from frame 0
    assert bench.n_subjects(A[0]) == 1, "precondition: they read as one blob"
    areas = np.full((T, K), 0.06, np.float32)   # both alive throughout
    lost, _ = bench.subject_loss(A, K, object_areas=areas)
    assert lost is None


def test_merge_midway_is_not_lost():
    """shakira/dance: they separate, cross, separate again."""
    A = _frames_two_blobs(lambda t: 0 if 30 <= t <= 50 else 30)
    areas = np.full((T, K), 0.06, np.float32)
    lost, _ = bench.subject_loss(A, K, object_areas=areas)
    assert lost is None


def test_real_loss_still_fires():
    """A subject that is actually keyed out must still be caught."""
    areas = np.full((T, K), 0.06, np.float32)
    areas[40:, 1] = 0.0
    A = _frames_two_blobs(lambda t: 30)
    lost, _ = bench.subject_loss(A, K, object_areas=areas)
    assert lost == 40


def test_gradual_shrink_is_not_a_loss():
    """butter's dolly-back: subjects shrink 4x and are never lost."""
    areas = np.stack([np.linspace(0.08, 0.02, T)] * K, 1).astype(np.float32)
    A = _frames_two_blobs(lambda t: 30)
    lost, _ = bench.subject_loss(A, K, object_areas=areas)
    assert lost is None


def test_butter_scale_of_shrink_is_not_a_loss():
    """The real butter case, and the one the first version of this got wrong.

    The camera dollies back and every dancer shrinks about 9x together. The
    original floor was a fraction of each subject's own SEED-frame area, so all
    four dropped under it at once and the metric reported a loss at frame 17
    while the contact sheet shows four dancers still there at frame 71.
    """
    areas = np.stack([np.linspace(0.090, 0.010, T)] * 4, 1).astype(np.float32)
    A = _frames_two_blobs(lambda t: 30)
    lost, _ = bench.subject_loss(A, 4, object_areas=areas)
    assert lost is None, "a uniform scene-wide shrink is not a lost subject"


def test_one_subject_shrinking_alone_is_a_loss():
    """The counterpart: if only ONE subject collapses, that is real."""
    areas = np.full((T, K), 0.08, np.float32)
    areas[:, 0] = np.linspace(0.08, 0.004, T)
    A = _frames_two_blobs(lambda t: 30)
    lost, _ = bench.subject_loss(A, K, object_areas=areas)
    assert lost is not None


def test_shrink_past_the_floor_is_a_loss():
    """Losing most of your share of the matte is gone, not small."""
    areas = np.full((T, K), 0.08, np.float32)
    areas[50:, 0] = 0.08 * 0.10
    A = _frames_two_blobs(lambda t: 30)
    lost, _ = bench.subject_loss(A, K, object_areas=areas)
    assert lost == 50


def test_no_object_areas_refuses_to_guess():
    """Without per-object data a merge and a loss are indistinguishable.

    And it must say so, not say 'never'. Those were the same value until
    2 Sep 2026, which would have let the product arm -- which fuses its seeds
    and so has no per-object areas -- report a clean pass on all fifteen clips
    without measuring anything.
    """
    A = _frames_two_blobs(lambda t: 0)
    lost, k = bench.subject_loss(A, K, object_areas=None)
    assert lost == bench.UNMEASURED
    assert lost is not None, "unmeasured must not collapse back onto 'never'"
    assert len(k) == T


def test_unmeasured_does_not_print_as_never():
    A = _frames_two_blobs(lambda t: 0)
    assert bench.score(A, K)["subj_lost_at"] == bench.UNMEASURED
    held = np.full((T, K), 0.06, np.float32)
    assert bench.score(A, K, object_areas=held)["subj_lost_at"] == "never"


def test_wrong_shaped_areas_refuse_to_guess():
    A = _frames_two_blobs(lambda t: 30)
    lost, _ = bench.subject_loss(A, K, object_areas=np.zeros((T, K + 1)))
    assert lost == bench.UNMEASURED


def test_nan_rows_are_carried_not_counted():
    """A frame the predictor skipped must not read as everyone vanishing."""
    areas = np.full((T, K), 0.06, np.float32)
    areas[20] = np.nan
    A = _frames_two_blobs(lambda t: 30)
    lost, _ = bench.subject_loss(A, K, object_areas=areas)
    assert lost is None
