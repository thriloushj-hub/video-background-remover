"""coverage_disagreement and area_jitter.

These exist because the 2026-08-23 benchmark table read as a 15/15 loss for v2
while the contact sheets showed v2 was more correct on ipman and butter and
broken on exactly one clip (interview).  A coverage comparison against an
over-inclusive baseline cannot express that, so the disagreement is decomposed
by *where* it sits rather than netted into one number.

The fixtures are synthetic on purpose: each one isolates a single real case
from that run, with the answer known by construction.
"""
import numpy as np
import pytest

from bench.metrics import area_jitter, coverage_disagreement

H = W = 200


def disc(cx, cy, r, h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)


def frames(a, n=8):
    return [a.copy() for _ in range(n)]


# ---- the ipman case: baseline is inflated, run is right ------------------- #

def test_baseline_halo_is_not_charged_as_a_hole():
    """v1 keys in a ring of background around the subject.

    That is the ipman signature -- v1 0.2453 against v2 0.1233 with both
    fighters visibly held.  It must read as halo, not as a defect.
    """
    run = disc(100, 100, 40)
    base = disc(100, 100, 60)                  # same subject, 2.25x the area
    d = coverage_disagreement(frames(base), frames(run))
    assert d["halo_frac"] > 0.5
    assert d["hole_frac"] < 0.02
    assert d["hole_big"] < 0.02


# ---- the interview case: the run tore a band through a held subject ------- #

def test_tear_inside_the_run_silhouette_is_a_hole():
    """A band torn through a face, with the subject otherwise held.

    Every existing metric read interview as clean: area_cv 0.0175, dropouts 0,
    subj_lost_at never, mean_area within 10% of v1.
    """
    base = disc(100, 100, 60)
    run = base.copy()
    run[:, 95:105] = 0.0                       # 10px tear, 5% of frame width
    d = coverage_disagreement(frames(base), frames(run))
    assert d["hole_frac"] > 0.05
    assert d["hole_big"] > 0.05
    assert d["halo_frac"] < 0.02


def test_halo_and_hole_are_reported_separately():
    """Both defects at once must not cancel.

    Netting them is exactly how the old table lost the distinction.
    """
    base = disc(100, 100, 60)
    run = disc(100, 100, 45)
    run[:, 98:102] = 0.0
    d = coverage_disagreement(frames(base), frames(run))
    assert d["halo_frac"] > 0.2
    assert d["hole_frac"] > 0.01


def test_speckle_does_not_accumulate_into_a_verdict():
    """Scattered single pixels are recovery noise, not a tear.

    hole_frac may see them; hole_big must not.  On the contact-sheet proxy this
    noise floor is about 2%, which is the same magnitude as interview's real
    tear -- the reason hole is quoted from persisted alphas only.
    """
    rng = np.random.default_rng(0)
    base = disc(100, 100, 60)
    run = base.copy()
    m = (rng.random((H, W)) < 0.02) & (base > 0.5)
    run[m] = 0.0
    d = coverage_disagreement(frames(base), frames(run))
    assert d["hole_big"] < 0.005


def test_identical_mattes_disagree_about_nothing():
    a = disc(100, 100, 50)
    d = coverage_disagreement(frames(a), frames(a))
    assert d == dict(halo_frac=0.0, hole_frac=0.0, hole_big=0.0,
                     excess_frac=0.0, excess_rim_px=0.0)


def test_excess_rim_is_a_width_not_a_fraction():
    """The 5.1c finding.

    On the 23 Aug set ``excess_frac`` spans 0.027-0.259 while the rim width is
    0.51-1.06 px on 14 of 15 clips -- the spread is perimeter, not coverage.
    Two subjects of very different size with the same one-pixel disagreement
    must report the same width and very different fractions.
    """
    out = []
    for r in (30, 70):
        base = disc(100, 100, r)
        run = disc(100, 100, r + 1)          # one pixel wider, both times
        out.append(coverage_disagreement(frames(base), frames(run)))
    assert out[0]["excess_frac"] > 2 * out[1]["excess_frac"]
    assert abs(out[0]["excess_rim_px"] - out[1]["excess_rim_px"]) < 0.25


def test_run_larger_than_baseline_is_excess_not_halo():
    """The baseline missing a subject must not read as the run being wrong."""
    d = coverage_disagreement(frames(disc(100, 100, 40)),
                              frames(disc(100, 100, 60)))
    assert d["excess_frac"] > 1.0
    assert d["halo_frac"] < 0.01 and d["hole_frac"] < 0.01


def test_empty_baseline_frames_are_skipped_not_divided_by():
    base = [np.zeros((H, W), np.float32)] * 4
    d = coverage_disagreement(base, frames(disc(100, 100, 40), 4))
    assert d["halo_frac"] == 0.0 and d["excess_frac"] == 0.0


def test_mismatched_lengths_and_shapes_are_errors():
    a = disc(100, 100, 40)
    with pytest.raises(ValueError):
        coverage_disagreement(frames(a, 3), frames(a, 4))
    with pytest.raises(ValueError):
        coverage_disagreement([a], [disc(50, 50, 20, 100, 100)])


# ---- area_jitter: the butter dolly-back must not read as instability ------ #

def test_smooth_dolly_back_is_not_jitter():
    """butter shrinks ~9x across the window under a camera pull-back.

    area_cv scores that 0.74 and this project has read it as matte instability
    three times.  area_jitter must not.
    """
    seq = [disc(100, 100, r) for r in np.linspace(60, 20, 72)]
    assert area_jitter(seq) < 0.02


def test_a_matte_that_drops_and_recovers_is_jitter():
    seq = [disc(100, 100, 50) for _ in range(24)]
    for i in (7, 8, 15):
        seq[i] = disc(100, 100, 20)
    assert area_jitter(seq) > 0.2


def test_jitter_separates_a_dolly_from_a_flicker():
    """The pair matters more than either threshold: area_cv scores the dolly
    HIGHER than the flicker, which is the inversion that misled us."""
    dolly = [disc(100, 100, r) for r in np.linspace(60, 20, 72)]
    flick = [disc(100, 100, 50) for _ in range(72)]
    for i in range(5, 72, 9):
        flick[i] = disc(100, 100, 20)
    assert area_jitter(flick) > 4 * area_jitter(dolly)


def test_jitter_refuses_a_too_short_sequence():
    assert area_jitter([disc(100, 100, 40)]) == 0.0


def test_a_shell_of_baseline_halo_is_not_counted_as_a_tear():
    """The ipman false positive.

    A thick inflated baseline leaves a ring of ``only_base`` hugging the run's
    silhouette.  Part of that ring is narrower than the closing disk, so it
    lands in hole_frac -- but it is bounded by run foreground on one side only.
    hole_big must reject it: the 23 Aug sheet shows both fighters held and no
    tear anywhere on ipman.
    """
    run = disc(100, 100, 40)
    base = disc(100, 100, 48)
    d = coverage_disagreement(frames(base), frames(run))
    assert d["halo_frac"] > 0.1
    assert d["hole_big"] < 0.01


def test_enclosure_test_does_not_reject_a_real_tear():
    """The same fixture as the interview case must survive enclosure."""
    base = disc(100, 100, 60)
    run = base.copy()
    run[:, 95:105] = 0.0
    assert coverage_disagreement(frames(base), frames(run))["hole_big"] > 0.05
