"""The product path must seed one object per person, like the benchmark does.

`pipeline.py` used to hand the engine a single fused mask. The engine then fell
back to splitting it by connected component, and two people who touch are one
component -- so on `codylexi` (two subjects, ONE component for the whole
window) the product path was seeding one object for two people. That is
precisely the configuration 3.2a proved loses a subject: given one identity
holding two blobs, the tracker converged onto one ipman fighter at frame 7.

These tests use fakes rather than the real detector, seeder and engine, so they
run on CPU in milliseconds and assert the wiring rather than the model.
"""
import numpy as np

from vbgr.config import Config
from vbgr.detect import Detection
from vbgr.pipeline import Pipeline
from vbgr.seed import SeedResult


H, W, T = 120, 200, 6


def _frame():
    return np.zeros((H, W, 3), np.uint8)


def _person(x0, x1):
    m = np.zeros((H, W), np.uint8)
    m[30:100, x0:x1] = 1
    return m


class _Detector:
    def detect(self, frame, conf=None):
        return ([Detection(box=(40, 30, 105, 100), conf=0.95, cls=0, label="person"),
                 Detection(box=(95, 30, 160, 100), conf=0.93, cls=0, label="person")],
                [])


class _Seeder:
    """Two people standing shoulder to shoulder: their masks touch."""
    notes = []

    def masks_from_boxes(self, frame, boxes):
        spans = [(40, 105), (95, 160)]
        return [_person(*spans[i]) * 255 for i in range(len(boxes))]

    def masks_from_text(self, frame, text):
        return []


class _Engine:
    """Records how it was prompted and reports per-object areas."""
    is_streaming = False
    degraded = {}

    class info:
        name = "fake"

    def __init__(self):
        self.seen = None
        self.object_areas = None

    def reset(self):
        pass

    def has(self, _cap):
        return False

    def matte(self, frames, seed_mask=None, n_warmup=10, seed_masks=None,
              **kw):
        self.seen = dict(seed_mask=seed_mask, seed_masks=seed_masks)
        n = len(seed_masks) if seed_masks else 1
        self.object_areas = np.full((len(frames), n), 0.05, np.float32)
        a = np.zeros((len(frames), H, W), np.float32)
        a[:, 30:100, 40:160] = 1.0
        return a


def _pipe():
    cfg = Config()
    cfg.shots.enabled = False
    cfg.reid.enabled = False
    cfg.verbose = False
    p = Pipeline(cfg)
    p._detector, p._seeder, p._engine = _Detector(), _Seeder(), _Engine()
    return p


def _run(p):
    frames = [_frame() for _ in range(T)]
    return p._run_shot(frames, 0)


def test_two_touching_people_are_seeded_as_two_objects():
    p = _pipe()
    _run(p)
    assert p._engine.seen["seed_masks"] is not None, \
        "product path fell back to one fused mask"
    assert len(p._engine.seen["seed_masks"]) == 2
    assert p._engine.seen["seed_mask"] is None


def test_the_split_covers_the_whole_refined_seed():
    """Nothing the seed stage added may be dropped on the way to the engine."""
    p = _pipe()
    _run(p)
    objs = p._engine.seen["seed_masks"]
    union = np.zeros((H, W), bool)
    for m in objs:
        union |= (m > 0)
    assert union.sum() > 0
    # each object is non-empty and they do not overlap
    assert all((m > 0).sum() > 0 for m in objs)
    assert ((objs[0] > 0) & (objs[1] > 0)).sum() == 0


def test_per_object_areas_reach_the_report():
    """Without these subj_lost_at cannot be measured on the product path."""
    p = _pipe()
    _alphas, rep = _run(p)
    assert rep.object_areas is not None
    assert len(rep.object_areas) == T
    assert len(rep.object_areas[0]) == 2


def test_one_person_still_uses_the_fused_mask():
    p = _pipe()
    p._detector = type("D", (), {"detect": lambda self, f, conf=None: (
        [Detection(box=(40, 30, 105, 100), conf=0.95, cls=0, label="person")], [])})()
    _run(p)
    assert p._engine.seen["seed_masks"] is None
    assert p._engine.seen["seed_mask"] is not None


def test_the_split_is_announced_not_silent():
    p = _pipe()
    _alphas, rep = _run(p)
    assert any("one per person" in n for n in rep.notes)


# --------------------------------------------------------------------------- #
# The recovery passes, which the 2 Sep fix missed the first time
# --------------------------------------------------------------------------- #

def test_directional_pass_seeds_per_person():
    """The 2 Sep run logged 'seeding 1 object(s) (auto-split)' from here.

    The forward matte had the fix and the recovery passes did not, because the
    seeding block had been copied rather than shared. All three now go through
    one helper.
    """
    p = _pipe()
    out = p._directional_pass([_frame() for _ in range(T)], anchor=0,
                              backward=False)
    assert out is not None
    assert p._engine.seen["seed_masks"] is not None
    assert len(p._engine.seen["seed_masks"]) == 2


def test_local_pass_seeds_per_person():
    p = _pipe()
    p.cfg.reid.scan_every = 2
    out = p._local_pass([_frame() for _ in range(T)], anchor=2, radius=1)
    assert out is not None
    assert p._engine.seen["seed_masks"] is not None
    assert len(p._engine.seen["seed_masks"]) == 2


def test_helper_falls_back_to_the_fused_mask_for_one_person():
    """Single-subject clips must behave exactly as before."""
    p = _pipe()
    seed = SeedResult(mask=_person(40, 105), kept=[],
                      per_person=[_person(40, 105)])
    a, n = p._matte_by_person([_frame() for _ in range(T)], seed,
                              seed.mask, 10)
    assert n == 1
    assert p._engine.seen["seed_masks"] is None
    assert p._engine.seen["seed_mask"] is not None
    assert len(a) == T
