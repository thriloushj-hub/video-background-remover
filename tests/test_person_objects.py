"""MattingConfig.person_objects: per_person (default) vs union (23 Sep)."""
import numpy as np
from vbgr.config import Config
from vbgr import pipeline as P


class _Eng:
    def __init__(self): self.calls = []
    def reset(self): pass
    def matte(self, frames, seed_mask=None, n_warmup=10, seed_masks=None, **k):
        self.calls.append((seed_mask, seed_masks))
        out = np.zeros((len(frames),) + frames[0].shape[:2], np.float32)
        # union pass holds the left column, per-person pass the right one
        if len(seed_masks) == 1: out[:, :, 0] = 1.0
        else: out[:, :, -1] = 0.5
        return out


class _Seed:
    def __init__(self, per): self.per_person = per


def _two_people():
    a = np.zeros((40, 60), np.uint8); a[5:35, 5:25] = 255
    b = np.zeros((40, 60), np.uint8); b[5:35, 30:55] = 255
    return a, b


def _run(mode):
    cfg = Config(); cfg.matting.person_objects = mode
    pipe = P.Pipeline(cfg); eng = _Eng(); pipe._engine = eng
    a, b = _two_people(); union = np.maximum(a, b)
    frames = [np.zeros((40, 60, 3), np.uint8)] * 3
    out, n = pipe._matte_by_person(frames, _Seed([a > 0, b > 0]), union, 2)
    pipe._last = (out, eng.calls)
    return eng.calls[0], n, union


def test_default_is_both():
    # 25 Sep: flipped to "both" with Thriloush's OK
    assert Config().matting.person_objects == "both"


def test_per_person_gives_one_object_each():
    (sm, sms), n, _ = _run("per_person")
    assert n == 2 and sm is None and len(sms) == 2


def test_union_gives_exactly_one_object_holding_everyone():
    (sm, sms), n, union = _run("union")
    assert n == 1 and sm is None and len(sms) == 1
    assert (np.asarray(sms[0]) == union).all()


def test_both_runs_two_passes_and_takes_the_max():
    cfg = Config(); cfg.matting.person_objects = "both"
    pipe = P.Pipeline(cfg); eng = _Eng(); pipe._engine = eng
    a, b = _two_people(); union = np.maximum(a, b)
    frames = [np.zeros((40, 60, 3), np.uint8)] * 3
    out, n = pipe._matte_by_person(frames, _Seed([a > 0, b > 0]), union, 2)
    assert len(eng.calls) == 2 and len(eng.calls[0][1]) == 1 and len(eng.calls[1][1]) == 2
    assert n == 2
    assert (out[:, :, 0] == 1.0).all() and (out[:, :, -1] == 0.5).all()
