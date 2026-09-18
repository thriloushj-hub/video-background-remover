"""Job 2 probe: the deficit arithmetic (bench/probe_engine_deficit.py)."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
pj = pytest.importorskip("probe_engine_deficit")


def _case(alpha_stop=30):
    T, H, W = 4, 360, 640
    seed = np.zeros((H, W), np.float32); seed[50:, 100:300] = 1        # to the bottom
    alphas = np.zeros((T, H, W), np.float32)
    alphas[:, 50:H - alpha_stop, 100:300] = 1
    meta = dict(shots=[dict(start=0, end=T, seed_frame=0)])
    return pj.analyse("c", "base", alphas, {0: seed}, meta)


def test_deficit_is_alpha_short_minus_seed_short():
    rows = {r["edge"]: r for r in _case(30)}
    assert rows["bottom"]["seed_short"] == 0
    assert rows["bottom"]["deficit_at_seed"] == 30
    assert rows["bottom"]["worst_deficit"] == 30
    # the top edge: the seed itself is 50 px short and the alpha matches it
    assert rows["top"]["deficit_at_seed"] == 0


def test_an_edge_the_seed_never_reaches_solidly_is_skipped():
    edges = {r["edge"] for r in _case()}
    assert "left" not in edges and "right" not in edges


def test_engine_following_faithfully_reads_zero():
    rows = {r["edge"]: r for r in _case(0)}
    assert rows["bottom"]["deficit_at_seed"] == 0
