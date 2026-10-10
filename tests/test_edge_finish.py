"""Output edge finish (2 Oct 2026): restyles the boundary, never adds or removes a region."""
import numpy as np
from vbgr import refine
from vbgr.config import IOConfig

def _alpha():
    a=np.zeros((120,160),np.float32); a[30:90,40:120]=1.0
    a[30:90,118:124]=np.linspace(1,0,6)[None,:]     # a soft right edge
    a[59,40:120]=0.3                                 # a one-pixel notch line across the body
    return a

def test_soft_is_identity():
    a=_alpha(); assert np.array_equal(refine.edge_finish(a,"soft"),a)

def test_hard_modes_are_binary_plus_antialias_and_keep_the_region():
    a=_alpha()
    for m in ("hard","hard_round","hard_v1"):
        o=refine.edge_finish(a,m,aa_sigma=0.0)
        assert set(np.unique(o)).issubset({0.0,1.0}), m
        assert o[60,80]==1.0 and o[10,10]==0.0, m
        o2=refine.edge_finish(a,m,aa_sigma=1.5)
        assert 0.0<o2[45,121]<1.0, m          # anti-aliased edge

def test_hard_round_fills_a_one_pixel_notch_hard_does_not():
    a=_alpha()
    assert refine.edge_finish(a,"hard",aa_sigma=0.0)[59,80]==0.0
    assert refine.edge_finish(a,"hard_round",aa_sigma=0.0)[59,80]==1.0

def test_default_is_soft():
    assert IOConfig().edge_finish=="soft"
