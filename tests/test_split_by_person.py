"""One obj_id per person on the PRODUCT path (3.2a, finally).

Until 2 Sep 2026 `pipeline.py` handed the engine one fused seed mask, so
SAM2MattingEngine fell back to splitting it by connected component -- and two
people who touch are one component.  3.2s measured that directly: `codylexi`
is two subjects and ONE component for its whole window, with shakira, dance
and tryguys close behind.  A VOS tracker given one identity holding two blobs
converges onto one of them, which is how ipman lost a fighter at frame 7
before 3.2a fixed it *for the benchmark path only*.

These pin the fix and, more importantly, pin the property that makes it safe:
the partition must lose no pixels, because the fused mask carries the text
prompt, the interactive head, held props and hole filling that `per_person`
does not.
"""
import numpy as np

from vbgr.engines.sam2matting import SAM2MattingEngine
from vbgr.seed import split_by_person


def _two_touching():
    fused = np.zeros((200, 200), np.uint8)
    fused[50:150, 40:110] = 1
    fused[50:150, 100:170] = 1
    a = np.zeros((200, 200), np.uint8); a[60:140, 50:95] = 1
    b = np.zeros((200, 200), np.uint8); b[60:140, 115:160] = 1
    return fused, [a, b]


def test_touching_people_get_one_object_each():
    fused, pp = _two_touching()
    # the behaviour being replaced: one component, so one obj_id for two people
    assert len(SAM2MattingEngine.split_objects(fused)) == 1
    out = split_by_person(fused, pp)
    assert len(out) == 2


def test_partition_loses_no_pixels_and_does_not_overlap():
    fused, pp = _two_touching()
    out = split_by_person(fused, pp)
    union = np.zeros_like(fused)
    for m in out:
        union |= (m > 0).astype(np.uint8)
    # exactly the fused mask back: nothing the seed stage added is dropped
    assert (union == fused).all()
    assert ((out[0] > 0) & (out[1] > 0)).sum() == 0


def test_extras_in_the_fused_mask_are_kept():
    """per_person is only the concept masks; the fused mask has more in it."""
    fused, pp = _two_touching()
    fused[45:50, 40:170] = 1              # e.g. hole fill / interactive head
    out = split_by_person(fused, pp)
    union = np.zeros_like(fused)
    for m in out:
        union |= (m > 0).astype(np.uint8)
    assert (union == fused).all()


def test_single_person_declines_to_split():
    fused, pp = _two_touching()
    assert split_by_person(fused, [pp[0]]) == []


def test_empty_inputs_decline_to_split():
    fused, pp = _two_touching()
    assert split_by_person(np.zeros_like(fused), pp) == []
    assert split_by_person(fused, []) == []
    assert split_by_person(fused, [np.zeros_like(pp[0]), np.zeros_like(pp[1])]) == []


def test_a_partition_that_collapses_to_one_object_is_not_a_split():
    """A relabelled union is not a split, and must not be returned as one."""
    fused = np.zeros((200, 200), np.uint8)
    fused[50:150, 40:110] = 1
    big = np.zeros((200, 200), np.uint8); big[60:140, 50:100] = 1
    tiny = np.zeros((200, 200), np.uint8); tiny[0:2, 0:2] = 1   # far, negligible
    out = split_by_person(fused, [big, tiny])
    assert out == []


def test_accepts_0_255_masks_as_well_as_0_1():
    fused, pp = _two_touching()
    out = split_by_person(fused * 255, [p * 255 for p in pp])
    assert len(out) == 2
    assert set(np.unique(out[0])) <= {0, 255}
