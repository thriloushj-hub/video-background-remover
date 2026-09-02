"""The result table must describe the run it is given, product arm included.

build_result_table.py was written on 2 Sep to stop the headline table being
hand-copied, and within hours the run it existed for had an arm it could not
render. These pin the shape so that does not repeat quietly: a missing product
arm has to SAY it is missing, because a table that silently shows only the
benchmark path is exactly how "every number is from the measured path" went
unnoticed for three weeks.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.build_result_table import build  # noqa: E402


def _run(tmp, with_product=True, failed=False):
    off = dict(area_cv=0.01, edge_soft=0.5, sharpness=0.66, dropouts=0,
               mean_area=0.30, subj_lost_at="never")
    row = {"off": off, "on": dict(off, edge_soft=0.61, sharpness=0.64),
           "n_seed": 2}
    if with_product:
        row["product"] = dict(off, mean_area=0.33, subj_lost_at="never")
        row["product_meta"] = {"n_seed": 2, "n_shots": 1, "seconds": 180.0}
    if failed:
        row["product_error"] = "RuntimeError: seed gate kept nobody"
    v1 = {"clipA": {"window": [0, 72], "mean_area": 0.34, "area_cv": 0.012,
                    "dropouts": 0, "subjects_f0": 2, "trusted": True}}
    bo = os.path.join(tmp, "bench_out")
    os.makedirs(bo, exist_ok=True)
    json.dump({"rows": {"clipA": row}, "failed": {}, "v1": v1},
              open(os.path.join(bo, "bench_results.json"), "w"))
    json.dump({"rows": {"clipA": {"halo_frac": 0.01, "hole_frac": 0.01,
                                  "hole_big": 0.0, "excess_frac": 0.0,
                                  "excess_rim_px": 0.0, "area_cv_v1": 0.012,
                                  "area_cv_v2": 0.01, "area_jitter_v1": 0.002,
                                  "area_jitter_v2": 0.003, "verdict": "agree"}},
               "skipped": {}},
              open(os.path.join(tmp, "coverage_disagreement_alpha.json"), "w"))
    json.dump({"clipA": {"halo_frac": 0.01, "halo_shell": 0.01,
                         "halo_solo": 0.0}},
              open(os.path.join(tmp, "halo_split.json"), "w"))
    return tmp


def test_product_arm_gets_its_own_table():
    with tempfile.TemporaryDirectory() as tmp:
        md, tally = build(_run(tmp))
        assert "The product path" in md
        assert "97.1%" in md, "coverage as a share of v1 must be shown"
        assert tally["product"] == 1


def test_a_run_without_a_product_arm_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        md, tally = build(_run(tmp, with_product=False))
        assert "The product path" not in md
        assert tally["product"] == 0


def test_a_failed_product_arm_is_named_not_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        md, tally = build(_run(tmp, with_product=False, failed=True))
        assert "FAILED" in md
        assert tally["product_failed"] == ["clipA"]


def test_the_two_paths_are_never_mixed_in_one_row():
    """Separate tables, deliberately: they are different code paths."""
    with tempfile.TemporaryDirectory() as tmp:
        md, _ = build(_run(tmp))
        cov = md.index("### Coverage and correctness")
        prod = md.index("### The product path")
        assert cov < prod
        # the coverage table's rows must not carry a product column
        header = md[cov:prod].split("\n")[2]
        assert "product" not in header
