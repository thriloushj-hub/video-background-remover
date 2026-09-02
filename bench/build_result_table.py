"""Assemble the 5.1 result table from a completed run, and nothing else.

This exists so the headline table is *derived*, not typed.  Every previous
version of "v2 vs v1" in this project was hand-copied out of a cell, and three
of them carried a retracted number for days after the retraction.

It reads only files that are already on disk:

    bench_out/bench_results.json          v1 rows, v2 off/on arms, seeds
    coverage_disagreement_<source>.json   halo / hole / excess / cv / jitter
    halo_split.json                       shell vs solo

and refuses to invent anything the run does not contain.  Columns v1 cannot
supply honestly (``edge_soft``, ``sharpness``) are not in the comparison table
at all -- v1's alpha is recovered with ``clip((d-12)/28)``, a ramp that pins
sharpness to 0.654-0.673 across fifteen very different clips, so those columns
describe the recovery and not v1's matte.  They appear only in the ablation,
which is v2 against v2.

    python bench/build_result_table.py --run bench/results/run_2026-08-30
"""
import argparse
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run):
    bo = os.path.join(run, "bench_out", "bench_results.json")
    d = json.load(open(bo))
    dis = json.load(open(os.path.join(run, "coverage_disagreement_alpha.json")))
    split_path = os.path.join(run, "halo_split.json")
    hs = json.load(open(split_path)) if os.path.exists(split_path) else {}
    return d["rows"], d["v1"], dis["rows"], hs, dis.get("skipped", {})


def fmt(x, n=4):
    return "--" if x is None else f"{x:.{n}f}"


def build(run):
    rows, v1, dis, hs, skipped = load(run)
    clips = sorted(rows)
    out = []
    A = ["| clip | v1 subj f0 | v2 seeds | mean_area v1 | mean_area v2 | "
         "halo shell | halo solo | hole_big | excess rim px |",
         "|---|---|---|---|---|---|---|---|---|"]
    B = ["| clip | area_cv v1 | area_cv v2 | jitter v1 | jitter v2 | "
         "v1 dropouts | v2 subj_lost_at |",
         "|---|---|---|---|---|---|---|"]
    C = ["| clip | edge_soft off | edge_soft on | sharpness off | sharpness on |",
         "|---|---|---|---|---|"]
    tally = {"shell_max": 0.0, "solo": [], "hole": [], "jit_v2_worse": 0,
             "cv_v2_better": 0, "n": 0}
    for c in clips:
        r, b, dd = rows[c], v1.get(c) or {}, dis.get(c, {})
        h = hs.get(c, {})
        tally["n"] += 1
        tally["shell_max"] = max(tally["shell_max"], h.get("halo_shell", 0.0))
        if h.get("halo_solo", 0) > 0.02:
            tally["solo"].append(c)
        if dd.get("hole_big", 0) >= 0.01:
            tally["hole"].append(c)
        if dd.get("area_jitter_v2", 0) > dd.get("area_jitter_v1", 0):
            tally["jit_v2_worse"] += 1
        if dd.get("area_cv_v2", 9) < dd.get("area_cv_v1", 0):
            tally["cv_v2_better"] += 1
        A.append(f"| {c} | {b.get('subjects_f0','--')} | {r['n_seed']} | "
                 f"{fmt(b.get('mean_area'))} | {fmt(r['off']['mean_area'])} | "
                 f"{fmt(h.get('halo_shell'))} | {fmt(h.get('halo_solo'))} | "
                 f"{fmt(dd.get('hole_big'))} | {fmt(dd.get('excess_rim_px'),2)} |")
        B.append(f"| {c} | {fmt(dd.get('area_cv_v1'))} | {fmt(dd.get('area_cv_v2'))} | "
                 f"{fmt(dd.get('area_jitter_v1'))} | {fmt(dd.get('area_jitter_v2'))} | "
                 f"{b.get('dropouts','--')} | {r['off']['subj_lost_at']} |")
        if "on" in r:
            C.append(f"| {c} | {fmt(r['off']['edge_soft'])} | {fmt(r['on']['edge_soft'])} | "
                     f"{fmt(r['off']['sharpness'])} | {fmt(r['on']['sharpness'])} |")
    out += ["### Coverage and correctness", ""] + A + [""]
    out += ["### Stability", ""] + B + [""]
    if len(C) > 2:
        out += ["### Ablation: the motion fix, v2 against v2", ""] + C + [""]
    if skipped:
        out += ["### Skipped", ""]
        out += [f"- `{c}` -- {w}" for c, w in skipped.items()] + [""]
    return "\n".join(out), tally


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    md, tally = build(a.run)
    path = a.out or os.path.join(a.run, "result_table.md")
    open(path, "w").write(md)
    print(md)
    print(f"\nclips {tally['n']}   max halo shell {tally['shell_max']:.4f}")
    print(f"clips with a standalone halo component > 0.02: {tally['solo'] or 'none'}")
    print(f"clips with hole_big >= 0.01: {tally['hole'] or 'none'}")
    print(f"v2 jitterier than v1 on {tally['jit_v2_worse']} of {tally['n']}; "
          f"v2 lower area_cv on {tally['cv_v2_better']} of {tally['n']}")
    print(f"\nmd -> {path}")
