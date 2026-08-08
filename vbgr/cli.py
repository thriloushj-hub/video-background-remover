#!/usr/bin/env python3
"""vbgr2 command line.

    vbgr2 engines                       list engines and their licences
    vbgr2 selftest                      exercise the whole pipeline on CPU
    vbgr2 run  -i clips/ -o results/    batch a folder
    vbgr2 one  clip.mp4                 single clip
    vbgr2 config > my.yaml              dump the default config
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np


def cmd_engines(a) -> None:
    from .engines.base import list_engines
    infos = list_engines()
    w = max(len(k) for k in infos) + 2
    print(f"{'key':<{w}}{'name':<24}{'licence':<24}{'mode':<11}commercial")
    print("-" * (w + 24 + 24 + 11 + 11))
    for k, i in sorted(infos.items()):
        print(f"{k:<{w}}{i.name:<24}{i.license:<24}{i.mode:<11}"
              f"{'yes' if i.commercial_ok else 'NO'}")
    print("\ncapabilities:")
    for k, i in sorted(infos.items()):
        print(f"  {k}: {', '.join(sorted(i.capabilities)) or '-'}")
        if i.notes:
            print(f"      {i.notes}")
    print("\nNote: the two highest-quality engines are non-commercial. "
          "See docs/LICENSING.md before shipping.")


def cmd_config(a) -> None:
    from .config import Config
    import yaml
    print(yaml.safe_dump(Config().to_dict(), sort_keys=False))


def cmd_selftest(a) -> None:
    """End-to-end run on synthetic footage with no models and no GPU.

    Exercises decode/encode frame parity, shot splitting, the memory gate, the
    flow-adaptive trimap, refinement, decontamination and compositing.  If this
    fails, do not start a GPU batch.
    """
    import cv2
    from fractions import Fraction
    from . import compose, decontaminate, motion, refine, shots, video_io
    from .config import Config
    from .engines.base import build_engine
    from .gate import QualityGate

    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")

    print("synthesising a 60-frame clip with a moving disc and a hard cut...")
    T, H, W = 60, 180, 320
    frames: List[np.ndarray] = []
    for t in range(T):
        bg = 40 if t < 30 else 200
        f = np.full((H, W, 3), bg, np.uint8)
        f[:, :, 1] = (bg + 30) % 255
        cx = int(60 + 3.5 * t)
        cv2.circle(f, (cx, H // 2), 30, (250, 250, 250), -1)
        frames.append(f)
    seed = np.zeros((H, W), np.uint8)
    cv2.circle(seed, (60, H // 2), 30, 1, -1)

    # -- shots ---------------------------------------------------------- #
    starts = shots.compute_scene_cuts(frames, 0.35, 64, 8)
    rngs = shots.shot_ranges(starts, T)
    check("shot split finds the cut", any(abs(s - 30) <= 2 for s in starts),
          f"starts={starts}")
    check("shot ranges tile every frame", sum(b - a for a, b in rngs) == T)

    # -- engine --------------------------------------------------------- #
    eng = build_engine("passthrough")
    al = eng.matte(frames, seed)
    check("engine returns one alpha per frame", len(al) == T,
          f"{len(al)} vs {T}")

    # -- gate ----------------------------------------------------------- #
    g = QualityGate()
    g.prime(al[0], frames[0])
    v_good = g.evaluate(al[1], frames[1])
    v_bad = g.evaluate(np.zeros_like(al[0]), frames[2])
    check("gate accepts a good frame", v_good.ok, str(v_good))
    check("gate rejects a collapsed frame", not v_bad.ok, str(v_bad))

    # -- motion --------------------------------------------------------- #
    fe = motion.FlowEstimator()
    fl = fe.flow(frames[0], frames[1])
    mag = fe.magnitude(fl)
    check("flow detects the disc motion", float(np.percentile(mag, 99)) > 1.0,
          f"p99={float(np.percentile(mag, 99)):.2f}")
    tri_fixed = motion.flow_adaptive_trimap(al[1], None, 4.0)
    tri_flow = motion.flow_adaptive_trimap(al[1], mag, 4.0, 0.9, 40.0)
    check("flow widens the trimap band",
          (tri_flow == 128).sum() > (tri_fixed == 128).sum(),
          f"{(tri_flow==128).sum()} > {(tri_fixed==128).sum()}")

    warped, trust = motion.flow_alpha_prior(al[0], frames[0], frames[1], fl)
    check("warp trust is high on a clean warp", float(trust.mean()) > 0.4,
          f"{float(trust.mean()):.3f}")

    # -- decontamination ------------------------------------------------ #
    soft = cv2.GaussianBlur(al[5], (0, 0), 3.0)
    F = decontaminate.decontaminate(frames[5], soft)
    band = (soft > 0.02) & (soft < 0.98)
    d_in = float(np.abs(F.astype(int) - frames[5].astype(int)).mean(2)[band].mean())
    d_out = float(np.abs(F.astype(int) - frames[5].astype(int)).mean(2)[~band].mean())
    check("decontamination changes the band, not the interior",
          d_in > d_out, f"band={d_in:.2f} interior={d_out:.2f}")

    # -- refine --------------------------------------------------------- #
    gf = refine.guided_filter(soft, frames[5])
    check("guided filter stays in range", 0 <= gf.min() and gf.max() <= 1)
    st = refine.TemporalStabilizer()
    st.step(frames[0], al[0])
    check("temporal stabiliser runs", st.step(frames[1], al[1]).shape == (H, W))

    # -- compositing ---------------------------------------------------- #
    g_img = compose.composite_over_color(frames[5], soft, (0, 177, 64), F)
    rec = compose.alpha_from_greenscreen(g_img, (0, 177, 64))
    corr = float(np.corrcoef(rec.ravel(), soft.ravel())[0, 1])
    check("alpha survives a green round-trip", corr > 0.9, f"corr={corr:.3f}")
    bgra = compose.to_bgra(frames[5], soft, F)
    check("BGRA has 4 channels", bgra.shape[2] == 4)

    # -- video I/O frame parity ----------------------------------------- #
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.mp4")
        w = video_io.VideoWriter(p, W, H, Fraction(24000, 1001))
        for f in frames:
            w.write(f)
        n = w.close(expect=T, strict=False)
        back = video_io.read_all(p)
        info = video_io.probe(p)
        check("writer wrote every frame", n == T, f"{n} vs {T}")
        check("decode returns every frame", len(back) == T, f"{len(back)} vs {T}")
        check("fps is preserved exactly", info.fps == __import__(
            "fractions").Fraction(24000, 1001), str(info.fps))

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def cmd_run(a) -> None:
    from .config import Config
    from .pipeline import run_batch
    cfg = Config.load(a.config) if a.config else Config()
    if a.input:
        cfg.io.input_dir = a.input
    if a.output:
        cfg.io.output_dir = a.output
    if a.engine:
        cfg.matting.engine = a.engine
    if a.mode:
        cfg.io.output_mode = a.mode
    if a.max_size:
        cfg.io.max_size = a.max_size
    reports = run_batch(cfg, require_commercial=a.require_commercial)
    print(f"\n{len(reports)} clip(s)")
    for r in reports:
        print("  " + r.summary())


def cmd_one(a) -> None:
    from .config import Config
    from .pipeline import Pipeline
    cfg = Config.load(a.config) if a.config else Config()
    if a.engine:
        cfg.matting.engine = a.engine
    if a.mode:
        cfg.io.output_mode = a.mode
    if a.max_size:
        cfg.io.max_size = a.max_size
    rep = Pipeline(cfg, a.require_commercial).run_clip(a.clip, a.output)
    print(rep.summary())
    for k, v in rep.outputs.items():
        print(f"  {k}: {v}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="vbgr2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("engines").set_defaults(func=cmd_engines)
    sub.add_parser("config").set_defaults(func=cmd_config)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)

    common = dict(engine=None, mode=None, max_size=None)
    r = sub.add_parser("run")
    r.add_argument("-i", "--input")
    r.add_argument("-o", "--output")
    r.add_argument("-c", "--config")
    r.add_argument("--engine")
    r.add_argument("--mode", choices=["green", "alpha", "transparent", "all"])
    r.add_argument("--max-size", type=int)
    r.add_argument("--require-commercial", action="store_true",
                   help="refuse to run a non-commercially-licensed engine")
    r.set_defaults(func=cmd_run)

    o = sub.add_parser("one")
    o.add_argument("clip")
    o.add_argument("-o", "--output", default="results")
    o.add_argument("-c", "--config")
    o.add_argument("--engine")
    o.add_argument("--mode", choices=["green", "alpha", "transparent", "all"])
    o.add_argument("--max-size", type=int)
    o.add_argument("--require-commercial", action="store_true")
    o.set_defaults(func=cmd_one)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
