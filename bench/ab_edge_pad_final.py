"""Does the frame-edge fix survive to the FINAL alpha?  (5.10, Job 1, 17 Sep)

Why this exists
---------------
`maskrcnn_edge_pad` (replicate-pad the frame before Mask R-CNN, crop the masks
back) is in the tree at default 0.  Every number supporting it was measured on
the SEED, and the matting engine loses another 1.5-2.5x on top (microsoft seed
19 px short -> alpha 36; ipman 19 -> 46).  So nothing yet shows that the seed
gain reaches the picture Runbo looks at.  This runs the shipped product path
(`Pipeline.run_clip`) once per arm and measures the alpha it hands back.

Arms are config only.  The default set is

    pad0         maskrcnn_edge_pad = 0   (shipped)
    pad64        maskrcnn_edge_pad = 64  (under test)
    pad0_repeat  maskrcnn_edge_pad = 0   again

`pad0_repeat` is the control.  The pipeline is not bit-deterministic (HANDOFF
s7 entry thirty-one: up to 0.00089 per frame on butter with nothing changed),
so pad0 vs pad0_repeat is the yardstick every pad64 difference is read against.
A difference that does not clear it is not a difference.

What is measured, per clip, per frame, per frame edge (bottom/top/left/right):

    cols        columns BOTH main arms hold solidly (mean alpha > 0.95) in the
                band 80-100 px in from that edge.  Shared across arms, so the
                A/B compares the same columns.  NOTE: a shared column is not
                promised to reach the edge -- on interview a sleeve slopes and
                floor appears below it (16 Sep retraction) -- so absolute values
                are not a defect score.  The ARM DIFFERENCE is the result.
    edge_alpha  mean SOFT alpha on the edge row, over cols
    band8       mean soft alpha over the 8 rows nearest the edge (the metric
                the 7-11 Sep defect numbers were quoted in)
    absent      share of cols below 0.5 on the edge row
    trunc       share of cols below 0.05 on the edge row (truncation, not
                softness -- the 15 Sep finding)
    short_px    median distance from the edge to the last >0.5 pixel
    w0 / w90    >0.5 width on the edge row / in a band 85-95 px in.  The
                dilation control: a recovering fix moves w0 and leaves w90
                flat; a dilating one moves both.

Plus, per clip:

    interior churn   mean |a_arm - a_pad0| over pixels >= 100 px from every
                     edge, and the share differing by > 0.5.  The frame-edge
                     fix should do nothing there; the control says how much
                     "nothing" is.
    integrity        standalone regions >= 0.5% of frame held by one arm and
                     not the other (dilated 15 px), per frame.  Every edge
                     metric above conditions on held foreground and cannot see
                     a subject vanish -- s7 entries 34-36.  This can.
    engine loss      at each shot's seed frame: the seed the engine was handed
                     (after erode_dilate) vs the alpha it returned, on the same
                     columns.  Job 2's quantity, recorded here for free.

And magenta look-sheets composited from the SOFT alpha (no threshold, no
brightness floor -- s7 entries 24 and 27): source | pad0 | pad64 | difference,
full frame plus a 2x zoom on the edge with the most held columns.

Usage (on the GPU VM):

    python -u bench/ab_edge_pad_final.py --config /content/pipe.json \
        --out /content/ab_edgepad edge/*.mp4

Only numbers, sheets and uint8 alphas are written; one zip comes off the VM.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

EDGES = ("bottom", "top", "left", "right")
BAND_LO, BAND_HI = 80, 100      # reference band, px in from the edge
W90_LO, W90_HI = 85, 95
INTERIOR = 100
MIN_COLS = 20
MIN_REGION = 0.005              # of frame; dance's accepted miss is ~0.004
MAGENTA = np.array([255, 0, 255], np.float32)   # BGR


# --------------------------------------------------------------------------- #
# metrics -- pure numpy, unit-tested in tests/test_ab_edge_pad.py
# --------------------------------------------------------------------------- #

def orient(a: np.ndarray, edge: str) -> np.ndarray:
    """View of `a` with `edge` as the LAST row.  Columns keep their meaning."""
    if edge == "bottom":
        return a
    if edge == "top":
        return a[::-1]
    if edge == "left":
        return a.T[::-1]          # last row = column 0
    if edge == "right":
        return a.T                # last row = column W-1
    raise ValueError(edge)


def ref_columns(arms: list[np.ndarray], edge: str) -> np.ndarray:
    """Columns every given arm holds solidly in the reference band."""
    ok = None
    for a in arms:
        b = orient(a, edge)
        D = b.shape[0]
        solid = b[D - BAND_HI:D - BAND_LO].mean(axis=0) > 0.95
        ok = solid if ok is None else (ok & solid)
    return np.nonzero(ok)[0]


def edge_stats(a: np.ndarray, edge: str, cols: np.ndarray) -> dict | None:
    if len(cols) < MIN_COLS:
        return None
    b = orient(a, edge)
    D = b.shape[0]
    last = b[-1, cols]
    held = b[:, cols] > 0.5
    # distance from the edge to the last held pixel, scanning the whole column
    rev = held[::-1]
    first = np.where(rev.any(axis=0), rev.argmax(axis=0), D)
    return dict(
        n=int(len(cols)),
        edge_alpha=float(last.mean()),
        band8=float(b[-8:, cols].mean()),
        absent=float((last < 0.5).mean()),
        trunc=float((last < 0.05).mean()),
        short_px=float(np.median(first)),
        w0=int((b[-1] > 0.5).sum()),
        w90=int((b[D - W90_HI:D - W90_LO] > 0.5).any(axis=0).sum()),
    )


def interior_churn(a: np.ndarray, ref: np.ndarray) -> dict:
    H, W = a.shape
    s = (slice(INTERIOR, H - INTERIOR), slice(INTERIOR, W - INTERIOR))
    d = np.abs(a[s] - ref[s])
    return dict(mean_abs=float(d.mean()), flip=float((d > 0.5).mean()))


def standalone_only(a: np.ndarray, b: np.ndarray, grow: int = 15) -> list[float]:
    """Areas (share of frame) of regions `a` holds that `b` does not come near."""
    A = (a > 0.5).astype(np.uint8)
    B = cv2.dilate((b > 0.5).astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    only = A & (1 - B)
    n, lab, st, _ = cv2.connectedComponentsWithStats(only, connectivity=8)
    tot = a.size
    return sorted((float(st[i, cv2.CC_STAT_AREA]) / tot for i in range(1, n)
                   if st[i, cv2.CC_STAT_AREA] / tot >= MIN_REGION), reverse=True)


def over_magenta(frame: np.ndarray, a: np.ndarray) -> np.ndarray:
    a3 = a[..., None].astype(np.float32)
    return (frame.astype(np.float32) * a3 + MAGENTA * (1 - a3)).clip(0, 255).astype(np.uint8)


def diff_panel(frame: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """Grey source; GREEN where pad64 holds more, RED where it holds less."""
    g = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = (g.astype(np.float32) * 0.35)
    d = (a1 - a0).astype(np.float32)
    out[..., 1] += np.clip(d, 0, 1) * 255
    out[..., 2] += np.clip(-d, 0, 1) * 255
    return out.clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# sheets
# --------------------------------------------------------------------------- #

def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _fit(img, w):
    return cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])),
                      interpolation=cv2.INTER_AREA)


def zoom_strip(img, edge, depth=160, scale=2):
    """The `depth` px along `edge`, rotated so the edge is at the bottom."""
    if edge == "bottom":
        s = img[-depth:]
    elif edge == "top":
        s = img[:depth][::-1]
    elif edge == "left":
        s = np.rot90(img[:, :depth], k=1)      # column 0 -> bottom row
    else:
        s = np.rot90(img[:, -depth:], k=-1)    # column W-1 -> bottom row
    s = np.ascontiguousarray(s)
    return cv2.resize(s, (s.shape[1] * scale, s.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def build_sheet(frame, arms: dict, fi: int, edge: str, stats_txt: str,
                path: str, panel_w: int = 640):
    a0, a1 = arms["pad0"][fi], arms["pad64"][fi]
    full = [_label(_fit(frame, panel_w), f"source f{fi}"),
            _label(_fit(over_magenta(frame, a0), panel_w), "pad 0 (shipped)"),
            _label(_fit(over_magenta(frame, a1), panel_w), "pad 64"),
            _label(_fit(diff_panel(frame, a0, a1), panel_w),
                   "green = 64 adds, red = 64 removes")]
    row1 = np.hstack(full)
    zs = [zoom_strip(frame, edge), zoom_strip(over_magenta(frame, a0), edge),
          zoom_strip(over_magenta(frame, a1), edge),
          zoom_strip(diff_panel(frame, a0, a1), edge)]
    zw = row1.shape[1]
    z = np.vstack([_fit(_label(x, n), zw) for x, n in zip(
        zs, [f"{edge} edge, 2x", "pad 0", "pad 64", "difference"])])
    head = np.zeros((40, zw, 3), np.uint8)
    cv2.putText(head, stats_txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([head, row1, z]), [cv2.IMWRITE_JPEG_QUALITY, 90])


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def _seed_mask_at(pipe, cfg, frame):
    from vbgr import refine
    from vbgr.detect import held_props, select_person_boxes
    from vbgr.seed import build_seed
    H, W = frame.shape[:2]
    people, props = pipe.detector.detect(frame)
    kept, _ = select_person_boxes(
        people, W, H, cfg.detect.person_rel_size_min, cfg.detect.box_score_ratio,
        cfg.detect.person_rel_area_min, cfg.detect.conf_abs_min)
    if not kept:
        return None
    seed = build_seed(frame, pipe.seeder, kept, held_props(props, kept), cfg.seed)
    return refine.erode_dilate(seed.mask, cfg.matting.r_erode,
                               cfg.matting.r_dilate).astype(np.float32)


def run_arm(base: dict, pad: int, clip: str, out_dir: str, band: bool = False):
    from vbgr.config import Config
    from vbgr.pipeline import Pipeline
    cfg = Config.from_dict(copy.deepcopy(base))
    cfg.seed.maskrcnn_edge_pad = int(pad)
    cfg.seed.maskrcnn_edge_pad_band = bool(band and pad)
    pipe = Pipeline(cfg)
    t0 = time.time()
    rep = pipe.run_clip(clip, out_dir=out_dir, keep_alphas=True)
    alphas = np.asarray(rep.alphas, np.float32)
    shots = [dict(start=s.start, end=s.end, seed_frame=s.seed_frame,
                  n_kept=s.n_kept, bad_frames=s.bad_frames,
                  reentries=s.reentries, notes=list(s.notes),
                  n_objects=(len(s.object_areas[0]) if s.object_areas else None))
             for s in rep.shots]
    seeds = {}
    from vbgr import video_io
    frames = video_io.read_all(clip, cfg.io.max_size)
    for s in rep.shots:
        fi = s.start + s.seed_frame
        m = _seed_mask_at(pipe, cfg, frames[fi])
        if m is not None:
            seeds[fi] = m
    meta = dict(pad=int(pad), band=bool(band and pad), seconds=round(time.time() - t0, 1),
                summary=rep.summary(), shots=shots,
                seeder_notes=list(getattr(pipe.seeder, "notes", [])))
    del pipe
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass
    return alphas, seeds, meta, frames


def summarise_clip(name, frames, arms, seeds, metas, out):
    T, H, W = arms["pad0"].shape
    res = dict(clip=name, frames=T, size=[W, H], meta=metas, edges={},
               churn={}, integrity={}, engine_loss={})

    # --- edges ------------------------------------------------------------ #
    for e in EDGES:
        per = {k: [] for k in arms}
        ncols = []
        for t in range(T):
            cols = ref_columns([arms["pad0"][t], arms["pad64"][t]], e)
            ncols.append(len(cols))
            for k, A in arms.items():
                per[k].append(edge_stats(A[t], e, cols))
        valid = [t for t in range(T) if per["pad0"][t] is not None]
        if len(valid) < 3:
            continue
        agg = {}
        for k in arms:
            agg[k] = {m: float(np.mean([per[k][t][m] for t in valid]))
                      for m in per[k][valid[0]]}
        d = [per["pad64"][t]["edge_alpha"] - per["pad0"][t]["edge_alpha"] for t in valid]
        c = ([per["pad0_repeat"][t]["edge_alpha"] - per["pad0"][t]["edge_alpha"]
              for t in valid] if "pad0_repeat" in arms else [])
        res["edges"][e] = dict(
            frames_measured=len(valid), mean_cols=float(np.mean([ncols[t] for t in valid])),
            arms=agg,
            delta_edge_alpha=dict(mean=float(np.mean(d)), min=float(np.min(d)),
                                  max=float(np.max(d)),
                                  frames_better=int(sum(x > 0.02 for x in d)),
                                  frames_worse=int(sum(x < -0.02 for x in d))),
            control_edge_alpha=(dict(mean=float(np.mean(c)),
                                     max_abs=float(np.max(np.abs(c)))) if c else None),
            per_frame=dict(t=valid, pad0=[per["pad0"][t]["edge_alpha"] for t in valid],
                           pad64=[per["pad64"][t]["edge_alpha"] for t in valid]))

    # --- interior churn & integrity --------------------------------------- #
    for k in arms:
        if k == "pad0":
            continue
        ch = [interior_churn(arms[k][t], arms["pad0"][t]) for t in range(T)]
        res["churn"][k] = dict(mean_abs=float(np.mean([x["mean_abs"] for x in ch])),
                               max_abs=float(np.max([x["mean_abs"] for x in ch])),
                               flip=float(np.mean([x["flip"] for x in ch])))
        lost = [standalone_only(arms["pad0"][t], arms[k][t]) for t in range(T)]
        new = [standalone_only(arms[k][t], arms["pad0"][t]) for t in range(T)]
        res["integrity"][k] = dict(
            frames_pad0_only=int(sum(bool(x) for x in lost)),
            max_pad0_only=float(max((x[0] for x in lost if x), default=0.0)),
            frames_arm_only=int(sum(bool(x) for x in new)),
            max_arm_only=float(max((x[0] for x in new if x), default=0.0)),
            worst_frames_pad0_only=[t for t, x in sorted(
                enumerate(lost), key=lambda p: -(p[1][0] if p[1] else 0))[:3] if x])

    # --- engine loss at seed frames ---------------------------------------- #
    for k in ("pad0", "pad64"):
        rows = []
        for fi, sm in seeds.get(k, {}).items():
            for e in EDGES:
                cols = ref_columns([arms["pad0"][fi], arms["pad64"][fi]], e)
                s_seed = edge_stats(sm, e, cols)
                s_alpha = edge_stats(arms[k][fi], e, cols)
                if s_seed and s_alpha:
                    rows.append(dict(frame=int(fi), edge=e, n=s_seed["n"],
                                     seed_short=s_seed["short_px"],
                                     alpha_short=s_alpha["short_px"],
                                     seed_edge=s_seed["edge_alpha"],
                                     alpha_edge=s_alpha["edge_alpha"]))
        res["engine_loss"][k] = rows

    # --- sheets ------------------------------------------------------------ #
    sheet_dir = os.path.join(out, "sheets")
    os.makedirs(sheet_dir, exist_ok=True)
    if res["edges"]:
        edge = max(res["edges"], key=lambda e: res["edges"][e]["mean_cols"]
                   * res["edges"][e]["frames_measured"])
        pf = res["edges"][edge]["per_frame"]
        deltas = np.array(pf["pad64"]) - np.array(pf["pad0"])
        pick = [0, T // 2, T - 1, pf["t"][int(np.argmax(deltas))],
                pf["t"][int(np.argmin(deltas))]]
    else:
        edge, pf, pick = "bottom", None, [0, T // 2, T - 1]
    # plus the worst integrity frames, so a vanished subject is looked at
    for k, v in res["integrity"].items():
        pick += v["worst_frames_pad0_only"][:1]
    seen = []
    for fi in pick:
        if fi in seen:
            continue
        seen.append(fi)
        txt = f"{name}  f{fi}  {edge} edge"
        if pf is not None and fi in pf["t"]:
            j = pf["t"].index(fi)
            txt += f"  edge alpha: pad0 {pf['pad0'][j]:.3f}  pad64 {pf['pad64'][j]:.3f}"
        build_sheet(frames[fi], arms, fi, edge, txt,
                    os.path.join(sheet_dir, f"{name}_f{fi:03d}.jpg"))
    res["sheet_edge"], res["sheet_frames"] = edge, seen
    return res


def print_table(all_res):
    print("\n==== AB_EDGEPAD_TABLE ====")
    print(f"{'clip':<22}{'edge':<7}{'cols':>6}{'pad0':>8}{'pad64':>8}{'delta':>8}"
          f"{'ctrl':>8}{'better/worse':>14}{'short0':>8}{'short64':>8}"
          f"{'w90 0':>7}{'w90 64':>7}")
    for r in all_res:
        for e, v in r["edges"].items():
            a = v["arms"]
            ctrl = v["control_edge_alpha"]["max_abs"] if v["control_edge_alpha"] else float("nan")
            print(f"{r['clip']:<22}{e:<7}{v['mean_cols']:>6.0f}"
                  f"{a['pad0']['edge_alpha']:>8.3f}{a['pad64']['edge_alpha']:>8.3f}"
                  f"{v['delta_edge_alpha']['mean']:>+8.3f}{ctrl:>8.3f}"
                  f"{v['delta_edge_alpha']['frames_better']:>7}/{v['delta_edge_alpha']['frames_worse']:<6}"
                  f"{a['pad0']['short_px']:>8.1f}{a['pad64']['short_px']:>8.1f}"
                  f"{a['pad0']['w90']:>7.0f}{a['pad64']['w90']:>7.0f}")
    print("\n-- interior churn vs pad0 (control first) and integrity --")
    for r in all_res:
        for k in r["churn"]:
            c, g = r["churn"][k], r["integrity"][k]
            print(f"{r['clip']:<22}{k:<12} churn {c['mean_abs']:.5f} (max {c['max_abs']:.5f})"
                  f" flip {c['flip']:.5f} | pad0-only frames {g['frames_pad0_only']}"
                  f" (max {g['max_pad0_only']:.4f})  arm-only frames {g['frames_arm_only']}"
                  f" (max {g['max_arm_only']:.4f})")
    print("\n-- engine loss at seed frames: seed short -> alpha short (px) --")
    for r in all_res:
        for k, rows in r["engine_loss"].items():
            for x in rows:
                print(f"{r['clip']:<22}{k:<7}f{x['frame']:<4}{x['edge']:<7}"
                      f"n {x['n']:>5}  seed {x['seed_short']:>6.1f} -> alpha {x['alpha_short']:>6.1f}"
                      f"   edge alpha seed {x['seed_edge']:.3f} -> {x['alpha_edge']:.3f}")
    print("==== END_TABLE ====")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", action="store_true",
                    help="run the pad64 arm in 5.10b band mode "
                         "(maskrcnn_edge_pad_band=True)")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the pad0_repeat arm (not recommended)")
    a = ap.parse_args()
    base = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    arms_def = [("pad0", 0), ("pad64", 64)] + ([] if a.no_control else [("pad0_repeat", 0)])
    all_res = []
    for clip in a.clips:
        name = os.path.splitext(os.path.basename(clip))[0]
        cdir = os.path.join(a.out, name)
        os.makedirs(cdir, exist_ok=True)
        arms, seeds, metas, frames = {}, {}, {}, None
        for label, pad in arms_def:
            print(f"\n=== {name} arm {label} (edge_pad={pad}, band={a.band and bool(pad)})", flush=True)
            al, sd, meta, frames = run_arm(base, pad, clip, os.path.join(cdir, "render_" + label),
                                           band=a.band)
            print(meta["summary"], flush=True)
            for s in meta["shots"]:
                print(f"   shot {s['start']}-{s['end']} seed_frame {s['seed_frame']} "
                      f"kept {s['n_kept']} objects {s['n_objects']} notes {s['notes']}", flush=True)
            arms[label], seeds[label], metas[label] = al, sd, meta
            np.savez_compressed(os.path.join(cdir, f"alpha_{label}.npz"),
                                a=(al * 255 + 0.5).astype(np.uint8))
        r = summarise_clip(name, frames, arms, seeds, metas, a.out)
        all_res.append(r)
        json.dump(all_res, open(os.path.join(a.out, "ab_edgepad.json"), "w"), indent=1)
        print_table([r])
    print_table(all_res)
    print("\nAB_EDGEPAD_DONE", flush=True)


if __name__ == "__main__":
    main()
