"""Locate the v1-not-v2 disagreement rather than just measuring its size.

Why this exists: on 7 Sep the blunt coverage measure flagged five clips at >1%
of frame and all five were dismissed as v1 holding furniture, on the strength
of a structure test and a look at the single worst frame.  Thriloush then found
two of them by eye in a minute: microsoft loses a strip along the bottom of the
subject, ipman shows green through a motion-blurred hand.  Both touch the held
subject, so `check_full_clip_solos.py` is blind to them by construction, and
both are inside a region the earlier check attributed to furniture.

So this splits the v1-not-v2 pixels into WHERE they are:

  bottom_edge  within `edge_px` of the bottom frame edge
  interior     enclosed by v2 foreground (a hole, at the 0.5 threshold)
  attached     touching the v2 subject but reaching open background
  separate     a standalone region, which is what solos already counts

    python bench/where_is_the_gap.py --v1 X_matte.mp4 --v2-alpha X_alpha.mp4 \
        --out-json X_gap.json [--offset N] [--every 1]
"""
import argparse, json, os, sys
import cv2, numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from vbgr.compose import alpha_from_greenscreen  # noqa: E402

GREEN = (151, 253, 119)


def classify(h1, h2, edge_px):
    """h1, h2 are boolean foreground masks. Returns per-class pixel counts."""
    gap = h1 & ~h2
    H, W = gap.shape
    out = dict(gap=int(gap.sum()), bottom_edge=0, interior=0, attached=0, separate=0)
    if out["gap"] == 0:
        return out
    bot = np.zeros_like(gap); bot[H - edge_px:, :] = True
    out["bottom_edge"] = int((gap & bot).sum())

    # Dilate v2 once; a gap component touching it is attached to the subject.
    k = np.ones((3, 3), np.uint8)
    near_v2 = cv2.dilate(h2.astype(np.uint8), k, iterations=2).astype(bool)
    n, lab = cv2.connectedComponents(gap.astype(np.uint8), connectivity=8)
    # A component is "interior" if dilating it stays inside v2 foreground.
    for i in range(1, n):
        comp = lab == i
        if not comp.any():
            continue
        ring = cv2.dilate(comp.astype(np.uint8), k, iterations=1).astype(bool) & ~comp
        c = int(comp.sum())
        if ring.any() and bool(h2[ring].all()):
            out["interior"] += c
        elif bool((comp & near_v2).any()):
            out["attached"] += c
        else:
            out["separate"] += c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2-alpha", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--edge-px", type=int, default=6)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    c1, c2 = cv2.VideoCapture(a.v1), cv2.VideoCapture(a.v2_alpha)
    for _ in range(max(0, a.offset)):
        c2.read()
    rows, n = [], 0
    while True:
        ok1, f1 = c1.read(); ok2, f2 = c2.read()
        if not (ok1 and ok2):
            break
        if n % a.every == 0:
            a1 = alpha_from_greenscreen(f1, GREEN)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if f2.ndim == 3 else f2
            a2 = cv2.resize(g2, (a1.shape[1], a1.shape[0]),
                            interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            r = classify(a1 > 0.5, a2 > 0.5, a.edge_px)
            r["frame"] = n
            r["px"] = int(a1.size)
            rows.append(r)
        n += 1
    c1.release(); c2.release()
    tot = {k: sum(r[k] for r in rows) for k in ("gap", "bottom_edge", "interior", "attached", "separate")}
    summary = {"clip": os.path.basename(a.v1), "frames_scored": len(rows), "totals": tot}
    if tot["gap"]:
        summary["share"] = {k: round(tot[k] / tot["gap"], 4)
                            for k in ("bottom_edge", "interior", "attached", "separate")}
    json.dump({"summary": summary, "per_frame": rows}, open(a.out_json, "w"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
