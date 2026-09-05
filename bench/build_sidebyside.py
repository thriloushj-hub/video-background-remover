"""Source | v1 | v2, one frame wide, for a person to watch rather than read.

Why this exists
---------------
Every number in this project is a column, and the 2 Sep RVM baseline showed
what those columns are worth: they cannot separate attempt 1 from attempt 2,
while the client separates them at a glance.  So the deliverable he reviews has
to be footage, and it has to be like for like.

"Like for like" is the whole difficulty.  v1 did not deliver a matte -- it
delivered a **green composite**, and its alpha has to be recovered from that
composite the same way `bench/split_halo.py` recovers it, through
`alpha_from_greenscreen`.  Compositing v1's recovered alpha and v2's real alpha
over the *same* background is the only comparison that is not rigged in either
direction: put v1 on its own green and v2 on a new background and v2 wins on
presentation alone.

Alignment is the second difficulty and it has already cost this project ten
days (see Ipman_Window_Void.md).  v1's matte does not always start at source
frame 0 -- on ipman it covers source 160-484 of 501 -- so the panels are built
only over the overlap, the offset is an explicit argument rather than a guess,
and the frame range actually used is printed and burned into the panel.  A
side-by-side of two different shots is worse than no side-by-side.

    python bench/build_sidebyside.py \
        --source  "output vids/ipman.mp4" \
        --v1      "output vids/ipman_matte.mp4" \
        --v2-alpha out/ipman_alpha.mp4 \
        --offset  160 \
        --bg      studio \
        --out     ipman_v1_vs_v2.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vbgr.compose import alpha_from_greenscreen  # noqa: E402

LABEL_H = 34
GREEN = (151, 253, 119)   # the key auto-detected on every delivered clip


def frames_of(path, limit=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
        if limit and len(out) >= limit:
            break
    cap.release()
    return out


def fps_of(path, default=24.0):
    cap = cv2.VideoCapture(path)
    v = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return v if v and v > 1 else default


def make_bg(kind, h, w):
    """A background chosen to make errors visible, not to flatter the matte.

    'studio' is a soft vertical gradient with a vignette -- what the footage
    would plausibly be used over, and mid-tone enough that both a bright halo
    and a dark fringe show.  'checker' is the honest one: a hard checkerboard
    makes any semi-transparent error unmissable, which is exactly why it is
    the wrong choice for a client-facing review and the right one for a check.
    """
    if kind not in ("studio", "checker") and os.path.exists(kind):
        img = cv2.imread(kind)
        if img is None:
            raise SystemExit(f"could not read background image: {kind}")
        ih, iw = img.shape[:2]
        s = max(w / iw, h / ih)
        img = cv2.resize(img, (int(round(iw * s)), int(round(ih * s))))
        y = (img.shape[0] - h) // 2
        x = (img.shape[1] - w) // 2
        return img[y:y + h, x:x + w]

    if kind == "checker":
        sq = max(16, h // 24)
        yy, xx = np.mgrid[0:h, 0:w]
        m = (((yy // sq) + (xx // sq)) % 2).astype(np.uint8)
        return np.dstack([np.where(m, 205, 150).astype(np.uint8)] * 3)

    top = np.array([88, 74, 62], np.float32)      # BGR, cool slate
    bot = np.array([182, 170, 158], np.float32)   # warm light grey
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = (top[None, None, :] * (1 - t) + bot[None, None, :] * t)
    bg = np.repeat(bg, w, axis=1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    bg *= np.clip(1.06 - 0.30 * r, 0.55, 1.0)[:, :, None]
    return np.clip(bg, 0, 255).astype(np.uint8)


def over(frame, alpha, bg):
    a = alpha[:, :, None].astype(np.float32)
    return np.clip(frame.astype(np.float32) * a
                   + bg.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


def label(panel, text, sub=""):
    h, w = panel.shape[:2]
    strip = np.full((LABEL_H, w, 3), 24, np.uint8)
    cv2.putText(strip, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (240, 240, 240), 1, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(strip, sub, (w - tw - 12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (150, 150, 150), 1, cv2.LINE_AA)
    return np.vstack([strip, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--v1", required=True, help="v1's delivered *_matte.mp4, "
                                                "which is a GREEN COMPOSITE")
    ap.add_argument("--v2-alpha", required=True,
                    help="v2 alpha as written by the product path "
                         "(<name>_alpha.mp4), greyscale")
    ap.add_argument("--offset", type=int, default=0,
                    help="source frame index of v1 matte frame 0. ipman is 160; "
                         "everything else measured so far is 0. Do not guess: "
                         "a wrong offset compares two different shots, which is "
                         "the defect that voided ipman for ten days")
    ap.add_argument("--bg", default="studio",
                    help="studio | checker | path to an image")
    ap.add_argument("--max-w", type=int, default=640,
                    help="width of ONE panel; the output is 3x this")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = frames_of(a.source)
    v1c = frames_of(a.v1)
    v2a = frames_of(a.v2_alpha)
    if not src or not v1c or not v2a:
        raise SystemExit("one of the inputs decoded to zero frames")

    # v2's alpha is produced from the source, so index i of v2 is source i.
    # v1's composite starts at source frame `offset`.
    lo = max(0, a.offset)
    hi = min(len(src), len(v2a), a.offset + len(v1c))
    if hi - lo < 2:
        raise SystemExit(
            f"no overlap: source {len(src)}f, v1 {len(v1c)}f at offset "
            f"{a.offset}, v2 {len(v2a)}f")

    h0, w0 = src[0].shape[:2]
    w = a.max_w
    h = int(round(h0 * w / w0))
    bg = make_bg(a.bg, h, w)
    fps = fps_of(a.source)

    rng = f"source {lo}-{hi - 1} of {len(src)}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = None
    for i in range(lo, hi):
        s = cv2.resize(src[i], (w, h), interpolation=cv2.INTER_AREA)
        c1 = cv2.resize(v1c[i - a.offset], (w, h), interpolation=cv2.INTER_AREA)
        a1 = alpha_from_greenscreen(c1, GREEN)
        g2 = cv2.cvtColor(v2a[i], cv2.COLOR_BGR2GRAY) if v2a[i].ndim == 3 else v2a[i]
        a2 = cv2.resize(g2, (w, h),
                        interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255

        panel = np.hstack([
            label(s, "source", rng if i == lo else ""),
            label(over(s, a1, bg), "v1  (alpha recovered from its green key)"),
            label(over(s, a2, bg), "v2  (real alpha, product pipeline)"),
        ])
        if vw is None:
            vw = cv2.VideoWriter(a.out, fourcc, fps,
                                 (panel.shape[1], panel.shape[0]))
        vw.write(panel)
    vw.release()
    print(f"{os.path.basename(a.out)}  {hi - lo} frames @ {fps:.3f}  "
          f"{rng}  offset {a.offset}  bg {a.bg}")


if __name__ == "__main__":
    main()
