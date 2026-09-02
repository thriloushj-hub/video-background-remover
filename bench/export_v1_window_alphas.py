"""Export v1's window alphas as PNGs, and optionally pack them for upload.

Why this exists
---------------
The product-path run of 2 Sep produced 240 MB of alphas on a Colab VM, and
they could not be brought down through the browser.  So the comparison went
the other way: v1's window alphas were carried *up*, and the scoring ran on
the VM beside the data.  That worked, and it was done from a scratch file in
/tmp, which is exactly how a method gets reinvented three weeks later.

The representation is deliberately the same one ``vbgr_bench.save_alphas``
persists for v2 -- 8-bit PNG, short side untouched, width capped at 960 -- so
neither side is ever compared at a resolution the other does not have.

    python bench/export_v1_window_alphas.py                 # -> bench/results/v1_alpha/
    python bench/export_v1_window_alphas.py --pack 8500000  # + upload-sized zips

``--pack`` splits the output into zips under the given byte budget, because
the browser upload path this was built for caps a single call at 10 MB.  On
the 15-clip set it produces two, at about 7.9 MB and 6.6 MB.

Caveat worth carrying: absolute ``halo_shell`` values computed from these PNGs
differ slightly from ones computed by resampling v1's video directly, because
the resize happens at a different point.  Compare arms measured the same way;
do not mix this against a run scored from the MP4s.
"""
import argparse
import json
import os
import sys
import zipfile

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vbgr.compose import alpha_from_greenscreen  # noqa: E402

# The v1 delivery folder. Machine-specific, so it comes from the environment:
# set VBGR_V1_DIR, or pass --v1-dir. Nothing here should hard-code one
# person's home directory -- this repo gets read by other people.
DEFAULT_V1_DIR = os.environ.get("VBGR_V1_DIR", "")
MAX_W = 960


def export(v1_dir, out_dir, baselines):
    base = json.load(open(baselines))
    os.makedirs(out_dir, exist_ok=True)
    done = []
    for clip, v in sorted(base.items()):
        # A voided baseline is not a missing one -- say which, and skip.
        if not v:
            print(f"{clip:12s} SKIPPED -- no v1 window baseline exists")
            continue
        if not v.get("trusted", True) or v.get("window") is None:
            print(f"{clip:12s} SKIPPED -- v1 baseline is voided, see its note")
            continue
        src = os.path.join(v1_dir, f"{clip}_matte.mp4")
        if not os.path.exists(src):
            print(f"{clip:12s} SKIPPED -- no matte file at {src}")
            continue
        lo, hi = v["window"]
        cap = cv2.VideoCapture(src)
        d = os.path.join(out_dir, clip)
        os.makedirs(d, exist_ok=True)
        i = n = 0
        while True:
            ok, f = cap.read()
            if not ok or i >= hi:
                break
            if lo <= i < hi:
                a = alpha_from_greenscreen(f, tuple(v["green_bgr"]))
                if a.shape[1] > MAX_W:
                    h = int(round(a.shape[0] * MAX_W / a.shape[1]))
                    a = cv2.resize(a, (MAX_W, h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"{d}/{n:04d}.png",
                            np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
                n += 1
            i += 1
        cap.release()
        if n != hi - lo:
            print(f"{clip:12s} WARNING: {n} frames, expected {hi - lo} -- "
                  f"the matte is shorter than the window")
        else:
            print(f"{clip:12s} {n} frames")
        done.append(clip)
    return done


def pack(out_dir, budget):
    clips = sorted(d for d in os.listdir(out_dir)
                   if os.path.isdir(os.path.join(out_dir, d)))
    parts, cur, size = [], [], 0
    for c in clips:
        d = os.path.join(out_dir, c)
        sz = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        if size + sz > budget and cur:
            parts.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += sz
    if cur:
        parts.append(cur)
    for i, part in enumerate(parts, 1):
        zp = os.path.join(os.path.dirname(out_dir), f"v1a_{i}.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for c in part:
                d = os.path.join(out_dir, c)
                for f in sorted(os.listdir(d)):
                    z.write(os.path.join(d, f), f"v1_alpha/{c}/{f}")
        print(f"  {os.path.basename(zp)}  {os.path.getsize(zp)/1e6:.2f} MB  "
              f"{len(part)} clips")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", default=os.environ.get("VBGR_V1_DIR", DEFAULT_V1_DIR))
    ap.add_argument("--out", default=os.path.join(_ROOT, "bench", "results", "v1_alpha"))
    ap.add_argument("--baselines", default=os.path.join(
        _ROOT, "bench", "results", "v1_window_baselines.json"))
    ap.add_argument("--pack", type=int, default=0,
                    help="also write zips under this many bytes each "
                         "(8500000 keeps each under the 10 MB upload cap)")
    a = ap.parse_args()
    if not os.path.isdir(a.v1_dir):
        raise SystemExit(f"v1 delivery folder not found: {a.v1_dir}\n"
                         f"pass --v1-dir or set VBGR_V1_DIR")
    done = export(a.v1_dir, a.out, a.baselines)
    print(f"\n{len(done)} clips -> {a.out}")
    if a.pack:
        pack(a.out, a.pack)
