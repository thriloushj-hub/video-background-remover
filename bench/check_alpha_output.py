"""4.1 -- does the transparent output actually carry alpha, end to end?

v1 shipped a `transparency.webm` that renders **green, not transparent** (0.5).
That is the defect this checks for, and it is not a thing you can check by
looking at the code: the only proof is to encode with the real writer, decode
the file back, and measure the alpha channel that comes out.

Run it against a persisted-alpha benchmark run:

    python bench/check_alpha_output.py --clip butter
    python bench/check_alpha_output.py --all

What it asserts, in order of what would actually have caught v1's bug:

1.  **The container declares an alpha-capable stream.**  For VP9 in WebM that
    is the Matroska `alpha_mode: 1` tag, **not** the pixel format: the alpha
    lives in per-block side data and `ffprobe` reports the base plane, so a
    perfectly good transparent file reads as `yuv420p`.  Checking `pix_fmt`
    here is how you conclude a working file is broken -- it cost an hour on
    29 Aug.
2.  **Background pixels decode to alpha 0.**  v1's file failed exactly here:
    every pixel came back opaque and the "transparent" region was green paint.
3.  **The decoded alpha matches what went in** (correlation and mean absolute
    error against the source alpha).  Encoding is lossy; losing the *shape* is
    not.
4.  **No green in the transparent region.**  If the background were keyed
    rather than cut, the RGB under alpha 0 would sit near v1's key
    (151, 253, 119).  This measures the distance and fails if it is close.
5.  **Frame count survives.**  `-vsync passthrough` is there to stop ffmpeg
    re-deriving the rate; this proves it worked.

The composite is written from the **decontaminated** foreground, so this
doubles as the 3.5 visual check on real v2 alphas -- the fringe test that until
now had only ever run on a synthetic (see Foreground_Decontamination).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import zipfile
from fractions import Fraction

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vbgr import compose, decontaminate, video_io  # noqa: E402

RUN = os.path.join(_ROOT, "bench/results/run_2026-08-27")
V1_GREEN = (151, 253, 119)
SCRATCH = os.path.expanduser("~/scratch/alpha_check")


def source_frames(clip):
    for z in sorted(glob.glob(os.path.join(_ROOT, "bench_payload/clips_*.zip"))):
        zf = zipfile.ZipFile(z)
        for n in zf.namelist():
            if n.startswith(clip + "_"):
                os.makedirs(SCRATCH, exist_ok=True)
                zf.extract(n, SCRATCH)
                cap = cv2.VideoCapture(os.path.join(SCRATCH, n))
                fr = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    fr.append(f)
                return fr
    return []


def decode_rgba(path, w, h):
    """Decode the file back to raw RGBA -- the only honest way to read alpha.

    ``-c:v libvpx-vp9`` is load-bearing.  ffmpeg's *default* vp9 decoder
    silently discards VP9's alpha side data and hands back a fully opaque
    frame, with no warning and exit code 0.  Decoding a good file that way
    reports alpha 255 everywhere, which is indistinguishable from the v1
    defect this script exists to catch.
    """
    p = subprocess.run(
        [video_io.FFMPEG, "-v", "error", "-nostdin",
         "-c:v", "libvpx-vp9", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:400])
    a = np.frombuffer(p.stdout, np.uint8)
    n = a.size // (w * h * 4)
    return a[:n * w * h * 4].reshape(n, h, w, 4)


def probe_stream(path):
    """Returns (codec, pix_fmt, alpha_mode).

    ``alpha_mode`` is the Matroska tag and is the only field that says whether
    the file can carry transparency; ``pix_fmt`` describes the base plane and
    reads ``yuv420p`` on a perfectly good alpha file.
    """
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-of", "json", path], capture_output=True)
    s = json.loads(p.stdout.decode())["streams"][0]
    tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
    return s.get("codec_name"), s.get("pix_fmt"), tags.get("alpha_mode")


def check(clip, n_frames=12, decontam=True):
    src = source_frames(clip)
    afs = sorted(glob.glob(f"{RUN}/bench_out/{clip}/alpha/*.png"))
    if not src or not afs:
        return None
    H, W = src[0].shape[:2]
    n = min(n_frames, len(src), len(afs))
    alphas, bgra = [], []
    for i in range(n):
        a = cv2.resize(cv2.imread(afs[i], 0), (W, H),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        fg = decontaminate.decontaminate(src[i], a) if decontam else None
        alphas.append(a)
        bgra.append(compose.to_bgra(src[i], a, foreground=fg))

    os.makedirs(SCRATCH, exist_ok=True)
    out = os.path.join(SCRATCH, f"{clip}_transparent.webm")
    w = video_io.VideoWriter(out, W, H, Fraction(24000, 1001), alpha=True)
    for f in bgra:
        w.write(f)
    wrote = w.close(expect=n, strict=False)

    codec, pix, alpha_mode = probe_stream(out)
    dec = decode_rgba(out, W, H)
    da = dec[..., 3].astype(np.float32) / 255
    sa = np.stack(alphas)
    m = min(len(da), len(sa))
    da, sa = da[:m], sa[:m]

    bg = sa < 0.01                       # definitely background in the input
    bg_alpha_mean = float(da[bg].mean()) if bg.any() else float("nan")
    bg_alpha_p99 = float(np.percentile(da[bg], 99)) if bg.any() else float("nan")
    if da.std() < 1e-6 or sa.std() < 1e-6:
        corr = 0.0          # a constant plane correlates with nothing
    else:
        corr = float(np.corrcoef(da.ravel(), sa.ravel())[0, 1])
    mae = float(np.abs(da - sa).mean())

    rgb = dec[:m, ..., :3].astype(np.float32)
    key_rgb = np.array(V1_GREEN[::-1], np.float32)     # BGR key -> RGB
    d_green = float(np.linalg.norm(rgb[bg] - key_rgb, axis=-1).mean()) if bg.any() else float("nan")

    return {"frames_written": wrote, "frames_decoded": int(len(dec)),
            "codec": codec, "pix_fmt": pix, "alpha_mode": alpha_mode,
            "bg_alpha_mean": round(bg_alpha_mean, 5),
            "bg_alpha_p99": round(bg_alpha_p99, 5),
            "alpha_corr": round(corr, 5),
            "alpha_mae": round(mae, 5),
            "bg_dist_from_v1_green": round(d_green, 1),
            "size_kb": round(os.path.getsize(out) / 1024, 1),
            "path": out}


VERDICTS = [
    ("alpha-capable stream", lambda r: str(r.get("alpha_mode")) == "1"
     or r["pix_fmt"] in ("yuva420p", "yuva444p10le")),
    ("background decodes transparent", lambda r: r["bg_alpha_p99"] <= 0.02),
    ("alpha shape survives", lambda r: r["alpha_corr"] >= 0.98 and r["alpha_mae"] <= 0.02),
    ("not a green key", lambda r: r["bg_dist_from_v1_green"] >= 60),
    ("frame count preserved", lambda r: r["frames_written"] == r["frames_decoded"]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(RUN, "alpha_output_check.json"))
    args = ap.parse_args()
    got = json.load(open(args.out)) if os.path.exists(args.out) else {}
    clips = (sorted(os.listdir(f"{RUN}/bench_out")) if args.all else [args.clip])
    for c in clips:
        if not os.path.isdir(f"{RUN}/bench_out/{c}") or c in got:
            continue
        r = check(c, args.frames)
        if r is None:
            continue
        r["verdicts"] = {name: bool(fn(r)) for name, fn in VERDICTS}
        got[c] = r
        ok = all(r["verdicts"].values())
        print(f"{c:12s} {'PASS' if ok else 'FAIL'}  {r['codec']}/{r['pix_fmt']}"
              f"(alpha_mode={r['alpha_mode']})  "
              f"bg_alpha p99 {r['bg_alpha_p99']:.4f}  corr {r['alpha_corr']:.4f}  "
              f"mae {r['alpha_mae']:.4f}  green_dist {r['bg_dist_from_v1_green']:.0f}")
        for name, v in r["verdicts"].items():
            if not v:
                print(f"    FAILED: {name}")
        json.dump(got, open(args.out, "w"), indent=1)
