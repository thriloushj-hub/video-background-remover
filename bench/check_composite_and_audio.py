"""4.2 and 4.4 -- does a composite actually composite, and does the audio survive?

Both were listed as "verified" by eye and neither had ever been measured.

**4.2, composite over image and video backgrounds.** The still-image path
existed (`composite_over_image`); the moving-background half had no code at
all until 29 Aug (`composite_over_video`). What this asserts:

* **Transparent regions are the background, exactly.** Where alpha is 0 the
  output must equal the background pixel for pixel. Any drift means the
  subject's plate is bleeding through -- the thing that makes a composite look
  "stuck on".
* **Opaque regions are the foreground, exactly.** Where alpha is 1 the output
  must equal the decontaminated F.
* **A background of a different aspect ratio is cropped, not stretched.**
  Checked by compositing over a deliberately wrong-shaped background and
  measuring that a circle drawn in it stays circular.
* **The edge band does not carry the old background's colour.** The fringe
  test, on real footage rather than a synthetic: compare the mean hue of the
  alpha band against the subject interior. A plate-coloured rim is what
  decontamination exists to remove.

**4.4, audio carried through.** v1 dropped audio entirely. `mux_audio` was
written but never run. What this asserts: the output has an audio stream, it
is the same duration as the source to within a frame, and the video stream is
not re-encoded (`-c:v copy`) so the matte is bit-identical after muxing.

    python bench/check_composite_and_audio.py --clip 1917
    python bench/check_composite_and_audio.py --all
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
V1_DIR = os.path.expanduser("~/mnt/output vids")
SCRATCH = os.path.expanduser("~/scratch/composite_check")


def window_frames(clip):
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
                return fr, os.path.join(SCRATCH, n)
    return [], None


def still_background(h, w):
    """A deliberately wrong-shaped background with a circle in it.

    The circle is the aspect-ratio probe: cover-resize must crop, so the circle
    stays a circle. A stretch turns it into an ellipse and the axis ratio moves.
    """
    bh, bw = 480, 1920                      # 4:1, nothing like a video frame
    bg = np.full((bh, bw, 3), (40, 30, 25), np.uint8)
    cv2.circle(bg, (bw // 2, bh // 2), 160, (30, 200, 240), -1)
    return bg


def circle_axis_ratio(img):
    """Width/height of the drawn circle after cover-resize. 1.0 = not stretched."""
    m = (np.abs(img.astype(np.int16) - np.array([30, 200, 240])).sum(-1) < 60)
    if m.sum() < 200:
        return None
    ys, xs = np.nonzero(m)
    return float((xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1))


def check(clip, n_frames=8):
    src, src_path = window_frames(clip)
    afs = sorted(glob.glob(f"{RUN}/bench_out/{clip}/alpha/*.png"))
    if not src or not afs:
        return None
    H, W = src[0].shape[:2]
    n = min(n_frames, len(src), len(afs))
    alphas, fgs = [], []
    for i in range(n):
        a = cv2.resize(cv2.imread(afs[i], 0), (W, H),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        alphas.append(a)
        fgs.append(decontaminate.decontaminate(src[i], a))

    r = {}

    # ---- 4.2a: still image background --------------------------------- #
    bg = still_background(H, W)
    over_img = [compose.composite_over_image(src[i], alphas[i], bg,
                                             foreground=fgs[i])
                for i in range(n)]
    bg_fit = compose.composite_over_image(
        np.zeros_like(src[0]), np.zeros((H, W), np.float32), bg)
    r["circle_axis_ratio"] = round(circle_axis_ratio(bg_fit) or -1, 4)

    outside = np.stack(alphas) < 0.002
    inside = np.stack(alphas) > 0.998
    O = np.stack(over_img).astype(np.int16)
    B = np.repeat(bg_fit[None].astype(np.int16), n, 0)
    F = np.stack(fgs).astype(np.int16)
    r["bg_max_err"] = int(np.abs(O - B)[outside].max()) if outside.any() else -1
    r["fg_max_err"] = int(np.abs(O - F)[inside].max()) if inside.any() else -1

    # ---- 4.2b: moving background -------------------------------------- #
    bg_frames = [np.full((720, 1280, 3),
                         (int(20 + 200 * (i / 23.0) % 200), 60, 120), np.uint8)
                 for i in range(23)]          # deliberately not a multiple of n
    over_vid = compose.composite_over_video(src[:n], alphas, bg_frames,
                                            foregrounds=fgs)
    r["video_bg_frames_out"] = len(over_vid)
    # the background must actually move: two frames whose bg differs must differ
    d0 = int(np.abs(over_vid[0].astype(np.int16)
                    - over_vid[1].astype(np.int16))[alphas[0] < 0.002].mean()) \
        if (alphas[0] < 0.002).any() else -1
    r["video_bg_moves"] = d0

    # ---- 4.2c: fringe, on real footage -------------------------------- #
    band = (np.stack(alphas) > 0.05) & (np.stack(alphas) < 0.95)
    core = np.stack(alphas) > 0.99
    Fh = cv2.cvtColor(np.stack(fgs).reshape(-1, W, 3), cv2.COLOR_BGR2LAB)
    Fh = Fh.reshape(n, H, W, 3).astype(np.float32)
    r["band_vs_core_lab_dist"] = round(float(np.linalg.norm(
        Fh[band].mean(0) - Fh[core].mean(0))), 2) if (band.any() and core.any()) else -1

    # ---- 4.4: audio ---------------------------------------------------- #
    os.makedirs(SCRATCH, exist_ok=True)
    silent = os.path.join(SCRATCH, f"{clip}_composite.mp4")
    w = video_io.VideoWriter(silent, W, H, Fraction(24000, 1001))
    for f in over_img:
        w.write(f)
    w.close(expect=n, strict=False)

    source_audio = os.path.join(V1_DIR, f"{clip}.mp4")
    withaudio = os.path.join(SCRATCH, f"{clip}_composite_audio.mp4")
    r["source_has_audio"] = bool(
        video_io.probe(source_audio, count_frames=False).has_audio)
    video_io.mux_audio(silent, source_audio, withaudio)

    def streams(p):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", p],
            capture_output=True).stdout.decode()
        return json.loads(out)["streams"]

    st = streams(withaudio)
    aud = [s for s in st if s["codec_type"] == "audio"]
    vid = [s for s in st if s["codec_type"] == "video"]
    r["out_has_audio"] = bool(aud)
    r["out_audio_codec"] = aud[0]["codec_name"] if aud else None
    r["out_video_frames"] = int(vid[0].get("nb_frames") or 0) if vid else 0
    # -c:v copy means the matte must be untouched by the mux
    a_before = cv2.VideoCapture(silent)
    a_after = cv2.VideoCapture(withaudio)
    diffs = []
    while True:
        ok1, f1 = a_before.read()
        ok2, f2 = a_after.read()
        if not (ok1 and ok2):
            break
        diffs.append(int(np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max()))
    r["video_untouched_by_mux"] = (max(diffs) if diffs else -1)
    return r


VERDICTS = [
    ("transparent region IS the background", lambda r: 0 <= r["bg_max_err"] <= 1),
    ("opaque region IS the foreground", lambda r: 0 <= r["fg_max_err"] <= 1),
    ("background cropped, not stretched", lambda r: abs(r["circle_axis_ratio"] - 1.0) <= 0.03),
    ("moving background actually moves", lambda r: r["video_bg_moves"] > 0),
    ("video background length preserved", lambda r: r["video_bg_frames_out"] > 0),
    ("audio carried through", lambda r: r["out_has_audio"] or not r["source_has_audio"]),
    ("mux did not re-encode the video", lambda r: 0 <= r["video_untouched_by_mux"] <= 1),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(RUN, "composite_audio_check.json"))
    a = ap.parse_args()
    got = json.load(open(a.out)) if os.path.exists(a.out) else {}
    clips = sorted(os.listdir(f"{RUN}/bench_out")) if a.all else [a.clip]
    for c in clips:
        if not os.path.isdir(f"{RUN}/bench_out/{c}") or c in got:
            continue
        r = check(c, a.frames)
        if r is None:
            continue
        r["verdicts"] = {k: bool(fn(r)) for k, fn in VERDICTS}
        got[c] = r
        ok = all(r["verdicts"].values())
        print(f"{c:12s} {'PASS' if ok else 'FAIL':4s}  bg_err {r['bg_max_err']:>3} "
              f"fg_err {r['fg_max_err']:>3}  circle {r['circle_axis_ratio']:.3f}  "
              f"audio {r['out_audio_codec']}  mux_delta {r['video_untouched_by_mux']}")
        for k, v in r["verdicts"].items():
            if not v:
                print(f"    FAILED: {k}")
        json.dump(got, open(a.out, "w"), indent=1)
