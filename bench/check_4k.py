"""4.3 -- does a 4K frame run without running out of memory?

Run this on the GPU VM. It needs no upload: it upscales a window clip that is
already there, so it can ride a warm VM at any point.

    python bench/check_4k.py                    # butter, 3840x1600
    python bench/check_4k.py --clip 1917 --width 3840

Why it is an upscale, stated plainly
------------------------------------
**There is no 4K footage in this project.** All 19 delivered sources and all 19
v1 mattes are 1920 wide -- checked 29 Aug, every one. The plan item said "the
3840/4096 clips run without running out of memory" and those clips do not
exist; the number came from an assumption, not from the delivery. That is the
same shape of problem as 3.3, whose test case also turned out not to be in the
set.

So this measures the thing 4.3 is actually about, which is **pixel count**, not
detail: VRAM and time scale with pixels, and an upscaled 3840x1600 frame costs
the tracker exactly what a native one would. It says nothing about matte
quality at 4K, and the report says so. Anything about real 4K detail needs real
4K footage, which is a Runs conversation.

Reported: peak CUDA memory, wall time, and whether the alpha comes back at the
input resolution -- a model that silently returns a downscaled alpha would pass
an OOM test and still be wrong.
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def upscale(src, dst, width):
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    height = int(round(width * h / w / 2)) * 2
    subprocess.check_call(
        ["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", src,
         "-vf", f"scale={width}:{height}:flags=lanczos",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-vsync", "passthrough", dst])
    return width, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="butter")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--win", default="/content/win")
    ap.add_argument("--frames", type=int, default=24)
    a = ap.parse_args()

    import torch
    from vbgr.engines import build_engine

    cand = [f for f in os.listdir(a.win) if f.startswith(a.clip + "_")]
    if not cand:
        raise SystemExit(f"no window clip for {a.clip} in {a.win}")
    src = os.path.join(a.win, cand[0])
    big = f"/content/{a.clip}_{a.width}.mp4"
    W, H = upscale(src, big, a.width)
    print(f"upscaled {cand[0]} -> {W}x{H}  (UPSCALED, not native 4K)")

    cap = cv2.VideoCapture(big)
    frames = []
    while len(frames) < a.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    print(f"{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")

    # seed: the whole first-frame subject area is not needed for a memory test,
    # a single centred box is enough to make the tracker do full-resolution work
    m = np.zeros((H, W), np.uint8)
    m[H // 5: H * 4 // 5, W // 3: W * 2 // 3] = 255

    eng = build_engine("sam2matting", repo_dir="/content/SAM2Matting",
                       checkpoint="/content/checkpoints/SAM2Matting-SAM3.pt",
                       device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        A = eng.matte(frames, seed_mask=None, seed_masks=[m])
        ok, err = True, None
    except RuntimeError as e:
        ok, err = False, f"{type(e).__name__}: {e}"
        A = None
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n4K PROBE  clip={a.clip}  {W}x{H}  frames={len(frames)}")
    print(f"  ran without OOM : {ok}")
    if err:
        print(f"  error           : {err[:300]}")
    print(f"  peak CUDA mem   : {peak:.2f} GB of {total:.1f} GB")
    print(f"  wall time       : {dt:.1f} s  ({dt / max(len(frames),1):.2f} s/frame)")
    if A is not None:
        print(f"  alpha shape     : {A.shape[1]}x{A.shape[2]} "
              f"({'native' if A.shape[1:] == (H, W) else 'NOT INPUT RESOLUTION'})")
    print("  NOTE: upscaled footage. This measures pixel cost, not 4K detail.")


if __name__ == "__main__":
    main()
