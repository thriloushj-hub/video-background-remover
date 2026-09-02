"""Run RVM -- attempt 1's method -- on OUR frozen benchmark windows.

Why not just use the interns' output folder
-------------------------------------------
`bg_remove_samples/output` holds four files named video.mp4 .. video3.mp4, on
footage that is not the delivery. There is no clip in common with the fifteen
windows every number in this project is measured on, so those files cannot be
a baseline -- they can only be looked at.

So instead of scavenging their outputs, reproduce their method on our windows.
That gives a genuine attempt-1 column, measured on exactly the frames v1 and v2
are measured on.

Fairness notes, because this is a comparison against someone else's work:
  * **resnet50**, the higher-quality of the two released variants, not the fast
    one. Runbo's complaint about this attempt was edge quality, so handicapping
    it on the axis under discussion would make the result meaningless.
  * `downsample_ratio=0.25`, which is the authors' own recommendation for
    1080p in the RVM README. Not tuned by us in either direction.
  * Alpha persisted the same way `vbgr_bench.save_alphas` does it -- 8-bit PNG,
    width capped at 960 -- so no side is compared at a resolution another does
    not have.

    python run_rvm.py --variant resnet50 --clips win --out rvm_alpha
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "RobustVideoMatting")
sys.path.insert(0, REPO)
MAX_W = 960


def load_model(variant, ckpt):
    from model import MattingNetwork
    m = MattingNetwork(variant).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m


def frames_of(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


@torch.no_grad()
def matte_clip(model, frames, downsample_ratio=0.25):
    """RVM is recurrent: the four rec states carry across frames and must not
    be reset mid-clip. Resetting them per frame is the classic way to make this
    model look far worse than it is."""
    rec = [None] * 4
    alphas = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        fgr, pha, *rec = model(t, *rec, downsample_ratio)
        a = pha[0, 0].cpu().numpy()
        alphas.append(a)
    return alphas


def save(alphas, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, a in enumerate(alphas):
        if a.shape[1] > MAX_W:
            h = int(round(a.shape[0] * MAX_W / a.shape[1]))
            a = cv2.resize(a, (MAX_W, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out_dir}/{i:04d}.png",
                    np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="resnet50",
                    choices=["resnet50", "mobilenetv3"])
    ap.add_argument("--clips", default="/home/claude/win")
    ap.add_argument("--out", default="/home/claude/rvm_alpha")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()

    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"rvm_{a.variant}.pth")
    model = load_model(a.variant, ckpt)
    torch.set_num_threads(os.cpu_count() or 4)

    files = sorted(f for f in os.listdir(a.clips) if f.endswith(".mp4"))
    for f in files:
        clip = f.split("_w")[0]
        if a.only and clip not in a.only:
            continue
        od = os.path.join(a.out, clip)
        if os.path.isdir(od) and len(os.listdir(od)) > 0:
            print(f"{clip:12s} already done, skipping", flush=True)
            continue
        t0 = time.time()
        fr = frames_of(os.path.join(a.clips, f))
        al = matte_clip(model, fr, a.ratio)
        save(al, od)
        cov = float(np.mean([(x > 0.5).mean() for x in al]))
        print(f"{clip:12s} {len(al)} frames  mean_area {cov:.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)
    print("RVM DONE")


if __name__ == "__main__":
    main()
