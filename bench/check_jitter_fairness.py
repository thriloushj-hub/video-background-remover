"""Is v2's higher area_jitter a real instability, or an artefact of
binarising a genuinely soft alpha at 0.5?

area_jitter counts pixels with alpha > 0.5.  v1's alpha comes from a hard
chroma ramp, clip((d-12)/28), so very few of its pixels sit near 0.5.  v2
produces real alpha, so a blurred limb has a whole band hovering around the
threshold -- and every frame those pixels flip is counted as area moving.

So measure the same jitter three ways:
  binary  : pixels > 0.5           (what area_jitter does)
  mass    : sum of alpha           (threshold-free)
  near    : share of pixels in 0.35-0.65, i.e. how much of each matte is
            sitting on the fence in the first place
"""
import json, os, sys, glob
import cv2, numpy as np
sys.path.insert(0, os.getcwd())
from vbgr.compose import alpha_from_greenscreen

V1D = os.path.expanduser("~/mnt/output vids")
RUN = "bench/results/run_2026-08-30"
BASE = json.load(open("bench/results/v1_window_baselines.json"))

def jit(series):
    a = np.asarray(series, np.float64); a = a[a > 0]
    return float(np.diff(np.log(a)).std()) if a.size >= 3 else 0.0

def v1_alphas(clip, lo, hi, key):
    cap = cv2.VideoCapture(os.path.join(V1D, f"{clip}_matte.mp4")); out=[]; i=0
    while True:
        ok, f = cap.read()
        if not ok or i >= hi: break
        if lo <= i < hi: out.append(alpha_from_greenscreen(f, tuple(key)))
        i += 1
    return out

print(f"{'clip':<11}{'binV1':>8}{'binV2':>8}{'massV1':>8}{'massV2':>8}{'nearV1':>8}{'nearV2':>8}")
rows={}
for clip in sorted(BASE):
    v = BASE[clip]
    if not v or not v.get("trusted", True): continue
    d = f"{RUN}/bench_out/{clip}/alpha"
    fs = sorted(glob.glob(d + "/*.png"))
    if not fs: continue
    lo, hi = v["window"]
    A1 = v1_alphas(clip, lo, hi, v["green_bgr"])
    if not A1: continue
    A2 = [cv2.imread(p, 0).astype(np.float32)/255 for p in fs]
    h, w = A2[0].shape
    A1 = [cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in A1[:len(A2)]]
    r = dict(
        bin_v1=jit([(x>0.5).mean() for x in A1]), bin_v2=jit([(x>0.5).mean() for x in A2]),
        mass_v1=jit([x.sum() for x in A1]),       mass_v2=jit([x.sum() for x in A2]),
        near_v1=float(np.mean([((x>0.35)&(x<0.65)).mean() for x in A1])),
        near_v2=float(np.mean([((x>0.35)&(x<0.65)).mean() for x in A2])),
    )
    rows[clip]=r
    print(f"{clip:<11}{r['bin_v1']:>8.4f}{r['bin_v2']:>8.4f}{r['mass_v1']:>8.4f}"
          f"{r['mass_v2']:>8.4f}{r['near_v1']:>8.4f}{r['near_v2']:>8.4f}")
json.dump(rows, open(f"{RUN}/jitter_probe.json","w"), indent=1)
b1=sum(r['bin_v2']>r['bin_v1'] for r in rows.values()); m1=sum(r['mass_v2']>r['mass_v1'] for r in rows.values())
print(f"\nv2 jitterier on binary: {b1}/{len(rows)}   on alpha mass: {m1}/{len(rows)}")
print("mean near-0.5 share  v1", round(np.mean([r['near_v1'] for r in rows.values()]),5),
      " v2", round(np.mean([r['near_v2'] for r in rows.values()]),5))
