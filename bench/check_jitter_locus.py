"""Where does v2's extra frame-to-frame change actually live?

If the extra jitter sits in the boundary band, it is edge shimmer on a real
alpha and it is not a subject moving in and out.  If it sits in the interior,
it is a defect.  Measured on the three clips with the biggest relative gap
plus the two hardest, so the answer isn't read off one clip.
"""
import glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.getcwd())
from vbgr.compose import alpha_from_greenscreen

V1D = os.path.expanduser("~/mnt/output vids")
RUN = "bench/results/run_2026-08-30"
BASE = json.load(open("bench/results/v1_window_baselines.json"))
CLIPS = ["microsoft", "tryguys", "interview", "butter", "ipman"]

def band_split(A):
    """mean |a_t - a_{t-1}| inside the boundary band vs the solid interior."""
    band_d, core_d = [], []
    for x, y in zip(A, A[1:]):
        core = ((x > 0.9) & (y > 0.9))
        solid = (x > 0.5).astype(np.uint8)
        er = cv2.erode(solid, np.ones((9, 9), np.uint8))
        dl = cv2.dilate(solid, np.ones((9, 9), np.uint8))
        band = (dl > 0) & (er == 0)
        d = np.abs(x - y)
        if band.any(): band_d.append(d[band].mean())
        if core.any(): core_d.append(d[core].mean())
    return float(np.mean(band_d)), float(np.mean(core_d))

print(f"{'clip':<11}{'band_v1':>9}{'band_v2':>9}{'core_v1':>9}{'core_v2':>9}")
for clip in CLIPS:
    v = BASE[clip]; lo, hi = v["window"]
    fs = sorted(glob.glob(f"{RUN}/bench_out/{clip}/alpha/*.png"))
    A2 = [cv2.imread(p, 0).astype(np.float32)/255 for p in fs]
    cap = cv2.VideoCapture(os.path.join(V1D, f"{clip}_matte.mp4")); A1=[]; i=0
    while True:
        ok, f = cap.read()
        if not ok or i >= hi: break
        if lo <= i < hi: A1.append(alpha_from_greenscreen(f, tuple(v["green_bgr"])))
        i += 1
    h, w = A2[0].shape
    A1 = [cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in A1[:len(A2)]]
    b1, c1 = band_split(A1); b2, c2 = band_split(A2)
    print(f"{clip:<11}{b1:>9.4f}{b2:>9.4f}{c1:>9.4f}{c2:>9.4f}")
