"""Does a change to the seeder LOSE a subject?  (5.10, 16 Sep)

Why this exists
---------------
Every metric this project has for the frame edge -- `seed_tail_fraction`,
`absent at edge`, `edge alpha`, `w0`, `w90`, the halo split -- conditions on
columns or regions that are ALREADY held foreground.  A subject that disappears
entirely has no such columns, so **none of them can see it**.  On 15-16 Sep that
gap let two wrong conclusions through in one session: first that a pad arm was
safe (it had not been checked in that direction at all), then that it had lost a
person (it had merged a bystander).  HANDOFF s7 entry twenty-seven is the same
shape one level up.

So: match every subject present in the baseline against the arm, by mask IoU
AND by coverage, and report three distinct outcomes rather than one.

  LOST    nothing in the arm covers this subject      -> a real regression
  MERGED  covered, but absorbed into another instance -> a 3.2a regression:
          one obj_id per person is the largest correctness fix in this project,
          and a merged pair is exactly the configuration that cost ipman a
          fighter at frame 7 before 3.2a
  HELD    matched one-to-one

And it reports each against the SEED GATE's own height test, because 3.3g/3.3h
decided that bystanders below `person_rel_size_min` are dropped on purpose --
1917's trench soldiers and leo's background figure.  Losing one of those is the
intended behaviour, not a defect, and a check that cannot tell the difference
will cry wolf.  That distinction is the whole point of this script.

Usage:  python bench/check_subject_integrity.py CLIP.mp4 [--pad 64] [--frames 0,24,48]
"""
import argparse, os, sys
import cv2, numpy as np, torch
from torchvision.models.detection import (maskrcnn_resnet50_fpn,
                                          MaskRCNN_ResNet50_FPN_Weights)

MIN_AREA = 0.005     # dance's accepted miss is ~0.4% of frame; at/above this we care
GATE_HR  = 0.60      # SeedConfig.person_rel_size_min
IOU_MATCH = 0.25
COV_MERGE = 0.60


def _det():
    m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
    if torch.cuda.is_available():
        m = m.cuda()
    return m


def people(det, img, score_min=0.80):
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
    if torch.cuda.is_available():
        t = t.cuda()
    with torch.no_grad():
        o = det([t])[0]
    s = (o["labels"] == 1) & (o["scores"] > score_min)
    return (o["masks"][s, 0].detach().cpu().numpy(),
            o["boxes"][s].detach().cpu().numpy().astype(float),
            o["scores"][s].detach().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--pad", type=int, default=64)
    ap.add_argument("--frames", default="0,24,48")
    ap.add_argument("--score-min", type=float, default=0.80)
    a = ap.parse_args()
    idx = [int(x) for x in a.frames.split(",")]
    det = _det()
    cap = cv2.VideoCapture(a.clip)
    real_lost = real_merged = 0
    print(f"{os.path.basename(a.clip)}  pad={a.pad}  gate h_ratio>={GATE_HR}\n")
    for fi in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]
        m0, b0, s0 = people(det, f, a.score_min)
        if len(m0) == 0:
            print(f"  f{fi}: no baseline subjects"); continue
        P = a.pad
        img = f if P == 0 else cv2.copyMakeBorder(f, P, P, P, P, cv2.BORDER_REPLICATE)
        m1, b1, s1 = people(det, img, a.score_min)
        if P and len(m1):
            m1 = m1[:, P:P + H, P:P + W]
            keep = [i for i in range(len(m1)) if (m1[i] > 0.5).sum() > 500]
            m1 = m1[keep]
        B1 = [(m1[j] > 0.5) for j in range(len(m1))]
        tallest = max(b[3] - b[1] for b in b0)
        print(f"  f{fi}: baseline {len(m0)} -> arm {len(B1)}")
        for i in range(len(m0)):
            A = (m0[i] > 0.5)
            if A.sum() / (H * W) < MIN_AREA:
                continue
            hr = (b0[i][3] - b0[i][1]) / tallest
            iou = max((((A & x).sum() / max(1, (A | x).sum())) for x in B1), default=0.0)
            cov = max((((A & x).sum() / max(1, A.sum())) for x in B1), default=0.0)
            if iou >= IOU_MATCH:
                verdict = "HELD"
            elif cov >= COV_MERGE:
                verdict = "MERGED"
            else:
                verdict = "LOST"
            gated = hr >= GATE_HR
            if verdict != "HELD" and gated:
                real_lost += verdict == "LOST"
                real_merged += verdict == "MERGED"
            tag = "" if verdict == "HELD" else ("   <<< REAL" if gated else "   (bystander, gate drops it anyway)")
            print(f"     score {s0[i]:.2f}  h_ratio {hr:.3f}  area {A.sum()/(H*W):.4f}  "
                  f"iou {iou:.2f} cov {cov:.2f}  {verdict}{tag}")
    cap.release()
    print(f"\nREAL losses (gate-passing): LOST {real_lost}, MERGED {real_merged}")
    return 1 if (real_lost or real_merged) else 0


if __name__ == "__main__":
    sys.exit(main())
