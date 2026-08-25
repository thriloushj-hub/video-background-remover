"""3.2 -- motion-blur fix A/B on the ipman fast-motion window.

    !cd /content && unzip -o vbgr_package.zip && python vbgr_ipman_ab.py all

Needs the v4 notebook cells 1-3 (SAM2Matting repo + checkpoints).  Everything
else it builds itself, so a recycled VM only costs the install.

Three arms, run in this order, all on the same 72 frames and the same
per-person seed masks (built once, cached to disk):

  single   one obj_id for every person unioned together -- what the adapter did
           before 2026-08-15.  Kept only as the ablation control, because the
           whole point is that it loses a subject.
  off      one obj_id PER PERSON, no motion fix.  This is the real A/B control.
  on       one obj_id per person, motion fix applied.

Why a separate script and not the pipeline
------------------------------------------
The motion fix lives in ``Pipeline._refine_sequence``, not in the engine.
Running the real pipeline would also drag in YOLO + SAM 3 seeding, which is a
second variable we do not want in an A/B.  So this replicates the *exact*
motion block from ``pipeline.py`` around the same engine call, with the same
MotionConfig / RefineConfig defaults.  Nothing is tuned.

Read the "on" arm honestly
--------------------------
SAM2Matting has no standalone (image, trimap) -> alpha head -- see
``SAM2MattingEngine.matte_frame``.  So band refinement falls back to a guided
filter: the widened band is *smoothed*, not re-matted.  The "on" arm is
therefore the flow-warped alpha prior plus a guided-filter band, NOT the motion
fix as designed.  Label it that way everywhere.

The A/B discipline
------------------
Every arm shares, byte for byte: the same frames, the same seed masks, the same
checkpoint, the same scoring code.  If an arm errors, that arm is void -- do
not "adjust a setting to get past it", because then the arms are no longer on
the same config and the comparison stops meaning anything.

Window provenance
-----------------
1b_ipman_fastwindow_3s.mp4 is source frames 318-389.  The ORIGINAL ipman window
was discarded: it straddled two hard cuts, so it measured what a memory
propagator does when carried across a cut, not matting quality.  This script
re-checks for cuts before it mattes anything and refuses to run if it finds
one.  See Shot_Boundaries.md.
"""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "/content")

CLIP = "/content/1b_ipman_fastwindow_3s.mp4"
WORK = "/content/ipman_ab"
FRAMES = f"{WORK}/frames"
REPO = "/content/SAM2Matting"
CKPT = "/content/checkpoints/SAM2Matting-SAM3.pt"

# v1 baseline on this exact window, native resolution, scored with this code.
V1_NATIVE = dict(area_cv=0.124, edge_soft=0.3792, dropouts=0, mean_area=0.2874)


# ---- scoring, identical to notebook cell 9 / bench/metrics.py ------------- #

def _band(a, w=6):
    hard = (a > 0.5).astype(np.uint8)
    if hard.max() == 0:
        return np.zeros_like(hard, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1,) * 2)
    return (cv2.dilate(hard, k) > 0) & (cv2.erode(hard, k) == 0)


def _edge_soft(a):
    b = _band(a)
    if not b.any():
        return 0.0
    v = a[b]
    return float(((v > 0.05) & (v < 0.95)).mean())


def _dropouts(A, drop=0.72, rec=0.9, win=24):
    ar = np.array([float((x > 0.5).sum()) for x in A])
    if len(ar) < win + 2:
        return 0
    n, i = 0, win
    while i < len(ar):
        med = float(np.median(ar[max(0, i - win):i]))
        if med > 0 and ar[i] < drop * med:
            j = i + 1
            while j < min(len(ar), i + win):
                if ar[j] > rec * med:
                    n += 1
                    i = j
                    break
                j += 1
            else:
                i = min(len(ar), i + win)
                continue
        i += 1
    return n


def n_subjects(a, min_frac=0.002):
    n, _l, s, _c = cv2.connectedComponentsWithStats((a > 0.5).astype(np.uint8), 8)
    return int(sum(1 for i in range(1, n) if s[i, 4] >= min_frac * a.size))


def subject_loss(A, n_seed):
    """First frame after which the subject count never returns to n_seed.

    ``dropouts`` cannot see this: it only counts a drop that RECOVERS within 24
    frames, so a person lost for good scores zero.  That is exactly what
    happened on the single-obj_id run, where mean_area halved and dropouts
    stayed at 0.
    """
    k = [n_subjects(a) for a in A]
    last = max((i for i, v in enumerate(k) if v >= n_seed), default=-1)
    return (None if last == len(A) - 1 else last + 1), k


def score(A, n_seed):
    frac = np.array([float((x > 0.5).mean()) for x in A])
    lost, _k = subject_loss(A, n_seed)
    return dict(area_cv=round(float(frac.std() / (frac.mean() + 1e-9)), 4),
                edge_soft=round(float(np.mean([_edge_soft(x) for x in A[::2]])), 4),
                dropouts=_dropouts(A),
                mean_area=round(float(frac.mean()), 4),
                subj_lost_at=("never" if lost is None else lost))


# ---- setup ---------------------------------------------------------------- #

def extract_frames():
    os.makedirs(FRAMES, exist_ok=True)
    if not os.listdir(FRAMES):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", CLIP,
                        "-q:v", "2", f"{FRAMES}/%06d.jpg"], check=True)
    names = sorted(os.listdir(FRAMES))
    return [cv2.imread(os.path.join(FRAMES, n)) for n in names]


def cut_guard(frames):
    """Refuse to run if the window contains a shot boundary."""
    from vbgr.shots import compute_scene_cuts, frame_distances
    starts = compute_scene_cuts(frames)
    hd, _ad = frame_distances(frames)
    print(f"shot starts: {starts}   (max hist dist {hd.max():.3f} at "
          f"frame {int(hd.argmax())})")
    if len(starts) > 1:
        sys.exit("ABORT: this window contains a cut at frames "
                 f"{starts[1:]}. Propagating a seed across a cut is the exact "
                 "defect that voided the previous ipman window. Pick a "
                 "different window before running anything.")
    print("cut guard: clean, single shot\n")


def build_seeds(frames):
    """Mask R-CNN + the PATCHED select_person_boxes, ONE MASK PER PERSON.

    Cached, so every arm and every re-run gets a byte-identical prompt.
    """
    H, W = frames[0].shape[:2]
    # seed_0.png, seed_1.png, ...  The digit in the glob matters: a bare
    # seed_*.png would also pick up union.png-style siblings and silently
    # inflate the subject count.
    cached = sorted(glob.glob(f"{WORK}/seed_[0-9]*.png"))
    if cached:
        seeds = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in cached]
        u = np.zeros_like(seeds[0])
        for m in seeds:
            u |= m
        print(f"seed: reused {len(seeds)} cached mask(s), union covers "
              f"{(u > 127).mean() * 100:.1f}% of frame")
        return seeds

    import torch
    import torchvision
    det = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT").eval().cuda()
    rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        out = det([torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).cuda()])[0]
    sel = (out["labels"] == 1) & (out["scores"] > 0.80)
    boxes = out["boxes"][sel].detach().cpu().numpy()
    masks = out["masks"][sel, 0].detach().cpu().numpy() > 0.5
    confs = out["scores"][sel].detach().cpu().numpy()

    from vbgr.detect import Detection, select_person_boxes
    from vbgr.config import DetectConfig
    cfg = DetectConfig()
    dets = [Detection(box=tuple(map(float, boxes[i])), conf=float(confs[i]),
                      cls=0, label="person") for i in range(len(boxes))]
    pos = {id(d): i for i, d in enumerate(dets)}
    kept, dropped = select_person_boxes(
        dets, W, H, cfg.person_rel_size_min, cfg.box_score_ratio,
        cfg.person_rel_area_min)
    keep = [pos[id(d)] for d in kept]
    print(f"{len(boxes)} people detected -> {len(keep)} kept, "
          f"{len(dropped)} dropped  "
          f"[height>={cfg.person_rel_size_min}, area>={cfg.person_rel_area_min}]")

    os.makedirs(WORK, exist_ok=True)
    seeds = []
    for n, i in enumerate(keep):
        m = masks[i].astype(np.uint8) * 255
        cv2.imwrite(f"{WORK}/seed_{n}.png", m)
        seeds.append(m)
        print(f"  subject {n}: {(m > 127).mean() * 100:.1f}% of frame")
    union = np.zeros((H, W), np.uint8)
    for m in seeds:
        union |= m
    cv2.imwrite(f"{WORK}/union.png", union)
    print(f"seed: {len(seeds)} mask(s) cached, union covers "
          f"{(union > 127).mean() * 100:.1f}% of frame")
    return seeds


# ---- the motion fix, lifted verbatim from Pipeline._refine_sequence ------- #

def apply_motion_fix(alphas, frames, eng):
    from vbgr import motion, refine
    from vbgr.config import MotionConfig, RefineConfig
    mcfg, rcfg = MotionConfig(), RefineConfig()

    flow = motion.FlowEstimator(mcfg.flow_method, mcfg.flow_scale_short_side)
    head = eng.matte_frame if (rcfg.band_refine
                               and eng.has("per_frame_head")) else None
    print(f"motion fix ON  | flow={mcfg.flow_method} "
          f"high_motion_px={mcfg.high_motion_px} "
          f"prior_w={mcfg.warp_prior_weight} "
          f"trimap={mcfg.trimap_base}+{mcfg.trimap_flow_gain}*mag "
          f"| band_refine head={'real' if head else 'NONE -> guided filter'}")
    if head is None:
        print("  NB: this arm is the flow-warped alpha prior plus a "
              "guided-filter band.\n      It is NOT the motion fix as "
              "designed -- SAM2Matting has no per-frame\n      matting head. "
              "Say so in the write-up.")

    out = [alphas[0].copy()]
    n_high = 0
    for i in range(1, len(frames)):
        prev, f = frames[i - 1], frames[i]
        a = alphas[i].copy()
        fl = flow.flow(prev, f)
        mag = flow.magnitude(fl, mcfg.flow_smooth_sigma)
        if float(np.percentile(mag, 95)) > mcfg.high_motion_px:
            n_high += 1
            warped, trust = motion.flow_alpha_prior(
                out[i - 1], prev, f, fl, mcfg.warp_consistency_thresh)
            a = motion.blend_with_prior(
                a, warped, trust, mag, mcfg.warp_prior_weight,
                mcfg.high_motion_px)
            if rcfg.band_refine:
                tri = motion.flow_adaptive_trimap(
                    a, mag, mcfg.trimap_base, mcfg.trimap_flow_gain,
                    mcfg.trimap_max)
                a = refine.refine_band(a, f, tri, head)
        out.append(a)
    print(f"high-motion frames: {n_high} of {len(frames)}\n")
    return np.stack(out)


# ---- contact sheet -------------------------------------------------------- #

def contact_sheet(frames, alphas, path, cols=6, rows=4):
    n = cols * rows
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    tw = 320
    th = int(tw * frames[0].shape[0] / frames[0].shape[1])
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, i in enumerate(idx):
        a = alphas[i][:, :, None]
        comp = (frames[i].astype(np.float32) * a
                + np.array([0, 255, 0], np.float32) * (1 - a)).astype(np.uint8)
        t = cv2.resize(comp, (tw, th))
        cv2.putText(t, f"{i}  n={n_subjects(alphas[i])}", (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"contact sheet -> {path}")


# ---- main ----------------------------------------------------------------- #

def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    arms = {"single": ["single"], "off": ["off"], "on": ["on"],
            "all": ["single", "off", "on"]}.get(mode)
    if arms is None:
        sys.exit("usage: python vbgr_ipman_ab.py [single|off|on|all]")

    for p in (CLIP, REPO, CKPT):
        if not os.path.exists(p):
            sys.exit(f"MISSING: {p}\n"
                     f"Run the v4 notebook cells 1-3 first (repos + "
                     f"checkpoints), and make sure the clip is in /content.")
    os.makedirs(WORK, exist_ok=True)

    frames = extract_frames()
    H, W = frames[0].shape[:2]
    print(f"{len(frames)} frames at {W}x{H}  (source frames 318-389)\n")

    cut_guard(frames)
    seeds = build_seeds(frames)
    n_seed = len(seeds)
    union = np.zeros((H, W), np.uint8)
    for m in seeds:
        union |= m
    print()

    from vbgr.engines import build_engine
    eng = build_engine("sam2matting", repo_dir=REPO, checkpoint=CKPT,
                       device="cuda")
    print(f"engine: {eng.info.name} | {eng.info.license} | mode "
          f"{eng.info.mode} | per_frame_head={eng.has('per_frame_head')}\n")

    runs, base_multi = {}, None
    for arm in arms:
        print("-" * 74)
        if arm == "single":
            print("ARM single -- every person unioned into obj_id=1 (the old "
                  "behaviour, kept\n           as the ablation control)")
            eng.reset()
            A = eng.matte(frames, seed_mask=union, seed_masks=[union])
        elif arm == "off":
            print(f"ARM off -- one obj_id per person ({n_seed}), no motion fix")
            eng.reset()
            A = eng.matte(frames, seed_mask=None, seed_masks=seeds)
            base_multi = A
        else:
            print(f"ARM on -- one obj_id per person ({n_seed}) + motion fix")
            if base_multi is None:
                eng.reset()
                base_multi = eng.matte(frames, seed_mask=None, seed_masks=seeds)
            A = apply_motion_fix(base_multi, frames, eng)
        assert len(A) == len(frames), f"{arm}: frame count mismatch"
        np.save(f"{WORK}/alpha_{arm}.npy", A.astype(np.float16))
        runs[arm] = score(A, n_seed)
        contact_sheet(frames, A, f"{WORK}/sheet_{arm}.jpg")
        lost, k = subject_loss(A, n_seed)
        print(f"subjects per frame: {k}")
        print(f"seeded {n_seed}; count first falls short for good at frame "
              f"{'never' if lost is None else lost}\n")

    # ---- table ----------------------------------------------------------- #
    print("=" * 74)
    print("3.2  ipman fast-motion window, native 1080p, 72 frames")
    print("=" * 74)
    cols = ["v1"] + arms
    print(f"{'metric':14}" + "".join(f"{c:>14}" for c in cols))
    for k in ("area_cv", "edge_soft", "dropouts", "mean_area", "subj_lost_at"):
        row = f"{k:14}{V1_NATIVE.get(k, '-'):>14}"
        for c in arms:
            row += f"{runs[c][k]:>14}"
        print(row)

    print("\nHow to read this")
    print("  subj_lost_at is the frame after which the matte never again holds")
    print(f"  all {n_seed} seeded subjects. 'never' is the pass condition, and it")
    print("  is the ONLY column that can tell a cleaner matte from a lost")
    print("  person -- mean_area falling looks identical either way, and")
    print("  dropouts scores 0 for a permanent loss by construction.")
    print("  edge_soft is not comparable to the v1 column: v1's alpha is")
    print("  recovered by inverting a green composite, which over-softens.")
    if "on" in runs and "off" in runs:
        print("\non - off:")
        for k in ("area_cv", "edge_soft", "dropouts", "mean_area"):
            print(f"  {k:12}{runs['on'][k] - runs['off'][k]:+.4f}")

    with open(f"{WORK}/results.json", "w") as fh:
        json.dump({"window": "ipman source 318-389", "n_seed": n_seed,
                   "v1_native": V1_NATIVE, "runs": runs}, fh, indent=2)
    print(f"\njson -> {WORK}/results.json")


if __name__ == "__main__":
    main()
