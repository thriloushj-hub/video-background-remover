"""Full benchmark: v2 against v1, on every clip's single-shot window.

    !cd /content && python vbgr_bench.py            # all clips, no motion fix
    !cd /content && python vbgr_bench.py --fix      # ... with the motion fix too
    !cd /content && python vbgr_bench.py --product  # ... and through pipeline.py
    !cd /content && python vbgr_bench.py --product --outputs   # ... writing watchable video
    !cd /content && python vbgr_bench.py 1917 dance # just these

``--product`` is the one that matters.  Every other arm calls the engine
directly, so shots, entrant recovery, track health, decontamination and output
writing are all absent from the measurement.  On butter that is `mean_area`
0.0901 measured against 0.1974 shipped, with v1 at 0.2011 -- the harness was
scoring 45% of v1's coverage where the product scores 98%.  Arms are additive
and independent: ``--product`` writes ``rows[clip]["product"]`` beside the
existing ``off``, and a failure in it can never take ``off`` down with it.

Needs the v4 notebook cells 1-3 (repos + checkpoints) and the window clips in
/content/win/.

What this is
------------
Until now v2 had numbers on three windows.  v1 has them on fifteen
(`bench/results/v1_window_baselines.json`).  This closes that gap so the
"measurably better than v1" claim can be made on the whole frozen set instead
of on the clips that happened to get attention.

Every window is a verified single shot -- the runner re-checks and refuses
otherwise, because two of the three original windows turned out to straddle
cuts.

What is comparable and what is not
----------------------------------
``area_cv``, ``dropouts`` and ``mean_area`` are ratios of thresholded area and
survive the green-composite recovery v1's alpha comes from, so they are
comparable to the v1 column.

``edge_soft`` and ``sharpness`` are NOT.  v1's alpha is reconstructed by
pushing chroma distance through ``clip((d - 12) / 28)``, and that ramp stamps
its own boundary profile on every clip alike -- measured sharpness across all
fifteen v1 windows spans only 0.654 to 0.673.  Those two columns are printed
for v2-to-v2 comparison and must never appear in a v2-beats-v1 claim.
"""
import json
import os
import subprocess
import sys
import traceback

import cv2
import numpy as np

sys.path.insert(0, "/content")

WIN = "/content/win"
WORK = "/content/bench_out"
REPO = "/content/SAM2Matting"
CKPT = "/content/checkpoints/SAM2Matting-SAM3.pt"
def _load_v1(path=None):
    """v1 per-window baselines.

    Tolerant of a missing file so this module can be imported (and its metrics
    unit-tested) off Colab.  main() still refuses to run without it -- a
    benchmark with no baseline column is not a benchmark.
    """
    for p in filter(None, [path,
                           os.environ.get("VBGR_V1_BASELINES"),
                           "/content/v1_window_baselines.json",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "bench", "results",
                                        "v1_window_baselines.json")]):
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
    return {}


V1 = _load_v1()


# ---- scoring ------------------------------------------------------------- #

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


def _sharp(a):
    b = _band(a)
    if not b.any():
        return 0.0
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, 3)
    return float(np.hypot(gx, gy)[b].mean())


def n_subjects(a, min_frac=0.002):
    """Connected components above ``min_frac`` of the frame.

    Kept for the contact-sheet labels only.  It is NOT a subject count -- see
    subject_loss.
    """
    n, _l, s, _c = cv2.connectedComponentsWithStats((a > 0.5).astype(np.uint8), 8)
    return int(sum(1 for i in range(1, n) if s[i, 4] >= min_frac * a.size))


# Distinct from None ("held all 72 frames") because the two used to be the
# same value and printed as the same word. See subject_loss.
UNMEASURED = "unmeasured"


def subject_loss(A, n_seed, object_areas=None, alive_frac=0.25, min_frac=1e-4):
    """Frame after which the matte stops holding all ``n_seed`` subjects.

    Returns ``(verdict, per_frame_component_count)`` where verdict is an int
    frame index, ``None`` for "held all the way", or the string ``UNMEASURED``
    for "there was nothing to measure with".

    **Those last two are not the same thing and used to be.**  Both came back
    as ``None`` and ``score`` printed both as ``never``, so an arm with no
    per-object areas -- the product path, which fuses its seeds -- would have
    reported a clean pass on all fifteen clips without measuring anything.
    That is the same shape as ``dropouts`` scoring 0 for a subject lost
    permanently: a metric that cannot fire reading as a metric that did not
    need to.

    **This used to count connected components and that was wrong.**  Two people
    standing shoulder to shoulder are one component, so the metric fired on
    every clip where anybody touched: on the 2026-08-23 benchmark it reported a
    loss on codylexi, shakira, dance and tryguys, and the contact sheets show
    every subject present in every frame of all four.  codylexi is the clearest
    case -- two subjects, one component for the entire window, nobody lost.

    ``object_areas`` is an (T, K) array of per-subject coverage straight from
    the tracker (``SAM2MattingEngine.object_areas``).  That is the honest
    signal: subject k is still held if subject k still has area, whether or not
    it is fused to a neighbour in the union.

    Presence is measured on each subject's **share of the total matted area**,
    not on its raw coverage.  The first version of this compared raw area to a
    fraction of the subject's own seed-frame area, and that was wrong: on
    butter the camera dollies back and every dancer shrinks about 9x, so all
    four fell under the floor together and the metric reported a loss at frame
    17 while the contact sheet plainly shows four dancers at frame 71.  A
    share is invariant to the whole scene changing scale, which is exactly the
    case that has to not fire.  A subject that is genuinely keyed out loses its
    share; a subject that merely gets smaller with everyone else does not.

    Without ``object_areas`` there is no way to tell a merge from a loss, so
    the fallback refuses to guess and returns UNMEASURED rather than the old
    false positive.  A wrong 'never' is a missing alarm; a wrong frame number
    is an alarm that sent three people to look at healthy footage.
    """
    k = [n_subjects(a) for a in A]
    if object_areas is None:
        return UNMEASURED, k

    oa = np.asarray(object_areas, np.float32)
    if oa.ndim != 2 or oa.shape[0] != len(A) or oa.shape[1] != n_seed:
        return UNMEASURED, k

    # carry the last measured row across frames the predictor never returned
    for t in range(len(oa)):
        if np.isnan(oa[t]).any() and t:
            oa[t] = np.where(np.isnan(oa[t]), oa[t - 1], oa[t])
    if np.isnan(oa).any():
        return UNMEASURED, k

    total = oa.sum(axis=1, keepdims=True)
    alive_anywhere = (total[:, 0] > min_frac)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, oa / total, 0.0)

    if not alive_anywhere[0]:
        return UNMEASURED, k                  # nothing seeded to lose
    floor = share[0] * alive_frac
    held = (share >= floor).all(axis=1) & alive_anywhere
    last = max((i for i, v in enumerate(held) if v), default=-1)
    return (None if last == len(A) - 1 else last + 1), k


def score(A, n_seed, cuts=(), object_areas=None):
    from bench.metrics import dropout_events        # the rewritten one
    frac = np.array([float((x > 0.5).mean()) for x in A])
    lost, _k = subject_loss(A, n_seed, object_areas=object_areas)
    return dict(area_cv=round(float(frac.std() / (frac.mean() + 1e-9)), 4),
                edge_soft=round(float(np.mean([_edge_soft(x) for x in A[::2]])), 4),
                sharpness=round(float(np.mean([_sharp(x) for x in A[::2]])), 4),
                dropouts=dropout_events(list(A), cuts=list(cuts))[0],
                mean_area=round(float(frac.mean()), 4),
                subj_lost_at=("never" if lost is None else lost))



# ---- per clip ------------------------------------------------------------ #

def load_frames(path, work):
    fd = f"{work}/frames"
    os.makedirs(fd, exist_ok=True)
    if not os.listdir(fd):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path,
                        "-q:v", "2", f"{fd}/%06d.jpg"], check=True)
    return [cv2.imread(os.path.join(fd, n)) for n in sorted(os.listdir(fd))]


def cut_guard(frames, name):
    from vbgr.shots import compute_scene_cuts
    starts = compute_scene_cuts(frames, warn_suppressed=True)
    if len(starts) > 1:
        raise RuntimeError(
            f"{name}: window contains a cut at {starts[1:]}. Refusing -- this "
            f"is the defect that voided the original ipman and butter rows.")


def build_seeds(frames, work, det):
    import torch
    H, W = frames[0].shape[:2]
    cached = sorted(__import__("glob").glob(f"{work}/seed_[0-9]*.png"))
    if cached:
        return [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in cached]
    rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        o = det([torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).cuda()])[0]
    sel = (o["labels"] == 1) & (o["scores"] > 0.80)
    boxes = o["boxes"][sel].detach().cpu().numpy()
    masks = o["masks"][sel, 0].detach().cpu().numpy() > 0.5
    confs = o["scores"][sel].detach().cpu().numpy()
    from vbgr.detect import Detection, drop_nested_duplicates, select_person_boxes
    from vbgr.config import DetectConfig
    cfg = DetectConfig()
    dets = [Detection(box=tuple(map(float, boxes[i])), conf=float(confs[i]),
                      cls=0, label="person") for i in range(len(boxes))]
    pos = {id(d): i for i, d in enumerate(dets)}
    kept, dropped = select_person_boxes(dets, W, H, cfg.person_rel_size_min,
                                        cfg.box_score_ratio, cfg.person_rel_area_min)
    # One obj_id is seeded per kept person below, so a person detected twice
    # -- once whole, once as a nested part -- becomes two objects tracked
    # against each other and SAM2 tears a seam down her middle.  interview.
    kept, dup = drop_nested_duplicates(
        kept, masks=[masks[pos[id(d)]] for d in kept],
        box_contain_max=cfg.duplicate_box_contain_max,
        mask_contain_max=cfg.duplicate_mask_contain_max)
    dropped = list(dropped) + list(dup)
    seeds = []
    for n, i in enumerate([pos[id(d)] for d in kept]):
        m = masks[i].astype(np.uint8) * 255
        cv2.imwrite(f"{work}/seed_{n}.png", m)
        seeds.append(m)
    print(f"    seed: {len(boxes)} detected -> {len(seeds)} kept, "
          f"{len(dropped)} dropped ({len(dup)} nested duplicate)")
    return seeds


def reseed_scan(frames, alphas, det, cfg=None, every=None, overlap_max=None):
    """Find people who walk in after frame 0, as (frame_idx, mask) pairs.

    The benchmark seeds once at frame 0, so an entrant is invisible to it
    forever. Measured on the 27 Aug run, butter loses three dancers to the
    camera pull-back and dance loses one, and all four scored as v1 halo --
    v1 holds them and we do not. See Halo_Was_Missing_Subjects.

    A person is only prompted **once**: after a scan accepts somebody, the
    running "held" mask is widened to include them, so the next scan does not
    hand the tracker a second copy of the same subject.
    """
    import torch
    from vbgr.detect import (Detection, drop_nested_duplicates, new_subjects,
                             select_person_boxes)
    from vbgr.config import DetectConfig
    cfg = cfg or DetectConfig()
    every = every or cfg.reseed_every
    overlap_max = cfg.reseed_overlap_max if overlap_max is None else overlap_max
    H, W = frames[0].shape[:2]
    out, claimed = [], None
    for i in range(every, len(frames), every):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            o = det([torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).cuda()])[0]
        sel = (o["labels"] == 1) & (o["scores"] > 0.80)
        boxes = o["boxes"][sel].detach().cpu().numpy()
        masks = o["masks"][sel, 0].detach().cpu().numpy() > 0.5
        confs = o["scores"][sel].detach().cpu().numpy()
        if not len(boxes):
            continue
        dets = [Detection(box=tuple(map(float, boxes[k])), conf=float(confs[k]),
                          cls=0, label="person") for k in range(len(boxes))]
        pos = {id(d): k for k, d in enumerate(dets)}
        kept, _ = select_person_boxes(dets, W, H, cfg.person_rel_size_min,
                                      cfg.box_score_ratio, cfg.person_rel_area_min)
        kept, _ = drop_nested_duplicates(
            kept, masks=[masks[pos[id(d)]] for d in kept],
            box_contain_max=cfg.duplicate_box_contain_max,
            mask_contain_max=cfg.duplicate_mask_contain_max)
        km = [masks[pos[id(d)]] for d in kept]
        held = alphas[i] if claimed is None else np.maximum(alphas[i], claimed)
        for j in new_subjects(km, held, overlap_max=overlap_max):
            out.append((i, (km[j].astype(np.uint8) * 255)))
            claimed = km[j].astype(np.float32) if claimed is None else \
                np.maximum(claimed, km[j].astype(np.float32))
    print(f"    reseed scan: every {every} frames -> {len(out)} new subject(s) "
          f"at {sorted({f for f, _ in out})}")
    return out


def motion_fix(alphas, frames, eng):
    from vbgr import motion, refine
    from vbgr.config import MotionConfig, RefineConfig
    m, r = MotionConfig(), RefineConfig()
    flow = motion.FlowEstimator(m.flow_method, m.flow_scale_short_side)
    head = eng.matte_frame if (r.band_refine and eng.has("per_frame_head")) else None
    out = [alphas[0].copy()]
    for i in range(1, len(frames)):
        prev, f = frames[i - 1], frames[i]
        a = alphas[i].copy()
        fl = flow.flow(prev, f)
        mag = flow.magnitude(fl, m.flow_smooth_sigma)
        if float(np.percentile(mag, 95)) > m.high_motion_px:
            warped, trust = motion.flow_alpha_prior(out[i - 1], prev, f, fl,
                                                    m.warp_consistency_thresh)
            a = motion.blend_with_prior(a, warped, trust, mag,
                                        m.warp_prior_weight, m.high_motion_px)
            if r.band_refine:
                tri = motion.flow_adaptive_trimap(a, mag, m.trimap_base,
                                                  m.trimap_flow_gain, m.trimap_max)
                a = refine.refine_band(a, f, tri, head)
        out.append(a)
    return np.stack(out)


ALPHA_MAX_W = 960


def save_alphas(alphas, out_dir, max_w=ALPHA_MAX_W):
    """Persist the run's alpha as 8-bit PNGs.

    The 23 Aug run kept only the contact sheets, so every later question about
    the mattes had to be answered off a 300px JPEG tile -- which is how
    butter's frame-17 "loss" went unexamined for a day, and why ``hole_frac``
    could not be validated on real data: the sheet proxy has a noise floor
    around 2% and reads a clean control (microsoft) at the same magnitude as
    interview's real torn face.

    A GPU run costs ~15 minutes and a fresh VM; a metric change costs seconds.
    Persisting the alpha means the second never needs the first again.

    Downscaled to ``max_w`` because the questions these answer are structural
    (is there a tear, is the baseline's extra area a halo) and those survive
    the resize -- interview's band is ~6% of frame width.  Boundary-detail
    metrics must still be read off the full-resolution run, not these.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, a in enumerate(alphas):
        if a.shape[1] > max_w:
            h = int(round(a.shape[0] * max_w / a.shape[1]))
            a = cv2.resize(a, (max_w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out_dir}/{i:04d}.png",
                    np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))


def sheet(frames, alphas, path, cols=6, rows=4):
    n = cols * rows
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    tw = 300
    th = int(tw * frames[0].shape[0] / frames[0].shape[1])
    out = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, i in enumerate(idx):
        a = alphas[i][:, :, None]
        comp = (frames[i].astype(np.float32) * a
                + np.array([0, 255, 0], np.float32) * (1 - a)).astype(np.uint8)
        t = cv2.resize(comp, (tw, th))
        cv2.putText(t, f"{i} n={n_subjects(alphas[i])}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        r, c = divmod(k, cols)
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---- main ---------------------------------------------------------------- #

def product_arm(clip_path, work, name, output_mode="alpha"):
    """Run the window through `pipeline.py` -- the code that would actually ship.

    Why this exists
    ---------------
    Every v2-vs-v1 number this project has produced came from this file
    calling the engine directly.  `vbgr_bench.py` has never touched
    `pipeline.py`, so shots, re-seeding, track health, entrant recovery,
    decontamination and output writing were all absent from the measurement --
    and on butter that is the difference between `mean_area` 0.0901 and
    0.1974 against v1's 0.2011.  The measured path covered 45% of v1 there;
    the shipped path covers 98%.

    So this arm is not a nicety.  Until it runs, no row in the result table is
    a product number on multi-subject footage, and the honest reading of the
    whole benchmark is "v2 as handicapped by its own harness".

    Scored with the same `score()` as every other arm, off the alphas the
    pipeline actually produced, so the columns are comparable by construction
    rather than by assertion.
    """
    from vbgr.config import Config
    from vbgr.pipeline import Pipeline

    cfg = Config()
    cfg.io.input_dir = os.path.dirname(clip_path)
    cfg.io.output_dir = f"{work}/product_out"
    # "alpha" is the default because the composite and the WebM are already
    # exercised by bench/check_alpha_output.py and check_composite_and_audio.py,
    # and writing all three per clip adds encode time to every run for nothing.
    #
    # "all" is for when the run has to produce something a person can WATCH.
    # Numbers are what the benchmark is for, but a client who named his failure
    # cases by looking at clips is going to judge this by looking at clips.
    cfg.io.output_mode = output_mode
    cfg.matting.engine = "sam2matting"
    cfg.matting.repo_dir = REPO
    cfg.matting.checkpoint = CKPT
    cfg.matting.device = "cuda"
    cfg.verbose = True

    pipe = Pipeline(cfg)
    rep = pipe.run_clip(clip_path, keep_alphas=True)
    A = [np.asarray(a, np.float32) for a in rep.alphas]

    # Per-subject areas come back per shot. Every benchmark window is a single
    # shot by construction (cut_guard refuses otherwise), so there is exactly
    # one to take -- but assert rather than assume, because "every window is
    # one shot" is a property of the window cutting, not of this function.
    oa, n_seed = None, sum(sh.n_kept for sh in rep.shots)
    if len(rep.shots) == 1 and rep.shots[0].object_areas is not None:
        oa = rep.shots[0].object_areas
    sc = score(A, n_seed, object_areas=oa)
    meta = dict(n_seed=n_seed,
                n_shots=len(rep.shots),
                seconds=round(rep.seconds, 1),
                bad_frames=sum(sh.bad_frames for sh in rep.shots),
                reseeds=sum(sh.reseeds for sh in rep.shots),
                reentries=sum(sh.reentries for sh in rep.shots),
                high_motion=sum(sh.high_motion_frames for sh in rep.shots),
                notes=[n for sh in rep.shots for n in sh.notes],
                degraded=dict(rep.degraded))
    return A, sc, meta


def main():
    if not V1:
        raise SystemExit(
            "no v1 window baselines found -- looked at $VBGR_V1_BASELINES, "
            "/content/v1_window_baselines.json and bench/results/. "
            "Refusing to run: a v2 column with no v1 column proves nothing.")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_fix = "--fix" in sys.argv
    do_reseed = "--reseed" in sys.argv
    do_product = "--product" in sys.argv
    # Write watchable output (green / alpha / transparent webm) from the
    # product arm, not just the alpha stack the metrics need.
    do_outputs = "--outputs" in sys.argv
    clips = args or [c for c in V1
                     if V1[c] and V1[c].get("trusted", True)
                     and V1[c].get("window") is not None]
    os.makedirs(WORK, exist_ok=True)

    import torchvision
    det = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT").eval().cuda()
    from vbgr.engines import build_engine
    eng = build_engine("sam2matting", repo_dir=REPO, checkpoint=CKPT, device="cuda")
    print(f"engine: {eng.info.name} | {eng.info.license} | "
          f"verified caps {eng.verified_capabilities}\n")

    rows, failed = {}, {}
    for nm in clips:
        v1 = V1.get(nm)
        if not v1:
            print(f"{nm}: no v1 window baseline, skipping"); continue
        if not v1.get("trusted", True) or v1.get("window") is None:
            # Voided, not missing.  Running it would produce a v2 column with
            # nothing valid beside it, which is how ipman's rows went wrong.
            print(f"{nm}: v1 baseline VOIDED -- see the note in "
                  f"v1_window_baselines.json. Skipping."); continue
        cand = [f for f in os.listdir(WIN) if f.startswith(nm + "_w")]
        if not cand:
            print(f"{nm}: no window clip in {WIN}, skipping"); continue
        print(f"--- {nm}  window {v1['window']}")
        work = f"{WORK}/{nm}"; os.makedirs(work, exist_ok=True)
        try:
            frames = load_frames(os.path.join(WIN, cand[0]), work)
            cut_guard(frames, nm)
            seeds = build_seeds(frames, work, det)
            if not seeds:
                raise RuntimeError("seed gate kept nobody")
            eng.reset()
            A = eng.matte(frames, seed_mask=None, seed_masks=seeds)
            assert len(A) == len(frames)
            oa = getattr(eng, "object_areas", None)
            if oa is None:
                print("    NOTE: engine reported no per-object areas, so "
                      "subj_lost_at cannot be measured on this clip")
            r = {"off": score(A, len(seeds), object_areas=oa)}
            sheet(frames, A, f"{work}/sheet_off.jpg")
            if do_reseed:
                # A SECOND ARM, not a replacement, and it must never be able to
                # take the off arm down with it.  On the 30 Aug run it did
                # exactly that: the engine hook raised, the exception reached
                # the outer handler, and butter and dance lost their *already
                # computed* off rows to a failure in an arm flagged unverified.
                # An experimental arm gets its own handler.
                try:
                    extra = reseed_scan(frames, A, det)
                    r["n_reseed"] = len(extra)
                    if extra:
                        eng.reset()
                        C = eng.matte(frames, seed_mask=None, seed_masks=seeds,
                                      extra_seeds=extra)
                        oc = getattr(eng, "object_areas", None)
                        r["reseed"] = score(C, len(seeds) + len(extra),
                                            object_areas=oc)
                        sheet(frames, C, f"{work}/sheet_reseed.jpg")
                        save_alphas(C, f"{work}/alpha_reseed")
                except Exception as e:                       # noqa: BLE001
                    r["reseed_error"] = f"{type(e).__name__}: {e}"
                    print(f"    reseed arm FAILED (off arm kept) -- "
                          f"{r['reseed_error']}")
                    eng.reset()
            if do_product:
                # Its own handler, always. On 30 Aug the reseed arm raised
                # inside the clip's main try and took butter's and dance's
                # already-computed off rows down with it. Nothing unverified
                # may be able to void the verified arm.
                try:
                    P, psc, pmeta = product_arm(
                        os.path.join(WIN, cand[0]), work, nm,
                        output_mode=("all" if do_outputs else "alpha"))
                    r["product"] = psc
                    r["product_meta"] = pmeta
                    save_alphas(P, f"{work}/alpha_product")
                    sheet(frames, P, f"{work}/sheet_product.jpg")
                    print(f"    product: {psc}")
                    print(f"    product meta: {pmeta}")
                except Exception as e:                       # noqa: BLE001
                    r["product_error"] = f"{type(e).__name__}: {e}"
                    print(f"    product arm FAILED (other arms kept) -- "
                          f"{r['product_error']}")
                    traceback.print_exc()
                eng.reset()
            if do_fix:
                B = motion_fix(A, frames, eng)
                # the motion fix reshapes the union, not the tracker's own
                # per-object alphas, so the off-arm areas still describe who
                # the tracker is holding
                r["on"] = score(B, len(seeds), object_areas=oa)
                sheet(frames, B, f"{work}/sheet_on.jpg")
                # The ON arm's alphas have never been written to disk, and that
                # is exactly what blocks the edge comparison against RVM: every
                # edge number we have is from the fix-OFF arm, while the fix is
                # measured at edge_soft +0.1098 (3.6). Without these there is no
                # way to ask whether v2's boundary is softer than RVM's with the
                # fix on, which is the open question this run exists to close.
                save_alphas(B, f"{work}/alpha_on")
            save_alphas(A, f"{work}/alpha")
            r["n_seed"] = len(seeds)
            r["alpha_dir"] = f"{nm}/alpha"
            # Persist the per-subject areas the subj_lost_at verdict was made
            # from. Without these the verdict cannot be checked without a full
            # GPU re-run, which is how butter's frame-17 "loss" went unexamined
            # in the first place. Rounded to keep the file readable.
            if oa is not None:
                r["object_areas"] = [[round(float(x), 5) for x in row]
                                     for row in oa]
            rows[nm] = r
            print(f"    v2 off: {r['off']}")
        except Exception as e:
            failed[nm] = f"{type(e).__name__}: {e}"
            print(f"    FAILED -- {failed[nm]}")
            traceback.print_exc()

    # ---- table ----------------------------------------------------------- #
    print("\n" + "=" * 104)
    print("v2 vs v1, every clip on its single-shot window   (comparable: area_cv, dropouts, mean_area)")
    print("=" * 104)
    hdr = f"{'clip':12}{'subj':>5}{'area_cv v1':>12}{'v2':>9}{'mean_area v1':>14}{'v2':>9}{'drop v1':>9}{'v2':>5}{'lost':>7}"
    print(hdr); print("-" * 104)
    for nm, r in rows.items():
        v1 = V1[nm]; o = r["off"]
        print(f"{nm:12}{r['n_seed']:>5}{v1['area_cv']:>12.4f}{o['area_cv']:>9.4f}"
              f"{v1['mean_area']:>14.4f}{o['mean_area']:>9.4f}"
              f"{v1['dropouts']:>9}{o['dropouts']:>5}{str(o['subj_lost_at']):>7}")
    if any("product" in r or "product_error" in r for r in rows.values()):
        print("\n" + "=" * 104)
        print("PRODUCT PATH (pipeline.py) vs v1 -- the numbers that describe "
              "what would actually ship")
        print("=" * 104)
        ph = (f"{'clip':12}{'subj':>5}{'mean_area v1':>14}{'bench':>9}"
              f"{'product':>9}{'v1 cov%':>9}{'lost':>11}{'shots':>7}{'s':>7}")
        print(ph); print("-" * 104)
        for nm, r in rows.items():
            if "product" not in r:
                if "product_error" in r:
                    print(f"{nm:12}  FAILED -- {r['product_error']}")
                continue
            p, m, v1 = r["product"], r["product_meta"], V1[nm]
            cov = 100.0 * p["mean_area"] / max(v1["mean_area"], 1e-9)
            print(f"{nm:12}{m['n_seed']:>5}{v1['mean_area']:>14.4f}"
                  f"{r['off']['mean_area']:>9.4f}{p['mean_area']:>9.4f}"
                  f"{cov:>8.1f}%{str(p['subj_lost_at']):>11}"
                  f"{m['n_shots']:>7}{m['seconds']:>7.0f}")
        print("\nv1 cov% is the product path's coverage as a share of v1's. It "
              "is NOT a quality score -- v1 over-includes, so 100% is not the "
              "target and neither is more. Read it beside the halo/hole split.")
    if failed:
        print("\nFAILED:")
        for k, v in failed.items():
            print(f"  {k}: {v}")
    json.dump({"rows": rows, "failed": failed, "v1": V1},
              open(f"{WORK}/bench_results.json", "w"), indent=2)
    print(f"\njson -> {WORK}/bench_results.json")
    print("subj = subjects the seed gate kept; lost = frame after which the matte "
          "stops holding all of them ('never' passes)")
    print("\nNOTE: area_cv and mean_area compare COVERAGE, and v1's matte is "
          "over-inclusive on at least ipman and butter, so a v2 loss in these "
          "columns is not a quality result. Run bench/score_disagreement.py "
          "against the persisted alphas before making any v2-vs-v1 claim.")


if __name__ == "__main__":
    main()
