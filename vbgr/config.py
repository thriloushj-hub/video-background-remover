"""Configuration for the vbgr2 video background removal pipeline.

Everything is a plain dataclass so it can be constructed in code, loaded from
YAML, or overridden from the CLI.  Defaults are the ones we believe are correct
for 1080p 24fps footage; see configs/ for per-scenario overrides.
"""
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #

@dataclass
class IOConfig:
    input_dir: str = "inputs"
    output_dir: str = "results"
    # "green" | "alpha" | "transparent" (webm/vp9) | "all"
    output_mode: Literal["green", "alpha", "transparent", "all"] = "all"
    green: tuple = (0, 177, 64)
    # Downscale so that min(w, h) <= max_size before matting. None = native.
    # Matting quality degrades badly below ~720 on the short side.
    max_size: Optional[int] = None
    # Encode CRF for the RGB outputs. Alpha is always written losslessly-ish.
    crf: int = 16
    # Write per-frame PNGs alongside the videos.
    save_frames: bool = False
    # Fail loudly rather than silently dropping frames (see docs/BUGS.md).
    strict_frame_count: bool = True
    # 2 Oct 2026 -- output edge finish, applied to the alpha just before the
    # composites are written (the alpha video gets it too, so every output
    # agrees).  Thriloush's standing target (28 Sep): "dont do soft alpha,
    # crisper edges".  v1's edge is MatAnyone 2's hard mask through an 8 px
    # erode+dilate, which is what reads as "crisp" to him.
    #   "soft"       : the engine's alpha as-is (every render before 2 Oct)
    #   "hard"       : cut at edge_threshold, blur edge_aa_sigma -- the 29 Sep
    #                  review finish; reads pixelated to him at 1:1
    #   "hard_round" : blur edge_round_sigma, re-cut, blur edge_aa_sigma --
    #                  rounds 1-2 px notches on the contour, then anti-aliases
    #   "hard_v1"    : cut, erode+dilate edge_morph_px, blur edge_aa_sigma
    # Judged on dance f24/f40 at 5x: hard_round is closest to v1; hard_v1
    # eats thin parts (as v1 does).  Default stays "soft" until his call.
    edge_finish: Literal["soft", "hard", "hard_round", "hard_v1"] = "soft"
    edge_threshold: float = 0.5
    edge_round_sigma: float = 2.0
    edge_aa_sigma: float = 1.5
    edge_morph_px: int = 8


@dataclass
class ShotConfig:
    """Shot splitting.  MatAnyone-style memory must not cross a cut."""
    enabled: bool = True
    # Bhattacharyya distance between consecutive HSV histograms.
    cut_threshold: float = 0.35
    hist_bins: int = 64
    # A "cut" that is only 1-2 frames long is a flash/strobe, not a cut.
    min_shot_len: int = 12
    # Guard against a hard cut being missed: also cut when mean abs frame
    # difference spikes far above the running median.
    use_absdiff_guard: bool = True
    absdiff_sigma: float = 6.0


@dataclass
class DetectConfig:
    """YOLO person detection used to seed and to scan for entrants."""
    model: str = "yolov8x.pt"
    conf: float = 0.25
    iou: float = 0.5
    # A person box shorter than this fraction of the tallest kept subject is
    # treated as a background bystander.  Height, not area: height survives
    # horizontal cropping when someone enters from the frame edge.
    person_rel_size_min: float = 0.60
    # Second gate, applied ONLY to boxes that do not touch a frame edge: a
    # person whose box area is below this fraction of the largest person's is a
    # distant bystander.  Height cannot separate near from far (1917's
    # background soldier is 0.658 of the foreground soldier's height but 0.149
    # of his area); area can, because area goes as roughly height squared.  The
    # edge exemption preserves the full-height-but-cropped entrant case above.
    # Set to 0.0 to disable and fall back to height-only behaviour.
    person_rel_area_min: float = 0.20
    # Prominence = conf * centrality * area, used only for ranking + a floor.
    box_score_ratio: float = 0.15
    # Absolute confidence floor, beside the relative one above.  The relative
    # floor is `box_score_ratio * top_conf`, which on a clip with one very
    # confident subject is permissive to the point of uselessness: on eddie it
    # is 0.15 * 0.94 = 0.141, and a 0.277 detection of a *blurred portrait
    # hanging on the wall* cleared it, producing two phantom re-entry events
    # (3.3a).  Measured over 284 kept detections across all 15 clips, that
    # phantom is the ONLY one below 0.755 -- so any floor in 0.28..0.75 drops
    # exactly it and nothing else.  0.50 sits 1.81x above the phantom and
    # 1.51x below the lowest genuine detection in the set.
    conf_abs_min: float = 0.50
    # Duplicate suppression: a detection this far inside another kept
    # detection is the same person found twice, not a second person.  One
    # obj_id is seeded per kept person, so a nested duplicate makes SAM2
    # track a person against herself and tears a seam down her middle --
    # this is the `interview` hole_big 0.055 defect.  Measured on the 15-clip
    # set the only nested pair is interview's (box 0.968 / mask 0.978); the
    # next highest is 0.283 / 0.001, so both thresholds sit in empty space.
    duplicate_box_contain_max: float = 0.80
    duplicate_mask_contain_max: float = 0.70
    # Re-seeding: scan for people the matte is not holding every N frames, and
    # treat a detection as new when at most this much of it overlaps what we
    # already have.  Coverage of the detection, not IoU -- a person standing
    # partly behind someone we hold still has most of their own pixels
    # outside the matte.  Off by default; the benchmark's frozen arm seeds
    # once at frame 0 and every run before 29 Aug did the same.
    reseed_every: int = 12
    reseed_overlap_max: float = 0.30
    # CLAHE is deliberately NOT used: it pushes frames out of YOLO's training
    # distribution and measurably hurt recall on bright/saturated footage.
    device: str = "auto"


@dataclass
class SeedConfig:
    """Seeding of the first frame of each shot.

    ``backend`` is **maskrcnn** by default (decided 30 Aug, 5.1k). Two
    reasons: neither SAM 3 backend is loadable -- ``sam3.pt`` is not in
    ultralytics' auto-download set and ``sam3.build_sam`` does not exist in
    the fork we install -- and, more importantly, Mask R-CNN is where **every
    validated seed in this project came from**. ``vbgr_bench.py`` has always
    seeded from it and never touched ``pipeline.py``, so the area gate (3.0),
    one-object-id-per-person (3.2a) and nested-duplicate suppression (3.2u)
    were all measured on these masks. See Pipeline_Never_Ran.
    """
    backend: Literal["maskrcnn", "sam3"] = "maskrcnn"
    # 5.7.  Close a seed mask's gap to the bottom of its OWN detection box,
    # when that gap is at most this fraction of the box height.  0.0 is off and
    # is the default, so the shipped path is unchanged until a measured run
    # says otherwise.  See seed.extend_short_tails for why bottom-only and why
    # capped: "extend the mask to its box" without those limits is the shape of
    # findings this project has already retracted.
    extend_tail_max: float = 0.0
    # 5.7, the look-ahead seed.  A shot is seeded from ONE frame and every later
    # frame inherits it until the next cut, so a frame the mask head happens to
    # fail on costs the whole shot.  On bilibili f1159 the mask stops 298 px
    # short of its box; one frame later it stops 74 px short, and the boots are
    # missing for exactly as long as the shot lasts.
    #
    # So: look at the first `lookahead_frames` frames and seed from the best of
    # them.  The guard is RELATIVE, not a global threshold, because a threshold
    # on the bad frame is what 3.3a showed cannot separate pose from defect:
    #   - only scan at all when frame 0's tail exceeds `lookahead_min_tail`;
    #   - only move when the best frame beats frame 0 by `lookahead_min_gain`.
    # A genuinely half-occluded subject (behind a desk) is equally occluded in
    # all of them, so there is no gain and nothing moves.  0 disables it.
    # Default 4, turned on 8 Sep after the sweep below.  Across the whole
    # delivery -- 19 clips, 79 shots -- it fires on three, and with the cast
    # guard on two: bilibili shot 11, where it is a large repair (alpha bottom
    # 587 -> 866 on the shot's first frame and every one of its sixteen frames
    # gaining 273-353 rows), and butter shot 8, which moves the matte less than
    # re-running the same code does.  Measured on full-length butter, both arms:
    # clip alpha mean 0.27846 -> 0.27847, against a run-to-run noise floor of
    # 0.00089 per frame taken from the seven shots that do not move at all.
    # Set to 0 to restore the pre-8-Sep behaviour exactly; the renders in the
    # delivery package predate this and were made with it off.
    lookahead_frames: int = 4
    lookahead_min_tail: float = 0.15
    lookahead_min_gain: float = 0.10
    # Never move the seed to a frame that holds FEWER people than frame 0.
    # The 8 Sep sweep over all 79 shots in the delivery found exactly one shot
    # where the best frame drops subjects -- butter shot 5, six kept at frame 0
    # and four at frame 3 -- so repairing the mask and losing two dancers are
    # the same move there.  Measured, that shot came back inside the pipeline's
    # own run-to-run noise because the entrant passes re-seed them; this guard
    # is for the footage where nothing catches them.  Cost is one comparison.
    lookahead_require_same_cast: bool = True
    # Mask R-CNN seeding. Boxes still come from the pipeline's own detector,
    # so the gates behave exactly as tested; the seeder only answers "what is
    # the mask inside this box". A box no instance matches is skipped with a
    # note rather than filled with its rectangle.
    maskrcnn_score_min: float = 0.80
    maskrcnn_match_iou: float = 0.30
    # Recovery-only relaxation (5.5).
    #
    # A heavily motion-blurred runner clears the pipeline's own person detector
    # -- the re-entry scan finds his box and reports cov=0.000 -- but does not
    # clear Mask R-CNN's mask head at 0.80. `masks_from_boxes` then returns
    # nothing for that box, `build_seed` drops the subject, and the local
    # re-entry pass re-mattes the cast it already had. On the full-length 1917
    # that is 97 of 724 frames where v1 holds a region v2 does not, peaking at
    # 12.3% of frame, with ten local passes firing and changing nothing.
    #
    # So: for a box the two-stage scan has ALREADY confirmed as an uncovered
    # person, and only after the strict pass has failed on that one box, retry
    # the match at this threshold. Everything else is unchanged -- the frame-0
    # cast (3.3g) still decides who is a subject, `match_iou` still decides
    # whether an instance belongs to the box, and a box that matches nothing is
    # still skipped rather than filled with its rectangle.
    maskrcnn_recover: bool = True
    maskrcnn_recover_score_min: float = 0.35
    # 5.9, the mask cutoff.  Mask R-CNN returns a soft probability map per
    # instance; `maskrcnn_mask_binarise` is what turns it into a mask, and 0.5
    # is torchvision's default.  On bilibili the DETECTION is right -- a box on
    # the whole figure at 0.93 -- and the mask inside it stops at the ankle,
    # because a dark boot on a dark floor scores under 0.5.  The shot then
    # inherits a subject with one foot for all sixteen of its frames.
    #
    # `maskrcnn_mask_recover` re-binarises only the boxes whose mask stops
    # short, once, at the lower cutoff, and keeps the result only if it reaches
    # further AND does not spill outside the box.  Default OFF until a GPU pass
    # says otherwise -- 5.7b was turned on from a plausible argument and had to
    # be reverted, so this one waits for numbers.
    maskrcnn_mask_binarise: float = 0.5
    maskrcnn_mask_recover: bool = False
    maskrcnn_mask_recover_binarise: float = 0.25
    maskrcnn_mask_recover_min_tail: float = 0.05
    maskrcnn_mask_recover_max_spill: float = 1.5
    # 5.10, the frame edge.  THE DEFECT: where a subject is cut by the edge of
    # frame the matte stops short of it -- and it is truncation, not softness.
    # On microsoft's bottom edge, of the 475 columns solidly foreground 80-100
    # px up, 81.3% are BELOW alpha 0.05 at the bottom row and 16.4% are above
    # 0.5.  "Mean alpha 0.52 in the bottom 8 rows" is a mean over a bimodal
    # distribution and it made the defect read as a soft edge for a week.
    #
    # THE CAUSE, and it is not the matting model: torchvision's
    # `expand_masks(mask, padding=1)` zero-pads Mask R-CNN's 28x28 ROI mask and
    # `paste_mask_in_image` bilinearly resizes it to the box, so every box edge
    # carries a ramp ~1 mask cell = box_dim/28 deep (measured on microsoft f0
    # at all four edges of both instances: d=0 values 0.378 / 0.329 / 0.175 /
    # 0.117, crossing 0.5 at 6-28 px, one cell 29-34 px).  Harmless everywhere
    # except the frame edge, where the person fills the box to the border.  The
    # boxes are additionally 2-11 px short of the edge.  Set-wide: the seed is
    # absent at the bottom row on all fifteen clips.
    #
    # NOTE this retracts "the seed is ruled out, Mask R-CNN reaches the border,
    # median 3 px short" -- that was measuring the BOX, not the MASK, the same
    # distinction 5.6 drew on bilibili.  And it is 5.6/5.9's defect set-wide,
    # so nothing in the mask-cutoff family can reach it.
    #
    # THE FIX: replicate-pad the frame before detection, crop the masks back.
    # microsoft, 4 frames, CPU, pad 64: seed reach 17 px short -> 0, absent at
    # the bottom row 1.00 -> 0.00, mean alpha at the edge 0.170 -> 0.957, edge
    # coverage 0 px -> 919 px.  It RECOVERS rather than dilates and that is
    # measured: edge width over width 90 px up saturates at 0.96 and stops
    # (dilation climbs past 1), width 90 px up is constant at ~956 px, total
    # area wobbles sub-1% non-monotonically, and the detection count is
    # unchanged at every pad width so the cast never moves.
    #
    # DEFAULT OFF.  5.7b was turned on from a plausible argument and had to be
    # reverted; this waits for a GPU pass that measures the FINAL ALPHA, since
    # everything above is the seed and the engine still loses another 1.5-2.5x
    # on top.  64 is where both reach and confidence plateau; 128 buys +0.016.
    # FLIPPED ON 18 Sep 2026, Thriloush's call, after the GPU A/B (band arm):
    # every edge equal or better on every frame of every tested clip, zero
    # regions lost, cast unchanged. run_2026-09-17_edgepad_gpu/BAND_RESULT.md.
    # The _to_send/ renders predate this -- set 0/False to reproduce them.
    # (20 Sep: this flip was briefly reverted by an edit made on a stale
    # checkout; the default-pinning tests caught it before anything rendered.)
    maskrcnn_edge_pad: int = 64
    # 24 Sep, the tryguys knee: two neighbours' Mask R-CNN masks leave the
    # seam between them unclaimed.  Close the seed inside overlapping person
    # boxes, but keep only pixels SAM 2.1 (prompted with the same boxes) also
    # calls person -- the ungated close webbed background 50-73% of the time.
    seam_sam: bool = True
    seam_sam_close: int = 121
    seam_sam_model: str = "sam2.1_b.pt"
    # 5.10b (17 Sep): the full pad fixed the bottom edge on the final alpha and
    # broke tryguys' right edge (GPU, run_2026-09-17_edgepad_gpu).  With this on,
    # the padded mask is only used inside an `maskrcnn_edge_pad`-wide band along
    # the frame border, OR-ed with the shipped mask; the interior and the cast
    # are the shipped ones.  Off until its own GPU A/B.
    # 5.10b (17 Sep): the padded mask is only used inside an
    # `maskrcnn_edge_pad`-wide band along the frame border, OR-ed with the
    # shipped mask; the interior and the cast are the shipped ones.
    # GPU A/B passed 17 Sep; on by default since 18 Sep.
    maskrcnn_edge_pad_band: bool = True
    mask_threshold: float = 0.0
    detect_threshold: float = 0.35
    concept_text: str = "person"
    concept_text_detect: bool = True
    concept_match_iou: float = 0.4
    # The interactive box->mask head always fires and recovers missed limbs and
    # held objects; its extra regions are then gated by appearance.
    force_interactive: bool = True
    interactive_mask_threshold: float = 0.0
    box_clip_margin: int = 8
    gate_background_regions: bool = True
    gate_bg_margin: float = 0.10
    # Hole handling
    fill_mask_holes: bool = True
    gap_fill: bool = True
    gap_bg_margin: float = 0.12
    gap_min_area: int = 64
    gap_max_frac: float = 0.05


@dataclass
class MattingConfig:
    """Which matting engine, and how its memory is driven."""
    # "matanyone2" | "sam2matting" | "rvm" | "passthrough"
    engine: str = "sam2matting"
    checkpoint: Optional[str] = None
    # Where the engine's upstream repo is cloned. SAM2Matting imports `sam3`
    # from its own tree, so without this the product path dies on
    # `ModuleNotFoundError: No module named 'sam3'` -- which is exactly what it
    # did the first time pipeline.py was ever run (30 Aug). The benchmark
    # passed repo_dir straight to build_engine and never needed a config field,
    # which is how the gap survived: another artefact of the two paths having
    # diverged. See Pipeline_Never_Ran.
    repo_dir: Optional[str] = None
    device: str = "auto"
    # fp16 halves memory and is visually lossless for alpha.
    half: bool = True

    n_warmup_static: int = 10
    n_warmup_motion: int = 10
    r_erode: int = 0
    r_dilate: int = 6
    # Write the clean motion-warmup frames into PERMANENT memory, not just
    # working memory.  Without this, a low-texture region (light shirt, waist)
    # slowly keys out over a long shot because recent frames outvote the single
    # first-frame anchor.
    drift_anchor_warmup: bool = True
    # Cap on permanent anchors so memory does not grow without bound.
    max_permanent_anchors: int = 24
    # Raise instead of warning when a declared engine capability turns out not
    # to exist on the installed API. Leave False for exploratory runs; set True
    # in CI and before any long batch or ablation, because a silently inactive
    # memory gate still emits counters and still produces numbers, and every
    # one of those numbers is meaningless.
    strict_capabilities: bool = False
    # How the kept people are handed to the engine.  "per_person" (default,
    # 3.2a) gives each person their own obj_id.  "union" seeds everyone as ONE
    # object.  23 Sep, tryguys: per-person objects back off the dark contact
    # zones between seated people (the missing lap), and union cut the missing
    # area 45-84% on three of four shots from identical seeds.  Off by default
    # until it has passed ipman, where one object lost a fighter at f7 (3.2a).
    # "both" runs per-person and union and takes the per-pixel max (2x time).
    person_objects: str = "both"
    # 25 Sep hybrid edge: take the silhouette from the tracker's own binary
    # mask (crisp, like v1) and keep the matting alpha only where it is strong
    # in a thin strip outside that line (hair, blur).  Default off.
    hybrid_edge: bool = False
    hybrid_band_px: int = 6
    hybrid_strong: float = 0.6
    # 25 Sep: fill background pockets on the bottom frame edge between two
    # foreground runs (knees running off-frame), gated per pocket by SAM 2.1-b.
    # Off by default until GPU-checked.  tryguys offline: missing -28%, extra +1%.
    bottom_pocket: bool = False  # 26 Sep: off again, fills the stool gap v1 leaves open
    bottom_pocket_gap: int = 300
    bottom_pocket_hmax: int = 200
    bottom_pocket_gate: float = 0.5
    # 26 Sep: temporal median on the bottom band (rows below
    # bottom_smooth_frac of the height), +-bottom_smooth_w frames, per shot.
    # tryguys: bottom-band flips 63k -> 41k (v1 55k), missing/extra unchanged.
    # 28 Sep: OFF.  On moving subjects the 7-frame median erases feet and
    # shins that travel more than their own width in 3 frames: missing vs v1
    # jumps 3-6x exactly at the band edge on dance (4% -> 11-30%), dance2,
    # dance3, shakira, butter; no effect on static clips.  It only ever passed
    # on tryguys, where nobody moves their feet.
    bottom_smooth: bool = False
    bottom_smooth_frac: float = 0.70
    bottom_smooth_w: int = 3


@dataclass
class TrackHealthConfig:
    """Score each streamed frame and notice when the track is lost.

    **This used to be MemoryGateConfig and it used to do two jobs.**  The first
    was memory gating: re-run a bad frame with ``commit_to_memory=False`` so it
    never poisoned the propagator's memory.  That job is gone (task 3.4,
    dropped 2026-08-23).  No engine we can actually use exposes the hook:
    MatAnyone 2's ``InferenceCore.step()`` takes no ``**kwargs`` at all, and
    SAM2Matting -- the model we chose -- has no ``step()`` whatsoever, so it
    runs batch and this whole path is unreachable on it.  Keeping a switch that
    silently did nothing is how a run ends up reporting "gate_rejects=47" for
    a gate that changed no pixels.

    The second job survives here because it never needed the engine's
    cooperation: score each frame, and after ``lost_after`` consecutive bad
    ones declare the track lost so the pipeline can re-seed.  That is what
    these settings now control, and the name says so.

    Old configs keying this as ``memory_gate`` still load -- see CONFIG_SECTIONS.
    """
    enabled: bool = True
    # Reject if mask area changes by more than this factor vs the running median.
    max_area_ratio: float = 1.6
    min_area_ratio: float = 0.55
    # Reject if IoU against the flow-warped previous alpha is too low: that
    # means the tracker jumped, which during smooth motion is always an error.
    min_flow_iou: float = 0.55
    # Reject frames whose alpha is suspiciously binary AND tiny (typical of a
    # collapsed / lost track).
    min_area_frac: float = 0.0008
    # How many consecutive rejects before we declare the track lost and hand
    # off to the re-prompting layer.
    lost_after: int = 8


@dataclass
class MotionConfig:
    """Fast-motion / motion-blur handling.

    A binary mask cannot represent a blurred limb: those pixels are genuinely
    semi-transparent.  And a memory propagator cannot *discover* a limb that
    was never in the seed.  So we do two things:
      1. widen the trimap unknown band in proportion to local flow magnitude,
         which gives the matting head room to find the blurred edge;
      2. warp the previous alpha forward by optical flow and use it as a prior,
         so a fast limb keeps a plausible alpha instead of vanishing.
    """
    enabled: bool = True
    # "dis" is ~10x faster than farneback at similar quality.
    flow_method: Literal["dis", "farneback"] = "dis"
    # Flow is computed at this short-side resolution and upsampled. 480 is
    # plenty for a band-width signal and keeps this off the critical path.
    flow_scale_short_side: int = 480

    # Trimap band width in px: w = clip(base + k * flow_mag, base, w_max)
    trimap_base: float = 4.0
    trimap_flow_gain: float = 0.9
    trimap_max: float = 40.0
    # Blur the flow magnitude map so the band width varies smoothly.
    flow_smooth_sigma: float = 9.0

    # Frames whose 95th-percentile flow magnitude exceeds this (px/frame) are
    # tagged high-motion and get the extra refinement pass.
    high_motion_px: float = 6.0
    # Weight of the flow-warped alpha prior in high-motion regions.
    warp_prior_weight: float = 0.5
    # Photometric consistency threshold (0-255) for trusting the warp.
    warp_consistency_thresh: float = 18.0


@dataclass
class ReIDConfig:
    """Re-entry after occlusion.

    The v1 pipeline decided "is this person new?" purely by overlap with the
    current matte, i.e. there was no identity at all.  So a subject who is
    occluded and re-enters is re-seeded as a stranger (fine) but can also be
    double-counted, mis-associated, or missed entirely if they re-enter behind
    an existing matte.  We keep a small DINOv3 feature pool per identity and
    match detections against it.
    """
    enabled: bool = True
    model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    device: str = "auto"
    # Embeddings kept per identity (only high-quality frames are banked).
    pool_size: int = 12
    # Cosine similarity above which a detection is the same identity.
    match_threshold: float = 0.62
    # Below this we treat it as a genuinely new person.
    new_identity_threshold: float = 0.45
    # Run the detector every N frames looking for uncovered/lost subjects.
    scan_every: int = 8
    # A detection counts as uncovered if the current matte covers less than
    # this fraction of its box.
    max_overlap: float = 0.35
    # Mask-coverage confirmation for anything the cheap box test flags.
    #
    # `max_overlap` above thresholds the fraction of a detection's BOUNDING
    # BOX the matte fills, and box fill is a function of pose, not of tracking:
    # measured on the 30 persisted seeds of correctly-held subjects, box
    # coverage runs 0.302..0.738, so 3 of 30 are already below 0.35 at frame 0
    # with nothing wrong.  That is where 19 of the 22 phantom re-entry events
    # came from (3.3a).
    #
    # Coverage of the detection's own MASK does not have that failure mode:
    # over the same 30 seeds it runs 0.802..0.975, and a subject the matte is
    # genuinely not holding scores ~0.  0.50 is 1.60x below the worst genuine
    # case and far above a real miss.
    #
    # The box test stays as the cheap pre-filter -- the seeder only runs on
    # boxes that fail it, which is what keeps this one detector pass per scan
    # on a fixed-cast clip.
    mask_max_overlap: float = 0.50
    min_area_frac: float = 0.0015
    # 1 Oct 2026, the dance wall.  Step 3 of `_recover_entrants` (the local
    # repair passes) used to re-test its pending frames with BOX coverage,
    # while step 1 only reached its verdict through mask confirmation.  That
    # silences the repair in exactly the case it exists for: a cast member the
    # matte is not holding has her box filled by the NEIGHBOURING object
    # spreading across the gap, so box coverage reads her as covered.
    # Measured on the 29 Sep dance render: 11 of 11 boxes flagged at
    # 0.004..0.071 box coverage during the scan read 0.386..0.616 in the final
    # alpha, all above `max_overlap` -- the defect satisfied its own repair
    # test.  True re-tests on the confirmed silhouette instead.  Set False to
    # reproduce every render before 1 Oct 2026.
    # NOT YET VALIDATED ON A GPU RUN -- see HANDOFF section 5.
    recheck_with_mask: bool = True
    # 2 Oct 2026, the dance wall, second attempt.  Recovery (the reverse pass
    # and the local passes) merges into the matte with np.maximum, so anything
    # the recovery matte paints is permanent.  On dance the per-person tracker
    # in the reverse pass spreads from one dancer onto her neighbour and paints
    # the floor between them as body -- 8179 px of wall at f60, one person
    # fewer than v1 on 20 of the first 72 frames, identical with person_objects
    # per_person or both (run_2026-10-01_pp).  With this on, recovery may only
    # ADD pixels that lie inside a `gate_dilate_px`-dilated union of the
    # detector's per-person silhouettes at the nearest scan frames (both
    # neighbours OR-ed, so motion between scans is covered).  The wall between
    # two people lies inside nobody's silhouette and is dropped; a recovered
    # limb lies inside its owner's and is kept.  Frames with no usable
    # silhouette are left UNGATED (old behaviour), never blanked.
    # Risk is the inverse of the 26 Sep pocket-fill: clipping thin limbs the
    # detector under-segments.  Judge on blob count AND missing-vs-v1.
    # OFF until its GPU A/B.
    gate_recovery_to_silhouette: bool = False
    gate_dilate_px: int = 40
    # When an identity is re-acquired, re-matte this many frames on each side
    # of the gap so the alpha stitches seamlessly.
    stitch_radius: int = 6


@dataclass
class RefineConfig:
    """Boundary refinement and temporal stabilisation of the final alpha."""
    # Run a per-frame matting head inside the flow-adaptive unknown band.
    # This is what actually recovers blurred limbs.
    band_refine: bool = True
    # Flow-guided temporal filter: blends alpha with the motion-compensated
    # previous alpha where the warp is photometrically consistent.  Reduces
    # flicker without smearing fast motion.
    temporal_filter: bool = True
    temporal_alpha: float = 0.35
    # Guided filter on the boundary band, edge-aware, cheap.
    guided_filter: bool = True
    guided_radius: int = 4
    guided_eps: float = 1e-4
    # Core solidification (20 Sep): force the subject's interior to alpha 1,
    # leaving everything within `radius` of a true boundary untouched.
    # Answers the full-length review finding that interiors fade to 0.3-0.7
    # under fast motion and occlusion (butter -30% solid area on f281-283)
    # while every IoU column stayed green.  OFF by default -- flipping it is
    # Thriloush's call, and means re-rendering the delivery.
    core_solidify: bool = False
    core_solidify_thresh: float = 0.25
    core_solidify_radius_frac: float = 0.00625   # x frame width; 12 px at 1920
    core_solidify_floor: float = 0.05            # gap guard: never solidify near-zero alpha
    # Post-stabiliser (21 Sep): a second flow-guided temporal pass run LAST,
    # after core_solidify, gated per pixel by stillness as well as warp trust.
    # Answers Runbo's "it seems to flicker a lot" on tryguys: the visible
    # flicker there is 0<->1 strobing in STATIC regions, where the shipped
    # smoother's trust gate is open but its 0.35 authority is too weak to hold
    # it.  Offline on the delivered alpha: tryguys large-swing blinks -66%,
    # ipman/dance soft band >=99.1% kept.  ON by default since 21 Sep
    # (Thriloush's call after the GPU check: tryguys large blinks -63%,
    # interview -34%, ipman unchanged; one ~2-frame smudge on ipman f452-454).
    # Set 0.0 to reproduce the 20 Sep delivery exactly.
    post_stabilize: float = 0.8
    post_stabilize_still_px: float = 6.0         # = MotionConfig.high_motion_px


@dataclass
class DecontamConfig:
    """Foreground colour decontamination.

    Even a perfect alpha composites badly if you use the *observed* pixel as
    the foreground colour, because in the boundary band the observed pixel is
    already a mix of subject and old background.  You get a coloured fringe.
    We solve I = a*F + (1-a)*B for F using multi-level fast foreground
    estimation (Germer et al. 2020).
    """
    enabled: bool = True
    regularization: float = 1e-5
    # Speed/accuracy dial, not a correctness limit. Upstream's defaults are
    # 10/2. Re-measured 2026-08-24 on the committed fixture in
    # tests/test_decontamination.py, as fraction of the fringe error left:
    #
    #     10/2   37.6%      20/3   17.5%      40/5   9.0%      120/30  8.4%
    #
    # so the solver is converged by about 40/5. An earlier version of this
    # comment quoted 62% / 40% from a synthetic that was never committed and
    # could not be reproduced; the fixture now lives in the test suite.
    #
    # On real 1920x800 footage 40/5 measured 2.84s/frame against 20/3's
    # 2.75s, i.e. roughly half the residual error for about 3% more time.
    # Worth revisiting 20/3 -> 40/5, but absolute timings are hardware
    # specific so re-measure before changing it.
    n_small_iterations: int = 20
    n_big_iterations: int = 3
    small_size: int = 32
    # Only bother where alpha is genuinely fractional.
    band_only: bool = True
    band_lo: float = 0.02
    band_hi: float = 0.98


@dataclass
class Config:
    io: IOConfig = field(default_factory=IOConfig)
    shots: ShotConfig = field(default_factory=ShotConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    matting: MattingConfig = field(default_factory=MattingConfig)
    track_health: TrackHealthConfig = field(default_factory=TrackHealthConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    decontam: DecontamConfig = field(default_factory=DecontamConfig)

    seed_value: int = 0
    verbose: bool = True

    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    #: old section name -> current one, for configs written before a rename
    RENAMED_SECTIONS = {"memory_gate": "track_health"}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        # A renamed section must be re-keyed BEFORE the field walk below, which
        # only looks up current field names and would otherwise drop an old
        # section on the floor without a word. Silently ignoring a setting the
        # user wrote is worse than failing.
        d = dict(d or {})
        for old_name, new_name in cls.RENAMED_SECTIONS.items():
            if old_name in d:
                if new_name in d:
                    raise ValueError(
                        f"config sets both '{old_name}' and '{new_name}'; "
                        f"'{old_name}' is the old name for the same section. "
                        f"Keep one.")
                warnings.warn(
                    f"config section '{old_name}' was renamed to '{new_name}' "
                    f"in task 3.4 (the memory-gating half was dropped; what "
                    f"remains is track-health scoring). Still honoured, but "
                    f"update the file.", DeprecationWarning, stacklevel=2)
                d[new_name] = d.pop(old_name)

        kw: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if dataclasses.is_dataclass(f.type) and isinstance(v, dict):
                kw[f.name] = f.type(**v)          # type: ignore[operator]
            elif isinstance(v, dict) and f.name in _SUB:
                kw[f.name] = _SUB[f.name](**v)
            else:
                kw[f.name] = v
        return cls(**kw)

    @classmethod
    def load(cls, path: str) -> "Config":
        import yaml
        with open(path, "r") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def save(self, path: str) -> None:
        import yaml
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


_SUB = {
    "io": IOConfig,
    "shots": ShotConfig,
    "detect": DetectConfig,
    "seed": SeedConfig,
    "matting": MattingConfig,
    "track_health": TrackHealthConfig,
    # Back-compat: configs written before 3.4 call this section memory_gate.
    # It is accepted and mapped, so an old yaml still loads.
    "memory_gate": TrackHealthConfig,
    "motion": MotionConfig,
    "reid": ReIDConfig,
    "refine": RefineConfig,
    "decontam": DecontamConfig,
}
