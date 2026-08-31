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
    # Mask R-CNN seeding. Boxes still come from the pipeline's own detector,
    # so the gates behave exactly as tested; the seeder only answers "what is
    # the mask inside this box". A box no instance matches is skipped with a
    # note rather than filled with its rectangle.
    maskrcnn_score_min: float = 0.80
    maskrcnn_match_iou: float = 0.30
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
    min_area_frac: float = 0.0015
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
