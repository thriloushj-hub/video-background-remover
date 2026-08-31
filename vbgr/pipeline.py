"""Pipeline orchestration.

Per clip:

    decode -> shot split -> per shot:
        seed (YOLO -> SAM 3)
        matte (engine, with memory gating where supported)
        re-entry recovery (ReID re-prompting, bidirectional local passes)
        refine (flow-adaptive trimap -> matting head -> temporal stabilise)
        decontaminate
    -> composite -> encode (frame parity asserted)

Design notes worth keeping in mind while reading:

* **Frame parity is asserted at every stage.**  Every array that represents
  "one entry per frame" is checked against ``len(frames)``.  The v1 audit found
  a clip that emitted 325 frames from a 501-frame source with no error, and the
  only reason it survived was that nothing ever counted.
* **Post-processing lives outside the engine** so a benchmark isolates the
  engine.
* **The expensive calls are gated behind cheap ones.**  YOLO gates SAM 3; flow
  magnitude gates band refinement.  On a fixed-cast talking-head clip almost
  none of the recovery machinery fires.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import compose, decontaminate, motion, refine, shots, video_io
from .config import Config
from .detect import Detection, PersonDetector, held_props, select_person_boxes
from .engines.base import MattingEngine, build_engine
from .gate import QualityGate
from .reid import IdentityBank, FeatureExtractor, uncovered_boxes
from .seed import SAM3Seeder, build_seed, build_seeder


# --------------------------------------------------------------------------- #

@dataclass
class ShotReport:
    start: int
    end: int
    n_kept: int = 0
    n_dropped: int = 0
    # Frames scored as bad by the track-health monitor. Diagnostic only: the
    # monitor's one action is to declare the track lost and trigger a reseed
    # (see `reseeds`). It no longer tries to withhold frames from the engine's
    # memory -- that was task 3.4's memory gate, dropped 2026-08-23 because no
    # usable engine exposes the hook.
    bad_frames: int = 0
    reseeds: int = 0
    reentries: int = 0
    high_motion_frames: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class ClipReport:
    name: str
    n_frames: int = 0
    fps: float = 0.0
    engine: str = ""
    shots: List[ShotReport] = field(default_factory=list)
    seconds: float = 0.0
    outputs: Dict[str, str] = field(default_factory=dict)

    # {kwarg: feature} for anything the engine declared but could not deliver
    degraded: Dict[str, Optional[str]] = field(default_factory=dict)

    def summary(self) -> str:
        bf = sum(s.bad_frames for s in self.shots)
        rs = sum(s.reseeds for s in self.shots)
        re_ = sum(s.reentries for s in self.shots)
        hm = sum(s.high_motion_frames for s in self.shots)
        # bad_frames is explicitly labelled diagnostic. The old field was
        # called gate_rejects, which read as "the gate did 47 things" when the
        # honest reading was "the gate spotted 47 things and did nothing".
        out = (f"{self.name}: {self.n_frames}f @{self.fps:.3f} "
               f"{len(self.shots)} shot(s) engine={self.engine} "
               f"bad_frames={bf}(diagnostic) reseeds={rs} reentries={re_} "
               f"high_motion={hm} {self.seconds:.1f}s")
        if self.degraded:
            out += ("\n  DEGRADED: " +
                    "; ".join(f"{k} -> {v or 'unknown feature'} OFF"
                              for k, v in self.degraded.items()))
        return out


# --------------------------------------------------------------------------- #

class Pipeline:
    def __init__(self, cfg: Config, require_commercial: bool = False):
        self.cfg = cfg
        self.require_commercial = require_commercial
        self._engine: Optional[MattingEngine] = None
        self._detector: Optional[PersonDetector] = None
        self._seeder = None
        self._flow = motion.FlowEstimator(cfg.motion.flow_method,
                                          cfg.motion.flow_scale_short_side)

    # -- lazy components ---------------------------------------------------- #

    @property
    def engine(self) -> MattingEngine:
        if self._engine is None:
            m = self.cfg.matting
            kw = dict(checkpoint=m.checkpoint, device=m.device, half=m.half)
            # only engines that support it take this; build_engine passes
            # through to __init__ so guard on the signature
            import inspect as _i
            from .engines.base import _REGISTRY
            cls = _REGISTRY.get(m.engine)
            if cls and "strict_capabilities" in _i.signature(cls).parameters:
                kw["strict_capabilities"] = m.strict_capabilities
            self._engine = build_engine(
                m.engine, require_commercial=self.require_commercial, **kw)
        return self._engine

    @property
    def detector(self) -> PersonDetector:
        if self._detector is None:
            d = self.cfg.detect
            self._detector = PersonDetector(d.model, d.conf, d.iou, d.device)
        return self._detector

    @property
    def seeder(self):
        """The configured seeder -- Mask R-CNN by default (5.1k).

        The SAM 3 path is still selectable via ``cfg.seed.backend``, but it
        has never loaded: see ``SeedSAM._ensure`` for the two reasons.
        """
        if self._seeder is None:
            self._seeder = build_seeder(self.cfg.seed)
        return self._seeder

    # ------------------------------------------------------------------ #
    # Clip
    # ------------------------------------------------------------------ #

    def run_clip(self, path: str, out_dir: Optional[str] = None) -> ClipReport:
        t0 = time.time()
        cfg = self.cfg
        out_dir = out_dir or cfg.io.output_dir
        name = os.path.splitext(os.path.basename(path))[0]

        info = video_io.probe(path)
        frames = video_io.read_all(path, cfg.io.max_size)
        if not frames:
            raise RuntimeError(f"{path}: decoded 0 frames")

        rep = ClipReport(name=name, n_frames=len(frames),
                         fps=float(info.fps), engine=self.engine.info.name)

        # -- shots ------------------------------------------------------- #
        if cfg.shots.enabled and len(frames) > cfg.shots.min_shot_len * 2:
            starts = shots.compute_scene_cuts(
                frames, cfg.shots.cut_threshold, cfg.shots.hist_bins,
                cfg.shots.min_shot_len, cfg.shots.use_absdiff_guard,
                cfg.shots.absdiff_sigma)
        else:
            starts = [0]
        ranges = shots.shot_ranges(starts, len(frames))   # asserts full coverage

        alphas = np.zeros((len(frames), *frames[0].shape[:2]), np.float32)
        for a, b in ranges:
            sr = self._run_shot(frames[a:b], a)
            alphas[a:b] = sr[0]
            rep.shots.append(sr[1])

        assert len(alphas) == len(frames), "alpha/frame count divergence"

        # -- output ------------------------------------------------------ #
        rep.outputs = self._write_outputs(name, path, frames, alphas, info, out_dir)
        rep.degraded = self.engine.degraded
        rep.seconds = time.time() - t0
        if cfg.verbose:
            print(rep.summary())
        return rep

    # ------------------------------------------------------------------ #
    # Shot
    # ------------------------------------------------------------------ #

    def _run_shot(self, frames: List[np.ndarray], offset: int
                  ) -> Tuple[np.ndarray, ShotReport]:
        cfg = self.cfg
        rep = ShotReport(start=offset, end=offset + len(frames))
        H, W = frames[0].shape[:2]

        # ---- seed -------------------------------------------------------- #
        people, props = self.detector.detect(frames[0])
        kept, dropped = select_person_boxes(
            people, W, H, cfg.detect.person_rel_size_min,
            cfg.detect.box_score_ratio, cfg.detect.person_rel_area_min)
        rep.n_kept, rep.n_dropped = len(kept), len(dropped)

        if not kept:
            rep.notes.append("no subject detected; emitting empty alpha")
            return np.zeros((len(frames), H, W), np.float32), rep

        seed = build_seed(frames[0], self.seeder, kept,
                          held_props(props, kept), cfg.seed)
        seed_mask = refine.erode_dilate(
            seed.mask, cfg.matting.r_erode, cfg.matting.r_dilate)
        rep.notes.extend(seed.notes)

        # ---- forward matting --------------------------------------------- #
        self.engine.reset()
        if self.engine.is_streaming:
            alphas, rep = self._matte_streaming(frames, seed_mask, rep)
        else:
            alphas = self.engine.matte(frames, seed_mask,
                                       n_warmup=cfg.matting.n_warmup_static)

        # ---- reverse pass for late entrants ------------------------------- #
        if cfg.reid.enabled:
            alphas, rep = self._recover_entrants(frames, alphas, rep)

        # ---- refinement ---------------------------------------------------- #
        alphas, rep = self._refine_sequence(frames, alphas, rep)

        assert len(alphas) == len(frames)
        return alphas, rep

    # ------------------------------------------------------------------ #
    # Streaming matting with the track-health monitor
    # ------------------------------------------------------------------ #

    def _matte_streaming(self, frames, seed_mask, rep: ShotReport):
        cfg = self.cfg
        gate = QualityGate(
            cfg.track_health.max_area_ratio, cfg.track_health.min_area_ratio,
            cfg.track_health.min_flow_iou, cfg.track_health.min_area_frac,
            lost_after=cfg.track_health.lost_after)

        a0 = self.engine.start(frames[0], seed_mask,
                               n_warmup=cfg.matting.n_warmup_static)
        gate.prime(a0, frames[0])

        out = np.empty((len(frames), *frames[0].shape[:2]), np.float32)
        out[0] = a0

        # Motion warmup: run the first real frames and bank them as permanent
        # anchors so a low-texture region cannot erode later in the shot.
        n_motion = min(cfg.matting.n_warmup_motion, len(frames) - 1)
        if cfg.matting.drift_anchor_warmup and self.engine.has("permanent_memory"):
            for i in range(1, n_motion + 1):
                self.engine.step(frames[i], permanent=True)

        prev = frames[0]
        for i in range(1, len(frames)):
            fl = self._flow.flow(prev, frames[i])
            a = self.engine.step(frames[i])
            v = gate.evaluate(a, frames[i], fl) if cfg.track_health.enabled \
                else None

            if v is not None and not v.ok:
                rep.bad_frames += 1
                # There is deliberately no commit_to_memory=False retry here.
                # That was the memory gate (3.4) and it is dropped: MatAnyone 2
                # takes no **kwargs on step(), and SAM2Matting has no step() at
                # all, so this branch never fired on anything we ship. What is
                # left is the part that never needed the engine's help --
                # noticing the track is gone and re-seeding.
                if gate.track_lost:
                    a, ok = self._reseed(frames[i], i, rep)
                    if ok:
                        gate.reset()
                        gate.prime(a, frames[i])

            out[i] = a
            prev = frames[i]

        assert len(out) == len(frames)
        return out, rep

    def _reseed(self, frame, idx: int, rep: ShotReport):
        """Re-detect and re-prompt after the gate declares the track lost."""
        H, W = frame.shape[:2]
        people, props = self.detector.detect(frame)
        kept, _ = select_person_boxes(
            people, W, H, self.cfg.detect.person_rel_size_min,
            self.cfg.detect.box_score_ratio,
            self.cfg.detect.person_rel_area_min)
        if not kept:
            return np.zeros((H, W), np.float32), False
        seed = build_seed(frame, self.seeder, kept,
                          held_props(props, kept), self.cfg.seed)
        m = refine.erode_dilate(seed.mask, self.cfg.matting.r_erode,
                                self.cfg.matting.r_dilate)
        rep.reseeds += 1
        if self.engine.is_streaming:
            return self.engine.step(frame, reseed_mask=m), True
        return m.astype(np.float32), True

    # ------------------------------------------------------------------ #
    # Re-entry recovery
    # ------------------------------------------------------------------ #

    def _recover_entrants(self, frames, alphas, rep: ShotReport):
        """Find subjects the forward pass never covered, and matte them in.

        Three mechanisms, cheapest first:

        1. **Coverage scan** every ``scan_every`` frames using only the
           detector.  If every box is already explained by the current matte,
           nothing else runs.
        2. **Reverse pass** seeded at the last frame.  Anyone present at the end
           -- including someone who walked in mid-shot and stayed -- is seeded
           where they clearly exist and propagated *backward* to their entry,
           which gives continuous coverage with no mid-shot gap.  This is the
           same bidirectional trick professional trackers use.
        3. **Local bidirectional pass** for anyone present at neither end,
           anchored on the frame where ReID is most confident.
        """
        cfg = self.cfg
        H, W = frames[0].shape[:2]
        bank = IdentityBank(
            FeatureExtractor(cfg.reid.model, cfg.reid.device),
            cfg.reid.pool_size, cfg.reid.match_threshold,
            cfg.reid.new_identity_threshold)

        # --- 1. scan for uncovered subjects ------------------------------- #
        pending: Dict[int, List[int]] = {}      # frame -> box indices
        boxes_at: Dict[int, List] = {}
        for i in range(0, len(frames), max(1, cfg.reid.scan_every)):
            people, _ = self.detector.detect(frames[i])
            kept, _ = select_person_boxes(
                people, W, H, cfg.detect.person_rel_size_min,
                cfg.detect.box_score_ratio,
                cfg.detect.person_rel_area_min)
            if not kept:
                continue
            unc = uncovered_boxes(alphas[i], [d.box for d in kept],
                                  cfg.reid.max_overlap, cfg.reid.min_area_frac)
            if unc:
                pending[i] = unc
                boxes_at[i] = kept
            # Bank embeddings of well-covered subjects so ReID has a reference.
            for j, d in enumerate(kept):
                if j not in unc:
                    bank.observe(bank.new_identity().ident if not bank.identities
                                 else next(iter(bank.identities)),
                                 frames[i], d.box, alphas[i] > 0.5, i)

        if not pending:
            return alphas, rep

        # --- 2. reverse pass if anyone is uncovered at the end ------------- #
        last = len(frames) - 1
        if any(k > len(frames) * 0.6 for k in pending):
            rev = self._directional_pass(frames, last, backward=True)
            if rev is not None:
                alphas = np.maximum(alphas, rev)
                rep.reentries += 1

        # --- 3. local bidirectional passes for the rest -------------------- #
        for i in sorted(pending):
            if uncovered_boxes(alphas[i], [boxes_at[i][j].box for j in pending[i]],
                               cfg.reid.max_overlap, cfg.reid.min_area_frac):
                loc = self._local_pass(frames, i, cfg.reid.stitch_radius)
                if loc is not None:
                    lo, hi, arr = loc
                    alphas[lo:hi] = np.maximum(alphas[lo:hi], arr)
                    rep.reentries += 1

        assert len(alphas) == len(frames)
        return alphas, rep

    def _directional_pass(self, frames, anchor: int, backward: bool
                          ) -> Optional[np.ndarray]:
        cfg = self.cfg
        H, W = frames[0].shape[:2]
        people, props = self.detector.detect(frames[anchor])
        kept, _ = select_person_boxes(people, W, H,
                                      cfg.detect.person_rel_size_min,
                                      cfg.detect.box_score_ratio,
                                      cfg.detect.person_rel_area_min)
        if not kept:
            return None
        seed = build_seed(frames[anchor], self.seeder, kept,
                          held_props(props, kept), cfg.seed)
        m = refine.erode_dilate(seed.mask, cfg.matting.r_erode,
                                cfg.matting.r_dilate)

        seq = frames[anchor::-1] if backward else frames[anchor:]
        self.engine.reset()
        a = self.engine.matte(seq, m, n_warmup=cfg.matting.n_warmup_static)
        self.engine.reset()

        out = np.zeros((len(frames), H, W), np.float32)
        if backward:
            out[:anchor + 1] = a[::-1]
        else:
            out[anchor:] = a
        return out

    def _local_pass(self, frames, anchor: int, radius: int):
        """Forward + backward from `anchor`, limited to a window around it."""
        cfg = self.cfg
        H, W = frames[0].shape[:2]
        people, props = self.detector.detect(frames[anchor])
        kept, _ = select_person_boxes(people, W, H,
                                      cfg.detect.person_rel_size_min,
                                      cfg.detect.box_score_ratio,
                                      cfg.detect.person_rel_area_min)
        if not kept:
            return None
        seed = build_seed(frames[anchor], self.seeder, kept,
                          held_props(props, kept), cfg.seed)
        m = refine.erode_dilate(seed.mask, cfg.matting.r_erode,
                                cfg.matting.r_dilate)

        lo = max(0, anchor - cfg.reid.scan_every - radius)
        hi = min(len(frames), anchor + cfg.reid.scan_every + radius)

        self.engine.reset()
        fwd = self.engine.matte(frames[anchor:hi], m,
                                n_warmup=cfg.matting.n_warmup_static)
        self.engine.reset()
        bwd = self.engine.matte(frames[anchor:lo - 1 if lo > 0 else None:-1], m,
                                n_warmup=cfg.matting.n_warmup_static)
        self.engine.reset()

        arr = np.zeros((hi - lo, H, W), np.float32)
        arr[anchor - lo:] = fwd
        arr[:anchor - lo + 1] = np.maximum(arr[:anchor - lo + 1], bwd[::-1])
        return lo, hi, arr

    # ------------------------------------------------------------------ #
    # Refinement
    # ------------------------------------------------------------------ #

    def _refine_sequence(self, frames, alphas, rep: ShotReport):
        cfg = self.cfg
        if not (cfg.motion.enabled or cfg.refine.band_refine or
                cfg.refine.temporal_filter or cfg.refine.guided_filter):
            return alphas, rep

        stab = refine.TemporalStabilizer(
            cfg.refine.temporal_alpha, cfg.motion.warp_consistency_thresh,
            self._flow) if cfg.refine.temporal_filter else None

        head = None
        if cfg.refine.band_refine and self.engine.has("per_frame_head"):
            head = self.engine.matte_frame

        out = np.empty_like(alphas)
        prev = frames[0]
        for i, f in enumerate(frames):
            a = alphas[i]

            if cfg.motion.enabled and i > 0:
                fl = self._flow.flow(prev, f)
                mag = self._flow.magnitude(fl, cfg.motion.flow_smooth_sigma)

                if float(np.percentile(mag, 95)) > cfg.motion.high_motion_px:
                    rep.high_motion_frames += 1
                    # Flow-warped prior: gives a fast limb a plausible alpha so
                    # the matting head has something to correct rather than
                    # nothing to find.
                    warped, trust = motion.flow_alpha_prior(
                        alphas[i - 1], prev, f, fl,
                        cfg.motion.warp_consistency_thresh)
                    a = motion.blend_with_prior(
                        a, warped, trust, mag, cfg.motion.warp_prior_weight,
                        cfg.motion.high_motion_px)

                    if cfg.refine.band_refine:
                        tri = motion.flow_adaptive_trimap(
                            a, mag, cfg.motion.trimap_base,
                            cfg.motion.trimap_flow_gain, cfg.motion.trimap_max)
                        a = refine.refine_band(a, f, tri, head)
            elif cfg.refine.guided_filter:
                a = refine.guided_filter(a, f, cfg.refine.guided_radius,
                                         cfg.refine.guided_eps)

            if stab is not None:
                a = stab.step(f, a)

            out[i] = a
            prev = f

        assert len(out) == len(alphas)
        return out, rep

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def _write_outputs(self, name, src_path, frames, alphas, info, out_dir
                       ) -> Dict[str, str]:
        cfg = self.cfg
        os.makedirs(out_dir, exist_ok=True)
        H, W = frames[0].shape[:2]
        modes = (["green", "alpha", "transparent"]
                 if cfg.io.output_mode == "all" else [cfg.io.output_mode])
        writers: Dict[str, video_io.VideoWriter] = {}
        paths: Dict[str, str] = {}

        for m in modes:
            ext = "webm" if m == "transparent" else "mp4"
            p = os.path.join(out_dir, f"{name}_{m}.{ext}")
            paths[m] = p
            writers[m] = video_io.VideoWriter(
                p, W, H, info.fps, crf=cfg.io.crf,
                alpha=(m == "transparent"))

        for i, f in enumerate(frames):
            a = alphas[i]
            F = (decontaminate.decontaminate(
                    f, a, cfg.decontam.regularization,
                    cfg.decontam.n_small_iterations,
                    cfg.decontam.n_big_iterations, cfg.decontam.small_size,
                    cfg.decontam.band_only, cfg.decontam.band_lo,
                    cfg.decontam.band_hi)
                 if cfg.decontam.enabled else f)

            if "green" in writers:
                writers["green"].write(
                    compose.composite_over_color(f, a, cfg.io.green, F))
            if "alpha" in writers:
                writers["alpha"].write(compose.alpha_to_bgr(a))
            if "transparent" in writers:
                writers["transparent"].write(compose.to_bgra(f, a, F))

        for m, w in writers.items():
            w.close(expect=len(frames), strict=cfg.io.strict_frame_count)

        # Carry the audio across -- v1 dropped it, which makes outputs useless
        # for cutting against the original.
        for m, p in list(paths.items()):
            if m == "transparent":
                continue
            muxed = p.replace(f"_{m}.", f"_{m}_av.")
            try:
                video_io.mux_audio(p, src_path, muxed)
                paths[m] = muxed
            except Exception:                              # noqa: BLE001
                pass
        return paths


# --------------------------------------------------------------------------- #

def run_batch(cfg: Config, require_commercial: bool = False,
              on_clip: Optional[Callable[[ClipReport], None]] = None
              ) -> List[ClipReport]:
    pipe = Pipeline(cfg, require_commercial)
    exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
    files = sorted(f for f in os.listdir(cfg.io.input_dir)
                   if f.lower().endswith(exts))
    reports = []
    for f in files:
        r = pipe.run_clip(os.path.join(cfg.io.input_dir, f))
        reports.append(r)
        if on_clip:
            on_clip(r)
    return reports
