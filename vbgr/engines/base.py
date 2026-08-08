"""The matting-engine interface.

Everything that differs between MatAnyone 2, SAM2Matting and RVM lives behind
this ABC.  Two reasons that matters:

1.  **Licensing is undecided.**  SAM2Matting is CC BY-NC-SA 4.0 and MatAnyone 2
    is NTU S-Lab -- both non-commercial.  If this ships as a product the matting
    engine has to be swapped, and that swap should be a config string, not a
    rewrite.  Every engine declares its licence, and ``build_engine`` refuses to
    return a non-commercial engine when ``require_commercial=True``.
2.  **Fair A/B.**  All post-processing (flow-adaptive trimap, temporal
    stabilisation, decontamination) sits *outside* the engine, so a benchmark
    run isolates the engine itself.

Streaming vs sequence engines
-----------------------------
This distinction is forced by the upstream code, not invented here:

* **MatAnyone 2** exposes ``InferenceCore.step(image, ...)`` -- genuinely
  frame-at-a-time.  We get full control: memory gating, permanent anchors, and
  mid-shot re-prompting all work.
* **SAM2Matting** exposes ``init_state(video_path=dir)`` +
  ``propagate_in_video(state)`` -- a generator over a *whole* frame directory.
  There is no supported way to interrupt it mid-generator, so per-frame memory
  gating is not available.  We work around it by cutting the timeline into
  chunks and re-seeding at chunk boundaries when the ReID layer reports a lost
  track.

Pretending both were streaming would produce an adapter that silently does
nothing.  So the capability is declared and the pipeline branches on it.
"""
from __future__ import annotations

import abc
import inspect
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set

import numpy as np


# --------------------------------------------------------------------------- #

@dataclass
class EngineInfo:
    name: str
    license: str
    commercial_ok: bool
    # "streaming" | "sequence"
    mode: str = "streaming"
    # {"mask_prompt","point_prompt","box_prompt","text_prompt",
    #  "permanent_memory","memory_gate","per_frame_head","auto_human"}
    capabilities: Set[str] = field(default_factory=set)
    url: str = ""
    notes: str = ""


class MattingEngine(abc.ABC):
    info: EngineInfo

    # -- degraded-feature tracking ------------------------------------------ #
    # Populated whenever a kwarg backing a declared capability turns out not to
    # exist on the installed API. The pipeline reads this and refuses to report
    # a feature as active when it isn't.

    @property
    def degraded(self) -> Dict[str, Optional[str]]:
        """{kwarg: feature it would have implemented}. Empty means all good."""
        return dict(getattr(self, "_degraded", {}))

    def note_degraded(self, kwarg: str, feature: Optional[str] = None) -> None:
        if not hasattr(self, "_degraded"):
            self._degraded = {}
        self._degraded[kwarg] = feature

    def feature_active(self, kwarg: str) -> bool:
        """False once `kwarg` has been found missing on this engine."""
        return kwarg not in getattr(self, "_degraded", {})

    # -- the one method every engine must provide --------------------------- #

    @abc.abstractmethod
    def matte(self,
              frames: Sequence[np.ndarray],
              seed_mask: Optional[np.ndarray],
              n_warmup: int = 10,
              progress: Optional[Callable[[int], None]] = None
              ) -> np.ndarray:
        """Matte a whole shot.

        Returns a ``(T, H, W)`` float32 array of alpha in [0, 1], with exactly
        ``len(frames)`` entries.  Warmup frames are consumed internally and
        never appear in the output -- the caller is guaranteed frame parity.
        """

    def reset(self) -> None:
        """Clear all memory. Must be called between shots."""

    # -- streaming engines additionally provide ----------------------------- #

    def start(self, frame_bgr: np.ndarray, seed_mask: np.ndarray,
              n_warmup: int = 10) -> np.ndarray:
        raise NotImplementedError(f"{self.info.name} is not a streaming engine")

    def step(self, frame_bgr: np.ndarray, *,
             commit_to_memory: bool = True,
             permanent: bool = False,
             reseed_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance one frame.

        ``commit_to_memory=False`` produces a prediction *without* writing it
        into working memory.  This is the memory gate: a frame scored as bad
        (occlusion, collapsed track) must not poison memory and drift the rest
        of the shot.

        ``permanent=True`` writes the frame into non-evicting memory.  Without
        several permanent anchors, a few dozen recent frames outvote the single
        first-frame anchor and a low-texture region -- a light shirt, a waist --
        erodes to nothing over a long shot.
        """
        raise NotImplementedError(f"{self.info.name} is not a streaming engine")

    def matte_frame(self, frame_bgr: np.ndarray,
                    trimap: np.ndarray) -> Optional[np.ndarray]:
        """Single-frame matting from a trimap, if the engine has such a head.

        SAM2Matting's progressive matting head goes here.  This is what makes
        the flow-adaptive trimap actually recover blurred limbs instead of just
        blurring the boundary.
        """
        return None

    # -- convenience -------------------------------------------------------- #

    @property
    def is_streaming(self) -> bool:
        return self.info.mode == "streaming"

    def has(self, cap: str) -> bool:
        return cap in self.info.capabilities


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, Any] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def _load_all() -> None:
    from . import matanyone2, sam2matting, rvm, passthrough  # noqa: F401


def build_engine(name: str, require_commercial: bool = False, **kwargs) -> MattingEngine:
    if name not in _REGISTRY:
        _load_all()
    if name not in _REGISTRY:
        raise KeyError(f"unknown engine {name!r}; have {sorted(_REGISTRY)}")

    cls = _REGISTRY[name]
    if require_commercial and not cls.INFO.commercial_ok:
        raise PermissionError(
            f"engine {cls.INFO.name!r} is licensed {cls.INFO.license!r} "
            f"(non-commercial) but require_commercial=True.\n"
            f"Swap to a commercially-licensable engine before shipping. "
            f"See docs/LICENSING.md.")
    return cls(**kwargs)


def list_engines() -> Dict[str, EngineInfo]:
    _load_all()
    return {k: v.INFO for k, v in _REGISTRY.items()}


# --------------------------------------------------------------------------- #
# Helpers shared by adapters
# --------------------------------------------------------------------------- #

def resolve_device(spec: str = "auto") -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class CapabilityError(RuntimeError):
    """A declared capability is not backed by the installed API."""


# Which config feature each risky kwarg implements. If one of these is dropped,
# the corresponding feature is silently OFF, which is far worse than a crash:
# the pipeline still reports gate_rejects, the ablation still produces numbers,
# and every one of them is wrong.
CRITICAL_KWARGS: Dict[str, str] = {
    "update_memory":   "memory_gate  (memory_gate.enabled)",
    "force_permanent": "permanent memory anchors  (matting.drift_anchor_warmup)",
    "commit_to_memory": "memory_gate  (memory_gate.enabled)",
}

_WARNED: set = set()


def kwarg_support(fn: Callable, name: str) -> str:
    """One of 'yes', 'no', 'unverifiable'.

    'unverifiable' means the function takes ``**kwargs``, so the name is
    *accepted* but may be silently ignored inside. Accepting an argument is not
    the same as honouring it, and for the memory-control kwargs that difference
    is the whole ballgame.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return "unverifiable"
    if name in sig.parameters:
        return "yes"
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return "unverifiable"
    return "no"


def filter_kwargs(fn: Callable, kwargs: Dict[str, Any],
                  engine: Optional["MattingEngine"] = None,
                  where: str = "") -> Dict[str, Any]:
    """Drop kwargs the installed version of `fn` does not accept, LOUDLY.

    These are third-party research repos under active development and their
    signatures move, so dropping an unsupported kwarg still beats a TypeError
    three hours into a batch. But dropping one in silence is how you end up
    with a memory gate that is switched off while still reporting rejections.
    Every drop is warned once per process and recorded on the engine, and the
    pipeline refuses to claim a feature is active when its kwarg was dropped.
    """
    kept, dropped = {}, {}
    for k, v in kwargs.items():
        if kwarg_support(fn, k) == "no":
            dropped[k] = v
        else:
            kept[k] = v

    for k in dropped:
        feature = CRITICAL_KWARGS.get(k)
        name = getattr(getattr(engine, "info", None), "name", "engine")
        key = (name, k, where)
        if engine is not None:
            engine.note_degraded(k, feature)
        if key in _WARNED:
            continue
        _WARNED.add(key)
        target = f"{getattr(fn, '__qualname__', fn)}{('  in ' + where) if where else ''}"
        if feature:
            msg = (f"\n{'!' * 78}\n"
                   f"! CAPABILITY NOT AVAILABLE: {name} does not accept {k!r}\n"
                   f"!   target : {target}\n"
                   f"!   feature: {feature} IS NOW OFF\n"
                   f"!   effect : the pipeline will still run and still print counters,\n"
                   f"!            but this feature is doing nothing. Any A/B that varies\n"
                   f"!            it is measuring noise. Do not trust those numbers.\n"
                   f"!   fix    : check the installed version's signature, or set\n"
                   f"!            strict_capabilities: true to make this an error.\n"
                   f"{'!' * 78}")
            print(msg, file=sys.stderr, flush=True)
            warnings.warn(f"{name}: {feature} disabled, {k!r} not accepted by "
                          f"{target}", RuntimeWarning, stacklevel=2)
        else:
            print(f"[{name}] note: dropped unsupported kwarg {k!r} for {target}",
                  file=sys.stderr, flush=True)
    return kept


def supports_kwarg(fn: Callable, name: str) -> bool:
    """True if the name is accepted. See kwarg_support for the 3-way answer."""
    return kwarg_support(fn, name) in ("yes", "unverifiable")


def require_kwarg(fn: Callable, name: str, engine: "MattingEngine",
                  strict: bool = False, where: str = "") -> bool:
    """Assert that a capability the engine *declares* is really there.

    Returns True if usable. With ``strict`` a mismatch raises instead, which is
    what you want in CI and before a long batch.
    """
    status = kwarg_support(fn, name)
    if status == "yes":
        return True
    feature = CRITICAL_KWARGS.get(name, name)
    detail = (f"{engine.info.name} declares a capability implemented by {name!r}, "
              f"but {getattr(fn, '__qualname__', fn)} "
              + ("takes **kwargs so it cannot be verified"
                 if status == "unverifiable" else "does not accept it")
              + f". Feature affected: {feature}.")
    if strict:
        raise CapabilityError(detail)
    engine.note_degraded(name, feature)
    key = (engine.info.name, name, where or "require")
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"\n!! {detail}\n", file=sys.stderr, flush=True)
        warnings.warn(detail, RuntimeWarning, stacklevel=2)
    return status == "unverifiable"


def to_alpha_2d(x) -> np.ndarray:
    """Coerce whatever an engine returned into HxW float32 in [0, 1]."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, (tuple, list)):
        x = x[-1]
    a = np.squeeze(np.asarray(x, np.float32))
    while a.ndim > 2:
        a = a[0] if a.shape[0] <= 4 else a[..., 0]
    if a.max() > 1.5:                      # someone handed us 0-255
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)
