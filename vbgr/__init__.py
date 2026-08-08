"""vbgr2 -- video background remover, v2.

Decoupled tracking and matting, with explicit fixes for the three failure
modes the v1 pipeline could not solve: re-entry after occlusion, fast/blurred
low-contrast limbs, and progressive memory drift.
"""
__version__ = "2.0.0"

from .config import Config                    # noqa: F401
