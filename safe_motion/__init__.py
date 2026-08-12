"""Small, fail-closed safety layer for UR5 VLA trajectories.

Keep package initialization side-effect free so ``python -m safe_motion.replay``
can load the CLI module exactly once. Public functions are imported from their
own modules explicitly in application code.
"""

from .config import SafetyConfig, Workspace

__all__ = ["SafetyConfig", "Workspace"]
