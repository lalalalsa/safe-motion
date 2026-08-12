"""Whole-body keep-in checks for the assignment's line-segment robot model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SafetyConfig
from .kinematics import chain_points


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    reason: str
    minimum_margin: float
    offending_node: int | None = None


def point_margins(points: np.ndarray, config: SafetyConfig) -> np.ndarray:
    """Signed distance to the nearest shrunken box face for every point."""
    points = np.asarray(points, dtype=float)
    lower = config.workspace.lower + config.safety_margin
    upper = config.workspace.upper - config.safety_margin
    return np.minimum(points - lower, upper - points).min(axis=1)


def check_full_body(q: np.ndarray, config: SafetyConfig) -> SafetyResult:
    """Check every moving joint and link against a convex axis-aligned box.

    An axis-aligned box is convex. Therefore, if both endpoints of each modeled
    link are inside it, the complete line segment is inside it as well. Checking
    every chain node is consequently exact for this simplified zero-thickness
    link model; it is not a claim about the real UR5 housing.
    """
    try:
        points = chain_points(q)[config.first_checked_node :]
        if not np.all(np.isfinite(points)):
            return SafetyResult(False, "non_finite_geometry", float("-inf"))
        margins = point_margins(points, config)
        index = int(np.argmin(margins))
        minimum = float(margins[index])
        if minimum < 0.0:
            return SafetyResult(False, "node_or_link_outside_workspace", minimum, index)
        return SafetyResult(True, "safe", minimum)
    except Exception as exc:
        return SafetyResult(False, f"geometry_error:{type(exc).__name__}", float("-inf"))
