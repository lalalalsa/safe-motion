"""Fail-closed velocity scaling safety filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SafetyConfig
from .geometry import SafetyResult, check_full_body
from .validation import inside_joint_limits


@dataclass(frozen=True)
class FilterResult:
    velocity: np.ndarray
    scale: float
    modified: bool
    stopped: bool
    reason: str
    minimum_margin: float


def nominal_velocity(q: np.ndarray, target: np.ndarray, dt: float, config: SafetyConfig) -> np.ndarray:
    raw = (np.asarray(target, dtype=float) - np.asarray(q, dtype=float)) / dt
    return np.clip(raw, -config.max_joint_velocity, config.max_joint_velocity)


def command_is_safe(
    q: np.ndarray, q_dot: np.ndarray, dt: float, config: SafetyConfig
) -> SafetyResult:
    """Validate joint and workspace safety throughout one control interval."""
    q = np.asarray(q, dtype=float)
    q_dot = np.asarray(q_dot, dtype=float)
    if q.shape != (6,) or q_dot.shape != (6,) or not np.all(np.isfinite(q_dot)):
        return SafetyResult(False, "invalid_candidate_velocity", float("-inf"))

    minimum_margin = float("inf")
    for fraction in np.linspace(0.0, 1.0, config.path_substeps + 1):
        candidate = q + fraction * q_dot * dt
        if not inside_joint_limits(candidate, config.joint_limits):
            return SafetyResult(False, "predicted_joint_limit_violation", float("-inf"))
        result = check_full_body(candidate, config)
        minimum_margin = min(minimum_margin, result.minimum_margin)
        if not result.safe:
            return SafetyResult(False, result.reason, minimum_margin, result.offending_node)
    return SafetyResult(True, "safe", minimum_margin)


def safety_filter(q: np.ndarray, q_dot_nom: np.ndarray, dt: float, config: SafetyConfig) -> FilterResult:
    """Choose the largest configured safe fraction; any error yields zero velocity."""
    zero = np.zeros(6, dtype=float)
    try:
        if not np.all(np.isfinite(q_dot_nom)):
            raise ValueError("non-finite nominal velocity")
        last_reason = "no_safe_scale"
        for scale in config.velocity_scales:
            velocity = float(scale) * np.asarray(q_dot_nom, dtype=float)
            result = command_is_safe(q, velocity, dt, config)
            if result.safe:
                stopped = bool(scale == 0.0)
                return FilterResult(
                    velocity=velocity,
                    scale=float(scale),
                    modified=bool(scale < 1.0),
                    stopped=stopped,
                    reason="safe" if scale == 1.0 else ("stopped" if stopped else "scaled_for_safety"),
                    minimum_margin=result.minimum_margin,
                )
            last_reason = result.reason
        return FilterResult(zero, 0.0, True, True, last_reason, float("-inf"))
    except Exception as exc:
        return FilterResult(
            zero, 0.0, True, True, f"filter_error:{type(exc).__name__}", float("-inf")
        )
