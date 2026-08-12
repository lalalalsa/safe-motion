"""Validate the complete VLA proposal before it can reach the robot."""

from __future__ import annotations

import numpy as np

from .config import SafetyConfig


class InputValidationError(ValueError):
    """The scenario violates the assignment's input contract."""


def inside_joint_limits(q: np.ndarray, limits: np.ndarray, atol: float = 1e-12) -> bool:
    q = np.asarray(q, dtype=float)
    limits = np.asarray(limits, dtype=float)
    return bool(
        q.shape == (6,)
        and limits.shape == (6, 2)
        and np.all(np.isfinite(q))
        and np.all(q >= limits[:, 0] - atol)
        and np.all(q <= limits[:, 1] + atol)
    )


def validate_config(config: SafetyConfig) -> None:
    """Reject malformed safety policy before it can weaken an invariant."""
    lower = config.workspace.lower
    upper = config.workspace.upper
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite([lower, upper])):
        raise InputValidationError("workspace bounds must be finite 3D values")
    if np.any(lower >= upper):
        raise InputValidationError("each workspace lower bound must be below its upper bound")

    limits = np.asarray(config.joint_limits, dtype=float)
    if limits.shape != (6, 2) or not np.all(np.isfinite(limits)):
        raise InputValidationError("joint_limits must be a finite array with shape (6, 2)")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise InputValidationError("each joint lower limit must be below its upper limit")

    velocities = np.asarray(config.max_joint_velocity, dtype=float)
    if velocities.shape != (6,) or not np.all(np.isfinite(velocities)):
        raise InputValidationError("max_joint_velocity must be a finite vector with shape (6,)")
    if np.any(velocities <= 0.0):
        raise InputValidationError("max_joint_velocity values must be positive")

    if not np.isfinite(config.jump_threshold) or config.jump_threshold <= 0.0:
        raise InputValidationError("jump_threshold must be finite and positive")
    if not np.isfinite(config.safety_margin) or config.safety_margin < 0.0:
        raise InputValidationError("safety_margin must be finite and non-negative")
    if np.any(2.0 * config.safety_margin >= upper - lower):
        raise InputValidationError("safety_margin leaves no valid workspace interior")
    if not isinstance(config.path_substeps, int) or isinstance(config.path_substeps, bool):
        raise InputValidationError("path_substeps must be an integer")
    if config.path_substeps < 1:
        raise InputValidationError("path_substeps must be at least 1")

    scales = np.asarray(config.velocity_scales, dtype=float)
    if scales.ndim != 1 or scales.size < 2 or not np.all(np.isfinite(scales)):
        raise InputValidationError("velocity_scales must be a finite one-dimensional sequence")
    if not np.isclose(scales[0], 1.0) or not np.isclose(scales[-1], 0.0):
        raise InputValidationError("velocity_scales must start at 1.0 and end at 0.0")
    if np.any(scales < 0.0) or np.any(scales > 1.0) or np.any(np.diff(scales) > 0.0):
        raise InputValidationError("velocity_scales must descend from 1.0 to 0.0")
    if config.first_checked_node != 1:
        raise InputValidationError("first_checked_node must be 1 to check the complete moving chain")


def validate_scenario(data: dict, config: SafetyConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return normalized inputs or raise before a MockRobot is constructed."""
    validate_config(config)
    try:
        q0 = np.asarray(data["joint_state"], dtype=float)
        chunk = np.asarray(data["action_chunk"], dtype=float)
        action_hz = float(data["action_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputValidationError(f"missing or malformed field: {exc}") from exc

    if q0.shape != (6,):
        raise InputValidationError(f"joint_state must have shape (6,), got {q0.shape}")
    if chunk.shape != (50, 6):
        raise InputValidationError(f"action_chunk must have shape (50, 6), got {chunk.shape}")
    if not np.isfinite(action_hz) or action_hz <= 0.0:
        raise InputValidationError("action_hz must be finite and positive")
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(chunk)):
        raise InputValidationError("joint_state and action_chunk must contain only finite values")
    if not inside_joint_limits(q0, config.joint_limits):
        raise InputValidationError("joint_state violates configured joint limits")

    below = chunk < config.joint_limits[:, 0]
    above = chunk > config.joint_limits[:, 1]
    if np.any(below | above):
        step, joint = np.argwhere(below | above)[0]
        raise InputValidationError(f"target at step {step}, joint {joint} violates joint limits")

    deltas = np.vstack([chunk[0] - q0, np.diff(chunk, axis=0)])
    bad = np.argwhere(np.abs(deltas) > config.jump_threshold)
    if bad.size:
        step, joint = bad[0]
        raise InputValidationError(
            f"trajectory jump at step {step}, joint {joint} exceeds "
            f"{config.jump_threshold:.3f} rad"
        )
    return q0, chunk, action_hz
