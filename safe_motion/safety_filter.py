"""Fail-closed velocity scaling safety filter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SafetyConfig
from .geometry import SafetyResult, check_full_body
from .validation import inside_joint_limits


@dataclass(frozen=True)
class CandidateAttempt:
    """One auditable grid or bisection candidate evaluation."""

    scale: float
    safe: bool
    reason: str
    minimum_margin: float
    offending_node: int | None
    stage: str


@dataclass(frozen=True)
class FilterResult:
    velocity: np.ndarray
    scale: float
    modified: bool
    stopped: bool
    reason: str
    minimum_margin: float
    attempts: tuple[CandidateAttempt, ...] = field(default_factory=tuple)


def nominal_velocity(q: np.ndarray, target: np.ndarray, dt: float, config: SafetyConfig) -> np.ndarray:
    raw = (np.asarray(target, dtype=float) - np.asarray(q, dtype=float)) / dt
    return np.clip(raw, -config.max_joint_velocity, config.max_joint_velocity)


def command_is_safe(
    q: np.ndarray, q_dot: np.ndarray, dt: float, config: SafetyConfig
) -> SafetyResult:
    """Validate joint and workspace safety throughout one control interval."""
    q = np.asarray(q, dtype=float)
    q_dot = np.asarray(q_dot, dtype=float)
    if (
        q.shape != (6,)
        or q_dot.shape != (6,)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(q_dot))
        or not np.isfinite(dt)
        or dt <= 0.0
    ):
        return SafetyResult(False, "invalid_candidate_velocity", float("-inf"))
    if np.any(np.abs(q_dot) > config.max_joint_velocity + 1e-12):
        return SafetyResult(False, "candidate_velocity_limit_violation", float("-inf"))

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


def _attempt(scale: float, result: SafetyResult, stage: str) -> CandidateAttempt:
    return CandidateAttempt(
        scale=float(scale),
        safe=result.safe,
        reason=result.reason,
        minimum_margin=result.minimum_margin,
        offending_node=result.offending_node,
        stage=stage,
    )


def safety_filter(q: np.ndarray, q_dot_nom: np.ndarray, dt: float, config: SafetyConfig) -> FilterResult:
    """Choose a verified-safe scale and refine its upper boundary.

    The configured scale grid first finds a safe bracket. Bisection then reduces
    unnecessary modification inside that bracket. Every returned command is
    explicitly checked; bisection never replaces the fail-closed predicate.
    """
    zero = np.zeros(6, dtype=float)
    attempts: list[CandidateAttempt] = []
    try:
        q_dot_nom = np.asarray(q_dot_nom, dtype=float)
        if q_dot_nom.shape != (6,) or not np.all(np.isfinite(q_dot_nom)):
            raise ValueError("non-finite nominal velocity")
        last_reason = "no_safe_scale"
        failed_scale_above: float | None = None
        best_scale: float | None = None
        best_result: SafetyResult | None = None

        for scale in config.velocity_scales:
            velocity = float(scale) * q_dot_nom
            result = command_is_safe(q, velocity, dt, config)
            attempts.append(_attempt(scale, result, "grid"))
            if result.safe:
                best_scale = float(scale)
                best_result = result
                break
            last_reason = result.reason

            failed_scale_above = float(scale)

        if best_scale is None or best_result is None:
            return FilterResult(
                zero, 0.0, True, True, last_reason, float("-inf"), tuple(attempts)
            )

        if (
            0.0 < best_scale < 1.0
            and failed_scale_above is not None
            and config.bisection_iterations > 0
        ):
            safe_scale = best_scale
            unsafe_scale = failed_scale_above
            for _ in range(config.bisection_iterations):
                candidate_scale = 0.5 * (safe_scale + unsafe_scale)
                result = command_is_safe(q, candidate_scale * q_dot_nom, dt, config)
                attempts.append(_attempt(candidate_scale, result, "bisection"))
                if result.safe:
                    safe_scale = candidate_scale
                    best_result = result
                else:
                    unsafe_scale = candidate_scale
            best_scale = safe_scale

        stopped = bool(best_scale == 0.0)
        return FilterResult(
            velocity=best_scale * q_dot_nom,
            scale=best_scale,
            modified=bool(best_scale < 1.0),
            stopped=stopped,
            reason="safe" if best_scale == 1.0 else ("stopped" if stopped else "scaled_for_safety"),
            minimum_margin=best_result.minimum_margin,
            attempts=tuple(attempts),
        )
    except Exception as exc:
        return FilterResult(
            zero,
            0.0,
            True,
            True,
            f"filter_error:{type(exc).__name__}",
            float("-inf"),
            tuple(attempts),
        )
