"""Configuration values kept in one place so hidden scenarios need no code changes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Workspace:
    x: tuple[float, float] = (-0.70, 0.70)
    y: tuple[float, float] = (-0.70, 0.70)
    z: tuple[float, float] = (0.05, 0.90)

    @classmethod
    def from_dict(cls, value: dict) -> "Workspace":
        return cls(tuple(value["x"]), tuple(value["y"]), tuple(value["z"]))

    @property
    def lower(self) -> np.ndarray:
        return np.array([self.x[0], self.y[0], self.z[0]], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.x[1], self.y[1], self.z[1]], dtype=float)


def _default_joint_limits() -> np.ndarray:
    # The assignment asks for configurable limits. ±2π is a common conservative
    # software range for this simplified UR5 model, not a real safety rating.
    return np.tile(np.array([-2.0 * np.pi, 2.0 * np.pi]), (6, 1))


@dataclass(frozen=True)
class SafetyConfig:
    workspace: Workspace = field(default_factory=Workspace)
    joint_limits: np.ndarray = field(default_factory=_default_joint_limits)
    max_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.full(6, 1.5, dtype=float)
    )
    jump_threshold: float = 0.75
    safety_margin: float = 1e-4
    path_substeps: int = 10
    velocity_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    bisection_iterations: int = 10
    # The fixed pedestal begins at z=0 while the example keep-in box begins at
    # z=0.05. We therefore check the moving chain from joint_1 onward.
    first_checked_node: int = 1

    @classmethod
    def from_scenario(cls, data: dict) -> "SafetyConfig":
        kwargs: dict = {}
        if "workspace" in data:
            kwargs["workspace"] = Workspace.from_dict(data["workspace"])
        if "joint_limits" in data:
            kwargs["joint_limits"] = np.asarray(data["joint_limits"], dtype=float)
        if "max_joint_velocity" in data:
            value = np.asarray(data["max_joint_velocity"], dtype=float)
            kwargs["max_joint_velocity"] = np.full(6, value) if value.ndim == 0 else value
        if "jump_threshold" in data:
            kwargs["jump_threshold"] = float(data["jump_threshold"])
        if "safety_margin" in data:
            kwargs["safety_margin"] = float(data["safety_margin"])
        if "path_substeps" in data:
            kwargs["path_substeps"] = data["path_substeps"]
        if "bisection_iterations" in data:
            kwargs["bisection_iterations"] = data["bisection_iterations"]
        return cls(**kwargs)
