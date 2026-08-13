"""Deterministic robot model specified by the assignment."""

from __future__ import annotations

import numpy as np

from .validation import inside_joint_limits


class MockRobot:
    def __init__(self, q0: np.ndarray, joint_limits: np.ndarray):
        self.q = np.asarray(q0, dtype=float).copy()
        self.joint_limits = np.asarray(joint_limits, dtype=float).copy()
        self.step_calls = 0
        if not inside_joint_limits(self.q, self.joint_limits):
            raise ValueError("invalid initial joint state")

    def get_joint_state(self) -> np.ndarray:
        return self.q.copy()

    def step(self, q_dot: np.ndarray, dt: float) -> np.ndarray:
        q_dot = np.asarray(q_dot, dtype=float)
        if q_dot.shape != (6,):
            raise ValueError("invalid command shape")
        if not np.all(np.isfinite(q_dot)):
            raise ValueError("non-finite command")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("invalid timestep")
        q_next = self.q + q_dot * dt
        if not inside_joint_limits(q_next, self.joint_limits):
            raise ValueError("joint limit violation")
        self.q = q_next
        self.step_calls += 1
        return self.q.copy()
