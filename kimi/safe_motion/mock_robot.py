"""确定性 Mock Robot（题目统一定义的数学模型）。

状态更新严格采用 q[t+1] = q[t] + q_dot[t] * dt。
假设：perfect joint tracking、fixed timestep、零通信延迟、零传感器噪声、
零执行器延迟；不考虑加速度、力矩、惯量、摩擦与真实底层控制器动力学。
"""
from __future__ import annotations

import numpy as np

from .config import N_JOINTS


class MockRobot:
    """与题目定义等价的最小 Mock Robot。"""

    def __init__(self, q0, joint_limits):
        self.q = np.asarray(q0, dtype=float)
        if self.q.shape != (N_JOINTS,):
            raise ValueError(f"q0 必须为 (6,)，得到 {self.q.shape}")
        self.joint_limits = np.asarray(joint_limits, dtype=float).reshape(N_JOINTS, 2)
        if not self._inside_limits(self.q):
            raise ValueError("初始关节状态越限")

    def _inside_limits(self, q) -> bool:
        return bool(np.all(q >= self.joint_limits[:, 0])
                    and np.all(q <= self.joint_limits[:, 1]))

    def get_joint_state(self):
        return self.q.copy()

    def step(self, q_dot, dt):
        q_dot = np.asarray(q_dot, dtype=float)

        if q_dot.shape != (N_JOINTS,):
            raise ValueError("invalid command shape")
        if not np.all(np.isfinite(q_dot)):
            raise ValueError("non-finite command")

        q_next = self.q + q_dot * dt

        if not self._inside_limits(q_next):
            raise ValueError("joint limit violation")

        self.q = q_next
        return self.q.copy()
