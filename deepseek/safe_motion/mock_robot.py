"""确定性 Mock Robot（题目统一定义的数学模型）。

严格采用 q[t+1] = q[t] + q_dot[t] * dt 的一阶积分；
假设 perfect joint tracking / fixed timestep / 零延迟零噪声零摩擦，
不建模加速度、力矩、惯性等真实底层控制器动力学。
"""
import numpy as np

from .config import as_joint_limits


def inside_joint_limits(q, joint_limits):
    """判断关节状态是否全部位于限位内（闭区间）。"""
    q = np.asarray(q, dtype=float)
    limits = as_joint_limits(joint_limits)
    return bool(np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]))


class MockRobot:
    """与题目等价的最小 Mock Robot。"""

    def __init__(self, q0, joint_limits):
        self.q = np.asarray(q0, dtype=float)
        if self.q.shape != (6,):
            raise ValueError("joint_state 必须为 6 维")
        self.joint_limits = as_joint_limits(joint_limits)

    def get_joint_state(self):
        return self.q.copy()

    def step(self, q_dot, dt):
        q_dot = np.asarray(q_dot, dtype=float)

        if q_dot.shape != (6,):
            raise ValueError("invalid command shape")
        if not np.all(np.isfinite(q_dot)):
            raise ValueError("non-finite command")

        q_next = self.q + q_dot * dt

        if not inside_joint_limits(q_next, self.joint_limits):
            raise ValueError("joint limit violation")

        self.q = q_next
        return self.q.copy()
