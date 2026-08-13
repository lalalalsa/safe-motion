"""UR5 标称正运动学（标准 DH 约定）。

单关节齐次变换（Craig 经典 DH）：
    A_i = RotZ(θ_i) · TransZ(d_i) · TransX(a_i) · RotX(α_i)

节点定义（O_k = 前 k 个变换累积后的坐标系原点）：
    O_0 = base  = (0, 0, 0)        固定基座原点
    O_1..O_6 = joint_1..joint_6    各关节坐标系原点
    tcp = O_6                       无工具偏置，与 joint_6 重合

运动机械臂的连杆为线段 O_1-O_2 … O_5-O_6。
O_0-O_1 是固定基座段（O_1 恒为 (0, 0, d_1)），不参与运动学检查——
面试方参考数据中 source_minimum_workspace_margin = d1 - z_min = 0.039159，
恰好是 O_1 的 margin，证实参考实现同样不检查 O_0。
"""
from __future__ import annotations

import numpy as np

from .config import DH_A, DH_ALPHA, DH_D, N_JOINTS

NODE_LABELS = ["base", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# 参与 workspace 检查的节点：O_1..O_6（运动本体），不含固定基座 O_0
CHECKED_NODE_INDICES = [1, 2, 3, 4, 5, 6]

# 参与检查的连杆线段（节点索引对）
LINK_SEGMENTS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

# 各段连杆长度（|O_k - O_{k-1}|，由 DH 参数决定的常量，用于测试不变量）
LINK_LENGTHS = np.array([0.089159, 0.425, 0.39225, 0.10915, 0.09465, 0.0823])


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """标准 DH 单关节齐次变换矩阵 (4, 4)。"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def forward_kinematics(q) -> dict:
    """正运动学。

    返回 dict：
      base / joint_1 .. joint_6 / tcp : (3,) 三维坐标
      points                          : (7, 3)，O_0..O_6 全部节点
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (N_JOINTS,):
        raise ValueError(f"q 必须为 (6,)，得到 {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("q 包含 NaN/Inf")

    points = np.zeros((7, 3))
    T = np.eye(4)
    for i in range(N_JOINTS):
        T = T @ dh_transform(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])
        points[i + 1] = T[:3, 3]

    out = {label: points[i].copy() for i, label in enumerate(NODE_LABELS)}
    out["tcp"] = points[6].copy()
    out["points"] = points
    return out


def node_label(index: int) -> str:
    """节点索引 → 可读标签（O_6 同时是 tcp）。"""
    if index == 6:
        return "tcp (joint_6)"
    return NODE_LABELS[index]
