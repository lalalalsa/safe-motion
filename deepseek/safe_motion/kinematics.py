"""UR5 标称正运动学（标准 DH）。

返回 8 个点的三维坐标：
    [base, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, tcp]

约定：
  - 使用题目给定的标称 DH 参数（标准 DH：Rz(theta) Tz(d) Tx(a) Rx(alpha)）；
  - 相邻节点之间的线段即视为机械臂连杆；
  - 不考虑真实出厂标定误差、工具 TCP 偏置与外壳复杂几何，故 tcp == joint_6。
"""
import numpy as np

from . import config


def dh_transform(q, a, d, alpha):
    """标准 DH 单关节齐次变换矩阵 (4, 4)。"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cq, sq = np.cos(q), np.sin(q)
    return np.array([
        [cq, -sq * ca,  sq * sa, a * cq],
        [sq,  cq * ca, -cq * sa, a * sq],
        [0.0,      sa,      ca,      d],
        [0.0,     0.0,     0.0,    1.0],
    ])


def forward_kinematics(q):
    """返回 (8, 3) 关节/tcp 三维坐标，索引 0..7 = base..tcp。"""
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise ValueError(f"joint 向量必须为 (6,)，得到 {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("joint 向量包含 NaN/Inf")

    T = np.eye(4)
    pts = [np.zeros(3)]  # base
    for i in range(6):
        T = T @ dh_transform(q[i], config.DH_A[i], config.DH_D[i], config.DH_ALPHA[i])
        pts.append(T[:3, 3].copy())  # joint_{i+1}
    pts.append(pts[-1].copy())       # tcp == joint_6（无工具偏置）
    return np.asarray(pts)


def link_segments(q):
    """返回 7 条连杆线段端点对：[(p0, p1), ...]，p0/p1 为 (3,) 坐标。"""
    pts = forward_kinematics(q)
    return [(pts[i], pts[i + 1]) for i in range(pts.shape[0] - 1)]
