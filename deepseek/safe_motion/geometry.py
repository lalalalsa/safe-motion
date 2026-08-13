"""全身几何检查：矩形 Keep-in Workspace 判定 + 连杆采样 + 带符号 margin。

核心思想：不能只检查 TCP。对每个机器人状态，需要检查
  - TCP
  - 六个关节位置
  - 各连杆上的中间采样点
全部位于 workspace 内才算安全。
"""
import numpy as np

from . import config
from .kinematics import forward_kinematics


def point_margin(p, workspace):
    """单点到矩形工作空间的带符号最小距离（m）。

    正 = 在内部（距最近边界）；负 = 已越界。
    """
    m = np.inf
    for i, axis in enumerate(("x", "y", "z")):
        lo, hi = workspace[axis]
        m = min(m, p[i] - lo, hi - p[i])
    return float(m)


def sample_link(p0, p1, spacing):
    """沿线段 [p0, p1] 等间距采样，返回 (n, 3) 点集（含两端点）。"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    n = max(1, int(np.ceil(length / spacing)))
    ts = np.linspace(0.0, 1.0, n + 1)
    return p0[None, :] + ts[:, None] * (p1 - p0)[None, :]


def full_body_points(q, spacing=None):
    """对位形 q 做全身采样，返回所有待检查点 (N, 3)。

    检查范围（题目第 5 节）：
      - TCP 与六个关节位置（即 FK 输出索引 1..7，不含固定底座 base）；
      - 各连杆上的中间采样点（joint_1→joint_2 … joint_6→tcp）。

    base（底座原点，z=0）与底座段（base→joint_1）为固定结构，
    不参与运动安全检查（z_min=0.05 已高于底座，若检查 base 将恒为负）。
    """
    spacing = config.LINK_SAMPLE_SPACING if spacing is None else spacing
    pts = forward_kinematics(q)
    samples = [pts[1:]]  # joint_1 .. tcp
    for i in range(1, pts.shape[0] - 1):  # 连杆 i=1..6
        samples.append(sample_link(pts[i], pts[i + 1], spacing))
    return np.vstack(samples)


def full_body_check(q, workspace, spacing=None):
    """全身安全检查。

    返回 dict：
      is_safe      : bool，是否全身安全（所有采样点 margin >= SAFETY_MARGIN）
      min_margin   : float，全身最小带符号 margin（越界时为负）
      worst_point  : (3,)，最危险的采样点坐标
      worst_link   : int，最危险点所属连杆索引（1..6），0 表示关节/tcp 点本身
    """
    spacing = config.LINK_SAMPLE_SPACING if spacing is None else spacing
    pts = forward_kinematics(q)

    min_margin = np.inf
    worst_point = None
    worst_link = 0

    # 关节/tcp 点（索引 1..7）
    for i in range(1, pts.shape[0]):
        m = point_margin(pts[i], workspace)
        if m < min_margin:
            min_margin, worst_point, worst_link = m, pts[i], 0

    # 连杆中间采样点（连杆 i=1..6：joint_i → joint_{i+1}）
    for i in range(1, pts.shape[0] - 1):
        for p in sample_link(pts[i], pts[i + 1], spacing):
            m = point_margin(p, workspace)
            if m < min_margin:
                min_margin, worst_point, worst_link = m, p, i

    is_safe = min_margin >= config.SAFETY_MARGIN
    return {
        "is_safe": is_safe,
        "min_margin": float(min_margin),
        "worst_point": np.asarray(worst_point, dtype=float),
        "worst_link": worst_link,
    }
