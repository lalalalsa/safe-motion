"""全身几何检查：矩形 Keep-in Workspace + 连杆采样 + 带符号 margin。

设计要点：
  - 不能只检查 TCP：每个状态都检查 6 个关节点 + TCP + 各连杆中间采样点；
  - 矩形 keep-in 区域是凸集，线段两端点在内部 ⟹ 整段在内部，
    因此对凸盒本检查在数学上是精确的；连杆采样作为通用兜底机制
    （防御浮点边界情形，并向非凸区域扩展时仍然可用）；
  - margin 定义：点到最近边界的带符号距离，负值 = 已越界。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import (CHECKED_NODE_INDICES, LINK_SEGMENTS,
                         forward_kinematics, node_label)

BOUNDARY_NAMES = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]


@dataclass(frozen=True)
class Workspace:
    """轴对齐矩形 Keep-in Workspace（单位 m）。"""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def from_dict(cls, d: dict) -> "Workspace":
        ws = cls(
            xmin=float(d["x"][0]), xmax=float(d["x"][1]),
            ymin=float(d["y"][0]), ymax=float(d["y"][1]),
            zmin=float(d["z"][0]), zmax=float(d["z"][1]),
        )
        for axis in ("x", "y", "z"):
            lo, hi = getattr(ws, f"{axis}min"), getattr(ws, f"{axis}max")
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                raise ValueError(f"workspace {axis} 区间非法: [{lo}, {hi}]")
        return ws

    def all_margins(self, pts: np.ndarray) -> np.ndarray:
        """(N, 3) 点集 → (N, 6) 各边界 margin（负 = 越出该边界）。"""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return np.column_stack([
            pts[:, 0] - self.xmin, self.xmax - pts[:, 0],
            pts[:, 1] - self.ymin, self.ymax - pts[:, 1],
            pts[:, 2] - self.zmin, self.zmax - pts[:, 2],
        ])

    def margins(self, pts: np.ndarray) -> np.ndarray:
        """(N, 3) 点集 → (N,) 最小 margin。"""
        return self.all_margins(pts).min(axis=1)

    def boundary_name(self, p: np.ndarray) -> str:
        """点 p 的最近（或越界）边界名。"""
        return BOUNDARY_NAMES[int(np.argmin(self.all_margins(p)[0]))]


def sample_segment(p0, p1, spacing: float) -> np.ndarray:
    """沿线段等间距采样（含两端点），相邻采样点距离 <= spacing。"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    n = max(1, int(np.ceil(length / spacing)))
    ts = np.linspace(0.0, 1.0, n + 1)
    return p0[None, :] + ts[:, None] * (p1 - p0)[None, :]


@dataclass
class BodyCheckResult:
    """一次全身检查的结果（可解释性：哪个点、越哪条边界、margin 多少）。"""

    safe: bool
    min_margin: float
    worst_label: str
    worst_position: np.ndarray
    worst_boundary: str
    n_points_checked: int


def check_state(q, workspace: Workspace, spacing: float = 0.02,
                safety_margin: float = 0.0, tol: float = 1e-12) -> BodyCheckResult:
    """对单个关节状态做全身检查。

    safe ⟺ 全身最小 margin >= safety_margin（tol 为浮点容忍）。
    """
    pts = forward_kinematics(q)["points"]

    node_idx = list(CHECKED_NODE_INDICES)
    node_pts = pts[node_idx]
    node_margins = workspace.margins(node_pts)

    # 节点优先：最差记录落在节点上时报告节点标签（而非 "link_x_y"），
    # 保证与参考分析中 "point": 3 之类的定位一致。
    ni = int(np.argmin(node_margins))
    best_margin = float(node_margins[ni])
    best_label = node_label(node_idx[ni])
    best_pos = node_pts[ni]

    n_checked = len(node_pts)
    for (i, j) in LINK_SEGMENTS:
        seg = sample_segment(pts[i], pts[j], spacing)
        seg_margins = workspace.margins(seg)
        n_checked += len(seg)
        si = int(np.argmin(seg_margins))
        if seg_margins[si] < best_margin - 1e-12:
            best_margin = float(seg_margins[si])
            best_label = f"link_{i}_{j}"
            best_pos = seg[si]

    return BodyCheckResult(
        safe=best_margin >= safety_margin - tol,
        min_margin=best_margin,
        worst_label=best_label,
        worst_position=np.asarray(best_pos, dtype=float),
        worst_boundary=workspace.boundary_name(best_pos),
        n_points_checked=n_checked,
    )


def check_motion(q_start, q_end, workspace: Workspace, substeps: int = 4,
                 spacing: float = 0.02, safety_margin: float = 0.0) -> BodyCheckResult:
    """检查 q_start → q_end 整段关节空间运动（子步插值检查，加分项）。

    对 linspace(0..1) 的中间状态逐个做全身检查，返回最差结果。
    不包含 t=0（当前状态由调用方保证已安全）。
    """
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    worst: BodyCheckResult | None = None
    for t in np.linspace(0.0, 1.0, substeps + 1)[1:]:
        r = check_state(q_start + t * (q_end - q_start), workspace,
                        spacing=spacing, safety_margin=safety_margin)
        if worst is None or r.min_margin < worst.min_margin:
            worst = r
    return worst
