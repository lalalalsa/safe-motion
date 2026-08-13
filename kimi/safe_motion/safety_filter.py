"""安全过滤（题目第 6、7 节）。

方法 A：速度缩放 / 回溯 + 二分细化。
  - 保持名义速度方向不变，依次尝试 scale ∈ {1.0, 0.75, 0.5, 0.25, 0.0}；
  - 对每个候选，预测 q_next = q + scale·q_dot_nom·dt，并沿该步做
    子步插值全身检查（含关节限位预检，先于 Mock Robot 的硬约束）；
  - 取能安全执行的最大 scale；在「通过的最大网格点」与「失败的上一档」
    之间做二分细化，使修改幅度接近最小；
  - 只有 scale=0 安全 → 停止（零速度）；连当前状态都不安全 → 抛
    SafetyFilterError（说明不变量已被破坏）。

Fail-closed（题目硬性要求）：
  filter_fail_closed() 兜底一切异常与非法输出，返回零速度。
  安全过滤失败时绝不回退到原始 VLA 动作。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RobotConfig
from .geometry import BodyCheckResult, Workspace, check_motion


class SafetyFilterError(RuntimeError):
    """安全过滤器自身无法给出可靠安全动作（输入非法或当前状态已不安全）。"""


@dataclass
class Attempt:
    """一次候选速度的评估记录（供 --explain 讲解）。"""

    scale: float
    safe: bool
    min_margin: float
    worst_label: str
    worst_boundary: str
    note: str


@dataclass
class FilterResult:
    q_dot: np.ndarray
    scale: float
    status: str  # "unmodified" | "modified" | "stopped"
    reason: str
    attempts: list = field(default_factory=list)


class SafetyFilter:
    def __init__(self, robot_cfg: RobotConfig, workspace: Workspace,
                 scales=(1.0, 0.75, 0.5, 0.25, 0.0), bisect_iters: int = 10,
                 substeps: int = 4, spacing: float = 0.02,
                 safety_margin: float = 1e-6):
        self.cfg = robot_cfg
        self.workspace = workspace
        self.scales = tuple(scales)
        self.bisect_iters = int(bisect_iters)
        self.substeps = int(substeps)
        self.spacing = float(spacing)
        self.safety_margin = float(safety_margin)

    # ------------------------------------------------------------------
    # 名义动作生成（题目第 6 节）
    # ------------------------------------------------------------------
    def nominal_velocity(self, q, q_target, dt: float) -> np.ndarray:
        """q_dot_nom = (q_target - q) / dt，再逐关节限速到 max_joint_velocity。"""
        q = np.asarray(q, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        v = (q_target - q) / dt
        return np.clip(v, -self.cfg.max_joint_velocity, self.cfg.max_joint_velocity)

    # ------------------------------------------------------------------
    # 单步安全性评估（关节限位预检 + 子步全身 workspace 检查）
    # ------------------------------------------------------------------
    def check_step(self, q, q_dot, dt: float):
        """返回 (safe, BodyCheckResult|None, note)。"""
        q = np.asarray(q, dtype=float)
        q_dot = np.asarray(q_dot, dtype=float)
        if q.shape != (6,) or q_dot.shape != (6,):
            return False, None, "维度非法"
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(q_dot))):
            return False, None, "含 NaN/Inf"

        q_next = q + q_dot * dt
        if (np.any(q_next < self.cfg.joint_lower)
                or np.any(q_next > self.cfg.joint_upper)):
            return False, None, "关节限位越界"

        r = check_motion(q, q_next, self.workspace, substeps=self.substeps,
                         spacing=self.spacing, safety_margin=self.safety_margin)
        note = ("ok" if r.safe else
                f"workspace 越界: {r.worst_label} @ {r.worst_boundary} "
                f"margin={r.min_margin:+.4f} m")
        return r.safe, r, note

    # ------------------------------------------------------------------
    # 速度缩放 / 回溯 + 二分细化
    # ------------------------------------------------------------------
    def filter(self, q, q_dot_nom, dt: float) -> FilterResult:
        q = np.asarray(q, dtype=float)
        q_dot_nom = np.asarray(q_dot_nom, dtype=float)
        if q.shape != (6,) or q_dot_nom.shape != (6,):
            raise SafetyFilterError("filter 输入维度非法")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(q_dot_nom))):
            raise SafetyFilterError("filter 输入含 NaN/Inf")

        attempts: list[Attempt] = []
        best: float | None = None
        fail_above: float | None = None

        for s in self.scales:
            safe, r, note = self.check_step(q, s * q_dot_nom, dt)
            attempts.append(Attempt(
                scale=s, safe=safe,
                min_margin=(r.min_margin if r is not None else float("nan")),
                worst_label=(r.worst_label if r is not None else "-"),
                worst_boundary=(r.worst_boundary if r is not None else "-"),
                note=note))
            if safe:
                best = s
                break
            fail_above = s

        if best is None:
            raise SafetyFilterError(
                "包括零速度在内的所有候选均不安全：当前状态很可能已经越界")

        if best == 0.0:
            return FilterResult(np.zeros(6), 0.0, "stopped",
                                "仅零速度安全，停止（hold position）", attempts)

        # 二分细化：在通过的最大网格点 best 与失败的上一档 fail_above 之间
        if best < 1.0 and fail_above is not None and self.bisect_iters > 0:
            lo, hi = best, fail_above
            for _ in range(self.bisect_iters):
                mid = 0.5 * (lo + hi)
                safe, r, note = self.check_step(q, mid * q_dot_nom, dt)
                if safe:
                    lo = mid
                else:
                    hi = mid
            best = lo
            safe, r, note = self.check_step(q, best * q_dot_nom, dt)
            attempts.append(Attempt(
                scale=best, safe=True,
                min_margin=(r.min_margin if r is not None else float("nan")),
                worst_label=(r.worst_label if r is not None else "-"),
                worst_boundary=(r.worst_boundary if r is not None else "-"),
                note=f"二分细化后采用（{self.bisect_iters} 轮）"))

        status = "unmodified" if best == 1.0 else "modified"
        reason = ("名义动作安全，原样执行" if status == "unmodified"
                  else f"速度缩放到 {best:.4f}（最小修改）")
        return FilterResult(best * q_dot_nom, best, status, reason, attempts)

    # ------------------------------------------------------------------
    # Fail-closed 包装：任何异常 / 非法输出 → 零速度
    # ------------------------------------------------------------------
    def filter_fail_closed(self, q, q_dot_nom, dt: float) -> FilterResult:
        try:
            res = self.filter(q, q_dot_nom, dt)
        except Exception as exc:  # noqa: BLE001 —— 安全层必须兜住一切
            return FilterResult(np.zeros(6), 0.0, "stopped",
                                f"安全过滤失败，fail-closed 停止: {exc}")
        if res.q_dot.shape != (6,) or not np.all(np.isfinite(res.q_dot)):
            return FilterResult(np.zeros(6), 0.0, "stopped",
                                "过滤器返回非法结果，fail-closed 停止",
                                res.attempts)
        return res
