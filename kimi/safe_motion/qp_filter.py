"""加分项：QP 安全过滤（题目第 7 节方法 B）。

与速度缩放（方法 A，保持方向、撞墙即停）不同，QP 求解的是：

    min_v  ||v - v_nom||^2
    s.t.   每个临近边界的节点：margin_i(q) + ∇margin_i · v·dt >= safety_margin
           关节速度上限、关节位置限位（线性盒约束）

即「在合法集合里找与名义动作欧氏距离最小的速度」。它可以在贴边时
沿边界滑行（把法向的推压转成切向运动），而不是直接停下。

工程分层（关键设计）：
  - QP 的约束是 FK 的**线性化**（一阶泰勒），只是「提案」；
    线性化误差在 0.15 rad 的单步上可达亚毫米级，因此 QP 使用更大的
    qp_safety_margin（默认 5 mm）吸收该误差——同时天然实现
    「workspace safety margin」加分项；
  - 提案必须通过非线性精确验证（SafetyFilter.check_step，含子步检查，
    阈值仍为 ~0）才会下发；验证不过或求解失败 → 回退速度缩放；
  - 任何异常 → fail-closed 零速度。
  线性化负责效率，非线性验证负责安全，回退链路保证可靠。
"""
from __future__ import annotations

import numpy as np

from .geometry import Workspace
from .kinematics import CHECKED_NODE_INDICES, forward_kinematics, node_label
from .safety_filter import Attempt, FilterResult, SafetyFilter


class QPFilter:
    def __init__(self, fallback: SafetyFilter, active_margin: float = 0.20,
                 qp_safety_margin: float = 5e-3, jac_eps: float = 1e-6):
        self.fallback = fallback          # SafetyFilter（验证 + 回退）
        self.cfg = fallback.cfg
        self.workspace: Workspace = fallback.workspace
        self.active_margin = float(active_margin)
        self.qp_safety_margin = float(qp_safety_margin)
        self.jac_eps = float(jac_eps)

    # replay 审计所需的参数透传
    @property
    def spacing(self):
        return self.fallback.spacing

    @property
    def safety_margin(self):
        return self.fallback.safety_margin

    # ------------------------------------------------------------------
    def check_step(self, q, q_dot, dt):
        """精确非线性验证：直接委托给 scaling 过滤器的 check_step。"""
        return self.fallback.check_step(q, q_dot, dt)

    def nominal_velocity(self, q, q_target, dt):
        return self.fallback.nominal_velocity(q, q_target, dt)

    # ------------------------------------------------------------------
    def _node_jacobians(self, q):
        """所有受检节点的位置雅可比（前向差分，6 次额外 FK）。

        返回 {node_index: (3, 6) Jacobian}。
        """
        base_pts = forward_kinematics(q)["points"]
        jac = {i: np.zeros((3, 6)) for i in CHECKED_NODE_INDICES}
        for j in range(6):
            dq = q.copy()
            dq[j] += self.jac_eps
            pts = forward_kinematics(dq)["points"]
            for i in CHECKED_NODE_INDICES:
                jac[i][:, j] = (pts[i] - base_pts[i]) / self.jac_eps
        return base_pts, jac

    def _build_constraints(self, q, dt, safety_margin):
        """构造线性化 margin 约束 A v >= b（仅活跃节点）与 v 的盒约束。"""
        ws = self.workspace
        base_pts, jac = self._node_jacobians(q)

        bounds_vec = np.array([
            [ws.xmin, ws.xmax], [ws.ymin, ws.ymax], [ws.zmin, ws.zmax]])
        rows_a, rows_b = [], []
        for i in CHECKED_NODE_INDICES:
            p = base_pts[i]
            margins6 = np.array([p[k] - bounds_vec[k, 0] for k in range(3)]
                                + [bounds_vec[k, 1] - p[k] for k in range(3)])
            if margins6.min() > self.active_margin:
                continue  # 离边界足够远，本步不会受限（精确验证兜底）
            for k in range(3):
                rows_a.append(jac[i][k, :])          # p_k - lo  >= sm
                rows_b.append((safety_margin - (p[k] - bounds_vec[k, 0])) / dt)
                rows_a.append(-jac[i][k, :])         # hi - p_k  >= sm
                rows_b.append((safety_margin - (bounds_vec[k, 1] - p[k])) / dt)

        # v 的盒约束：速度上限 ∩ 位置限位推出的一步可达集
        lo = np.maximum(-self.cfg.max_joint_velocity,
                        (self.cfg.joint_lower - q) / dt)
        hi = np.minimum(+self.cfg.max_joint_velocity,
                        (self.cfg.joint_upper - q) / dt)
        if rows_a:
            return np.asarray(rows_a), np.asarray(rows_b), lo, hi
        return np.zeros((0, 6)), np.zeros(0), lo, hi

    # ------------------------------------------------------------------
    def filter(self, q, q_dot_nom, dt) -> FilterResult:
        q = np.asarray(q, dtype=float)
        v_nom = np.asarray(q_dot_nom, dtype=float)

        # 快路径：名义动作本身就安全 → 原样执行
        safe, r, note = self.fallback.check_step(q, v_nom, dt)
        if safe:
            return FilterResult(v_nom.copy(), 1.0, "unmodified",
                                "名义动作安全，原样执行",
                                [Attempt(1.0, True, r.min_margin, r.worst_label,
                                         r.worst_boundary, "ok")])

        # QP：找离 v_nom 最近的线性化合法速度
        # （线性化约束用 qp_safety_margin 吸收一阶泰勒误差）
        try:
            from scipy.optimize import LinearConstraint, minimize
            A, b, lo, hi = self._build_constraints(q, dt,
                                                   self.qp_safety_margin)
            v0 = np.clip(v_nom, lo, hi)
            constraints = ([LinearConstraint(A, b, np.inf)]
                           if A.shape[0] else [])
            sol = minimize(lambda v: 0.5 * float(np.sum((v - v_nom) ** 2)),
                           v0, jac=lambda v: v - v_nom, method="SLSQP",
                           bounds=list(zip(lo, hi)), constraints=constraints)
            if not sol.success or not np.all(np.isfinite(sol.x)):
                raise RuntimeError(f"QP 未收敛: {sol.message}")
            v_qp = sol.x
        except Exception:
            # QP 不可用 → 回退速度缩放
            return self._fallback_result(q, v_nom, dt, "QP 求解失败")

        # 精确验证 QP 提案（含子步全身检查）
        ratio = float(np.linalg.norm(v_qp) / np.linalg.norm(v_nom)) \
            if np.linalg.norm(v_nom) > 0 else 0.0
        ok, r, note = self.fallback.check_step(q, v_qp, dt)
        if ok:
            return FilterResult(v_qp, ratio, "modified",
                                f"QP 最小修改解（|v|/|v_nom|={ratio:.3f}）",
                                [Attempt(ratio, True, r.min_margin,
                                         r.worst_label, r.worst_boundary,
                                         "QP(SLSQP) 提案，非线性验证通过")])
        return self._fallback_result(q, v_nom, dt,
                                     f"QP 提案未通过精确验证（{note}）")

    def _fallback_result(self, q, v_nom, dt, why: str) -> FilterResult:
        res = self.fallback.filter(q, v_nom, dt)
        res.reason = f"{why}，回退速度缩放；{res.reason}"
        return res

    def filter_fail_closed(self, q, q_dot_nom, dt) -> FilterResult:
        try:
            res = self.filter(q, q_dot_nom, dt)
        except Exception as exc:  # noqa: BLE001
            return FilterResult(np.zeros(6), 0.0, "stopped",
                                f"QP 过滤失败，fail-closed 停止: {exc}")
        if res.q_dot.shape != (6,) or not np.all(np.isfinite(res.q_dot)):
            return FilterResult(np.zeros(6), 0.0, "stopped",
                                "过滤器返回非法结果，fail-closed 停止",
                                res.attempts)
        return res
