"""输入检查（题目第 3 节）：非法输入不得进入 Mock Robot。

检查项：
  - joint_state 维度必须为 6；
  - action_chunk.shape 必须为 (50, 6)；
  - 所有数值必须为有限值（NaN / Inf 拒绝）；
  - action_hz > 0；
  - joint_state → 第一个目标点、相邻轨迹点之间的突跳不得超过阈值；
  - 当前关节状态与全部关节目标必须在关节位置范围内。

设计决策：关节目标越界 **拒绝**（而非钳位）。题目第 3 节把
「关节目标不得超过配置的关节位置范围」列为输入检查项，并明确
「非法输入不得进入 Mock Robot」；钳位会让 VLA 的非法目标静默变成
另一个动作，拒绝则把问题显式暴露给上游策略层。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (DEFAULT_WORKSPACE, N_JOINTS, N_POINTS, RobotConfig)


class InputValidationError(ValueError):
    """输入非法：轨迹整体拒绝，不得进入 Mock Robot。"""


@dataclass
class Scenario:
    """通过检查的合法场景。"""

    name: str
    action_hz: float
    joint_state: np.ndarray
    action_chunk: np.ndarray
    workspace_dict: dict

    @property
    def dt(self) -> float:
        return 1.0 / self.action_hz


def _as_float_array(value, what: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{what}: 无法解析为数值数组（{exc}）") from exc
    return arr


def validate_input(data: dict, cfg: RobotConfig) -> Scenario:
    """校验场景数据，非法时抛 InputValidationError，合法时返回 Scenario。"""
    if not isinstance(data, dict):
        raise InputValidationError(f"场景必须是 dict，得到 {type(data).__name__}")

    for key in ("action_hz", "joint_state", "action_chunk"):
        if key not in data:
            raise InputValidationError(f"缺少必需字段: {key}")

    # action_hz > 0 且有限
    try:
        hz = float(data["action_hz"])
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"action_hz 非法: {data['action_hz']!r}") from exc
    if not np.isfinite(hz) or hz <= 0:
        raise InputValidationError(f"action_hz 必须为正有限值，得到 {hz}")

    # joint_state：维度 6、有限、在关节限位内
    q0 = _as_float_array(data["joint_state"], "joint_state")
    if q0.shape != (N_JOINTS,):
        raise InputValidationError(f"joint_state 维度必须为 (6,)，得到 {q0.shape}")
    if not np.all(np.isfinite(q0)):
        raise InputValidationError("joint_state 包含 NaN/Inf")
    if np.any(q0 < cfg.joint_lower) or np.any(q0 > cfg.joint_upper):
        bad = int(np.argmax(np.maximum(cfg.joint_lower - q0, q0 - cfg.joint_upper)))
        raise InputValidationError(
            f"joint_state 超出关节限位: J{bad + 1} = {q0[bad]:.4f} rad")

    # action_chunk：shape (50, 6)、有限
    chunk = _as_float_array(data["action_chunk"], "action_chunk")
    if chunk.shape != (N_POINTS, N_JOINTS):
        raise InputValidationError(
            f"action_chunk shape 必须为 ({N_POINTS}, {N_JOINTS})，得到 {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        bad = np.argwhere(~np.isfinite(chunk))[0]
        raise InputValidationError(
            f"action_chunk 包含 NaN/Inf（首个出现于第 {bad[0]} 点 J{bad[1] + 1}）")

    # 突跳：joint_state → 第一个目标点
    jump0 = np.abs(chunk[0] - q0)
    if jump0.max() > cfg.max_joint_step:
        j = int(np.argmax(jump0))
        raise InputValidationError(
            f"起始突跳超限: J{j + 1} 变化 {jump0[j]:.4f} rad > 阈值 {cfg.max_joint_step}")

    # 突跳：相邻轨迹点
    jumps = np.abs(np.diff(chunk, axis=0))
    if jumps.size and jumps.max() > cfg.max_joint_step:
        i, j = np.unravel_index(int(np.argmax(jumps)), jumps.shape)
        raise InputValidationError(
            f"相邻点突跳超限: 第 {i}→{i + 1} 点 J{j + 1} 变化 "
            f"{jumps[i, j]:.4f} rad > 阈值 {cfg.max_joint_step}")

    # 关节目标不得越界
    below = chunk < cfg.joint_lower[None, :]
    above = chunk > cfg.joint_upper[None, :]
    if np.any(below) or np.any(above):
        i, j = np.unravel_index(int(np.argmax(below | above)), chunk.shape)
        raise InputValidationError(
            f"关节目标越界: 第 {i} 点 J{j + 1} = {chunk[i, j]:.4f} rad，"
            f"限位 [{cfg.joint_lower[j]:.4f}, {cfg.joint_upper[j]:.4f}]")

    # workspace（可选，默认题目示例）
    ws = data.get("workspace", DEFAULT_WORKSPACE)
    try:
        ws = {axis: [float(ws[axis][0]), float(ws[axis][1])] for axis in ("x", "y", "z")}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise InputValidationError(f"workspace 结构非法（{exc}）") from exc
    for axis in ("x", "y", "z"):
        lo, hi = ws[axis]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            raise InputValidationError(f"workspace {axis} 区间非法: [{lo}, {hi}]")

    return Scenario(
        name=str(data.get("name", "scenario")),
        action_hz=hz,
        joint_state=q0,
        action_chunk=chunk,
        workspace_dict=ws,
    )
