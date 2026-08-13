"""全局配置：UR5 标称 DH 参数、机器人限制、默认工作空间与安全阈值。

所有阈值都是配置项（场景 JSON / 测试可覆盖），代码中不出现
针对特定场景坐标的魔数——隐藏场景必须能直接处理。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# UR5 标称 DH 参数（题目给定，标准 DH 约定，勿改动）
# ---------------------------------------------------------------------------
DH_A = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0])
DH_D = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

N_JOINTS = 6
N_POINTS = 50  # action_chunk 固定 50 点

# 默认矩形 Keep-in Workspace（题目示例，单位 m）
DEFAULT_WORKSPACE = {"x": [-0.70, 0.70], "y": [-0.70, 0.70], "z": [0.05, 0.90]}


@dataclass(frozen=True)
class RobotConfig:
    """机器人限制与输入检查阈值。

    joint_lower / joint_upper : 关节位置限位（rad），UR5 标称 ±2π
    max_joint_velocity        : 逐关节速度上限（rad/s），取 UR5 标称 180°/s
    max_joint_step            : 输入检查的相邻点突跳阈值（rad）
    """

    joint_lower: np.ndarray
    joint_upper: np.ndarray
    max_joint_velocity: np.ndarray
    max_joint_step: float = 0.5

    @classmethod
    def ur5(cls) -> "RobotConfig":
        return cls(
            joint_lower=-2.0 * np.pi * np.ones(N_JOINTS),
            joint_upper=+2.0 * np.pi * np.ones(N_JOINTS),
            max_joint_velocity=np.pi * np.ones(N_JOINTS),
        )

    @property
    def joint_limits(self) -> np.ndarray:
        """(6, 2) 形式的限位，供 Mock Robot 使用。"""
        return np.column_stack([self.joint_lower, self.joint_upper])
