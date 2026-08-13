"""输入检查 + 名义速度生成 + 全身安全过滤（速度缩放/回溯）。

职责划分：
  - 输入检查（check_input）：识别「数据非法」（NaN/Inf、shape 错误、
    维度错误、action_hz 非法、突跳超阈值），这些轨迹应被拒绝，
    不得进入 Mock Robot。
  - 关节范围越界：视为「可修正的不安全目标」，由安全过滤在名义速度
    生成前把目标钳位到限位边界（最小修改），而非直接拒绝。
  - 安全过滤（safety_filter）：速度缩放/回溯，找到下一控制周期仍全身
    安全的最大速度；若无法得到可靠安全动作，则输出零速度（fail-safe）。
"""
import numpy as np

from . import config
from .config import as_joint_limits
from .geometry import full_body_check
from .mock_robot import inside_joint_limits


class InputValidationError(ValueError):
    """数据非法，轨迹应被整体拒绝，不得进入 Mock Robot。"""


def check_input(joint_state, action_chunk, action_hz=None, joint_limits=None,
                max_joint_step=None):
    """输入合法性检查。

    返回 (fatal, joint_limit_exceeded)：
      - fatal : list[str]，致命违规（非法输入），非空时应拒绝执行；
      - joint_limit_exceeded : bool，是否存在关节目标越界（可钳位修正）。
    """
    joint_limits = config.JOINT_LIMITS if joint_limits is None else joint_limits
    max_joint_step = config.MAX_JOINT_STEP if max_joint_step is None else max_joint_step
    action_hz = config.ACTION_HZ if action_hz is None else action_hz

    fatal = []
    js = np.asarray(joint_state, dtype=float)
    chunk = np.asarray(action_chunk, dtype=float)

    # 1. joint_state 维度
    if js.shape != (6,):
        fatal.append(f"joint_state 维度 {js.shape} != (6,)")

    # 2. action_chunk shape
    if chunk.shape != (50, 6):
        fatal.append(f"action_chunk shape {chunk.shape} != (50, 6)")

    # 3. 数值有限
    if not np.all(np.isfinite(js)):
        fatal.append("joint_state 包含 NaN/Inf")
    if not np.all(np.isfinite(chunk)):
        fatal.append("action_chunk 包含 NaN/Inf")

    # 4. action_hz
    if not (action_hz > 0 and np.isfinite(action_hz)):
        fatal.append(f"action_hz 非法: {action_hz}")

    joint_limit_exceeded = False

    # 仅在形状合法时继续做数值层面的检查
    if not fatal and js.shape == (6,) and chunk.shape == (50, 6):
        # 5. 突跳检查：起始点 + 相邻点
        combined = np.vstack([js[None, :], chunk])  # (51, 6)
        max_step = float(np.max(np.abs(np.diff(combined, axis=0))))
        if max_step > max_joint_step:
            fatal.append(f"关节突跳 {max_step:.4f} rad 超过阈值 {max_joint_step} rad")

        # 6. 关节目标范围（非致命：可钳位修正）
        limits = as_joint_limits(joint_limits)
        if np.any(chunk < limits[:, 0]) or np.any(chunk > limits[:, 1]):
            joint_limit_exceeded = True

    return fatal, joint_limit_exceeded


def clamp_target(q_target, joint_limits):
    """把关节目标钳位到限位内（对越界目标做最小修改）。"""
    limits = as_joint_limits(joint_limits)
    return np.clip(np.asarray(q_target, dtype=float), limits[:, 0], limits[:, 1])


def nominal_velocity(q, q_target, dt=None, max_joint_velocity=None):
    """名义关节速度：q_dot_nom = clamp((q_target - q)/dt, ±max_vel)。"""
    dt = config.DT if dt is None else dt
    max_vel = config.MAX_JOINT_VELOCITY if max_joint_velocity is None else max_joint_velocity
    q = np.asarray(q, dtype=float)
    q_target = np.asarray(q_target, dtype=float)
    q_dot = (q_target - q) / dt
    return np.clip(q_dot, -max_vel, max_vel)


def safety_filter(q, q_dot_nom, workspace, dt=None, joint_limits=None,
                  scales=None, spacing=None):
    """全身安全过滤（速度缩放 / 回溯）。

    依次尝试 VELOCITY_SCALES 中的缩放系数，返回能使下一状态
    q_next = q + scale * q_dot_nom * dt 全身安全的「最大」缩放后的速度。

    返回 (q_dot_safe, info)，info 含：
      scale     : float，最终采用的缩放系数
      modified  : bool，是否修改了名义动作（scale < 1.0）
      stopped   : bool，是否被迫停止（scale == 0.0，即无法得到安全动作）
      min_margin: float，预测下一状态的全身最小 margin
    """
    dt = config.DT if dt is None else dt
    joint_limits = config.JOINT_LIMITS if joint_limits is None else joint_limits
    scales = config.VELOCITY_SCALES if scales is None else scales

    q = np.asarray(q, dtype=float)
    q_dot_nom = np.asarray(q_dot_nom, dtype=float)

    # 名义速度本身必须有限（输入检查已保证，这里兜底）
    if not np.all(np.isfinite(q_dot_nom)):
        return np.zeros(6), _info(0.0, True, True, -np.inf)

    for scale in scales:
        q_next = q + scale * q_dot_nom * dt

        # 关节限位
        if not inside_joint_limits(q_next, joint_limits):
            continue

        # 全身 workspace 安全
        res = full_body_check(q_next, workspace, spacing)
        if res["is_safe"]:
            return scale * q_dot_nom, _info(
                scale, modified=scale < 1.0, stopped=(scale == 0.0),
                min_margin=res["min_margin"])

    # 所有缩放（含 0）都不安全：当前位形已越界或算法异常 → 输出零速度
    return np.zeros(6), _info(0.0, True, True, -np.inf)


def _info(scale, modified, stopped, min_margin):
    return {
        "scale": float(scale),
        "modified": bool(modified),
        "stopped": bool(stopped),
        "min_margin": float(min_margin),
    }
