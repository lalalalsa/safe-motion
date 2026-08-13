"""安全过滤（速度缩放/回溯）测试。"""
import numpy as np
import pytest

from safe_motion import config
from safe_motion.safety_filter import (clamp_target, nominal_velocity,
                                       safety_filter)

WS = config.DEFAULT_WORKSPACE
JL = config.JOINT_LIMITS

# 一个全身安全位形（real_01 初始）
SAFE_Q = np.array([1.7011667490005493, -1.718811337147848, -2.1258776823626917,
                   -0.9176800886737269, 1.5363986492156982, 1.810360312461853])


def test_nominal_velocity_clamped():
    q = np.zeros(6)
    target = np.array([1.0, 0, 0, 0, 0, 0])
    q_dot = nominal_velocity(q, target, dt=0.05, max_joint_velocity=3.14)
    # 理想速度 20 rad/s，被限制到 3.14
    assert q_dot[0] == np.float64(3.14)


def test_nominal_velocity_normal():
    q = np.zeros(6)
    target = np.array([0.01, 0, 0, 0, 0, 0])
    q_dot = nominal_velocity(q, target, dt=0.05, max_joint_velocity=3.14)
    assert q_dot[0] == pytest.approx(0.2)


def test_safe_action_kept():
    """安全位形附近的小名义速度不应被修改（scale=1）。"""
    q_dot_nom = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    q_dot_safe, info = safety_filter(SAFE_Q, q_dot_nom, WS, joint_limits=JL)
    assert info["scale"] == 1.0
    assert info["modified"] is False
    np.testing.assert_allclose(q_dot_safe, q_dot_nom, atol=1e-12)


def test_boundary_action_modified():
    """朝 workspace 边界外的大速度应被缩放或停止。"""
    # 大速度推动 TCP 大幅外移，必然越界
    q_dot_nom = np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
    q_dot_safe, info = safety_filter(SAFE_Q, q_dot_nom, WS, joint_limits=JL)
    assert info["scale"] < 1.0
    assert info["modified"] is True


def test_fail_safe_zero_velocity():
    """若所有缩放都不安全，必须输出零速度。"""
    # 构造一个当前位形已越界的情况：直接给一个越界位形
    bad_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # q=0 时 TCP 越界
    q_dot_nom = np.array([0.1, 0, 0, 0, 0, 0])
    q_dot_safe, info = safety_filter(bad_q, q_dot_nom, WS, joint_limits=JL)
    # 此时即使 scale=0，q_next=bad_q 也不安全 → 输出零速度并标记停止
    np.testing.assert_allclose(q_dot_safe, np.zeros(6), atol=1e-12)
    assert info["stopped"] is True


def test_output_always_finite():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(-2, 2, 6)
        q_dot_nom = rng.uniform(-4, 4, 6)
        q_dot_safe, _ = safety_filter(q, q_dot_nom, WS, joint_limits=JL)
        assert np.all(np.isfinite(q_dot_safe))


def test_clamp_target():
    limits = np.array([[-1.0, 1.0]] * 6)
    t = np.array([2.0, -3.0, 0.5, 0, 0, 0])
    c = clamp_target(t, limits)
    np.testing.assert_allclose(c, [1.0, -1.0, 0.5, 0, 0, 0])
