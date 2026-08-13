"""UR5 标称正运动学测试。"""
import numpy as np
import pytest

from safe_motion.kinematics import forward_kinematics, link_segments
from safe_motion import config

# q = 0 位形的参考坐标（独立手算得到，用于锁定 FK 正确性）
ZERO_POSE = {
    "base": [0.0, 0.0, 0.0],
    "j1": [0.0, 0.0, 0.089159],
    "j2": [-0.425, 0.0, 0.089159],
    "j3": [-0.81725, 0.0, 0.089159],
    "j4": [-0.81725, -0.10915, 0.089159],
    "j5": [-0.81725, -0.10915, -0.005491],
    "j6": [-0.81725, -0.19145, -0.005491],
    "tcp": [-0.81725, -0.19145, -0.005491],
}

# 连杆长度不变量（UR5 标称几何）
LINK_LENGTHS = [0.089159, 0.425, 0.39225, 0.10915, 0.09465, 0.0823, 0.0]


def test_fk_output_shape():
    pts = forward_kinematics(np.zeros(6))
    assert pts.shape == (8, 3)


def test_fk_zero_pose_reference():
    pts = forward_kinematics(np.zeros(6))
    for i, name in enumerate(["base", "j1", "j2", "j3", "j4", "j5", "j6", "tcp"]):
        np.testing.assert_allclose(pts[i], ZERO_POSE[name], atol=1e-6)


def test_link_length_invariant():
    """任意位形下相邻关节点距离等于标称连杆长度。"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, 6)
        pts = forward_kinematics(q)
        for i in range(7):
            d = np.linalg.norm(pts[i + 1] - pts[i])
            assert d == pytest.approx(LINK_LENGTHS[i], abs=1e-6)


def test_tcp_equals_joint6():
    """本题无工具 TCP 偏置，故 tcp == joint_6。"""
    q = np.array([0.3, -0.7, 1.2, -0.4, 0.8, -0.2])
    pts = forward_kinematics(q)
    np.testing.assert_allclose(pts[6], pts[7], atol=1e-12)


def test_joint1_affects_rotation_only():
    """改变 J1 只绕 base Z 轴旋转整体，TCP 到 Z 轴距离不变。"""
    q0 = np.array([0.0, -0.7, 1.2, -0.4, 0.8, -0.2])
    q1 = q0.copy()
    q1[0] = 1.3
    tcp0 = forward_kinematics(q0)[-1]
    tcp1 = forward_kinematics(q1)[-1]
    r0 = np.linalg.norm(tcp0[:2])  # 到 Z 轴距离
    r1 = np.linalg.norm(tcp1[:2])
    assert r1 == pytest.approx(r0, abs=1e-9)


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        forward_kinematics(np.zeros(5))
    with pytest.raises(ValueError):
        forward_kinematics(np.array([np.nan, 0, 0, 0, 0, 0]))


def test_link_segments_count():
    segs = link_segments(np.zeros(6))
    assert len(segs) == 7
