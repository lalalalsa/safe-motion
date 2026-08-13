"""FK 正确性：独立手算参考值 + 不变量 + 面试方参考数据交叉验证。"""
import numpy as np
import pytest

from safe_motion.kinematics import LINK_LENGTHS, forward_kinematics


def test_fk_zero_pose_tcp():
    """q=0 的 TCP 由 DH 参数手工推出：(-(a2+a3), -(d4+d6), d1-d5)。"""
    fk = forward_kinematics(np.zeros(6))
    np.testing.assert_allclose(fk["tcp"], [-0.81725, -0.19145, -0.005491],
                               atol=1e-9)
    np.testing.assert_allclose(fk["base"], [0.0, 0.0, 0.0], atol=1e-12)


def test_fk_returns_all_nodes():
    fk = forward_kinematics(np.zeros(6))
    for key in ("base", "joint_1", "joint_2", "joint_3", "joint_4",
                "joint_5", "joint_6", "tcp"):
        assert key in fk and fk[key].shape == (3,)
    assert fk["points"].shape == (7, 3)


def test_joint1_origin_constant():
    """O_1 恒为 (0, 0, d1)，与 q 无关。"""
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-2 * np.pi, 2 * np.pi, 6)
        np.testing.assert_allclose(forward_kinematics(q)["joint_1"],
                                   [0.0, 0.0, 0.089159], atol=1e-12)


def test_link_lengths_invariant():
    """相邻节点距离恒等于 DH 决定的连杆长度（强不变量）。"""
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, 6)
        pts = forward_kinematics(q)["points"]
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        np.testing.assert_allclose(d, LINK_LENGTHS, atol=1e-10)


def test_fk_matches_reference_real_01(real_01):
    """与面试方独立参考实现对比：frame 49 的 TCP 坐标。"""
    ref = real_01["reference_analysis"]["worst"]
    q = np.array(real_01["action_chunk"])[ref["frame"]]
    fk = forward_kinematics(q)
    np.testing.assert_allclose(fk["tcp"], ref["tcp_position"], atol=1e-6)


def test_fk_matches_reference_real_02(real_02):
    ref = real_02["reference_analysis"]["worst"]
    q = np.array(real_02["action_chunk"])[ref["frame"]]
    fk = forward_kinematics(q)
    np.testing.assert_allclose(fk["tcp"], ref["tcp_position"], atol=1e-6)


def test_fk_matches_reference_real_03(real_03):
    """real_03 同时锁定 TCP 与中间关节点（point 3）坐标。"""
    ref = real_03["reference_analysis"]["worst"]
    q = np.array(real_03["action_chunk"])[ref["frame"]]
    fk = forward_kinematics(q)
    np.testing.assert_allclose(fk["tcp"], ref["tcp_position"], atol=1e-6)
    np.testing.assert_allclose(fk["points"][ref["point"]], ref["position"],
                               atol=1e-6)


def test_fk_rejects_bad_input():
    with pytest.raises(ValueError):
        forward_kinematics(np.zeros(5))
    with pytest.raises(ValueError):
        forward_kinematics([0.0] * 5 + [np.nan])
