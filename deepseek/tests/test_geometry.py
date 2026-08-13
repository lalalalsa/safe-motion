"""全身几何检查测试。"""
import numpy as np
import pytest

from safe_motion import config
from safe_motion.geometry import full_body_check, point_margin, sample_link
from safe_motion.kinematics import forward_kinematics

WS = config.DEFAULT_WORKSPACE


def test_point_margin_inside_positive():
    p = np.array([0.0, 0.0, 0.5])
    assert point_margin(p, WS) > 0


def test_point_margin_outside_negative():
    p = np.array([0.0, 0.0, 0.01])  # z 低于 0.05
    assert point_margin(p, WS) < 0


def test_point_margin_exact():
    p = np.array([0.0, 0.0, 0.1])
    assert point_margin(p, WS) == pytest.approx(0.05)  # 距 z_min=0.05


def test_sample_link_spacing():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    pts = sample_link(p0, p1, 0.1)
    assert pts[0].tolist() == p0.tolist()
    assert pts[-1].tolist() == p1.tolist()
    # 相邻采样点间距不超过 spacing
    assert np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)) <= 0.1 + 1e-9


def test_safe_pose_known_margin():
    """real_01 初始位形应全身安全，且最小 margin ≈ 0.039159（对齐参考值）。"""
    q = np.array([1.7011667490005493, -1.718811337147848, -2.1258776823626917,
                  -0.9176800886737269, 1.5363986492156982, 1.810360312461853])
    res = full_body_check(q, WS)
    assert res["is_safe"] is True
    assert res["min_margin"] == pytest.approx(0.039159, abs=1e-3)


def test_base_not_checked():
    """底座 base（z=0）不应参与检查，否则任何位形都会因 z<0.05 而恒不安全。"""
    q = np.zeros(6)
    res = full_body_check(q, WS)
    # q=0 时 TCP 在 z=-0.005，越界，所以应为不安全（但最坏点不是 base）
    assert res["worst_link"] >= 0


def test_mid_link_unsafe_detected(real_03):
    """题目最重要的测试：TCP 安全但中间连杆越界时必须被识别为不安全。

    real_03 最后一帧：TCP margin > 0，但 joint_3 越过 x_min。
    """
    q = real_03["action_chunk"][-1]
    fk = forward_kinematics(q)
    tcp_margin = point_margin(fk[-1], real_03["workspace"])
    full = full_body_check(q, real_03["workspace"])

    assert tcp_margin > 0            # TCP 仍在安全区内
    assert full["min_margin"] < 0    # 但全身（连杆）已越界
    assert full["is_safe"] is False  # 必须识别为不安全
