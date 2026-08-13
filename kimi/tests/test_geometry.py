"""全身几何检查：margin 语义、采样、凸性、参考场景的最差点复核。"""
import numpy as np
import pytest

from safe_motion.geometry import (Workspace, check_state, sample_segment)

WS = Workspace.from_dict({"x": [-0.7, 0.7], "y": [-0.7, 0.7], "z": [0.05, 0.9]})


def test_margin_inside_outside():
    assert WS.margins([[0.0, 0.0, 0.5]])[0] == pytest.approx(0.4)
    assert WS.margins([[0.8, 0.0, 0.5]])[0] == pytest.approx(-0.1)
    assert WS.boundary_name([0.8, 0.0, 0.5]) == "x_max"
    assert WS.boundary_name([0.0, 0.0, 0.03]) == "z_min"


def test_sample_segment_spacing():
    seg = sample_segment([0, 0, 0], [0.425, 0, 0], spacing=0.02)
    gaps = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    assert gaps.max() <= 0.02 + 1e-12
    np.testing.assert_allclose(seg[0], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(seg[-1], [0.425, 0, 0], atol=1e-12)


def test_convexity_sampling_consistent():
    """凸盒内任意两点连线的采样点必然全部在盒内（采样不会假阳性）。"""
    rng = np.random.default_rng(2)
    lo = np.array([-0.7, -0.7, 0.05])
    hi = np.array([0.7, 0.7, 0.9])
    for _ in range(200):
        p0 = rng.uniform(lo, hi)
        p1 = rng.uniform(lo, hi)
        seg = sample_segment(p0, p1, 0.02)
        assert np.all(WS.margins(seg) >= -1e-12)


def test_check_state_real_03_mid_body_violation(real_03):
    """real_03 frame 49：TCP 在界内，joint_3 越 x_min —— 必须被识别。"""
    ref = real_03["reference_analysis"]["worst"]
    q = np.array(real_03["action_chunk"])[ref["frame"]]
    r = check_state(q, WS)
    assert not r.safe
    assert r.min_margin == pytest.approx(ref["margin"], abs=1e-6)
    assert r.worst_boundary == "x_min"
    assert r.worst_label == "joint_3"  # 最差的是中间关节，不是 TCP
    # 同时 TCP margin 应为正（TCP 安全不能掩盖连杆越界）
    tcp_margin = WS.margins([ref["tcp_position"]])[0]
    assert tcp_margin > 0


def test_check_state_real_01_tcp_violation(real_01):
    ref = real_01["reference_analysis"]["worst"]
    q = np.array(real_01["action_chunk"])[ref["frame"]]
    r = check_state(q, WS)
    assert not r.safe
    assert r.min_margin == pytest.approx(ref["margin"], abs=1e-6)
    assert r.worst_boundary == "z_min"
    assert r.worst_label == "tcp (joint_6)"


def test_base_excluded_from_check(real_01):
    """基座 O_0 在 z=0（z_min 之下）是固定结构，不参与检查：
    安全初始位形的 min margin 恰为 O_1 的 0.039159。"""
    r = check_state(np.array(real_01["joint_state"]), WS)
    assert r.safe
    assert r.min_margin == pytest.approx(0.039159, abs=1e-6)
    assert r.worst_label == "joint_1"
