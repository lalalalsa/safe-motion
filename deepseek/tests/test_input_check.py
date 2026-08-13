"""输入合法性检查测试。"""
import numpy as np

from safe_motion.safety_filter import check_input

VALID_JS = np.zeros(6)
VALID_CHUNK = np.zeros((50, 6))


def _chunk(n=50, dim=6, value=0.0):
    return np.full((n, dim), value, dtype=float)


def test_valid_input():
    fatal, jl = check_input(VALID_JS, VALID_CHUNK)
    assert fatal == []
    assert jl is False


def test_nan_input_fatal():
    chunk = _chunk()
    chunk[10, 2] = np.nan
    fatal, _ = check_input(VALID_JS, chunk)
    assert any("NaN" in f for f in fatal)


def test_inf_input_fatal():
    chunk = _chunk()
    chunk[0, 0] = np.inf
    fatal, _ = check_input(VALID_JS, chunk)
    assert any("NaN/Inf" in f for f in fatal)


def test_wrong_chunk_shape_fatal():
    fatal, _ = check_input(VALID_JS, _chunk(49, 6))
    assert any("(50, 6)" in f for f in fatal)

    fatal, _ = check_input(VALID_JS, _chunk(50, 5))
    assert any("(50, 6)" in f for f in fatal)


def test_wrong_joint_state_dim_fatal():
    fatal, _ = check_input(np.zeros(5), VALID_CHUNK)
    assert any("joint_state" in f for f in fatal)


def test_bad_hz_fatal():
    fatal, _ = check_input(VALID_JS, VALID_CHUNK, action_hz=0.0)
    assert any("action_hz" in f for f in fatal)
    fatal, _ = check_input(VALID_JS, VALID_CHUNK, action_hz=-5.0)
    assert any("action_hz" in f for f in fatal)


def test_jump_exceeds_threshold_fatal():
    # 相邻点突跳 1.0 rad > 阈值 0.5 rad
    chunk = _chunk()
    chunk[1, 0] = 1.0
    fatal, _ = check_input(VALID_JS, chunk, max_joint_step=0.5)
    assert any("突跳" in f for f in fatal)


def test_start_jump_exceeds_threshold_fatal():
    # 起始点到第一目标点突跳超阈值
    js = np.zeros(6)
    chunk = _chunk()
    chunk[0, 3] = 2.0  # 起始突跳 2.0 rad
    fatal, _ = check_input(js, chunk, max_joint_step=0.5)
    assert any("突跳" in f for f in fatal)


def test_joint_limit_exceeded_non_fatal():
    # 目标「平滑」越过关节限位（不触发突跳检查），应标记为可钳位而非致命拒绝
    js = np.array([6.2, 0, 0, 0, 0, 0])  # 接近 J1 上限 2π ≈ 6.283
    chunk = np.full((50, 6), 0.0)
    chunk[:, 0] = 6.2
    chunk[0, 0] = 6.4  # 越过 2π，但起始/相邻突跳仅 0.2 < 0.5
    fatal, jl = check_input(js, chunk)
    assert fatal == []
    assert jl is True
