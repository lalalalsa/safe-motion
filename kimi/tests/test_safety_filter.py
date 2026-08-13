"""安全过滤：限速、回溯、二分细化、关节限位预检、fail-closed。"""
import numpy as np
import pytest

from safe_motion.config import RobotConfig
from safe_motion.geometry import Workspace
from safe_motion.safety_filter import (FilterResult, SafetyFilter,
                                       SafetyFilterError)

CFG = RobotConfig.ur5()
WS = Workspace.from_dict({"x": [-0.7, 0.7], "y": [-0.7, 0.7], "z": [0.05, 0.9]})
DT = 0.05


def make_filter(**kw):
    return SafetyFilter(CFG, WS, **kw)


def test_nominal_velocity_clamped():
    sf = make_filter()
    v = sf.nominal_velocity(np.zeros(6), np.ones(6), DT)  # 原始 20 rad/s
    np.testing.assert_allclose(v, np.full(6, np.pi), atol=1e-12)


def test_nominal_velocity_normal():
    sf = make_filter()
    q = np.zeros(6)
    target = np.full(6, 0.05)
    np.testing.assert_allclose(sf.nominal_velocity(q, target, DT),
                               np.ones(6), atol=1e-12)


def test_safe_action_unmodified(real_01):
    sf = make_filter()
    q = np.array(real_01["joint_state"])
    target = np.array(real_01["action_chunk"])[0]
    v_nom = sf.nominal_velocity(q, target, DT)
    res = sf.filter(q, v_nom, DT)
    assert res.status == "unmodified" and res.scale == 1.0
    np.testing.assert_allclose(res.q_dot, v_nom, atol=1e-12)


def test_boundary_action_scaled(real_01):
    """frame 25（margin +9 mm，安全）直指 frame-49 深度越界目标：
    必须缩放，且缩放后的单步必须真的安全。"""
    sf = make_filter()
    chunk = np.array(real_01["action_chunk"])
    v_nom = sf.nominal_velocity(chunk[25], chunk[49], DT)
    res = sf.filter(chunk[25], v_nom, DT)
    assert res.status == "modified"
    assert 0.0 < res.scale < 1.0
    ok, _, _ = sf.check_step(chunk[25], res.q_dot, DT)
    assert ok  # 过滤结果本身必须真的安全


def test_bisection_refines_scale(real_01):
    """同一干预点：粗网格给 0.25，二分细化应显著提高 scale（≈0.36）且仍安全。"""
    chunk = np.array(real_01["action_chunk"])
    v_nom = make_filter().nominal_velocity(chunk[25], chunk[49], DT)
    coarse = make_filter(bisect_iters=0).filter(chunk[25], v_nom, DT)
    fine = make_filter(bisect_iters=10).filter(chunk[25], v_nom, DT)
    assert coarse.scale == 0.25
    assert 0.30 < fine.scale < 0.50
    assert fine.scale > coarse.scale
    ok, _, _ = make_filter().check_step(chunk[25], fine.q_dot, DT)
    assert ok


def test_joint_limit_preempted(real_01):
    """J6 不动机械臂几何（只转法兰），用它构造确定的限位边界：
    q6=6.20 rad，v6=3 rad/s → 单步 0.15 rad，最大可行 scale ≈ 0.5546。"""
    sf = make_filter()
    q = np.array(real_01["joint_state"])
    q[5] = 6.20
    v = np.array([0, 0, 0, 0, 0, 3.0])
    res = sf.filter(q, v, DT)
    assert res.status == "modified"
    assert 0.54 < res.scale < 0.56
    assert q[5] + res.q_dot[5] * DT <= 2 * np.pi + 1e-12


def test_only_zero_velocity_safe_stops(real_01):
    """frame 27（margin +2.4 mm，贴边但仍安全）继续压向深度越界目标：
    任何非零缩放都会越界，仅零速度安全 → stopped。"""
    sf = make_filter()
    chunk = np.array(real_01["action_chunk"])
    v_nom = sf.nominal_velocity(chunk[27], chunk[49], DT)
    res = sf.filter(chunk[27], v_nom, DT)
    assert res.status == "stopped"
    np.testing.assert_allclose(res.q_dot, np.zeros(6), atol=1e-15)


def test_filter_raises_when_current_state_unsafe(real_01):
    """当前状态已越界（连 scale=0 都不安全）→ filter 抛错，
    fail_closed 兜成零速度停止。"""
    sf = make_filter()
    q_bad = np.array(real_01["action_chunk"])[49]  # TCP 已越界
    with pytest.raises(SafetyFilterError):
        sf.filter(q_bad, np.zeros(6), DT)
    res = sf.filter_fail_closed(q_bad, np.zeros(6), DT)
    assert res.status == "stopped"
    np.testing.assert_allclose(res.q_dot, np.zeros(6), atol=1e-15)


def test_fail_closed_on_internal_exception(real_01, monkeypatch):
    """人为制造过滤器内部异常（模拟面试方 fail-safe 验收）。"""
    import safe_motion.safety_filter as sfmod

    def boom(*a, **k):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(sfmod, "check_motion", boom)
    sf = make_filter()
    q = np.array(real_01["joint_state"])
    res = sf.filter_fail_closed(q, np.ones(6), DT)
    assert res.status == "stopped"
    np.testing.assert_allclose(res.q_dot, np.zeros(6), atol=1e-15)
    assert "fail-closed" in res.reason


def test_fail_closed_on_garbage_output(real_01, monkeypatch):
    """过滤器返回 NaN（被污染的实现）→ fail-closed 停止。"""
    sf = make_filter()
    monkeypatch.setattr(
        SafetyFilter, "filter",
        lambda self, q, v, dt: FilterResult(np.full(6, np.nan), 1.0,
                                            "unmodified", "lies"))
    q = np.array(real_01["joint_state"])
    res = sf.filter_fail_closed(q, np.ones(6), DT)
    assert res.status == "stopped"
    np.testing.assert_allclose(res.q_dot, np.zeros(6), atol=1e-15)


def test_output_always_finite(real_01):
    """对真实场景全部 50 步：输出速度恒有限、恒不超速。"""
    sf = make_filter()
    q = np.array(real_01["joint_state"])
    for target in np.array(real_01["action_chunk"]):
        v_nom = sf.nominal_velocity(q, target, DT)
        res = sf.filter_fail_closed(q, v_nom, DT)
        assert np.all(np.isfinite(res.q_dot))
        assert np.max(np.abs(res.q_dot)) <= np.pi + 1e-12
        q = q + res.q_dot * DT  # 开环推进仅用于本性质测试
