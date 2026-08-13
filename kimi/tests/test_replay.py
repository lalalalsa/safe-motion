"""闭环 Replay 集成测试：5 个必备场景 + fail-safe + 随机轨迹不变量。

测试验证的是「最终执行轨迹是否安全」，而不是「函数能跑」。
参考 margin 数值来自面试方场景文件中的 reference_analysis（独立实现），
用于交叉验证，不参与任何安全逻辑的拟合。
"""
import json

import numpy as np
import pytest

from safe_motion.replay import run_scenario
from safe_motion.safety_filter import FilterResult, SafetyFilter
from safe_motion.validation import InputValidationError

from conftest import SCENARIOS, load

TOL = 1e-6  # 执行轨迹 margin 容忍（浮点量级）


def run(name, **kw):
    kw.setdefault("make_plot", False)
    return run_scenario(SCENARIOS / name, **kw)


# ---------------------------------------------------------------- 5 个必备场景

def test_free_space_unmodified():
    """安全轨迹：完整执行 50 步，过滤器不做任何修改，执行 == 名义。"""
    r = run("free_space.json")
    s = r["summary"]
    assert s["executed_steps"] == 50
    assert s["modified_steps"] == 0
    assert s["stopped_steps"] == 0
    assert s["minimum_workspace_margin"] > 0.03
    assert s["max_tcp_deviation_m"] < 1e-9
    chunk = np.array(load("free_space.json")["action_chunk"])
    np.testing.assert_allclose(r["executed"][1:], chunk, atol=1e-12)


def test_workspace_boundary_contained():
    """real_01（下 z 边界）：VLA 越界，执行轨迹全程在界内。"""
    r = run("workspace_boundary.json")
    s = r["summary"]
    assert s["executed_steps"] == 50
    assert s["modified_steps"] + s["stopped_steps"] >= 1
    # 未过滤轨迹的 min margin 与面试方参考值一致（-0.04120，确实越界）
    assert s["nominal_minimum_workspace_margin"] == pytest.approx(
        -0.04119871814725225, abs=1e-6)
    # 实际执行轨迹全程安全
    assert np.all(r["executed_margins"] >= -TOL)
    assert s["maximum_joint_velocity"] <= np.pi + 1e-9
    assert s["max_tcp_deviation_m"] > 0.005  # 确实做了实质性修改


def test_real_02_upper_z_contained():
    r = run("real_02_upper_z_boundary.json")
    s = r["summary"]
    assert s["nominal_minimum_workspace_margin"] == pytest.approx(
        -0.02479730792141177, abs=1e-6)
    assert np.all(r["executed_margins"] >= -TOL)
    assert s["modified_steps"] + s["stopped_steps"] >= 1


def test_tcp_safe_mid_link_unsafe_detected():
    """real_03：TCP 在界内但中间关节越界。不仅最终轨迹要安全，
    还必须证明「检测能力来自中间关节/连杆而非 TCP」。"""
    r = run("tcp_safe_mid_link_unsafe.json")
    s = r["summary"]
    assert s["nominal_minimum_workspace_margin"] == pytest.approx(
        -0.06098336283158656, abs=1e-6)
    assert np.all(r["executed_margins"] >= -TOL)
    assert s["modified_steps"] + s["stopped_steps"] >= 1
    # 至少一次被拒绝的候选，其最差点不是 TCP —— 全身检查发挥作用的直接证据
    assert any(
        (not a["safe"]) and a["worst_label"] != "tcp (joint_6)"
        for rec in r["records"] for a in rec["attempts"]
    ), "没有任何一次干预是由中间关节/连杆触发的"


def test_joint_limit_rejected_before_robot():
    """joint_limit 场景：输入检查拒绝，轨迹不得进入 Mock Robot。"""
    with pytest.raises(InputValidationError, match="关节目标越界"):
        run("joint_limit.json")


def test_invalid_input_nan_rejected():
    with pytest.raises(InputValidationError, match="NaN"):
        run("invalid_input_nan.json")


# ------------------------------------------------------------------ fail-safe

def test_fail_safe_when_filter_crashes(monkeypatch):
    """人为制造过滤器每步崩溃：系统受控停止（零速度），机器人原地不动，
    而不是崩溃或继续执行原动作。"""
    def crash(self, q, v, dt):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(SafetyFilter, "filter", crash)
    r = run("free_space.json")
    s = r["summary"]
    assert s["executed_steps"] == 50
    assert s["stopped_steps"] == 50
    q0 = np.array(load("free_space.json")["joint_state"])
    np.testing.assert_allclose(r["executed"][-1], q0, atol=1e-12)
    assert s["minimum_workspace_margin"] > 0.03  # 原地保持，始终安全


def test_fail_safe_when_filter_lies(monkeypatch):
    """过滤器被污染、对越界动作也返回「安全」并放行原速度：
    step 前的信任边界独立复核必须否决，执行轨迹仍然全程安全。"""
    def lie(self, q, v, dt):
        return FilterResult(np.asarray(v, dtype=float), 1.0,
                            "unmodified", "过滤器说谎：无条件放行")

    monkeypatch.setattr(SafetyFilter, "filter", lie)
    r = run("workspace_boundary.json")
    s = r["summary"]
    assert np.all(r["executed_margins"] >= -TOL)
    assert s["stopped_steps"] >= 1  # 复核确实拦截过


# ------------------------------------------------------------------ 输出契约

def test_replay_output_fields():
    s = run("free_space.json")["summary"]
    for key in ("total_steps", "executed_steps", "modified_steps",
                "stopped_steps", "minimum_workspace_margin",
                "maximum_joint_velocity", "final_joint_state"):
        assert key in s
    assert s["total_steps"] == 50
    assert len(s["final_joint_state"]) == 6


# ---------------------------------------------------------- 随机轨迹不变量

def _random_scenario(rng, q0, step_std):
    increments = np.clip(rng.normal(0.0, step_std, (50, 6)), -0.4, 0.4)
    chunk = np.array(q0)[None, :] + np.cumsum(increments, axis=0)
    chunk = np.clip(chunk, -6.0, 6.0)  # 保持在关节限位内（合法输入）
    diffs = np.diff(np.vstack([np.asarray(q0)[None, :], chunk]), axis=0)
    assert np.abs(diffs).max() < 0.5  # 保持在突跳阈值内（合法输入）
    return {
        "name": "random",
        "action_hz": 20.0,
        "joint_state": list(q0),
        "action_chunk": chunk.tolist(),
        "workspace": {"x": [-0.7, 0.7], "y": [-0.7, 0.7], "z": [0.05, 0.9]},
    }


@pytest.mark.parametrize("seed", range(12))
def test_random_trajectories_stay_safe(seed, real_01):
    """任意合法输入：执行轨迹全程全身安全、速度有限且不超速。

    step_std 混合温和（0.03）与激进（0.15）轨迹；激进轨迹会大幅越界，
    过滤器必须修改/停止，且执行结果始终安全。
    """
    rng = np.random.default_rng(seed)
    q0 = np.array(real_01["joint_state"])
    for step_std in (0.03, 0.15):
        data = _random_scenario(rng, q0, step_std)
        r = run_scenario(data, make_plot=False)
        s = r["summary"]
        assert s["executed_steps"] == 50
        assert np.all(r["executed_margins"] >= -TOL)
        assert s["maximum_joint_velocity"] <= np.pi + 1e-9
        for rec in r["records"]:
            assert np.all(np.isfinite(rec["q_dot_safe"]))
