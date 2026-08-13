"""端到端 Replay 测试：验证「最终执行轨迹是否安全」这一核心不变量。"""
import numpy as np
import pytest

from safe_motion import config
from safe_motion.geometry import full_body_check
from safe_motion.mock_robot import inside_joint_limits
from safe_motion.replay import run_scenario


def test_free_space_unmodified(free_space):
    """安全轨迹应基本不被修改。"""
    report = run_scenario(free_space)
    assert report["rejected"] is False
    assert report["executed_steps"] == 50
    assert report["modified_steps"] == 0
    assert report["stopped_steps"] == 0
    assert report["minimum_workspace_margin"] >= 0


def test_workspace_boundary_limited(real_01):
    """VLA 试图越界，但最终执行轨迹必须被限制在安全区内。"""
    report = run_scenario(real_01)
    assert report["rejected"] is False
    # 原始 VLA 轨迹确实存在越界（安全过滤被触发）
    assert report["modified_steps"] + report["stopped_steps"] > 0
    # 执行后全身安全
    assert report["minimum_workspace_margin"] >= 0
    for q in report["executed_trajectory"]:
        assert full_body_check(q, real_01["workspace"])["is_safe"]


def test_upper_z_boundary_limited(real_02):
    report = run_scenario(real_02)
    assert report["rejected"] is False
    assert report["minimum_workspace_margin"] >= 0
    for q in report["executed_trajectory"]:
        assert full_body_check(q, real_02["workspace"])["is_safe"]


def test_tcp_safe_mid_link_unsafe_handled(real_03):
    """TCP 安全但连杆越界时，执行轨迹仍必须保持全身安全。"""
    report = run_scenario(real_03)
    assert report["rejected"] is False
    assert report["minimum_workspace_margin"] >= 0
    # 中间连杆越界必须触发修改/停止，不能因为 TCP 安全而放行
    assert report["modified_steps"] + report["stopped_steps"] > 0
    for q in report["executed_trajectory"]:
        assert full_body_check(q, real_03["workspace"])["is_safe"]


def test_joint_limit_respected(joint_limit):
    """关节目标越界时，Mock Robot 不得进入非法关节状态。"""
    report = run_scenario(joint_limit)
    assert report["rejected"] is False
    # 输入检查应标记越界
    assert report["joint_limit_exceeded"] is True
    # 执行后所有关节都在限位内
    for q in report["executed_trajectory"]:
        assert inside_joint_limits(q, joint_limit["joint_limits"])


def test_invalid_input_rejected():
    """NaN / 错误 shape / 非 50 点必须被拒绝，不得进入 Mock Robot。"""
    base = {
        "name": "invalid",
        "action_hz": 20.0,
        "workspace": config.DEFAULT_WORKSPACE,
        "joint_limits": config.JOINT_LIMITS,
    }

    # NaN
    s = dict(base, joint_state=[0] * 6, action_chunk=np.zeros((50, 6)))
    s["action_chunk"][5, 0] = np.nan
    assert run_scenario(s)["rejected"] is True

    # Inf
    s = dict(base, joint_state=[0] * 6, action_chunk=np.zeros((50, 6)))
    s["action_chunk"][0, 1] = np.inf
    assert run_scenario(s)["rejected"] is True

    # 错误 shape
    s = dict(base, joint_state=[0] * 6, action_chunk=np.zeros((49, 6)))
    assert run_scenario(s)["rejected"] is True

    # 非 50 点
    s = dict(base, joint_state=[0] * 6, action_chunk=np.zeros((40, 6)))
    assert run_scenario(s)["rejected"] is True

    # 突跳
    s = dict(base, joint_state=[0] * 6, action_chunk=np.zeros((50, 6)))
    s["action_chunk"][0, 0] = 10.0  # 起始突跳 10 rad
    assert run_scenario(s)["rejected"] is True


def test_replay_output_fields():
    """Replay 报告应包含题目要求的全部统计字段。"""
    from safe_motion.replay import load_scenario
    import os
    sc = load_scenario(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scenarios", "free_space.json"))
    report = run_scenario(sc)
    for key in ("total_steps", "executed_steps", "modified_steps", "stopped_steps",
                "minimum_workspace_margin", "maximum_joint_velocity", "final_joint_state",
                "nominal_trajectory", "executed_trajectory"):
        assert key in report, f"缺少字段 {key}"
