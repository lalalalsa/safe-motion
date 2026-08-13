"""闭环 Replay：把 50 点 VLA 轨迹逐步执行到 Mock Robot 上。

流程（题目第 9 节）：
    q = robot.get_joint_state()
    for target in action_chunk:        # 恰好 50 个点
        q_dot_nom  = nominal_control(q, target)
        q_dot_safe = safety_filter(q, q_dot_nom, workspace)
        q          = robot.step(q_dot_safe, dt=0.05)
        check_full_body_safety(q)

后续控制点基于「实际 Mock Robot 当前状态」继续执行，而不是假设
前一目标已被完美到达。
"""
import json
import os

import numpy as np

from . import config
from .geometry import full_body_check
from .mock_robot import MockRobot
from .safety_filter import (InputValidationError, check_input, clamp_target,
                            nominal_velocity, safety_filter)


def load_scenario(path):
    """从 JSON 读取场景，填充默认值。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return normalize_scenario(data)


def normalize_scenario(data):
    """规范化场景字段，缺省项使用 config 默认值。"""
    return {
        "name": data.get("name", data.get("description", "scenario")),
        "joint_state": np.asarray(data["joint_state"], dtype=float),
        "action_chunk": np.asarray(data["action_chunk"], dtype=float),
        "action_hz": float(data.get("action_hz", config.ACTION_HZ)),
        "workspace": config.as_workspace(data.get("workspace", config.DEFAULT_WORKSPACE)),
        "joint_limits": config.as_joint_limits(data.get("joint_limits", config.JOINT_LIMITS)),
        "max_joint_velocity": float(data.get("max_joint_velocity", config.MAX_JOINT_VELOCITY)),
        "max_joint_step": float(data.get("max_joint_step", config.MAX_JOINT_STEP)),
        "link_sample_spacing": float(data.get("link_sample_spacing", config.LINK_SAMPLE_SPACING)),
    }


def run_scenario(scenario):
    """执行一条轨迹的闭环 Replay，返回报告 dict。"""
    sc = normalize_scenario(scenario) if isinstance(scenario, dict) else scenario
    joint_state = np.asarray(sc["joint_state"], dtype=float)
    action_chunk = np.asarray(sc["action_chunk"], dtype=float)
    action_hz = sc["action_hz"]
    dt = 1.0 / action_hz
    workspace = sc["workspace"]
    joint_limits = sc["joint_limits"]
    max_vel = sc["max_joint_velocity"]
    spacing = sc["link_sample_spacing"]

    # ---------- 输入检查 ----------
    fatal, jl_exceeded = check_input(
        joint_state, action_chunk, action_hz, joint_limits, sc["max_joint_step"])
    if fatal:
        return {
            "name": sc["name"],
            "rejected": True,
            "reasons": fatal,
            "joint_limit_exceeded": jl_exceeded,
            "total_steps": action_chunk.shape[0],
            "executed_steps": 0,
            "modified_steps": 0,
            "stopped_steps": 0,
            "minimum_workspace_margin": None,
            "maximum_joint_velocity": 0.0,
            "final_joint_state": joint_state.copy(),
            "nominal_trajectory": action_chunk.copy(),
            "executed_trajectory": joint_state[None, :].copy(),
        }

    # ---------- 闭环执行 ----------
    robot = MockRobot(joint_state, joint_limits)

    executed = [joint_state.copy()]
    modified_steps = 0
    stopped_steps = 0
    min_margin = np.inf
    max_observed_vel = 0.0
    details = []

    for target in action_chunk:
        q = robot.get_joint_state()

        # 越界目标钳位到限位（最小修改，保证不越限）
        target = clamp_target(target, joint_limits)

        q_dot_nom = nominal_velocity(q, target, dt, max_vel)
        max_observed_vel = max(max_observed_vel, float(np.max(np.abs(q_dot_nom))))

        q_dot_safe, info = safety_filter(q, q_dot_nom, workspace, dt, joint_limits)

        if info["stopped"]:
            stopped_steps += 1
        elif info["modified"]:
            modified_steps += 1

        q_next = robot.step(q_dot_safe, dt)
        executed.append(q_next)

        # 全身安全复核（题目要求 step 后仍要 check_full_body_safety）
        chk = full_body_check(q_next, workspace, spacing)
        min_margin = min(min_margin, chk["min_margin"])

        details.append({
            "step": len(details),
            "q": q.tolist(),
            "target": target.tolist(),
            "q_dot_nom": q_dot_nom.tolist(),
            "q_dot_safe": q_dot_safe.tolist(),
            "scale": info["scale"],
            "modified": info["modified"],
            "stopped": info["stopped"],
            "margin": chk["min_margin"],
        })

    executed = np.asarray(executed)
    return {
        "name": sc["name"],
        "rejected": False,
        "reasons": [],
        "joint_limit_exceeded": jl_exceeded,
        "total_steps": action_chunk.shape[0],
        "executed_steps": executed.shape[0] - 1,
        "modified_steps": modified_steps,
        "stopped_steps": stopped_steps,
        "minimum_workspace_margin": float(min_margin),
        "maximum_joint_velocity": max_observed_vel,
        "final_joint_state": executed[-1],
        "nominal_trajectory": action_chunk.copy(),
        "executed_trajectory": executed,
        "details": details,
    }


def print_report(report):
    """终端友好打印 Replay 结果。"""
    if report["rejected"]:
        print(f"[REJECTED] 非法输入，轨迹未进入 Mock Robot：")
        for r in report["reasons"]:
            print(f"  - {r}")
        return

    print(f"scenario          : {report['name']}")
    print(f"total_steps       : {report['total_steps']}")
    print(f"executed_steps    : {report['executed_steps']}")
    print(f"modified_steps    : {report['modified_steps']}")
    print(f"stopped_steps     : {report['stopped_steps']}")
    print(f"min_ws_margin (m) : {report['minimum_workspace_margin']:.4f}")
    print(f"max_joint_vel     : {report['maximum_joint_velocity']:.4f} rad/s")
    print(f"final_joint_state : {np.round(report['final_joint_state'], 4).tolist()}")
    if report["joint_limit_exceeded"]:
        print("  (注：VLA 存在越界关节目标，已钳位到限位内)")


if __name__ == "__main__":
    # 兼容题目推荐调用：python -m safe_motion.replay scenarios/free_space.json
    import argparse

    parser = argparse.ArgumentParser(description="SafeMotion 轨迹回放")
    parser.add_argument("scenario", help="场景 JSON 路径")
    parser.add_argument("--plot", action="store_true", help="生成 3D 可视化图")
    parser.add_argument("--out", default="artifacts", help="输出目录")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    report = run_scenario(scenario)
    print_report(report)

    if args.plot:
        from .visualize import plot_replay
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, f"{report['name']}.png".replace(" ", "_"))
        plot_replay(scenario, report, out_path)
        print(f"\n3D 图已保存: {out_path}")
