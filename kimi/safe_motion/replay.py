"""闭环 Replay（题目第 9、11 节）+ CLI（第 12 节演示）。

50 点 action_chunk 逐点执行，每个控制周期：
    q          = robot.get_joint_state()      # 基于实际状态，非开环
    q_dot_nom  = nominal_velocity(q, target)  # 限速后的名义速度
    q_dot_safe = filter_fail_closed(...)      # 安全过滤（fail-closed）
    信任边界复核：step 前对 q_dot_safe 再独立验证一次
    q          = robot.step(q_dot_safe, dt)   # Mock Robot 积分
    执行后审计：check_state(q) 记录全身 margin

三层防线：输入检查 → 安全过滤 → step 前独立复核。
任何一层失效都退化为零速度停止，绝不执行未经验证的动作。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import RobotConfig
from .geometry import Workspace, check_state
from .kinematics import forward_kinematics
from .mock_robot import MockRobot
from .safety_filter import SafetyFilter
from .validation import InputValidationError, Scenario, validate_input


def run_scenario(source, robot_cfg: RobotConfig | None = None,
                 filter_kwargs: dict | None = None,
                 filter_method: str = "scaling",
                 artifacts_dir: str | None = None,
                 make_plot: bool = True, show: bool = False) -> dict:
    """执行一个场景，返回 {"summary", "records", "executed", ...}。

    source 可以是场景 JSON 路径或已解析的 dict。
    filter_method: "scaling"（速度缩放，默认）| "qp"（QP 最小修改，加分项）。
    输入非法时抛 InputValidationError —— 此时 Mock Robot 尚未创建，
    轨迹不可能进入机器人。
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = source

    cfg = robot_cfg or RobotConfig.ur5()
    scenario: Scenario = validate_input(data, cfg)  # ← 非法输入在此被拒绝

    ws = Workspace.from_dict(scenario.workspace_dict)
    sf = SafetyFilter(cfg, ws, **(filter_kwargs or {}))
    if filter_method == "qp":
        from .qp_filter import QPFilter
        sf = QPFilter(sf)
    elif filter_method != "scaling":
        raise ValueError(f"未知 filter_method: {filter_method}")
    robot = MockRobot(scenario.joint_state, cfg.joint_limits)
    dt = scenario.dt
    chunk = scenario.action_chunk

    # 初始状态审计（若起始位形已不安全，过滤器将全程 hold）
    audit0 = check_state(scenario.joint_state, ws, spacing=sf.spacing,
                         safety_margin=sf.safety_margin)

    records = []
    executed = [robot.get_joint_state()]
    executed_margins = [audit0.min_margin]
    aborted = False

    for i in range(chunk.shape[0]):
        q = robot.get_joint_state()
        target = chunk[i]

        v_nom = sf.nominal_velocity(q, target, dt)
        res = sf.filter_fail_closed(q, v_nom, dt)
        v_safe, status, reason = res.q_dot, res.status, res.reason

        # 信任边界复核：step 前独立重验即将下发的动作
        try:
            ok, _, note = sf.check_step(q, v_safe, dt)
        except Exception as exc:  # noqa: BLE001
            ok, note = False, f"复核抛出异常: {exc}"
        if not ok:
            v_safe = np.zeros(6)
            status = "stopped"
            reason = f"step 前复核否决了过滤器输出（{note}），改发零速度"

        try:
            q_next = robot.step(v_safe, dt)
        except Exception as exc:  # noqa: BLE001 —— Mock Robot 拒绝（如限位）
            records.append(_record(i, q, target, v_nom, np.zeros(6), "stopped",
                                   f"Mock Robot 拒绝执行: {exc}；本周期停止",
                                   res.attempts, None))
            aborted = True
            break

        audit = check_state(q_next, ws, spacing=sf.spacing,
                            safety_margin=sf.safety_margin)
        executed.append(q_next)
        executed_margins.append(audit.min_margin)
        records.append(_record(i, q, target, v_nom, v_safe, status, reason,
                               res.attempts, audit))

    executed = np.asarray(executed)
    executed_margins = np.asarray(executed_margins, dtype=float)

    # 名义（未过滤）轨迹的全身 margin：把每个目标点当作状态做静态检查，
    # 用于量化「如果没有 SafeMotion 会有多危险」。
    nominal_margins = np.array([
        check_state(t, ws, spacing=sf.spacing).min_margin for t in chunk
    ])

    tcp_exec = np.array([forward_kinematics(q)["tcp"] for q in executed])
    tcp_nom = np.array([forward_kinematics(t)["tcp"] for t in chunk])
    n_pairs = min(len(tcp_exec) - 1, len(tcp_nom))
    max_tcp_dev = float(np.max(np.linalg.norm(
        tcp_exec[1:n_pairs + 1] - tcp_nom[:n_pairs], axis=1))) if n_pairs else 0.0

    velocities = np.array([r["q_dot_safe"] for r in records], dtype=float)
    statuses = [r["status"] for r in records]
    interventions = [r["index"] for r in records if r["status"] != "unmodified"]

    summary = {
        "scenario": scenario.name,
        "filter_method": filter_method,
        "total_steps": int(chunk.shape[0]),
        "executed_steps": int(len(executed) - 1),
        "modified_steps": int(statuses.count("modified")),
        "stopped_steps": int(statuses.count("stopped")),
        "aborted": aborted,
        "minimum_workspace_margin": float(np.min(executed_margins)),
        "nominal_minimum_workspace_margin": float(np.min(nominal_margins)),
        "maximum_joint_velocity": (float(np.max(np.abs(velocities)))
                                   if len(velocities) else 0.0),
        "max_tcp_deviation_m": max_tcp_dev,
        "first_intervention_step": (interventions[0] if interventions else None),
        "final_joint_state": [float(x) for x in executed[-1]],
        "initial_state_safe": bool(audit0.safe),
    }

    artifacts = {}
    display_name = (scenario.name if filter_method == "scaling"
                    else f"{scenario.name}_{filter_method}")
    if artifacts_dir:
        artifacts = _save_artifacts(artifacts_dir, display_name, chunk, executed,
                                    executed_margins, nominal_margins, records,
                                    summary, ws, make_plot, show)
    elif show:
        from .visualize import plot_3d
        plot_3d(None, display_name, ws, chunk, executed, records, show=True)

    return {"summary": summary, "records": records, "executed": executed,
            "nominal_margins": nominal_margins,
            "executed_margins": executed_margins, "artifacts": artifacts,
            "scenario": scenario, "workspace": ws}


def _record(i, q, target, v_nom, v_safe, status, reason, attempts, audit):
    return {
        "index": int(i),
        "q_start": np.asarray(q, dtype=float).tolist(),
        "q_target": np.asarray(target, dtype=float).tolist(),
        "q_dot_nom": np.asarray(v_nom, dtype=float).tolist(),
        "q_dot_safe": np.asarray(v_safe, dtype=float).tolist(),
        "status": status,
        "reason": reason,
        "attempts": [a.__dict__ for a in attempts],
        "margin_after": (audit.min_margin if audit is not None else None),
        "worst_after": (audit.worst_label if audit is not None else None),
    }


def _save_artifacts(out_dir, name, chunk, executed, exec_m, nom_m, records,
                    summary, ws, make_plot, show):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_")

    npz = out / f"{safe_name}_trajectories.npz"
    np.savez(npz, action_chunk=chunk, executed=executed,
             executed_margins=exec_m, nominal_margins=nom_m)

    summary_path = out / f"{safe_name}_summary.json"
    summary_path.write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    artifacts = {"trajectories": str(npz), "summary": str(summary_path)}
    if make_plot:
        from .visualize import plot_3d, plot_margins
        p3d = out / f"{safe_name}_3d.png"
        plot_3d(str(p3d), name, ws, chunk, executed, records, show=show)
        pm = out / f"{safe_name}_margins.png"
        plot_margins(str(pm), name, nom_m, exec_m)
        artifacts.update({"plot_3d": str(p3d), "plot_margins": str(pm)})
    return artifacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(s: dict):
    print(f"scenario            : {s['scenario']}   [filter={s['filter_method']}]")
    print(f"total_steps         : {s['total_steps']}")
    print(f"executed_steps      : {s['executed_steps']}")
    print(f"modified_steps      : {s['modified_steps']}")
    print(f"stopped_steps       : {s['stopped_steps']}")
    print(f"first_intervention  : {s['first_intervention_step']}")
    print(f"min_ws_margin (m)   : {s['minimum_workspace_margin']:+.6f}   (执行轨迹)")
    print(f"nominal_min_margin  : {s['nominal_minimum_workspace_margin']:+.6f}   (未过滤的 VLA 轨迹)")
    print(f"max_tcp_deviation   : {s['max_tcp_deviation_m']:.4f} m")
    print(f"max_joint_vel       : {s['maximum_joint_velocity']:.4f} rad/s   (实际执行)")
    print(f"final_joint_state   : {np.round(s['final_joint_state'], 4).tolist()}")
    if not s["initial_state_safe"]:
        print("!! 初始状态已不安全，过滤器全程 hold")


def _print_explanation(result: dict, step: int):
    """题目 12.4：完整讲解一个发生干预的控制点。"""
    rec = next((r for r in result["records"] if r["index"] == step), None)
    if rec is None:
        print(f"没有第 {step} 步的记录"); return
    print(f"\n===== 第 {step} 步安全干预讲解 =====")
    print(f"当前 q        : {np.round(rec['q_start'], 4).tolist()}")
    print(f"VLA target    : {np.round(rec['q_target'], 4).tolist()}")
    print(f"q_dot_nom     : {np.round(rec['q_dot_nom'], 4).tolist()}  (已限速)")
    print("候选评估（速度缩放回溯 + 二分细化）:")
    for a in rec["attempts"]:
        mark = "✓ 安全" if a["safe"] else "✗ 不安全"
        margin = f"{a['min_margin']:+.4f}" if a["min_margin"] == a["min_margin"] else "  n/a  "
        print(f"  scale={a['scale']:.4f}  {mark}  margin={margin} m"
              f"  worst={a['worst_label']} @ {a['worst_boundary']}"
              f"  [{a['note']}]")
    print(f"结论          : {rec['status']} — {rec['reason']}")
    print(f"q_dot_safe    : {np.round(rec['q_dot_safe'], 4).tolist()}")
    if rec["margin_after"] is not None:
        print(f"执行后审计    : margin={rec['margin_after']:+.6f} m"
              f"  (worst: {rec['worst_after']})")


def main(argv=None) -> int:
    try:  # Windows 终端默认 GBK，保证中文输出不乱码
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(
        prog="python -m safe_motion.replay",
        description="SafeMotion 闭环 Replay")
    parser.add_argument("scenario", help="场景 JSON 路径")
    parser.add_argument("--method", choices=["scaling", "qp"], default="scaling",
                        help="安全过滤方法（默认 scaling；qp 为加分项）")
    parser.add_argument("--artifacts", default="artifacts", help="输出目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成图")
    parser.add_argument("--show", action="store_true", help="弹窗显示 3D 图")
    parser.add_argument("--explain", type=int, default=None, metavar="STEP",
                        help="详细讲解第 STEP 步的安全干预")
    args = parser.parse_args(argv)

    try:
        result = run_scenario(args.scenario, filter_method=args.method,
                              artifacts_dir=args.artifacts,
                              make_plot=not args.no_plot, show=args.show)
    except InputValidationError as exc:
        print(f"[INPUT REJECTED] 非法输入，未进入 Mock Robot:\n  {exc}")
        return 2

    _print_summary(result["summary"])
    if result["artifacts"]:
        print("artifacts         : " + ", ".join(result["artifacts"].values()))
    if args.explain is not None:
        _print_explanation(result, args.explain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
