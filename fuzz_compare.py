"""跨版本模糊测试：deepseek vs kimi，同一批全新 100 组数据。

防数据泄露设计：
  - 数据 seed 由命令行传入（--seed），脚本内不写死任何固定 seed；
  - 数据只在内存中确定性生成，测完即输出，不留可被 agent 预知的数据文件；
  - 两版用同一 seed 跑两次，数据完全一致，横向对比公平。

运行（seed 由测试方私下指定，两版必须同 seed）：
    python fuzz_compare.py --target deepseek --seed <secret>
    python fuzz_compare.py --target kimi     --seed <secret>

输出：fuzz_compare_results/fuzz_<target>.csv / _summary.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fuzz_compare_results")

VERSIONS = {
    "codex": os.path.join(HERE, "codex"),
    "deepseek": os.path.join(HERE, "deepseek"),
    "kimi": os.path.join(HERE, "kimi"),
}

# real_01 初始位形（题目数据内，全身安全，margin ≈ 0.039 m）
SAFE_BASE = np.array([
    1.7011667490005493, -1.718811337147848, -2.1258776823626917,
    -0.9176800886737269, 1.5363986492156982, 1.810360312461853,
])
LIMITS = np.column_stack([-2.0 * np.pi * np.ones(6), 2.0 * np.pi * np.ones(6)])
WORKSPACE = {"x": [-0.70, 0.70], "y": [-0.70, 0.70], "z": [0.05, 0.90]}


# ---------------------------------------------------------------------------
# 数据生成（确定性，仅依赖 seed）
# ---------------------------------------------------------------------------
def _safe_start(rng, fk_check, max_try=3000):
    """在 SAFE_BASE 附近 rejection sampling 一个全身安全起始位形。"""
    for _ in range(max_try):
        q = SAFE_BASE + rng.normal(0.0, 0.03, 6)
        q = np.clip(q, LIMITS[:, 0], LIMITS[:, 1])
        if fk_check(q):
            return q
    return SAFE_BASE.copy()


def build_groups(rng, fk_check):
    """生成 100 组：88 随机游走 + 8 非法输入 + 4 关节越界。"""
    groups = []
    gid = 0
    for _ in range(88):
        sigma = float(10 ** rng.uniform(np.log10(0.0015), np.log10(0.10)))
        q0 = _safe_start(rng, fk_check)
        chunk = []
        q = q0.copy()
        for _ in range(50):
            q = q + rng.normal(0.0, sigma, 6)
            chunk.append(q)
        groups.append({"gid": gid, "kind": "random_walk", "sigma": sigma,
                       "joint_state": q0, "action_chunk": np.asarray(chunk)})
        gid += 1

    base = SAFE_BASE.copy()
    small = SAFE_BASE + rng.normal(0.0, 0.01, (50, 6))
    # 8 组非法输入
    invalid = []
    c = small.copy(); c[10, 3] = np.nan
    invalid.append(("invalid_nan", base, c))
    c = small.copy(); c[20, 1] = np.inf
    invalid.append(("invalid_inf", base, c))
    invalid.append(("invalid_shape_short", base, SAFE_BASE + rng.normal(0, 0.01, (30, 6))))
    wide = np.column_stack([np.tile(SAFE_BASE, (50, 1)), np.zeros(50)])
    invalid.append(("invalid_shape_wide", base, wide))
    invalid.append(("invalid_js_dim", SAFE_BASE[:5], small))
    invalid.append(("invalid_hz", base, small, 0.0))
    jump = small.copy(); jump[0] = base + 1.0  # 起始突跳 1.0 rad > 0.5
    invalid.append(("invalid_jump", base, jump))
    js_nan = base.copy(); js_nan[0] = np.nan
    invalid.append(("invalid_js_nan", js_nan, small))
    for kind, js, chunk, *rest in invalid:
        hz = rest[0] if rest else 20.0
        groups.append({"gid": gid, "kind": kind, "sigma": None,
                       "joint_state": js, "action_chunk": np.asarray(chunk),
                       "action_hz": hz})
        gid += 1

    # 4 组关节越界
    for jj, target in [(5, 7.0), (5, -7.0), (2, 7.0), (0, -7.0)]:
        chunk = []
        for i in range(50):
            q = SAFE_BASE.copy()
            q[jj] = SAFE_BASE[jj] + (target - SAFE_BASE[jj]) * i / 49.0
            chunk.append(q)
        groups.append({"gid": gid, "kind": "joint_limit", "sigma": None,
                       "joint_state": base, "action_chunk": np.asarray(chunk)})
        gid += 1
    return groups


# ---------------------------------------------------------------------------
# 版本加载与 API 适配
# ---------------------------------------------------------------------------
def _load(target):
    sys.path.insert(0, VERSIONS[target])
    import safe_motion
    import safe_motion.geometry  # noqa: F401  子模块需显式导入
    import safe_motion.kinematics  # noqa: F401
    import safe_motion.replay  # noqa: F401
    return safe_motion


def _fk_func(sm, target):
    if target == "deepseek":
        return lambda q: sm.kinematics.forward_kinematics(q)[-1]
    return lambda q: sm.kinematics.forward_kinematics(q)["tcp"]


def _fk_check_func(sm, target):
    if target == "deepseek":
        def chk(q):
            return sm.geometry.full_body_check(q, WORKSPACE)["is_safe"]
    elif target == "kimi":
        ws = sm.geometry.Workspace.from_dict(WORKSPACE)
        def chk(q):
            return sm.geometry.check_state(q, ws).safe
    else:  # codex
        cfg = sm.config.SafetyConfig()
        def chk(q):
            return sm.geometry.check_full_body(q, cfg).safe
    return chk


def run_one(sm, target, g, fk, method="scaling"):
    """适配两版 API，返回统一格式的结果 dict。"""
    row = {
        "group_id": g["gid"], "kind": g["kind"],
        "sigma": round(g["sigma"], 5) if g["sigma"] is not None else "",
        "input_valid": "", "rejected": None, "reject_reason": "",
        "total_steps": int(g["action_chunk"].shape[0]),
        "executed_steps": None, "modified_steps": None, "stopped_steps": None,
        "joint_dev_mean": "", "joint_dev_max": "", "tcp_dev_mean": "",
        "tcp_dev_max": "", "final_gap": "", "min_margin": "", "max_vel": "",
        "violations": "", "verdict": "",
    }
    scenario = {
        "name": f"fz_{g['gid']:03d}_{g['kind']}",
        "action_hz": float(g.get("action_hz", 20.0)),
        "joint_state": np.asarray(g["joint_state"], dtype=float).tolist(),
        "action_chunk": np.asarray(g["action_chunk"], dtype=float).tolist(),
    }
    row["input_valid"] = bool(
        np.asarray(g["joint_state"]).shape == (6,)
        and np.asarray(g["action_chunk"]).shape == (50, 6)
        and np.all(np.isfinite(g["joint_state"]))
        and np.all(np.isfinite(g["action_chunk"]))
        and g.get("action_hz", 20.0) > 0)

    if target == "deepseek":
        try:
            report = sm.replay.run_scenario(scenario)
        except Exception as e:  # noqa: BLE001
            row["verdict"] = "CRASH"
            row["violations"] = f"exception:{type(e).__name__}:{e}"
            return row
        if report["rejected"]:
            row["rejected"] = True
            row["reject_reason"] = "; ".join(report["reasons"])
            row["verdict"] = "REJECTED"
            return row
        row["rejected"] = False
        row["executed_steps"] = report["executed_steps"]
        row["modified_steps"] = report["modified_steps"]
        row["stopped_steps"] = report["stopped_steps"]
        row["min_margin"] = round(report["minimum_workspace_margin"], 5)
        row["max_vel"] = round(report["maximum_joint_velocity"], 5)
        nominal = np.asarray(report["nominal_trajectory"])
        executed = np.asarray(report["executed_trajectory"])
        details = report.get("details", [])
        q_dot_safe = [d["q_dot_safe"] for d in details]
    elif target == "kimi":
        from safe_motion.validation import InputValidationError
        try:
            result = sm.replay.run_scenario(scenario, make_plot=False,
                                            filter_method=method)
        except InputValidationError as e:
            row["rejected"] = True
            row["reject_reason"] = str(e)
            row["verdict"] = "REJECTED"
            return row
        except Exception as e:  # noqa: BLE001
            row["verdict"] = "CRASH"
            row["violations"] = f"exception:{type(e).__name__}:{e}"
            return row
        summary = result["summary"]
        row["rejected"] = False
        row["executed_steps"] = summary["executed_steps"]
        row["modified_steps"] = summary["modified_steps"]
        row["stopped_steps"] = summary["stopped_steps"]
        row["min_margin"] = round(summary["minimum_workspace_margin"], 5)
        records = result["records"]
        nom_vel = np.abs(np.array([r["q_dot_nom"] for r in records]))
        row["max_vel"] = round(float(np.max(nom_vel)), 5) if len(records) else 0.0
        nominal = np.asarray(g["action_chunk"], dtype=float)
        executed = np.asarray(result["executed"])
        q_dot_safe = [r["q_dot_safe"] for r in records]
    else:  # codex
        from safe_motion.validation import InputValidationError
        try:
            result = sm.replay.run_scenario(scenario)
        except InputValidationError as e:
            row["rejected"] = True
            row["reject_reason"] = str(e)
            row["verdict"] = "REJECTED"
            return row
        except Exception as e:  # noqa: BLE001
            row["verdict"] = "CRASH"
            row["violations"] = f"exception:{type(e).__name__}:{e}"
            return row
        row["rejected"] = False
        row["executed_steps"] = result.executed_steps
        row["modified_steps"] = result.modified_steps
        row["stopped_steps"] = result.stopped_steps
        row["min_margin"] = round(result.minimum_workspace_margin, 5)
        records = result.records
        nom_vel = np.abs(np.array([r.q_dot_nominal for r in records]))
        row["max_vel"] = round(float(np.max(nom_vel)), 5) if records else 0.0
        nominal = np.asarray(result.nominal_joint_trajectory)
        executed = np.asarray(result.executed_joint_trajectory)
        q_dot_safe = [r.q_dot_safe for r in records]

    # 统一偏差计算
    step_exec = executed[1:]
    n = min(len(step_exec), len(nominal))
    dev_joint = np.linalg.norm(step_exec[:n] - nominal[:n], axis=1)
    nom_tcp = np.array([fk(q) for q in nominal[:n]])
    exec_tcp = np.array([fk(q) for q in step_exec[:n]])
    dev_tcp = np.linalg.norm(exec_tcp - nom_tcp, axis=1)
    row["joint_dev_mean"] = round(float(np.mean(dev_joint)), 5)
    row["joint_dev_max"] = round(float(np.max(dev_joint)), 5)
    row["tcp_dev_mean"] = round(float(np.mean(dev_tcp)), 5)
    row["tcp_dev_max"] = round(float(np.max(dev_tcp)), 5)
    row["final_gap"] = round(float(np.linalg.norm(executed[-1] - nominal[-1])), 5)

    # 安全不变量
    problems = []
    if row["min_margin"] is not None and row["min_margin"] < 0:
        problems.append(f"workspace_violation(margin={row['min_margin']})")
    for q in executed:
        if not (np.all(q >= LIMITS[:, 0]) and np.all(q <= LIMITS[:, 1])):
            problems.append("joint_limit_violation")
            break
    for v in q_dot_safe:
        if not np.all(np.isfinite(v)):
            problems.append("nonfinite_velocity")
            break
    row["violations"] = "; ".join(problems)
    row["verdict"] = "FAIL" if problems else "PASS"
    return row


def main():
    ap = argparse.ArgumentParser(description="SafeMotion 跨版本模糊测试")
    ap.add_argument("--target", choices=["codex", "deepseek", "kimi"], required=True)
    ap.add_argument("--seed", type=int, required=True, help="秘密种子（两版必须一致）")
    ap.add_argument("--method", default="scaling", choices=["scaling", "qp"],
                    help="kimi 版的过滤方法（scaling=速度缩放，qp=二次规划加分项）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sm = _load(args.target)
    fk = _fk_func(sm, args.target)
    fk_check = _fk_check_func(sm, args.target)
    groups = build_groups(rng, fk_check)

    rows = [run_one(sm, args.target, g, fk, args.method) for g in groups]

    os.makedirs(OUT, exist_ok=True)
    fields = list(rows[0].keys())
    suffix = "" if args.method == "scaling" else f"_{args.method}"
    csv_path = os.path.join(OUT, f"fuzz_{args.target}{suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_fail = sum(1 for r in rows if r["verdict"] == "FAIL")
    n_reject = sum(1 for r in rows if r["verdict"] == "REJECTED")
    n_crash = sum(1 for r in rows if r["verdict"] == "CRASH")
    executed = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    summary = {
        "target": args.target, "seed": args.seed, "total": n,
        "pass": n_pass, "fail": n_fail, "rejected": n_reject, "crash": n_crash,
        "safety_pass_rate": round(n_pass / n, 4),
        "mean_joint_dev": round(float(np.mean([r["joint_dev_mean"] for r in executed])), 5) if executed else None,
        "mean_tcp_dev": round(float(np.mean([r["tcp_dev_mean"] for r in executed])), 5) if executed else None,
        "mean_modified_steps": round(float(np.mean([r["modified_steps"] for r in executed])), 2) if executed else None,
        "mean_stopped_steps": round(float(np.mean([r["stopped_steps"] for r in executed])), 2) if executed else None,
        "violation_rows": [r["group_id"] for r in rows if r["violations"]],
        "crash_rows": [r["group_id"] for r in rows if r["verdict"] == "CRASH"],
    }
    json_path = os.path.join(OUT, f"fuzz_{args.target}{suffix}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"SafeMotion fuzz  [target={args.target}, seed={args.seed}]")
    print("=" * 60)
    print(f"total={n}  pass={n_pass}  fail={n_fail}  "
          f"rejected={n_reject}  crash={n_crash}")
    print(f"safety_pass_rate = {summary['safety_pass_rate']:.2%}")
    if executed:
        print(f"mean_joint_dev    = {summary['mean_joint_dev']} rad")
        print(f"mean_tcp_dev      = {summary['mean_tcp_dev']} m")
        print(f"mean_modified     = {summary['mean_modified_steps']} steps")
        print(f"mean_stopped      = {summary['mean_stopped_steps']} steps")
    if summary["violation_rows"]:
        print(f"violation rows    = {summary['violation_rows']}")
    else:
        print("violation rows    = (none)")
    print(f"\nCSV  -> {csv_path}")
    print(f"JSON -> {json_path}")


if __name__ == "__main__":
    main()
