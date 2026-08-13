"""生成自建场景并拷贝命名场景。

- free_space.json          : 自建安全轨迹（小幅正弦微扰），生成时用本项目的
                             FK + 全身检查验证 min margin 达标（FK 本身由
                             tests 用面试方参考数据交叉验证，形成可信链）；
- workspace_boundary.json  : 命名拷贝 real_01（下 z 边界）；
- tcp_safe_mid_link_unsafe.json : 命名拷贝 real_03；
- joint_limit.json         : J6 平滑 ramp 越过 +2π（相邻步 0.145 rad <
                             突跳阈值 0.5 rad），必须被输入检查以「关节目标
                             越界」拒绝——演示突跳检查拦不住、范围检查能拦住；
- invalid_input_nan.json   : 轨迹中部注入 NaN，必须被拒绝。

用法: python scripts/make_scenarios.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_motion.config import RobotConfig  # noqa: E402
from safe_motion.geometry import Workspace, check_state  # noqa: E402
from safe_motion.validation import InputValidationError, validate_input  # noqa: E402

SC = ROOT / "scenarios"
CFG = RobotConfig.ur5()


def load(name):
    return json.loads((SC / name).read_text(encoding="utf-8"))


def dump(name, data):
    (SC / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"written: scenarios/{name}")


def make_free_space():
    real01 = load("real_01_lower_z_boundary.json")
    q0 = np.array(real01["joint_state"])
    ws = Workspace.from_dict(real01["workspace"])
    base_amp = np.array([0.05, 0.04, 0.06, 0.05, 0.03, 0.05])
    i = np.arange(50)

    chunk = None
    for scale in (1.0, 0.7, 0.5, 0.3, 0.2):
        cand = q0[None, :] + (base_amp * scale)[None, :] * np.sin(
            2 * np.pi * (i + 1)[:, None] / 25.0)
        min_margin = min(check_state(row, ws).min_margin for row in cand)
        # 0.038 < 0.039159（O_1 肩关节的恒定 margin 上限），留出余量
        if min_margin >= 0.038:
            chunk = cand
            print(f"free_space: amplitude scale={scale}, min margin={min_margin:.4f} m")
            break
    if chunk is None:
        raise SystemExit("free_space 生成失败：找不到足够安全的振幅")

    data = {
        "name": "free_space",
        "description": "Synthetic gentle sinusoid around a measured safe pose.",
        "action_hz": 20.0,
        "joint_state": q0.tolist(),
        "action_chunk": chunk.tolist(),
        "workspace": real01["workspace"],
        "provenance": {"source_kind": "synthetic_sinusoid_around_real_01_joint_state"},
    }
    validate_input(data, CFG)  # 自建场景也必须通过输入检查
    dump("free_space.json", data)


def make_named_copies():
    wb = load("real_01_lower_z_boundary.json")
    wb["name"] = "workspace_boundary"
    dump("workspace_boundary.json", wb)

    tm = load("real_03_tcp_safe_mid_link_unsafe.json")
    tm["name"] = "tcp_safe_mid_link_unsafe"
    dump("tcp_safe_mid_link_unsafe.json", tm)


def make_joint_limit():
    real02 = load("real_02_upper_z_boundary.json")
    q0 = np.array(real02["joint_state"])
    chunk = np.repeat(q0[None, :], 50, axis=0)
    chunk[:, 5] = np.linspace(q0[5], 7.4, 50)  # J6 平滑 ramp 越过 +2π ≈ 6.2832
    data = {
        "name": "joint_limit",
        "description": "J6 ramps smoothly past the +2pi joint limit.",
        "action_hz": 20.0,
        "joint_state": q0.tolist(),
        "action_chunk": chunk.tolist(),
        "workspace": real02["workspace"],
        "provenance": {"source_kind": "synthetic_joint_limit_ramp"},
    }
    try:
        validate_input(data, CFG)
    except InputValidationError as exc:
        print(f"joint_limit 按预期被拒绝: {exc}")
    else:
        raise SystemExit("joint_limit 场景应当被输入检查拒绝！")
    dump("joint_limit.json", data)


def make_invalid_nan():
    bad = load("real_01_lower_z_boundary.json")
    bad["name"] = "invalid_input_nan"
    bad["action_chunk"][25][2] = float("nan")
    try:
        validate_input(bad, CFG)
    except InputValidationError as exc:
        print(f"invalid_input_nan 按预期被拒绝: {exc}")
    else:
        raise SystemExit("NaN 场景应当被输入检查拒绝！")
    (SC / "invalid_input_nan.json").write_text(
        json.dumps(bad, indent=2), encoding="utf-8")
    print("written: scenarios/invalid_input_nan.json")


if __name__ == "__main__":
    make_free_space()
    make_named_copies()
    make_joint_limit()
    make_invalid_nan()
    print("done.")
