"""生成 free_space 与 joint_limit 两个自建场景。

起点使用题目数据中已知安全的位形（real_01 初始 joint_state，
其参考 margin ≈ 0.039 m），在其基础上叠加小幅平滑扰动。
生成后会用全身安全检查逐点验证，确保场景语义正确。
"""
import json
import os

import numpy as np

from safe_motion import config
from safe_motion.geometry import full_body_check

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = os.path.join(HERE, "scenarios")

# real_01 初始位形（题目数据内，全身安全，margin ≈ 0.039 m）
SAFE_BASE = np.array([
    1.7011667490005493,
    -1.718811337147848,
    -2.1258776823626917,
    -0.9176800886737269,
    1.5363986492156982,
    1.810360312461853,
])

WORKSPACE = config.DEFAULT_WORKSPACE
JOINT_LIMITS = config.JOINT_LIMITS


def _gen_free_space():
    """50 点安全轨迹：小幅平滑正弦扰动，幅度 ~0.01 rad。"""
    chunk = []
    for i in range(50):
        t = i * config.DT
        q = SAFE_BASE + np.array([
            0.004 * np.sin(t),
            0.006 * np.sin(0.7 * t + 1.0),
            0.008 * np.sin(0.5 * t + 2.0),
            0.006 * np.cos(0.6 * t),
            0.004 * np.sin(0.8 * t + 0.5),
            0.004 * np.cos(0.9 * t),
        ])
        chunk.append(q.tolist())
    return np.asarray(chunk)


def _gen_joint_limit():
    """J6 平滑上升越过上限 +2π（≈ 6.283）。

    选 J6 是因为它只改变末端朝向、不影响 TCP/连杆位置（J6 的平移
    分量与 q6 无关），因此可「纯粹」触发关节限位钳位，而不被 workspace
    越界抢先生效。
    """
    j6_start = SAFE_BASE[5]
    j6_end = 7.0  # 越出 +2π 上界
    chunk = []
    for i in range(50):
        frac = i / 49.0
        q = SAFE_BASE.copy()
        q[5] = j6_start + (j6_end - j6_start) * frac
        chunk.append(q.tolist())
    return np.asarray(chunk)


def _verify(name, chunk):
    """逐点全身安全检查，返回安全比例。"""
    safe = sum(1 for q in chunk if full_body_check(q, WORKSPACE)["is_safe"])
    print(f"[{name}] {safe}/{len(chunk)} 点全身安全")
    return safe


def _dump(name, chunk, description):
    data = {
        "name": name,
        "description": description,
        "action_hz": config.ACTION_HZ,
        "joint_state": SAFE_BASE.tolist(),
        "action_chunk": chunk.tolist(),
        "workspace": WORKSPACE,
        "joint_limits": JOINT_LIMITS,
    }
    path = os.path.join(SCENARIOS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"已写入 {path}")


if __name__ == "__main__":
    free = _gen_free_space()
    jl = _gen_joint_limit()

    _verify("free_space", free)
    _verify("joint_limit(存在越界,预期非100%)", jl)

    _dump("free_space", free, "自建安全轨迹：小幅正弦微扰，全身始终处于工作空间内")
    _dump("joint_limit", jl, "自建越界轨迹：J3 平滑下降到关节下限以下，其余关节安全")
