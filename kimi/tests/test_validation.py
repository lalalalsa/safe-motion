"""输入检查：合法放行、各类非法拒绝（NaN/Inf/shape/点数/突跳/越界/hz）。"""
import copy

import numpy as np
import pytest

from safe_motion.config import RobotConfig
from safe_motion.validation import (InputValidationError, validate_input)

CFG = RobotConfig.ur5()


def test_valid_passes(valid_scenario):
    sc = validate_input(valid_scenario, CFG)
    assert sc.action_chunk.shape == (50, 6)
    assert sc.dt == pytest.approx(0.05)


def _mutate(valid_scenario, fn):
    data = copy.deepcopy(valid_scenario)
    fn(data)
    return data


CASES = {
    "nan": lambda d: d["action_chunk"][10].__setitem__(3, float("nan")),
    "inf": lambda d: d["action_chunk"][10].__setitem__(3, float("inf")),
    "neg_inf": lambda d: d["action_chunk"][10].__setitem__(3, float("-inf")),
    "not_50_points": lambda d: d.__setitem__("action_chunk", d["action_chunk"][:49]),
    "wrong_joint_dim": lambda d: d.__setitem__(
        "action_chunk", [row[:5] for row in d["action_chunk"]]),
    "joint_state_dim5": lambda d: d.__setitem__("joint_state",
                                                d["joint_state"][:5]),
    "hz_zero": lambda d: d.__setitem__("action_hz", 0.0),
    "hz_negative": lambda d: d.__setitem__("action_hz", -20.0),
    "hz_nan": lambda d: d.__setitem__("action_hz", float("nan")),
    "start_jump": lambda d: d["action_chunk"][0].__setitem__(
        0, d["action_chunk"][0][0] + 1.0),
    "successive_jump": lambda d: d["action_chunk"][20].__setitem__(
        1, d["action_chunk"][20][1] + 1.0),
    "joint_state_beyond_limit": lambda d: d.__setitem__(
        "joint_state", [7.0] + d["joint_state"][1:]),
    "missing_chunk": lambda d: d.pop("action_chunk"),
    "bad_workspace": lambda d: d.__setitem__(
        "workspace", {"x": [0.5, -0.5], "y": [-0.7, 0.7], "z": [0.05, 0.9]}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_invalid_inputs_rejected(valid_scenario, case):
    with pytest.raises(InputValidationError):
        validate_input(_mutate(valid_scenario, CASES[case]), CFG)


def test_joint_target_beyond_limit_rejected(valid_scenario):
    """关节目标平滑 ramp 越过限位（相邻步小于突跳阈值）：
    突跳检查拦不住，必须由范围检查拒绝。"""
    data = copy.deepcopy(valid_scenario)
    q0 = np.array(data["joint_state"])
    chunk = np.array(data["action_chunk"])
    chunk[:, 5] = np.linspace(q0[5], 7.4, 50)  # J6 → 7.4 rad > +2π
    assert np.abs(np.diff(chunk, axis=0)).max() < CFG.max_joint_step
    data["action_chunk"] = chunk.tolist()
    with pytest.raises(InputValidationError, match="关节目标越界"):
        validate_input(data, CFG)
