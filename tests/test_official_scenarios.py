import json
from pathlib import Path

import numpy as np

from safe_motion.config import SafetyConfig
from safe_motion.geometry import check_full_body
from safe_motion.kinematics import chain_points
from safe_motion.replay import run_scenario


SCENARIOS = Path(__file__).parents[1] / "scenarios"


def load(name):
    return json.loads((SCENARIOS / name).read_text())


def test_all_official_scenarios_finish_with_safe_actual_states():
    for path in sorted(SCENARIOS.glob("real_*.json")):
        data = json.loads(path.read_text())
        config = SafetyConfig.from_scenario(data)
        result = run_scenario(data, config)
        assert result.executed_steps == 50
        assert result.modified_steps > 0
        assert result.minimum_workspace_margin >= 0.0
        assert all(check_full_body(np.asarray(q), config).safe
                   for q in result.executed_joint_trajectory)


def test_official_mid_link_case_would_fool_tcp_only_check():
    data = load("real_03_tcp_safe_mid_link_unsafe.json")
    config = SafetyConfig.from_scenario(data)
    unsafe_target = np.asarray(data["action_chunk"])[-1]
    points = chain_points(unsafe_target)[1:]
    tcp = points[-1]
    assert np.all(tcp >= config.workspace.lower)
    assert np.all(tcp <= config.workspace.upper)
    assert not check_full_body(unsafe_target, config).safe


def test_reference_first_unsafe_frames_are_detected():
    for path in sorted(SCENARIOS.glob("real_*.json")):
        data = json.loads(path.read_text())
        config = SafetyConfig.from_scenario(data)
        expected = data["reference_analysis"]["first_unsafe_frame"]
        first = next(
            i for i, q in enumerate(data["action_chunk"])
            if not check_full_body(np.asarray(q), config).safe
        )
        assert first == expected
