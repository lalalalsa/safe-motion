import json
from pathlib import Path

import numpy as np
import pytest

from safe_motion.config import SafetyConfig
from safe_motion.geometry import check_full_body
from safe_motion.safety_filter import FilterResult
from safe_motion.replay import run_scenario


ROOT = Path(__file__).parents[1]


def load(name):
    return json.loads((ROOT / "scenarios" / name).read_text())


def test_free_space_executes_all_points_without_modification():
    result = run_scenario(load("free_space.json"))
    assert result.executed_steps == 50
    assert result.modified_steps == 0
    assert result.stopped_steps == 0
    np.testing.assert_allclose(result.final_joint_state, result.nominal_joint_trajectory[-1])


def test_every_executed_state_remains_full_body_safe():
    data = load("workspace_boundary.json")
    config = SafetyConfig.from_scenario(data)
    result = run_scenario(data, config)
    assert result.executed_steps == 50
    assert result.modified_steps > 0
    for q in result.executed_joint_trajectory:
        assert check_full_body(np.asarray(q), config).safe


def test_filter_exception_becomes_zero_velocity(monkeypatch):
    data = load("free_space.json")

    def broken_filter(*args, **kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr("safe_motion.replay.safety_filter", broken_filter)
    result = run_scenario(data)
    assert result.stopped_steps == 50
    assert all(record.reason == "filter_error:RuntimeError" for record in result.records)
    np.testing.assert_allclose(result.final_joint_state, data["joint_state"])
