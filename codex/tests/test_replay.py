import json
from pathlib import Path

import numpy as np
import pytest

from safe_motion.config import SafetyConfig
from safe_motion.geometry import check_full_body
from safe_motion.safety_filter import FilterResult, command_is_safe
from safe_motion.replay import print_explanation, run_scenario
from safe_motion.visualization import plot_workspace_margins


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


def test_malformed_filter_output_becomes_zero_velocity(monkeypatch):
    data = load("free_space.json")

    def malformed_filter(*args, **kwargs):
        return FilterResult(
            np.full(6, np.nan), 1.0, False, False, "malformed", 1.0
        )

    monkeypatch.setattr("safe_motion.replay.safety_filter", malformed_filter)
    result = run_scenario(data)

    assert result.stopped_steps == 50
    assert all(record.reason == "filter_error:ValueError" for record in result.records)
    np.testing.assert_allclose(result.final_joint_state, data["joint_state"])


def test_filter_cannot_change_nominal_direction(monkeypatch):
    data = load("free_space.json")

    def policy_breaking_filter(q, q_dot_nom, dt, cfg):
        return FilterResult(
            -np.asarray(q_dot_nom), 1.0, False, False, "changed_direction", 1.0
        )

    monkeypatch.setattr("safe_motion.replay.safety_filter", policy_breaking_filter)
    result = run_scenario(data)

    nonzero_records = [
        record for record in result.records
        if not np.allclose(record.q_dot_nominal, 0.0, rtol=0.0, atol=1e-12)
    ]
    assert nonzero_records
    assert all(record.stopped for record in nonzero_records)
    assert all(record.reason == "filter_error:ValueError" for record in nonzero_records)
    np.testing.assert_allclose(result.final_joint_state, data["joint_state"])


def test_filter_lie_is_rejected_before_robot_step(monkeypatch):
    """A well-shaped but unsafe filter output must never cross the robot boundary."""
    data = load("real_03_tcp_safe_mid_link_unsafe.json")
    config = SafetyConfig.from_scenario(data)

    def lying_filter(q, q_dot_nom, dt, cfg):
        return FilterResult(
            np.asarray(q_dot_nom), 1.0, False, False, "unsafe_filter_lie", 1.0
        )

    def checked_step(self, q_dot, dt):
        assert command_is_safe(self.q, q_dot, dt, config).safe
        q_next = self.q + np.asarray(q_dot) * dt
        self.q = q_next
        self.step_calls += 1
        return self.q.copy()

    monkeypatch.setattr("safe_motion.replay.safety_filter", lying_filter)
    monkeypatch.setattr("safe_motion.replay.MockRobot.step", checked_step)
    result = run_scenario(data, config)

    assert any(
        record.reason.startswith("pre_execution_recheck_failed:")
        for record in result.records
    )
    assert all(
        check_full_body(np.asarray(q), config).safe
        for q in result.executed_joint_trajectory
    )


def test_pre_execution_recheck_exception_fails_closed(monkeypatch):
    data = load("free_space.json")

    def broken_recheck(*args, **kwargs):
        raise RuntimeError("injected trust-boundary failure")

    monkeypatch.setattr("safe_motion.replay.command_is_safe", broken_recheck)
    result = run_scenario(data)

    assert result.stopped_steps == 50
    assert all(
        record.reason == "pre_execution_recheck_failed:recheck_error:RuntimeError"
        for record in result.records
    )
    np.testing.assert_allclose(result.final_joint_state, data["joint_state"])


def test_explanation_and_margin_plot_are_demo_ready(tmp_path, capsys):
    data = load("real_03_tcp_safe_mid_link_unsafe.json")
    result = run_scenario(data)

    assert result.nominal_minimum_workspace_margin < 0.0
    assert result.minimum_workspace_margin >= 0.0
    assert len(result.nominal_workspace_margins) == 50
    assert len(result.executed_workspace_margins) == 51

    print_explanation(result, 46)
    output = capsys.readouterr().out
    assert "Safety intervention at step 46" in output
    assert "bisection" in output
    assert "node=" in output

    destination = tmp_path / "margins.png"
    plot_workspace_margins(result, destination)
    assert destination.is_file()
    assert destination.stat().st_size > 10_000
