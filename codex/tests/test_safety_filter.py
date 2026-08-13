import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from safe_motion.config import SafetyConfig
from safe_motion.safety_filter import command_is_safe, nominal_velocity, safety_filter


Q_HOME = np.array([0.0, -1.35, 1.65, -1.55, -1.57, 0.0])
SCENARIOS = Path(__file__).parents[1] / "scenarios"


def test_safe_zero_command_remains_unmodified():
    result = safety_filter(Q_HOME, np.zeros(6), 0.05, SafetyConfig())
    assert not result.modified
    assert result.scale == 1.0


def test_non_finite_nominal_command_fails_closed():
    result = safety_filter(Q_HOME, np.full(6, np.nan), 0.05, SafetyConfig())
    assert result.stopped
    np.testing.assert_array_equal(result.velocity, np.zeros(6))


def test_every_returned_command_is_finite_and_safe():
    rng = np.random.default_rng(7)
    config = SafetyConfig()
    for _ in range(100):
        nominal = rng.uniform(-config.max_joint_velocity, config.max_joint_velocity)
        result = safety_filter(Q_HOME, nominal, 0.05, config)
        assert np.all(np.isfinite(result.velocity))
        assert command_is_safe(Q_HOME, result.velocity, 0.05, config).safe


def test_command_recheck_rejects_velocity_limit_violation():
    config = SafetyConfig()
    result = command_is_safe(Q_HOME, np.full(6, 2.0), 0.05, config)
    assert not result.safe
    assert result.reason == "candidate_velocity_limit_violation"


def test_bisection_reduces_modification_without_weakening_safety():
    data = json.loads(
        (SCENARIOS / "real_03_tcp_safe_mid_link_unsafe.json").read_text()
    )
    config = SafetyConfig.from_scenario(data)
    chunk = np.asarray(data["action_chunk"], dtype=float)
    q = chunk[45]
    q_dot = nominal_velocity(q, chunk[46], 1.0 / data["action_hz"], config)

    coarse = safety_filter(
        q, q_dot, 1.0 / data["action_hz"],
        replace(config, bisection_iterations=0),
    )
    refined = safety_filter(q, q_dot, 1.0 / data["action_hz"], config)

    assert coarse.scale == 0.5
    assert coarse.scale < refined.scale < 0.75
    assert any(attempt.stage == "bisection" for attempt in refined.attempts)
    assert refined.minimum_margin >= 0.0
    assert command_is_safe(q, refined.velocity, 1.0 / data["action_hz"], config).safe
