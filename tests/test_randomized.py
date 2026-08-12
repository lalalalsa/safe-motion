"""Small deterministic fuzz suite retained as a reproducible hidden-test proxy."""

import copy
import json
from pathlib import Path

import numpy as np

from safe_motion.config import SafetyConfig
from safe_motion.geometry import check_full_body
from safe_motion.replay import run_scenario
from safe_motion.validation import inside_joint_limits


SCENARIOS = Path(__file__).parents[1] / "scenarios"


def test_random_smooth_trajectory_mutations_preserve_execution_invariants():
    rng = np.random.default_rng(20260812)
    for path in sorted(SCENARIOS.glob("real_*.json")):
        original = json.loads(path.read_text())
        for _ in range(10):
            data = copy.deepcopy(original)
            sequence = np.vstack([data["joint_state"], data["action_chunk"]])
            time = np.linspace(0.0, 1.0, 51)
            amplitude = rng.uniform(0.002, 0.03, size=6)
            phase = rng.uniform(0.0, 2.0 * np.pi, size=6)
            sequence += np.sin(2.0 * np.pi * time[:, None] + phase) * amplitude
            data["joint_state"] = sequence[0].tolist()
            data["action_chunk"] = sequence[1:].tolist()
            data["action_hz"] = float(rng.uniform(12.0, 35.0))
            data["max_joint_velocity"] = rng.uniform(0.6, 2.2, size=6).tolist()

            config = SafetyConfig.from_scenario(data)
            result = run_scenario(data, config)
            commands = np.asarray([record.q_dot_safe for record in result.records])

            assert result.executed_steps == 50
            assert np.all(np.isfinite(commands))
            assert np.all(np.abs(commands) <= config.max_joint_velocity + 1e-12)
            assert all(inside_joint_limits(np.asarray(q), config.joint_limits)
                       for q in result.executed_joint_trajectory)
            assert all(check_full_body(np.asarray(q), config).safe
                       for q in result.executed_joint_trajectory)
