import numpy as np

from safe_motion.config import SafetyConfig
from safe_motion.safety_filter import command_is_safe, safety_filter


Q_HOME = np.array([0.0, -1.35, 1.65, -1.55, -1.57, 0.0])


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
