import numpy as np
import pytest

from safe_motion.config import SafetyConfig
from safe_motion.mock_robot import MockRobot


def test_exact_euler_update_and_copy_semantics():
    config = SafetyConfig()
    robot = MockRobot(np.zeros(6), config.joint_limits)
    q = robot.step(np.ones(6), 0.05)
    np.testing.assert_allclose(q, np.full(6, 0.05))
    q[:] = 99
    np.testing.assert_allclose(robot.get_joint_state(), np.full(6, 0.05))


def test_non_finite_command_is_rejected():
    robot = MockRobot(np.zeros(6), SafetyConfig().joint_limits)
    with pytest.raises(ValueError, match="non-finite"):
        robot.step(np.full(6, np.nan), 0.05)
