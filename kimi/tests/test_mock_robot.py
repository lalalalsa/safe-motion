"""Mock Robot 契约：题目统一定义的行为。"""
import numpy as np
import pytest

from safe_motion.config import RobotConfig
from safe_motion.mock_robot import MockRobot

LIMITS = RobotConfig.ur5().joint_limits


def test_step_integrates():
    r = MockRobot(np.zeros(6), LIMITS)
    q = r.step(np.full(6, 0.2), 0.05)
    np.testing.assert_allclose(q, np.full(6, 0.01), atol=1e-15)
    np.testing.assert_allclose(r.get_joint_state(), q, atol=1e-15)


def test_step_wrong_shape_raises():
    r = MockRobot(np.zeros(6), LIMITS)
    with pytest.raises(ValueError, match="shape"):
        r.step(np.zeros(5), 0.05)


def test_step_non_finite_raises():
    r = MockRobot(np.zeros(6), LIMITS)
    with pytest.raises(ValueError, match="non-finite"):
        r.step([np.nan] * 6, 0.05)


def test_step_joint_limit_violation_raises():
    r = MockRobot(np.full(6, 6.28), LIMITS)
    with pytest.raises(ValueError, match="joint limit"):
        r.step(np.full(6, 1.0), 0.05)
    # 状态未被破坏
    np.testing.assert_allclose(r.get_joint_state(), np.full(6, 6.28),
                               atol=1e-15)


def test_init_out_of_limits_raises():
    with pytest.raises(ValueError):
        MockRobot(np.full(6, 7.0), LIMITS)
