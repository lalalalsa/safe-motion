"""Mock Robot 行为测试。"""
import numpy as np
import pytest

from safe_motion.mock_robot import MockRobot, inside_joint_limits
from safe_motion import config


def test_step_basic():
    r = MockRobot(np.zeros(6), config.JOINT_LIMITS)
    q = r.step(np.ones(6) * 0.1, 0.05)
    np.testing.assert_allclose(q, np.ones(6) * 0.005, atol=1e-12)


def test_step_wrong_shape_raises():
    r = MockRobot(np.zeros(6), config.JOINT_LIMITS)
    with pytest.raises(ValueError):
        r.step(np.zeros(5), 0.05)


def test_step_non_finite_raises():
    r = MockRobot(np.zeros(6), config.JOINT_LIMITS)
    with pytest.raises(ValueError):
        r.step(np.array([np.nan, 0, 0, 0, 0, 0]), 0.05)


def test_step_joint_limit_violation_raises():
    limits = np.array([[-1.0, 1.0]] * 6)
    r = MockRobot(np.array([0.9, 0, 0, 0, 0, 0]), limits)
    with pytest.raises(ValueError):
        r.step(np.array([10.0, 0, 0, 0, 0, 0]), 0.05)  # 会越出 1.0


def test_inside_joint_limits():
    limits = np.array([[-1.0, 1.0]] * 6)
    assert inside_joint_limits(np.zeros(6), limits)
    assert not inside_joint_limits(np.array([1.5, 0, 0, 0, 0, 0]), limits)
