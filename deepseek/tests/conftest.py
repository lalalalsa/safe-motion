"""测试公共 fixture：场景路径、加载工具。"""
import os

import pytest

from safe_motion import config
from safe_motion.replay import load_scenario, run_scenario

SCENARIOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenarios")


def scenario_path(name):
    return os.path.join(SCENARIOS, name)


@pytest.fixture
def workspace():
    return config.DEFAULT_WORKSPACE


@pytest.fixture
def joint_limits():
    return config.JOINT_LIMITS


@pytest.fixture
def real_01():
    return load_scenario(scenario_path("real_01_lower_z_boundary.json"))


@pytest.fixture
def real_02():
    return load_scenario(scenario_path("real_02_upper_z_boundary.json"))


@pytest.fixture
def real_03():
    return load_scenario(scenario_path("real_03_tcp_safe_mid_link_unsafe.json"))


@pytest.fixture
def free_space():
    return load_scenario(scenario_path("free_space.json"))


@pytest.fixture
def joint_limit():
    return load_scenario(scenario_path("joint_limit.json"))
