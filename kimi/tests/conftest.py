import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"


def load(name):
    return json.loads((SCENARIOS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def real_01():
    return load("real_01_lower_z_boundary.json")


@pytest.fixture(scope="session")
def real_02():
    return load("real_02_upper_z_boundary.json")


@pytest.fixture(scope="session")
def real_03():
    return load("real_03_tcp_safe_mid_link_unsafe.json")


@pytest.fixture()
def valid_scenario(real_01):
    """一份合法场景的深拷贝，供非法用例做变异。"""
    return copy.deepcopy(real_01)
