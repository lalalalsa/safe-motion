from dataclasses import replace

import numpy as np
import pytest

from safe_motion.config import SafetyConfig, Workspace
from safe_motion.validation import InputValidationError, validate_config, validate_scenario


def valid_data():
    q = np.array([0.0, -1.35, 1.65, -1.55, -1.57, 0.0])
    return {"action_hz": 20.0, "joint_state": q, "action_chunk": np.tile(q, (50, 1))}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_rejected(value):
    data = valid_data()
    data["action_chunk"][3, 2] = value
    with pytest.raises(InputValidationError):
        validate_scenario(data, SafetyConfig())


@pytest.mark.parametrize("shape", [(49, 6), (50, 5), (300,)])
def test_wrong_chunk_shape_is_rejected(shape):
    data = valid_data()
    data["action_chunk"] = np.zeros(shape)
    with pytest.raises(InputValidationError):
        validate_scenario(data, SafetyConfig())


def test_large_jump_is_rejected():
    data = valid_data()
    data["action_chunk"][20:, 0] += 1.0
    with pytest.raises(InputValidationError, match="jump"):
        validate_scenario(data, SafetyConfig())


@pytest.mark.parametrize(
    "config",
    [
        replace(SafetyConfig(), max_joint_velocity=np.full(6, -1.0)),
        replace(SafetyConfig(), max_joint_velocity=np.array([1.0, 1.0])),
        replace(SafetyConfig(), max_joint_velocity=np.array([1, 1, 1, np.nan, 1, 1])),
        replace(SafetyConfig(), workspace=Workspace(x=(0.7, -0.7))),
        replace(SafetyConfig(), path_substeps=0),
        replace(SafetyConfig(), safety_margin=-0.1),
        replace(SafetyConfig(), velocity_scales=(1.0, 0.5)),
        replace(SafetyConfig(), first_checked_node=2),
    ],
)
def test_malformed_safety_config_is_rejected(config):
    with pytest.raises(InputValidationError):
        validate_config(config)
