from dataclasses import replace

import numpy as np

from safe_motion.config import SafetyConfig, Workspace
from safe_motion.geometry import check_full_body
from safe_motion.kinematics import chain_points


Q_HOME = np.array([0.0, -1.35, 1.65, -1.55, -1.57, 0.0])


def test_home_state_is_safe():
    assert check_full_body(Q_HOME, SafetyConfig()).safe


def test_tcp_inside_does_not_hide_middle_link_violation():
    # Deterministically search a small grid for the semantic condition; the
    # production checker itself contains no scenario-specific coordinates.
    found = None
    for q2 in np.linspace(-2.4, -0.2, 16):
        for q3 in np.linspace(0.2, 2.6, 16):
            q = np.array([0.0, q2, q3, -1.2, -1.57, 0.0])
            points = chain_points(q)[1:]
            tcp = points[-1]
            ws = Workspace(x=(-0.7, 0.7), y=(-0.7, 0.7), z=(0.05, 0.90))
            tcp_safe = np.all(tcp >= ws.lower) and np.all(tcp <= ws.upper)
            result = check_full_body(q, replace(SafetyConfig(), workspace=ws))
            if tcp_safe and not result.safe:
                found = result
                break
        if found:
            break
    assert found is not None
    assert found.reason == "node_or_link_outside_workspace"
