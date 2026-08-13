"""Generate deterministic 50-point scenarios through the public interfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
Q_HOME = np.array([0.0, -1.35, 1.65, -1.55, -1.57, 0.0])


def write(name: str, q0: np.ndarray, targets: np.ndarray, **extra) -> None:
    data = {
        "action_hz": 20.0,
        "joint_state": q0.tolist(),
        "action_chunk": targets.tolist(),
        "workspace": {"x": [-0.70, 0.70], "y": [-0.70, 0.70], "z": [0.05, 0.90]},
        "jump_threshold": 0.75,
        **extra,
    }
    (ROOT / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    t = np.linspace(0.0, 1.0, 50)
    free = Q_HOME + np.column_stack(
        [0.12 * t, -0.08 * t, 0.08 * t, -0.06 * t, 0.03 * t, 0.10 * t]
    )
    write("free_space.json", Q_HOME, free)

    # A broad, smooth shoulder/elbow move. It is legal in joint space but the
    # filter must stop/scale when its modeled arm approaches a workspace face.
    boundary_goal = Q_HOME + np.array([0.0, 0.62, -0.55, 0.25, 0.0, 0.0])
    boundary = Q_HOME + t[:, None] * (boundary_goal - Q_HOME)
    write("workspace_boundary.json", Q_HOME, boundary)


if __name__ == "__main__":
    main()
