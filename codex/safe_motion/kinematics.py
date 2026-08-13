"""Nominal UR5 forward kinematics using exactly the assignment DH table."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

A = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0], dtype=float)
D = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823], dtype=float)
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)


def dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Standard DH transform from frame i-1 to frame i."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def forward_kinematics(q: np.ndarray) -> OrderedDict[str, np.ndarray]:
    """Map six joint angles to base, joint and TCP Cartesian positions."""
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite vector with shape (6,)")

    transform = np.eye(4)
    result: OrderedDict[str, np.ndarray] = OrderedDict()
    result["base"] = transform[:3, 3].copy()
    for index in range(6):
        transform = transform @ dh_transform(A[index], D[index], ALPHA[index], q[index])
        result[f"joint_{index + 1}"] = transform[:3, 3].copy()
    # No additional tool offset is specified, so TCP equals the final DH frame.
    result["tcp"] = transform[:3, 3].copy()
    return result


def chain_points(q: np.ndarray) -> np.ndarray:
    """Return points in chain order; joint_6 and TCP may coincide by definition."""
    return np.vstack(list(forward_kinematics(q).values()))
