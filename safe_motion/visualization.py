"""Small 3D plot designed for the interview demo, not for control."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Replay is a command-line artifact generator; a non-interactive backend keeps
# it deterministic on laptops, CI and headless interview environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SafetyConfig
from .kinematics import chain_points
from .replay import ReplayResult


def _tcp_path(joint_trajectory: np.ndarray) -> np.ndarray:
    return np.vstack([chain_points(q)[-1] for q in joint_trajectory])


def _draw_box(ax, lower: np.ndarray, upper: np.ndarray) -> None:
    corners = np.array(
        [[x, y, z] for x in [lower[0], upper[0]]
         for y in [lower[1], upper[1]] for z in [lower[2], upper[2]]]
    )
    for i, p in enumerate(corners):
        for j, q in enumerate(corners):
            if j > i and np.count_nonzero(p != q) == 1:
                ax.plot(*zip(p, q), color="gray", alpha=0.45, linewidth=1)


def plot_replay(result: ReplayResult, config: SafetyConfig, path: str | Path) -> None:
    nominal = np.asarray(result.nominal_joint_trajectory)
    executed = np.asarray(result.executed_joint_trajectory)
    nominal_tcp = _tcp_path(nominal)
    executed_tcp = _tcp_path(executed)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _draw_box(ax, config.workspace.lower, config.workspace.upper)
    ax.plot(*nominal_tcp.T, color="tab:red", label="VLA nominal TCP", linewidth=2)
    ax.plot(*executed_tcp.T, color="tab:green", label="SafeMotion executed TCP", linewidth=2)

    nominal_final_chain = chain_points(nominal[-1])
    ax.plot(*nominal_final_chain.T, "o--", color="tab:red", alpha=0.7,
            label="Unsafe nominal final arm")
    final_chain = chain_points(executed[-1])
    ax.plot(*final_chain.T, "o-", color="tab:blue", label="Safe executed final arm")
    modified = [record.step for record in result.records if record.modified]
    if modified:
        index = modified[0] + 1
        ax.scatter(*executed_tcp[index], color="gold", edgecolor="black", s=70,
                   label=f"First intervention: step {modified[0]}")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("UR5 VLA trajectory: nominal vs fail-closed execution")
    ax.legend(loc="upper left")
    ax.set_box_aspect((1.4, 1.4, 0.85))
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
