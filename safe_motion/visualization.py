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


def plot_workspace_margins(result: ReplayResult, path: str | Path) -> None:
    """Plot nominal danger against the margin of states actually executed."""
    nominal = np.asarray(result.nominal_workspace_margins, dtype=float)
    # Drop the initial state so both series align at target/control-step indices
    # 0..49. The full 51-state history remains available in ReplayResult.
    executed = np.asarray(result.executed_workspace_margins[1:], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1,
               label="Configured safety boundary")
    nominal_steps = np.arange(len(nominal))
    executed_steps = np.arange(len(executed))
    ax.plot(nominal_steps, nominal, "o-", color="tab:red", markersize=3,
            linewidth=1.4, label="VLA nominal full-body margin")
    ax.plot(executed_steps, executed, "o-", color="tab:green", markersize=3,
            linewidth=1.8, label="SafeMotion executed margin")
    ax.fill_between(nominal_steps, nominal, 0.0, where=nominal < 0.0,
                    color="tab:red", alpha=0.15)
    ax.set_xlabel("control step")
    ax.set_ylabel("minimum full-body margin [m]")
    ax.set_title("Full-body workspace margin: nominal vs executed")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
