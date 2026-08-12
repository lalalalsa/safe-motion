"""Closed-loop replay: validate once, then filter and execute exactly 50 targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import SafetyConfig
from .geometry import check_full_body
from .kinematics import chain_points
from .mock_robot import MockRobot
from .safety_filter import FilterResult, nominal_velocity, safety_filter
from .validation import validate_scenario


@dataclass
class StepRecord:
    step: int
    target: list[float]
    q_before: list[float]
    q_dot_nominal: list[float]
    q_dot_safe: list[float]
    scale: float
    modified: bool
    stopped: bool
    reason: str
    minimum_margin: float
    q_after: list[float]


@dataclass
class ReplayResult:
    total_steps: int
    executed_steps: int
    modified_steps: int
    stopped_steps: int
    minimum_workspace_margin: float
    maximum_joint_velocity: float
    final_joint_state: list[float]
    nominal_joint_trajectory: list[list[float]]
    executed_joint_trajectory: list[list[float]]
    records: list[StepRecord]

    def summary(self) -> dict:
        data = asdict(self)
        data.pop("records")
        data.pop("nominal_joint_trajectory")
        data.pop("executed_joint_trajectory")
        return data


def run_scenario(data: dict, config: SafetyConfig | None = None) -> ReplayResult:
    """Execute a scenario while preserving fail-closed safety invariants."""
    config = config or SafetyConfig.from_scenario(data)
    q0, action_chunk, action_hz = validate_scenario(data, config)
    dt = 1.0 / action_hz

    initial_safety = check_full_body(q0, config)
    if not initial_safety.safe:
        raise ValueError(
            f"unsafe initial state: {initial_safety.reason}, "
            f"margin={initial_safety.minimum_margin:.6f}"
        )

    robot = MockRobot(q0, config.joint_limits)
    records: list[StepRecord] = []
    executed = [q0.copy()]
    minimum_margin = initial_safety.minimum_margin
    maximum_velocity = 0.0

    for step, target in enumerate(action_chunk):
        q_before = robot.get_joint_state()
        q_dot_nom = nominal_velocity(q_before, target, dt, config)
        try:
            decision = safety_filter(q_before, q_dot_nom, dt, config)
            # Defence in depth: a malformed filter result is treated as failure.
            if decision.velocity.shape != (6,) or not np.all(np.isfinite(decision.velocity)):
                raise ValueError("invalid filter output")
        except Exception as exc:
            decision = FilterResult(
                np.zeros(6), 0.0, True, True,
                f"filter_error:{type(exc).__name__}", float("-inf")
            )

        q_after = robot.step(decision.velocity, dt)
        post = check_full_body(q_after, config)
        if not post.safe:
            # Reaching this branch means an invariant in the filter/robot model
            # is broken. Do not continue a trajectory after detecting it.
            raise RuntimeError(
                f"post-execution safety invariant failed at step {step}: {post.reason}"
            )

        minimum_margin = min(minimum_margin, post.minimum_margin)
        maximum_velocity = max(maximum_velocity, float(np.max(np.abs(decision.velocity))))
        executed.append(q_after.copy())
        records.append(
            StepRecord(
                step=step,
                target=target.tolist(),
                q_before=q_before.tolist(),
                q_dot_nominal=q_dot_nom.tolist(),
                q_dot_safe=decision.velocity.tolist(),
                scale=decision.scale,
                modified=decision.modified,
                stopped=decision.stopped,
                reason=decision.reason,
                minimum_margin=post.minimum_margin,
                q_after=q_after.tolist(),
            )
        )

    return ReplayResult(
        total_steps=50,
        executed_steps=len(records),
        modified_steps=sum(record.modified for record in records),
        stopped_steps=sum(record.stopped for record in records),
        minimum_workspace_margin=float(minimum_margin),
        maximum_joint_velocity=float(maximum_velocity),
        final_joint_state=robot.get_joint_state().tolist(),
        nominal_joint_trajectory=action_chunk.tolist(),
        executed_joint_trajectory=np.asarray(executed).tolist(),
        records=records,
    )


def load_scenario(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_replay(result: ReplayResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    data = load_scenario(args.scenario)
    result = run_scenario(data)
    print(json.dumps(result.summary(), indent=2))
    if args.output:
        save_replay(result, args.output)
    if args.plot:
        from .visualization import plot_replay

        plot_replay(result, SafetyConfig.from_scenario(data), args.plot)


if __name__ == "__main__":
    main()
