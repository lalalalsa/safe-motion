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
from .safety_filter import (
    CandidateAttempt,
    FilterResult,
    command_is_safe,
    nominal_velocity,
    safety_filter,
)
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
    attempts: list[CandidateAttempt]
    minimum_margin: float
    q_after: list[float]


@dataclass
class ReplayResult:
    total_steps: int
    executed_steps: int
    modified_steps: int
    stopped_steps: int
    minimum_workspace_margin: float
    nominal_minimum_workspace_margin: float
    maximum_joint_velocity: float
    final_joint_state: list[float]
    nominal_joint_trajectory: list[list[float]]
    executed_joint_trajectory: list[list[float]]
    nominal_workspace_margins: list[float]
    executed_workspace_margins: list[float]
    records: list[StepRecord]

    def summary(self) -> dict:
        data = asdict(self)
        data.pop("records")
        data.pop("nominal_joint_trajectory")
        data.pop("executed_joint_trajectory")
        data.pop("nominal_workspace_margins")
        data.pop("executed_workspace_margins")
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
    executed_margins = [initial_safety.minimum_margin]
    minimum_margin = initial_safety.minimum_margin
    maximum_velocity = 0.0

    for step, target in enumerate(action_chunk):
        q_before = robot.get_joint_state()
        q_dot_nom = nominal_velocity(q_before, target, dt, config)
        try:
            decision = safety_filter(q_before, q_dot_nom, dt, config)
            # Validate both command safety data and the scaling-policy contract
            # before any field can be trusted by the robot or audit log.
            if not isinstance(decision, FilterResult):
                raise ValueError("invalid filter output")
            if decision.velocity.shape != (6,) or not np.all(np.isfinite(decision.velocity)):
                raise ValueError("invalid filter output")
            if not np.isfinite(decision.scale) or not 0.0 <= decision.scale <= 1.0:
                raise ValueError("invalid filter scale")
            if not np.allclose(
                decision.velocity, decision.scale * q_dot_nom, rtol=0.0, atol=1e-12
            ):
                raise ValueError("filter output violates scaling policy")
            if decision.modified != (decision.scale < 1.0):
                raise ValueError("inconsistent modified flag")
            if decision.stopped != (decision.scale == 0.0):
                raise ValueError("inconsistent stopped flag")
        except Exception as exc:
            decision = FilterResult(
                np.zeros(6), 0.0, True, True,
                f"filter_error:{type(exc).__name__}", float("-inf")
            )

        # Treat the filter as an untrusted component at the command boundary.
        # A plausible-looking but unsafe command is replaced with zero velocity
        # before it can reach the robot. The post-step check below remains an
        # invariant audit rather than the first time danger is discovered.
        try:
            verification = command_is_safe(q_before, decision.velocity, dt, config)
        except Exception as exc:
            verification_reason = f"recheck_error:{type(exc).__name__}"
        else:
            verification_reason = None if verification.safe else verification.reason
        if verification_reason is not None:
            decision = FilterResult(
                np.zeros(6),
                0.0,
                True,
                True,
                f"pre_execution_recheck_failed:{verification_reason}",
                float("-inf"),
                decision.attempts,
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
        executed_margins.append(post.minimum_margin)
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
                attempts=list(decision.attempts),
                minimum_margin=post.minimum_margin,
                q_after=q_after.tolist(),
            )
        )

    nominal_margins = [
        check_full_body(target, config).minimum_margin for target in action_chunk
    ]

    return ReplayResult(
        total_steps=50,
        executed_steps=len(records),
        modified_steps=sum(record.modified for record in records),
        stopped_steps=sum(record.stopped for record in records),
        minimum_workspace_margin=float(minimum_margin),
        nominal_minimum_workspace_margin=float(min(nominal_margins)),
        maximum_joint_velocity=float(maximum_velocity),
        final_joint_state=robot.get_joint_state().tolist(),
        nominal_joint_trajectory=action_chunk.tolist(),
        executed_joint_trajectory=np.asarray(executed).tolist(),
        nominal_workspace_margins=[float(value) for value in nominal_margins],
        executed_workspace_margins=[float(value) for value in executed_margins],
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


def print_explanation(result: ReplayResult, step: int) -> None:
    """Print the complete decision trail for one control step."""
    if step < 0 or step >= len(result.records):
        raise ValueError(f"step must be in [0, {len(result.records) - 1}], got {step}")

    record = result.records[step]
    print(f"\n=== Safety intervention at step {step} ===")
    print(f"q_before     : {np.round(record.q_before, 5).tolist()}")
    print(f"VLA target   : {np.round(record.target, 5).tolist()}")
    print(f"q_dot_nominal: {np.round(record.q_dot_nominal, 5).tolist()}")
    print("candidate evaluations:")
    for attempt in record.attempts:
        verdict = "SAFE" if attempt.safe else "UNSAFE"
        margin = (
            f"{attempt.minimum_margin:+.6f} m"
            if np.isfinite(attempt.minimum_margin)
            else "n/a"
        )
        node = (
            str(attempt.offending_node + 1)
            if attempt.offending_node is not None
            else "-"
        )
        print(
            f"  {attempt.stage:9s} scale={attempt.scale:.6f} "
            f"{verdict:6s} margin={margin} node={node} reason={attempt.reason}"
        )
    print(f"decision     : {record.reason}, scale={record.scale:.6f}")
    print(f"q_dot_safe   : {np.round(record.q_dot_safe, 5).tolist()}")
    print(f"post margin  : {record.minimum_margin:+.6f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--margin-plot", type=Path)
    parser.add_argument("--explain", type=int, metavar="STEP")
    args = parser.parse_args()

    data = load_scenario(args.scenario)
    result = run_scenario(data)
    print(json.dumps(result.summary(), indent=2))
    if args.output:
        save_replay(result, args.output)
    if args.plot:
        from .visualization import plot_replay

        plot_replay(result, SafetyConfig.from_scenario(data), args.plot)
    if args.margin_plot:
        from .visualization import plot_workspace_margins

        plot_workspace_margins(result, args.margin_plot)
    if args.explain is not None:
        print_explanation(result, args.explain)


if __name__ == "__main__":
    main()
