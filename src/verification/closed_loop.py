"""The retry loop: try a step, measure what really happened, adjust, and try again.

It only compares the robot's measured result with the plan taken from the video. It never
sees hand-written labels, so it can't cheat.
"""

import logging
from dataclasses import dataclass, replace
from typing import Protocol

from src.schemas.plan import Plan, Relation, Step
from src.simulation import primitives
from src.simulation.primitives import StepResult
from src.simulation.robot_env import MotionConfig, RobotEnv
from src.simulation.scene import SceneConfig

SQUEEZE_STEP = 0.01
SPEED_FACTOR = 0.7
logger = logging.getLogger(__name__)


class Robot(Protocol):
    def run_step(self, step: Step, motion: MotionConfig) -> StepResult: ...

    def measure(self, step: Step) -> Relation: ...


@dataclass(frozen=True)
class ClosedLoopResult:
    success: bool
    attempts: dict[str, int]
    failures: tuple[str, ...]

    @property
    def total_attempts(self) -> int:
        return sum(self.attempts.values())


def failure_reason(result: StepResult) -> str:
    return "pick" if not result.picked else "placement"


def next_motion(base: MotionConfig, reason: str, retry: int) -> MotionConfig:
    """Choose the adjustment: grip harder after a dropped pick, go slower after a bad place."""
    if reason == "pick":
        return replace(base, grip_squeeze=base.grip_squeeze + SQUEEZE_STEP * retry)
    return replace(base, hand_speed=base.hand_speed * SPEED_FACTOR**retry)


def _attempt_step(
    robot: Robot, step: Step, base: MotionConfig, limit: int
) -> tuple[bool, int, tuple[str, ...]]:
    motion = base
    failures: tuple[str, ...] = ()
    for attempt in range(1, limit + 1):
        result = robot.run_step(step, motion)
        if result.success:
            return True, attempt, failures
        reason = failure_reason(result)
        failures = (*failures, reason)
        logger.info("'%s' attempt %d failed: %s", step.object, attempt, reason)
        motion = next_motion(base, reason, attempt)
    return False, limit, failures


def run_closed_loop(
    robot: Robot, plan: Plan, base: MotionConfig, max_attempts: int
) -> ClosedLoopResult:
    """Run every step with retries, then redo any object that later got knocked away."""
    attempts: dict[str, int] = {}
    failures: tuple[str, ...] = ()
    for step in plan.steps:
        done, used, new_failures = _attempt_step(robot, step, base, max_attempts)
        attempts[step.object] = used
        failures = (*failures, *new_failures)
        if not done:
            return ClosedLoopResult(False, attempts, failures)

    for step in plan.steps:
        if robot.measure(step) is step.relation:
            continue
        failures = (*failures, "disturbed")
        done, used, new_failures = _attempt_step(robot, step, base, max_attempts)
        attempts[step.object] += used
        failures = (*failures, *new_failures)
        if not done:
            return ClosedLoopResult(False, attempts, failures)

    return ClosedLoopResult(True, attempts, failures)


class SimulatedRobot:
    """Connects the loop to the physics simulator."""

    def __init__(self, env: RobotEnv, config: SceneConfig) -> None:
        self._env = env
        self._config = config

    def run_step(self, step: Step, motion: MotionConfig) -> StepResult:
        self._env.motion = motion
        return primitives.run_step(self._env, self._config, step)

    def measure(self, step: Step) -> Relation:
        return primitives.measure(self._env, self._config, step)
