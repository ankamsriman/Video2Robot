"""The robot's two actions, pick and place, and running a whole plan with them."""

import logging
from dataclasses import dataclass

import numpy as np

from src.schemas.plan import Plan, Relation, Step
from src.simulation.robot_env import RobotEnv, hand_pose_for_width
from src.simulation.scene import ItemConfig, SceneConfig
from src.verification.relations import (
    Box,
    classify_relation,
    free_slot,
    placement_target,
)

APPROACH_SPEED = 0.10
MIN_LIFT = 0.05
MAX_HOLD_DISTANCE = 0.06
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    step: Step
    picked: bool
    reached: bool
    measured_relation: Relation | None

    @property
    def success(self) -> bool:
        """Judged on where the object ended up. `reached` is kept as a diagnostic only."""
        return self.picked and self.measured_relation is self.step.relation


@dataclass(frozen=True)
class ExecutionResult:
    plan: Plan
    steps: tuple[StepResult, ...]
    final: tuple[Relation, ...]

    @property
    def success(self) -> bool:
        """Every step worked, and everything is still where it belongs at the end."""
        expected = tuple(step.relation for step in self.plan.steps)
        completed = len(self.steps) == len(self.plan.steps)
        return (
            completed
            and all(step.success for step in self.steps)
            and self.final == expected
        )


def _item(config: SceneConfig, name: str) -> ItemConfig:
    for item in (*config.objects, *config.containers):
        if item.name == name:
            return item
    raise KeyError(f"'{name}' is not in the scene")


def _is_container(config: SceneConfig, name: str) -> bool:
    return any(item.name == name for item in config.containers)


def _live_box(env: RobotEnv, config: SceneConfig, name: str) -> Box:
    """Describe an item's current box. Container bodies are based at the table top."""
    item = _item(config, name)
    half = tuple(length / 2 for length in item.size)
    x, y, z = env.object_position(name)
    center_z = z + half[2] if _is_container(config, name) else z
    return Box(center=(float(x), float(y), float(center_z)), half_size=half)


def _hover_position(x: float, y: float, env: RobotEnv, table: float) -> np.ndarray:
    return np.array([x, y, table + env.motion.hover_height])


def pick(env: RobotEnv, config: SceneConfig, name: str) -> bool:
    """Grasp an object from above and lift it. Returns whether it is really held."""
    table = config.table.top_height
    x, y, start_z = env.object_position(name)
    width = _item(config, name).size[0]
    env.set_fingers(hand_pose_for_width(env.motion.open_width))
    reached = env.move_hand(_hover_position(x, y, env, table))
    reached &= env.move_hand(
        np.array([x, y, table + env.motion.grip_height]), APPROACH_SPEED
    )
    env.set_fingers(hand_pose_for_width(width - env.motion.grip_squeeze))
    env.advance(env.motion.settle_time)
    reached &= env.move_hand(_hover_position(x, y, env, table), APPROACH_SPEED)

    position = env.object_position(name)
    lifted = position[2] - start_z >= MIN_LIFT
    close = np.linalg.norm(position - env.hand_position) <= MAX_HOLD_DISTANCE
    if not reached:
        logger.info("Hand fell short while picking '%s'", name)
    return bool(lifted and close)


def _target_xyz(
    env: RobotEnv,
    config: SceneConfig,
    step: Step,
    placed: tuple[str, ...],
) -> tuple[float, float, float]:
    obj = _live_box(env, config, step.object)
    reference = _live_box(env, config, step.reference)
    x, y, z = placement_target(obj, reference, step.relation, env.motion.side_gap)
    if step.relation is Relation.IN:
        occupants = tuple(_live_box(env, config, name) for name in placed)
        x, y = free_slot(obj, reference, occupants)
    return (x, y, z)


def place(
    env: RobotEnv, config: SceneConfig, step: Step, placed: tuple[str, ...] = ()
) -> bool:
    """Carry the held object to its target, lower it, and let go."""
    table = config.table.top_height
    x, y, z = _target_xyz(env, config, step, placed)
    grip_offset = env.hand_position - env.object_position(step.object)
    hand_x, hand_y = x + grip_offset[0], y + grip_offset[1]

    reached = env.move_hand(_hover_position(hand_x, hand_y, env, table))
    drop_z = z + grip_offset[2] + env.motion.release_clearance
    reached &= env.move_hand(np.array([hand_x, hand_y, drop_z]), APPROACH_SPEED)
    width = _item(config, step.object).size[0]
    env.set_fingers(hand_pose_for_width(width + env.motion.release_margin))
    env.advance(env.motion.settle_time)
    reached &= env.move_hand(
        _hover_position(hand_x, hand_y, env, table), APPROACH_SPEED
    )
    env.set_fingers(hand_pose_for_width(env.motion.open_width))
    return bool(reached)


def measure(env: RobotEnv, config: SceneConfig, step: Step) -> Relation:
    return classify_relation(
        _live_box(env, config, step.object),
        _live_box(env, config, step.reference),
        container=_is_container(config, step.reference),
    )


def final_relations(
    env: RobotEnv, config: SceneConfig, plan: Plan
) -> tuple[Relation, ...]:
    """Measure, after the fact, where every planned object actually ended up."""
    return tuple(measure(env, config, step) for step in plan.steps)


def occupants(
    env: RobotEnv, config: SceneConfig, reference: str, exclude: str
) -> tuple[str, ...]:
    """Find which objects are really inside a container right now, by measuring."""
    container = _live_box(env, config, reference)
    return tuple(
        item.name
        for item in config.objects
        if item.name != exclude
        and classify_relation(_live_box(env, config, item.name), container)
        is Relation.IN
    )


def run_step(env: RobotEnv, config: SceneConfig, step: Step) -> StepResult:
    """Pick and place one object, then measure where it really ended up."""
    placed = ()
    if _is_container(config, step.reference):
        placed = occupants(env, config, step.reference, step.object)
    if not pick(env, config, step.object):
        logger.warning("Could not pick '%s'", step.object)
        return StepResult(step, picked=False, reached=False, measured_relation=None)
    try:
        reached = place(env, config, step, placed)
    except ValueError as error:
        logger.warning("Could not place '%s': %s", step.object, error)
        return StepResult(step, picked=True, reached=False, measured_relation=None)
    measured = measure(env, config, step)
    return StepResult(step, picked=True, reached=reached, measured_relation=measured)


def execute_plan(env: RobotEnv, config: SceneConfig, plan: Plan) -> ExecutionResult:
    """Run all steps once, with no retries, stopping at the first failure."""
    results: tuple[StepResult, ...] = ()
    for step in plan.steps:
        outcome = run_step(env, config, step)
        results = (*results, outcome)
        logger.info("%s -> %s", step, "ok" if outcome.success else "FAILED")
        if not outcome.success:
            break
    return ExecutionResult(plan, results, final_relations(env, config, plan))
