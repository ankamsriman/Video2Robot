"""Build the triangle with tall, real-size glasses, measuring and correcting each placement.

Taller glasses tip more easily than the short cups, so the robot levels each glass, aims
from where the glasses really are, and opens its fingers gently before lifting away.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.perception.hand_trajectory import CarryLimits, CarryProfile, robot_carry_path
from src.simulation.cup_flip import horizontal_error, rotation_between
from src.simulation.cup_task import SETTLE_AFTER_BUILD, TriangleResult, cup_pose
from src.simulation.primitives import pick_steps
from src.simulation.robot_env import RobotEnv, hand_pose_for_width
from src.simulation.scene import SceneConfig
from src.verification.steps import (
    StepCheck,
    glass_at,
    glass_held,
    glass_upright,
    others_still,
)
from src.verification.triangle import (
    TriangleConfig,
    triangle_failures,
    triangle_targets,
)

LOWER_SPEED = 0.04
AIM_ROUNDS = 3
AIM_TOLERANCE = 0.001
AIM_SECONDS = 1.0
LEVEL_ROUNDS = 2
LIFT_AFTER_RELEASE = 0.16
LEVEL_SECONDS = 1.0
LEVEL_SPEED = 0.3
logger = logging.getLogger(__name__)


def _positions(env: RobotEnv, names: tuple[str, ...], moving: str) -> dict:
    """Where every glass but the one being moved is now."""
    return {name: env.object_position(name).copy() for name in names if name != moving}


@dataclass(frozen=True)
class CopiedCarries:
    """The hand's path for each glass, read from a video, and what keeps them safe."""

    profiles: tuple[CarryProfile | None, CarryProfile | None, CarryProfile | None]
    limits: CarryLimits


def _level(env: RobotEnv, name: str) -> None:
    """Measure how far the held glass leans, and turn the hand to make it stand straight."""
    for _ in range(LEVEL_ROUNDS):
        rotation = env.data.xmat[env.model.body(name).id].reshape(3, 3)
        correction = rotation_between(rotation[:, 2], np.array([0.0, 0.0, 1.0]))
        hand = env.data.site_xmat[env.hand_site].reshape(3, 3)
        env.move_hand(
            env.hand_position,
            rotation=correction @ hand,
            twist_speed=LEVEL_SPEED,
            keep_waist=True,
            min_seconds=LEVEL_SECONDS,
        )


def _aim(env: RobotEnv, name: str, target: np.ndarray) -> None:
    """Measure the held glass, and nudge the hand until it's over the target."""
    for _ in range(AIM_ROUNDS):
        error = horizontal_error(env.object_position(name), target)
        if float(np.linalg.norm(error)) < AIM_TOLERANCE:
            return
        env.move_hand(
            env.hand_position + error,
            speed=LOWER_SPEED,
            keep_waist=True,
            min_seconds=AIM_SECONDS,
        )


def place_glass(
    env: RobotEnv,
    config: SceneConfig,
    name: str,
    target: np.ndarray,
    profile: CarryProfile | None = None,
    limits: CarryLimits | None = None,
) -> tuple[float, tuple[StepCheck, ...]]:
    """Carry the glass over the target, set it down accurately, and let go gently.

    With a profile, the carry follows the path copied from a video. Returns how much of that
    path's height had to be limited, from 0 to 1, and checks of the carry, place and release.
    """
    table = config.table.top_height
    hover = table + env.motion.hover_height
    offset = env.hand_position - env.object_position(name)
    over_target = np.array([target[0] + offset[0], target[1] + offset[1], hover])
    limited = 0.0
    if profile is None or limits is None:
        env.move_hand(over_target)
    else:
        carry = robot_carry_path(profile, env.hand_position, over_target, table, limits)
        env.follow_path(carry.points, carry.seconds)
        limited = carry.limited_fraction
    env.advance(env.motion.settle_time)
    carried = glass_held(name, "carry", env.object_position(name), env.hand_position)
    _level(env, name)
    _aim(env, name, target)
    offset = env.hand_position - env.object_position(name)
    final = target + offset + [0.0, 0.0, env.motion.release_clearance]
    env.move_hand(final, speed=LOWER_SPEED, keep_waist=True)
    env.advance(env.motion.settle_time)
    _aim(env, name, target)
    placed = glass_at(name, "place", env.object_position(name), target)
    env.relax_fingers()
    env.advance(env.motion.settle_time)
    width = next(item.size[0] for item in config.objects if item.name == name)
    env.set_fingers(hand_pose_for_width(width + env.motion.release_margin))
    env.advance(env.motion.settle_time)
    # A glass on top of the stack reaches almost to the hover height, so the hand must
    # rise well above it, or the fingers knock it over on the way back.
    lifted = env.hand_position + [0.0, 0.0, LIFT_AFTER_RELEASE]
    env.move_hand(lifted, speed=LOWER_SPEED, keep_waist=True)
    released = glass_upright(name, "release", cup_pose(env, name).up)
    return limited, (carried, placed, released)


def build_glass_triangle(
    env: RobotEnv,
    config: SceneConfig,
    triangle: TriangleConfig,
    order: tuple[str, str, str],
    arms: tuple[str, str, str] | None = None,
    carries: CopiedCarries | None = None,
) -> TriangleResult:
    """Place two glasses side by side, then the third across their measured tops.

    If `arms` is given, each glass uses its own arm. Otherwise one arm does them all.
    If `carries` is given, each glass is carried along the path copied from a video.
    """
    table = config.table.top_height
    targets = tuple(np.array(t) for t in triangle_targets(triangle, table))
    picked: tuple[bool, ...] = ()
    limited: tuple[float, ...] = ()
    steps: tuple[StepCheck, ...] = ()
    for index, (name, target) in enumerate(zip(order, targets, strict=True)):
        if arms is not None:
            env.use_arm(arms[index])
        others = _positions(env, order, name)
        reach, grip, lift = pick_steps(env, config, name)
        held = grip.passed and lift.passed
        picked = (*picked, held)
        steps = (*steps, reach, grip, lift)
        if not held:
            logger.warning("Could not pick '%s'", name)
            break
        if index == 2:
            lower = (env.object_position(order[0]) + env.object_position(order[1])) / 2
            target = np.array([lower[0], lower[1], target[2]])
        profile = carries.profiles[index] if carries else None
        amount, placing = place_glass(
            env, config, name, target, profile, carries.limits if carries else None
        )
        limited = (*limited, amount)
        steps = (
            *steps,
            *placing,
            others_still(name, others, _positions(env, order, name)),
        )
    env.move_hand(np.array(env.motion.home_position))
    env.advance(SETTLE_AFTER_BUILD)
    left, right, top = (cup_pose(env, name) for name in order)
    failures = triangle_failures(left, right, top, triangle, table)
    return TriangleResult(picked, failures, limited, steps)
