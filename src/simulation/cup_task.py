"""Build the cup triangle in physics: two cups side by side, and a third across them."""

import logging
from dataclasses import dataclass

import numpy as np

from src.simulation.primitives import pick
from src.simulation.robot_env import RobotEnv, hand_pose_for_width
from src.simulation.scene import SceneConfig
from src.verification.steps import StepCheck
from src.verification.triangle import (
    CupPose,
    TriangleConfig,
    triangle_failures,
    triangle_targets,
)

LOWER_SPEED = 0.06
SETTLE_AFTER_BUILD = 2.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriangleResult:
    picked: tuple[bool, ...]
    failures: tuple[str, ...]
    carry_limited: tuple[float, ...] = ()
    steps: tuple[StepCheck, ...] = ()

    @property
    def success(self) -> bool:
        return all(self.picked) and not self.failures


def cup_pose(env: RobotEnv, name: str) -> CupPose:
    rotation = env.data.xmat[env.model.body(name).id].reshape(3, 3)
    return CupPose(
        position=tuple(float(v) for v in env.object_position(name)),
        up=tuple(float(v) for v in rotation[:, 2]),
    )


def place_at(
    env: RobotEnv, config: SceneConfig, name: str, target: tuple[float, float, float]
) -> None:
    """Lower the held cup so its centre reaches the target, then let go and lift clear."""
    table = config.table.top_height
    offset = env.hand_position - env.object_position(name)
    x, y, z = target[0] + offset[0], target[1] + offset[1], target[2] + offset[2]
    width = next(item.size[0] for item in config.objects if item.name == name)
    hover = table + env.motion.hover_height
    env.move_hand(np.array([x, y, hover]))
    env.move_hand(np.array([x, y, z + env.motion.release_clearance]), speed=LOWER_SPEED)
    env.set_fingers(hand_pose_for_width(width + env.motion.release_margin))
    env.advance(env.motion.settle_time)
    env.move_hand(np.array([x, y, hover]), speed=LOWER_SPEED)


def build_triangle(
    env: RobotEnv,
    config: SceneConfig,
    triangle: TriangleConfig,
    order: tuple[str, str, str],
) -> TriangleResult:
    """Place the first two cups side by side, then the third on top, and measure."""
    table = config.table.top_height
    targets = triangle_targets(triangle, table)
    picked: tuple[bool, ...] = ()
    for name, target in zip(order, targets, strict=True):
        held = pick(env, config, name)
        picked = (*picked, held)
        if not held:
            logger.warning("Could not pick '%s'", name)
            break
        place_at(env, config, name, target)
    home = np.array(env.motion.home_position)
    env.move_hand(home)
    env.advance(SETTLE_AFTER_BUILD)
    left, right, top = (cup_pose(env, name) for name in order)
    return TriangleResult(picked, triangle_failures(left, right, top, triangle, table))
