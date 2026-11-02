"""Flip a cup with one hand and stand it opening-up on the triangle.

The hand grips with the palm beside the cup, not above it. Twisting the wrist 180 degrees
turns the cup over, like a spit, and the palm ends on the cup's other side, never under it.
So nothing is in the way when the cup is lowered. The flip is programmed, not learned.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from src.simulation.cup_task import TriangleResult, build_triangle, cup_pose
from src.simulation.ik import palm_down_rotation
from src.simulation.robot_env import RobotEnv
from src.simulation.scene import SceneConfig
from src.verification.triangle import (
    CupPose,
    TriangleConfig,
    triangle_failures,
)

WRIST_ROLL_JOINT = "right_wrist_roll_joint"
MAX_FLIP_ANGLE_DEGREES = 15.0
RESTING_TOLERANCE = 0.015
CENTRING_TOLERANCE = 0.02
SLOW_SPEED = 0.06
LEVEL_ROUNDS = 2
LEVEL_SECONDS = 2.5
LEVEL_SPEED = 0.3
AXIS_EPSILON = 1e-6
SETTLE_AFTER_BUILD = 2.0
MIN_LIFT = 0.05
MAX_HOLD_DISTANCE = 0.06
AIM_ROUNDS = 3
AIM_TOLERANCE = 0.001
AIM_SECONDS = 1.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlipConfig:
    grip_height: float
    grip_depth: float
    approach_clearance: float
    open_pose: tuple[float, ...]
    close_pose: tuple[float, ...]
    release_pose: tuple[float, ...]
    start_roll: float
    hand_yaw: float
    twist_angle: float
    twist_pose: Mapping[str, float]
    pose_seconds: float
    twist_seconds: float
    hover_height: float
    twist_speed: float
    place_clearance: float
    release_drift: tuple[float, float]
    lower_speed: float
    back_out_distance: float
    lift_after_release: float
    finish_position: tuple[float, float, float]


@dataclass(frozen=True)
class FlipResult:
    triangle: TriangleResult
    gripped: bool
    failures: tuple[str, ...]

    @property
    def success(self) -> bool:
        return self.gripped and not self.failures


def load_flip_config(path: Path) -> FlipConfig:
    raw = yaml.safe_load(path.read_text())["flip"]
    poses = (
        "open_pose",
        "close_pose",
        "release_pose",
        "finish_position",
        "release_drift",
    )
    return FlipConfig(**{**raw, **{name: tuple(raw[name]) for name in poses}})


def hand_rotation(roll: float, yaw: float = 0.0) -> np.ndarray:
    """A level hand, turned `roll` radians around its length from palm-down.

    `yaw` points it sideways. The cup is round, so any level direction flips it equally.
    """
    turn = Rotation.from_euler("x", roll).as_matrix()
    heading = Rotation.from_euler("z", yaw).as_matrix()
    return heading @ turn @ palm_down_rotation("right")


def slerp_rotation(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    turning = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((start, end))))
    return turning(fraction).as_matrix()


def flip_failures(
    cup: CupPose, top_cup: CupPose, triangle: TriangleConfig
) -> tuple[str, ...]:
    """List what's wrong with the fourth cup. An empty list means success."""
    problems: list[str] = []
    expected = top_cup.position[2] + triangle.cup_height
    if abs(cup.position[2] - expected) > RESTING_TOLERANCE:
        problems.append("fourth cup isn't resting on the top cup")
    offset = np.array(cup.position[:2]) - np.array(top_cup.position[:2])
    if float(np.linalg.norm(offset)) > CENTRING_TOLERANCE:
        problems.append("fourth cup isn't centred on the top cup")
    angle = float(np.degrees(np.arccos(np.clip(-cup.up[2], -1.0, 1.0))))
    if angle > MAX_FLIP_ANGLE_DEGREES:
        problems.append("top cup isn't flipped")
    return tuple(problems)


def grip_from_side(
    env: RobotEnv, config: SceneConfig, flip: FlipConfig, name: str
) -> bool:
    """Grip a cup with the palm beside it, and lift it. Returns whether it's held."""
    table = config.table.top_height
    side = hand_rotation(flip.start_roll, flip.hand_yaw)
    x, y, start_z = env.object_position(name)
    palm_direction = side[:2, 1]
    # Come down beside the cup, as going straight in would clip its wide base with the
    # palm. Then slide in, with the fingertips reaching past the cup's centre.
    bx, by = np.array([x, y]) - flip.approach_clearance * palm_direction
    x, y = np.array([x, y]) + flip.grip_depth * palm_direction
    hover = table + flip.hover_height
    grip = table + flip.grip_height
    env.set_fingers(flip.open_pose)
    env.move_hand(
        np.array([bx, by, hover]), rotation=side, twist_speed=flip.twist_speed
    )
    env.move_hand(np.array([bx, by, grip]), speed=SLOW_SPEED)
    env.move_hand(np.array([x, y, grip]), speed=SLOW_SPEED)
    env.set_fingers(flip.close_pose)
    env.advance(env.motion.settle_time)
    env.move_hand(np.array([x, y, hover]), speed=SLOW_SPEED)
    position = env.object_position(name)
    lifted = position[2] - start_z >= MIN_LIFT
    close = np.linalg.norm(position - env.hand_position) <= MAX_HOLD_DISTANCE
    return bool(lifted and close)


def is_holding(env: RobotEnv, name: str) -> bool:
    distance = np.linalg.norm(env.object_position(name) - env.hand_position)
    return bool(distance <= MAX_HOLD_DISTANCE)


def twist_over(env: RobotEnv, flip: FlipConfig, name: str) -> None:
    """Move to the twist pose, then turn the cup over by rotating only the wrist.

    In this pose the forearm is level and the other wrist joints are zero, so turning the
    wrist turns the cup exactly over. Solving the twist with inverse kinematics instead
    swung the whole arm violently, and threw the cup.
    """
    env.move_joints(flip.twist_pose, flip.pose_seconds)
    env.advance(env.motion.settle_time)
    angle = env.joint_angle(WRIST_ROLL_JOINT)
    env.move_joints({WRIST_ROLL_JOINT: angle + flip.twist_angle}, flip.twist_seconds)
    env.advance(env.motion.settle_time)
    if is_holding(env, name):
        level_cup(env, name)


def level_cup(env: RobotEnv, name: str) -> None:
    """Measure how far the held cup leans, and turn the hand to make it hang straight."""
    for _ in range(LEVEL_ROUNDS):
        if not is_holding(env, name):
            logger.warning("Stopped levelling, as '%s' is no longer held", name)
            return
        rotation = env.data.xmat[env.model.body(name).id].reshape(3, 3)
        correction = rotation_between(rotation[:, 2], np.array([0.0, 0.0, -1.0]))
        hand = env.data.site_xmat[env.hand_site].reshape(3, 3)
        env.move_hand(
            env.hand_position,
            rotation=correction @ hand,
            twist_speed=LEVEL_SPEED,
            keep_waist=True,
            min_seconds=LEVEL_SECONDS,
        )


def rotation_between(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The smallest rotation that turns one direction into another."""
    axis = np.cross(start, end)
    size = float(np.linalg.norm(axis))
    if size < AXIS_EPSILON:
        return np.eye(3)
    angle = np.arctan2(size, float(start @ end))
    return Rotation.from_rotvec(axis / size * angle).as_matrix()


def placement_target(
    top_position: tuple[float, float, float],
    cup_height: float,
    release_drift: tuple[float, float],
) -> np.ndarray:
    """Where the held cup should go: on the measured top cup, aimed against the drift.

    The top cup settles up to 1.5 cm from its planned spot, and the cup shifts a few
    millimetres the same way each time the fingers open.
    """
    x, y, z = top_position
    return np.array([x - release_drift[0], y - release_drift[1], z + cup_height])


def horizontal_error(cup_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """How far the cup must still move sideways to be over the target."""
    error = target - cup_position
    return np.array([error[0], error[1], 0.0])


def correct_aim(
    env: RobotEnv, flip: FlipConfig, name: str, target: np.ndarray
) -> None:
    """Measure the cup just above the tower, and nudge the hand until it's on target."""
    for _ in range(AIM_ROUNDS):
        error = horizontal_error(env.object_position(name), target)
        if float(np.linalg.norm(error)) < AIM_TOLERANCE:
            return
        env.move_hand(
            env.hand_position + error,
            speed=flip.lower_speed,
            keep_waist=True,
            min_seconds=AIM_SECONDS,
        )


def place_flipped(
    env: RobotEnv, flip: FlipConfig, name: str, target: np.ndarray
) -> None:
    """Lower the flipped cup onto the target, correct its aim, let go, and lift clear."""
    level_cup(env, name)
    offset = env.hand_position - env.object_position(name)
    final = target + offset + [0.0, 0.0, flip.place_clearance]
    env.move_hand(final, speed=flip.lower_speed, keep_waist=True)
    env.advance(env.motion.settle_time)
    correct_aim(env, flip, name, target)
    env.relax_fingers()
    env.set_fingers(flip.release_pose)
    env.advance(env.motion.settle_time)
    env.set_fingers(flip.open_pose)
    env.advance(env.motion.settle_time)
    # The flipped cup is widest at the top, so lifting straight up would catch the rim
    # and carry it away. So the hand centres itself to touch nothing, backs out sideways,
    # and only then lifts.
    hand = env.data.site_xmat[env.hand_site].reshape(3, 3)
    palm_direction = hand[:, 1]
    along_hand = hand[:, 0]
    miss = float((env.object_position(name) - env.hand_position) @ along_hand)
    centred = env.hand_position + miss * along_hand
    env.move_hand(centred, speed=flip.lower_speed, keep_waist=True, min_seconds=1.0)
    backed_out = env.hand_position - flip.back_out_distance * palm_direction
    env.move_hand(backed_out, speed=flip.lower_speed, keep_waist=True)
    lift = env.hand_position + [0.0, 0.0, flip.lift_after_release]
    env.move_hand(lift, speed=flip.lower_speed, keep_waist=True)
    env.move_hand(
        np.array(flip.finish_position), speed=flip.lower_speed, keep_waist=True
    )


def build_flipped_tower(
    env: RobotEnv,
    config: SceneConfig,
    triangle: TriangleConfig,
    flip: FlipConfig,
    order: tuple[str, str, str],
    flipped_name: str,
    build_base: Callable[..., TriangleResult] = build_triangle,
) -> FlipResult:
    """Build the triangle, then flip the fourth cup and stand it on top."""
    table = config.table.top_height
    built = build_base(env, config, triangle, order)
    if not built.success:
        return FlipResult(built, False, ("triangle failed",))
    held = grip_from_side(env, config, flip, flipped_name)
    if not held:
        return FlipResult(built, False, ("couldn't grip the fourth cup",))
    twist_over(env, flip, flipped_name)
    if not is_holding(env, flipped_name):
        logger.warning("Dropped '%s' during the twist", flipped_name)
        return FlipResult(built, True, ("dropped the fourth cup in the twist",))
    top_position = tuple(env.object_position(order[2]))
    target = placement_target(top_position, triangle.cup_height, flip.release_drift)
    place_flipped(env, flip, flipped_name, target)
    env.advance(SETTLE_AFTER_BUILD)
    left, right, top_cup = (cup_pose(env, name) for name in order)
    problems = (
        *triangle_failures(left, right, top_cup, triangle, table),
        *flip_failures(cup_pose(env, flipped_name), top_cup, triangle),
    )
    return FlipResult(built, True, problems)
