"""Turn a hand seen by a webcam into targets for the robot's hand, for live control.

MediaPipe gives 21 points on the hand twice: in the image, and in metres around the hand's
centre with the camera's axes (x right, y down, z away from the camera). From these:

- Where the robot's hand goes comes from where your wrist is in the image, and from how
  big your hand looks, which grows as it comes closer. One camera gives no true depth, so
  forward and back is the roughest of the three directions.
- How the robot's wrist turns copies how your hand has turned since calibration.
- How far each robot finger closes copies how bent your thumb, index and middle finger are.

You face the camera, and the robot is seen from behind, so it moves the way you do.
"""

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from src.simulation.robot_env import hand_pose_for_width

WRIST = 0
INDEX_BASE = 5
MIDDLE_BASE = 9
PINKY_BASE = 17
THUMB_TIP = 4
# Facing the camera: robot forward is toward the camera (-z), robot left is the image's
# right (+x, your left), and robot up is the image's up (-y). Rows are robot x, y, z.
CAMERA_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass(frozen=True)
class TeleopConfig:
    """How hand motion maps to robot motion. Distances are metres, angles radians."""

    gain_sideways: float
    gain_height: float
    gain_forward: float
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    smoothing: float
    max_joint_speed: float
    max_tilt: float
    finger_bend_open: float
    finger_bend_closed: float
    thumb_reach_open: float
    thumb_reach_closed: float
    open_width: float
    closed_width: float
    calibration_frames: int


@dataclass(frozen=True)
class Calibration:
    """Your hand's pose that matches the robot hand's starting pose."""

    wrist: np.ndarray
    scale: float
    frame: np.ndarray
    position: np.ndarray


def load_teleop_config(path: Path) -> TeleopConfig:
    raw = yaml.safe_load(path.read_text())["teleop"]
    return TeleopConfig(
        **{
            **raw,
            "workspace_min": tuple(raw["workspace_min"]),
            "workspace_max": tuple(raw["workspace_max"]),
        }
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def hand_frame(world: np.ndarray) -> np.ndarray:
    """The hand's orientation as a rotation: along the fingers, across the palm, out of it."""
    along = _unit(world[MIDDLE_BASE] - world[WRIST])
    across = world[INDEX_BASE] - world[PINKY_BASE]
    across = _unit(across - (across @ along) * along)
    return np.column_stack((along, across, np.cross(along, across)))


def image_scale(image_pixels: np.ndarray, world: np.ndarray) -> float:
    """Pixels per metre of hand: it grows as the hand nears the camera, whichever way it faces.

    The world points use the camera's axes, so their x and y spread is the hand's true
    outline, and comparing it with the outline in pixels gives the scale.
    """
    pixels = image_pixels - image_pixels.mean(axis=0)
    metres = world[:, :2] - world[:, :2].mean(axis=0)
    return float(np.sqrt(np.sum(pixels**2) / np.sum(metres**2)))


def wrist_target(
    wrist: np.ndarray, scale: float, calibration: Calibration, config: TeleopConfig
) -> np.ndarray:
    """Where the robot's hand should be, from the wrist in image fractions and the hand's scale."""
    moved = wrist - calibration.wrist
    forward = config.gain_forward * (1.0 - calibration.scale / scale)
    offset = np.array(
        (forward, config.gain_sideways * moved[0], -config.gain_height * moved[1])
    )
    return np.clip(
        calibration.position + offset, config.workspace_min, config.workspace_max
    )


def robot_rotation(
    start_frame: np.ndarray,
    frame: np.ndarray,
    neutral: np.ndarray,
    config: TeleopConfig,
) -> np.ndarray:
    """Turn the robot hand from its neutral pose by however your hand has turned, capped."""
    turned = CAMERA_TO_ROBOT @ (frame @ start_frame.T) @ CAMERA_TO_ROBOT.T
    turn = Rotation.from_matrix(turned).as_rotvec()
    angle = float(np.linalg.norm(turn))
    if angle > config.max_tilt:
        turn = turn * (config.max_tilt / angle)
    return Rotation.from_rotvec(turn).as_matrix() @ neutral


def _bend(world: np.ndarray, base: int) -> float:
    """Total bend along a finger, in radians: about zero when straight."""
    chain = world[[WRIST, base, base + 1, base + 2, base + 3]]
    bones = [_unit(bone) for bone in np.diff(chain, axis=0)]
    return float(sum(np.arccos(np.clip(a @ b, -1.0, 1.0)) for a, b in pairwise(bones)))


def _fraction(value: float, start: float, end: float) -> float:
    return float(np.clip((value - start) / (end - start), 0.0, 1.0))


def finger_curls(world: np.ndarray, config: TeleopConfig) -> tuple[float, float, float]:
    """How closed your thumb, index and middle finger are, each from 0 (open) to 1 (closed).

    The thumb's closure is how near its tip comes to the middle knuckle, in palm widths.
    """
    palm_width = float(np.linalg.norm(world[INDEX_BASE] - world[PINKY_BASE]))
    reach = float(np.linalg.norm(world[THUMB_TIP] - world[MIDDLE_BASE])) / palm_width
    thumb = _fraction(reach, config.thumb_reach_open, config.thumb_reach_closed)
    index, middle = (
        _fraction(
            _bend(world, base), config.finger_bend_open, config.finger_bend_closed
        )
        for base in (INDEX_BASE, MIDDLE_BASE)
    )
    return thumb, index, middle


def finger_pose(
    curls: tuple[float, float, float], config: TeleopConfig
) -> tuple[float, ...]:
    """Robot finger angles, right hand, with each robot finger following its own curl."""
    thumb, index, middle = (
        hand_pose_for_width(
            config.open_width + curl * (config.closed_width - config.open_width)
        )
        for curl in curls
    )
    return (0.0, thumb[1], thumb[2], index[3], index[4], middle[5], middle[6])


def smooth(previous: np.ndarray, new: np.ndarray, weight: float) -> np.ndarray:
    """Move part of the way toward a new reading, which steadies a jittery tracker."""
    return previous + weight * (new - previous)
