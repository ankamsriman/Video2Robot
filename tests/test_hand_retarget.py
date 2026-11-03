"""Turning a tracked hand into targets for the robot's hand, for live control."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.simulation.robot_env import hand_pose_for_width
from src.teleop.hand_retarget import (
    CAMERA_TO_ROBOT,
    Calibration,
    TeleopConfig,
    finger_curls,
    finger_pose,
    hand_frame,
    image_scale,
    load_teleop_config,
    robot_rotation,
    smooth,
    wrist_target,
)

CONFIG = TeleopConfig(
    gain_sideways=0.5,
    gain_height=0.4,
    gain_forward=0.6,
    workspace_min=(0.2, -0.4, 0.76),
    workspace_max=(0.6, 0.4, 1.15),
    smoothing=0.35,
    max_joint_speed=3.0,
    max_tilt=1.2,
    finger_bend_open=0.6,
    finger_bend_closed=3.2,
    thumb_reach_open=1.3,
    thumb_reach_closed=0.6,
    open_width=0.10,
    closed_width=0.02,
    calibration_frames=15,
)
HOME = np.array([0.30, -0.10, 0.98])


def _hand(curl: float = 0.0, thumb_tucked: bool = False) -> np.ndarray:
    """A made-up right hand in its own frame: x across the palm, y along the fingers."""
    points = np.zeros((21, 3))
    bases = {5: (0.03, 0.09), 9: (0.0, 0.095), 13: (-0.02, 0.09), 17: (-0.04, 0.08)}
    for base, (x, y) in bases.items():
        points[base] = (x, y, 0.0)
        tip = points[base].copy()
        angle = 0.0
        for joint, length in enumerate((0.04, 0.025, 0.02)):
            angle += curl * 1.3
            tip = tip + length * np.array([0.0, np.cos(angle), -np.sin(angle)])
            points[base + 1 + joint] = tip
    points[1] = (0.02, 0.02, 0.0)
    points[2] = (0.045, 0.04, 0.0)
    points[3] = (0.07, 0.05, 0.0)
    points[4] = (0.0, 0.08, -0.02) if thumb_tucked else (0.095, 0.06, 0.0)
    return points - points.mean(axis=0)


def _project(world: np.ndarray, focal: float, depth: float) -> np.ndarray:
    """A camera's view of the hand, in pixels, ignoring depth within the hand."""
    return focal * world[:, :2] / depth + np.array([320.0, 240.0])


def test_hand_frame_is_a_rotation() -> None:
    frame = hand_frame(_hand(0.3))

    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-9)
    assert np.linalg.det(frame) == pytest.approx(1.0)


def test_hand_frame_turns_with_the_hand() -> None:
    turn = Rotation.from_euler("xyz", (0.4, -0.3, 0.7)).as_matrix()
    hand = _hand()

    assert np.allclose(hand_frame(hand @ turn.T), turn @ hand_frame(hand), atol=1e-9)


def test_camera_to_robot_is_a_rotation() -> None:
    assert np.linalg.det(CAMERA_TO_ROBOT) == pytest.approx(1.0)


def test_hand_looks_bigger_when_closer_whatever_way_it_faces() -> None:
    turned = _hand() @ Rotation.from_euler("xy", (0.6, 0.5)).as_matrix().T

    far = image_scale(_project(turned, 600.0, 0.5), turned)
    near = image_scale(_project(turned, 600.0, 0.35), turned)

    assert far == pytest.approx(600.0 / 0.5)
    assert near / far == pytest.approx(0.5 / 0.35)


def _calibration() -> Calibration:
    return Calibration(
        wrist=np.array([0.5, 0.5]), scale=100.0, frame=np.eye(3), position=HOME
    )


def test_moving_your_hand_to_your_left_moves_the_robot_hand_to_its_left() -> None:
    # The camera faces you, so your left is the image's right.
    target = wrist_target(np.array([0.6, 0.5]), 100.0, _calibration(), CONFIG)

    assert target[1] == pytest.approx(HOME[1] + 0.05)
    assert target[[0, 2]] == pytest.approx(HOME[[0, 2]])


def test_raising_your_hand_raises_the_robot_hand() -> None:
    target = wrist_target(np.array([0.5, 0.4]), 100.0, _calibration(), CONFIG)

    assert target[2] == pytest.approx(HOME[2] + 0.04)


def test_moving_toward_the_camera_reaches_forward() -> None:
    target = wrist_target(np.array([0.5, 0.5]), 125.0, _calibration(), CONFIG)

    assert target[0] == pytest.approx(HOME[0] + 0.6 * (1 - 100 / 125))


def test_the_target_stays_inside_the_workspace() -> None:
    target = wrist_target(np.array([0.5, 2.0]), 100.0, _calibration(), CONFIG)

    assert target[2] == pytest.approx(CONFIG.workspace_min[2])


def test_the_robot_keeps_its_neutral_wrist_while_your_hand_is_still() -> None:
    neutral = Rotation.from_euler("y", 1.2).as_matrix()

    assert np.allclose(robot_rotation(np.eye(3), np.eye(3), neutral, CONFIG), neutral)


def test_turning_your_hand_to_your_left_turns_the_robot_hand_left() -> None:
    # Seen from above, turning left is anticlockwise about "up", which is -y for the camera.
    turned = Rotation.from_rotvec((0.0, -0.3, 0.0)).as_matrix()

    result = robot_rotation(np.eye(3), turned, np.eye(3), CONFIG)

    assert Rotation.from_matrix(result).as_rotvec() == pytest.approx((0.0, 0.0, 0.3))


def test_a_large_wrist_turn_is_capped() -> None:
    turned = Rotation.from_rotvec((0.0, -2.5, 0.0)).as_matrix()

    result = robot_rotation(np.eye(3), turned, np.eye(3), CONFIG)

    assert np.linalg.norm(Rotation.from_matrix(result).as_rotvec()) == pytest.approx(
        1.2
    )


def test_an_open_hand_has_straight_fingers() -> None:
    curls = finger_curls(_hand(0.0), CONFIG)

    assert curls == pytest.approx((0.0, 0.0, 0.0), abs=0.05)


def test_a_fist_has_curled_fingers() -> None:
    thumb, index, middle = finger_curls(_hand(1.0, thumb_tucked=True), CONFIG)

    assert thumb > 0.9
    assert index > 0.9
    assert middle > 0.9


def test_finger_pose_goes_from_open_to_closed() -> None:
    assert finger_pose((0.0, 0.0, 0.0), CONFIG) == pytest.approx(
        hand_pose_for_width(0.10)
    )
    assert finger_pose((1.0, 1.0, 1.0), CONFIG) == pytest.approx(
        hand_pose_for_width(0.02)
    )


def test_each_robot_finger_follows_its_own_finger() -> None:
    open_pose = hand_pose_for_width(0.10)
    closed_pose = hand_pose_for_width(0.02)

    pose = finger_pose((0.0, 1.0, 0.0), CONFIG)

    assert pose[3:5] == pytest.approx(closed_pose[3:5])
    assert pose[1:3] == pytest.approx(open_pose[1:3])
    assert pose[5:7] == pytest.approx(open_pose[5:7])


def test_smoothing_moves_part_of_the_way() -> None:
    assert smooth(np.zeros(3), np.ones(3), 0.25) == pytest.approx(np.full(3, 0.25))


def test_settings_load_from_the_glass_config() -> None:
    config = load_teleop_config(Path("configs/cups_real.yaml"))

    assert len(config.workspace_min) == 3
    assert all(
        low < high
        for low, high in zip(config.workspace_min, config.workspace_max, strict=True)
    )
