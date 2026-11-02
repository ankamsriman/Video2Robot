"""Using the left arm: every right-arm setting becomes its mirror image."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.simulation.cup_flip import flip_for_arm, hand_rotation, load_flip_config
from src.simulation.ik import palm_down_rotation, ready_poses
from src.simulation.mirror import (
    mirror_arm_pose,
    mirror_fingers,
    mirror_joint_targets,
    mirror_point,
    mirror_rotation,
)
from src.simulation.robot_env import RobotEnv, load_motion_config, motion_for_arm
from src.simulation.scene import build_model, load_scene_config
from src.verification.triangle import load_triangle_config, triangle_targets

CONFIG = Path("configs/flip_real.yaml")


def test_only_the_side_to_side_arm_joints_change_sign() -> None:
    assert mirror_arm_pose((1, 2, 3, 4, 5, 6, 7)) == (1, -2, -3, 4, -5, 6, -7)


def test_all_finger_joints_but_the_thumb_base_change_sign() -> None:
    assert mirror_fingers((1, 2, 3, 4, 5, 6, 7)) == (1, -2, -3, -4, -5, -6, -7)


def test_named_joints_move_to_the_other_arm_and_mirror() -> None:
    mirrored = mirror_joint_targets(
        {
            "waist_yaw_joint": 0.4,
            "waist_pitch_joint": -0.1,
            "right_shoulder_pitch_joint": -1.0,
            "right_shoulder_roll_joint": 0.2,
            "right_elbow_joint": 0.1,
            "right_wrist_roll_joint": -1.2,
        }
    )

    assert mirrored == {
        "waist_yaw_joint": -0.4,
        "waist_pitch_joint": -0.1,
        "left_shoulder_pitch_joint": -1.0,
        "left_shoulder_roll_joint": -0.2,
        "left_elbow_joint": 0.1,
        "left_wrist_roll_joint": 1.2,
    }


def test_a_point_mirrors_across_the_robots_centre_line() -> None:
    assert mirror_point((0.3, -0.1, 0.9)) == (0.3, 0.1, 0.9)


def test_the_mirrored_palm_down_hand_is_the_left_palm_down_hand() -> None:
    mirrored = mirror_rotation(palm_down_rotation("right", 0.2))

    assert mirrored == pytest.approx(palm_down_rotation("left", 0.2))


def test_a_mirrored_hand_rotation_is_still_a_proper_rotation() -> None:
    rotation = hand_rotation(1.57, -0.6, "left")

    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation == pytest.approx(mirror_rotation(hand_rotation(1.57, -0.6)))


def test_the_right_arm_settings_are_untouched_by_the_right_arm() -> None:
    flip = load_flip_config(CONFIG)
    motion = load_motion_config(CONFIG)

    assert flip_for_arm(flip, "right") == flip
    assert motion_for_arm(motion, "right") == motion


def test_left_arm_flip_settings_mirror_the_right_ones() -> None:
    flip = load_flip_config(CONFIG)

    left = flip_for_arm(flip, "left")

    assert left.twist_angle == -flip.twist_angle
    assert left.release_drift == (flip.release_drift[0], -flip.release_drift[1])
    assert left.finish_position[1] == -flip.finish_position[1]
    assert "left_wrist_roll_joint" in left.twist_pose


def test_the_two_arms_swap_ready_poses_as_mirror_images() -> None:
    right, left = ready_poses("right"), ready_poses("left")

    assert left["left"] == mirror_arm_pose(right["right"])
    assert left["right"] == mirror_arm_pose(right["left"])


def test_the_first_glass_goes_on_the_side_nearest_the_chosen_arm_last() -> None:
    triangle = load_triangle_config(CONFIG)
    mirrored = dataclasses.replace(triangle, first_on_left=False)

    first_right_arm = triangle_targets(triangle, 0.72)[0]
    first_left_arm = triangle_targets(mirrored, 0.72)[0]

    assert first_right_arm[1] > triangle.center[1]
    assert first_left_arm[1] < triangle.center[1]


def test_the_left_arm_hand_is_a_true_mirror_of_the_right_arm_hand() -> None:
    scene = load_scene_config(CONFIG)
    motion = load_motion_config(CONFIG)

    right = RobotEnv(build_model(scene), motion)
    left = RobotEnv(build_model(scene), motion_for_arm(motion, "left"))

    assert left.hand_position == pytest.approx(
        mirror_point(tuple(right.hand_position)), abs=0.01
    )
