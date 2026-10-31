"""Motion helpers and an end-to-end physics check of a single pick and place."""

import numpy as np
import pytest

from src.schemas.plan import Plan, Relation, Step
from src.simulation.ik import palm_down_rotation, waist_yaw_for_target
from src.simulation.primitives import execute_plan, final_relations, pick
from src.simulation.robot_env import (
    RobotEnv,
    hand_pose_for_width,
    load_motion_config,
    min_jerk,
)
from src.simulation.scene import build_model, load_scene_config


def test_min_jerk_starts_at_zero_and_ends_at_one() -> None:
    assert min_jerk(0.0) == 0.0
    assert min_jerk(1.0) == 1.0
    assert min_jerk(0.5) == pytest.approx(0.5)


def test_min_jerk_clamps_values_outside_the_range() -> None:
    assert min_jerk(-1.0) == 0.0
    assert min_jerk(2.0) == 1.0


@pytest.mark.integration
def test_g1_puts_an_object_in_a_container_using_real_physics() -> None:
    config = load_scene_config()
    env = RobotEnv(build_model(config), load_motion_config())
    plan = Plan(steps=[Step(object="apple", reference="box", relation=Relation.IN)])

    result = execute_plan(env, config, plan)

    assert result.success, result
    assert result.steps[0].measured_relation is Relation.IN


@pytest.mark.integration
def test_final_check_notices_an_object_moved_after_its_step() -> None:
    config = load_scene_config()
    env = RobotEnv(build_model(config), load_motion_config())
    plan = Plan(steps=[Step(object="apple", reference="box", relation=Relation.IN)])
    execute_plan(env, config, plan)

    free_joint = env.model.joint("apple_free").qposadr[0]
    env.data.qpos[free_joint + 1] += 0.30
    env.advance(0.2)

    assert final_relations(env, config, plan)[0] is not Relation.IN


@pytest.mark.integration
def test_first_hand_motion_is_smooth_and_leaves_other_objects_alone() -> None:
    config = load_scene_config()
    env = RobotEnv(build_model(config), load_motion_config())
    lemon_before = env.object_position("lemon")
    lime_before = env.object_position("lime")
    apple = env.object_position("apple")
    pick_height = config.table.top_height + env.motion.hover_height

    env.move_hand(np.array([apple[0], apple[1], pick_height]))

    assert np.linalg.norm(env.object_position("lemon") - lemon_before) < 0.005
    assert np.linalg.norm(env.object_position("lime") - lime_before) < 0.005


def test_torso_faces_straight_ahead_for_a_target_in_front_of_the_right_shoulder() -> (
    None
):
    assert waist_yaw_for_target(0.30, -0.10, "right") == pytest.approx(0.0, abs=1e-6)


def test_torso_turns_toward_targets_on_either_side() -> None:
    assert waist_yaw_for_target(0.30, 0.10, "right") > 0.3
    assert waist_yaw_for_target(0.30, -0.30, "right") < -0.3


@pytest.mark.integration
def test_move_hand_reports_failure_when_the_table_physically_blocks_the_hand() -> None:
    config = load_scene_config()
    env = RobotEnv(build_model(config), load_motion_config())
    below_table = np.array([0.30, -0.10, config.table.top_height - 0.10])

    assert env.move_hand(below_table) is False


def test_hand_pose_matches_a_calibrated_width_exactly() -> None:
    assert hand_pose_for_width(0.05) == pytest.approx(
        (0.0, 0.25, -1.45, 0.85, 0.70, 0.85, 0.70)
    )


def test_hand_pose_interpolates_between_calibrated_widths() -> None:
    narrow = hand_pose_for_width(0.05)
    wide = hand_pose_for_width(0.07)

    middle = hand_pose_for_width(0.06)

    assert middle[1] == pytest.approx((narrow[1] + wide[1]) / 2)


def test_hand_pose_stays_within_calibrated_range_for_extreme_widths() -> None:
    assert hand_pose_for_width(0.5) == hand_pose_for_width(0.10)
    assert hand_pose_for_width(0.0) == hand_pose_for_width(0.02)


def test_index_and_middle_fingers_move_together() -> None:
    pose = hand_pose_for_width(0.06)

    assert pose[3:5] == pose[5:7]


@pytest.mark.integration
def test_gripping_keeps_the_object_nearly_upright() -> None:
    config = load_scene_config()
    env = RobotEnv(build_model(config), load_motion_config())

    assert pick(env, config, "apple")

    rotation = env.data.xmat[env.model.body("apple").id].reshape(3, 3)
    assert np.degrees(np.arccos(rotation[2, 2])) < 8.0


def test_hand_pitch_tips_the_fingers_downward_without_distorting_the_rotation() -> None:
    rotation = palm_down_rotation("right", pitch=0.2)

    assert rotation[2, 0] == pytest.approx(-np.sin(0.2))
    assert rotation.T @ rotation == pytest.approx(np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.integration
def test_free_standing_robot_balances_on_its_feet_while_idle() -> None:
    """Far reaches do topple it, so results use a pinned pelvis (see PLAN.md, Phase 1)."""
    config = load_scene_config()
    env = RobotEnv(build_model(config, pinned=False), load_motion_config())

    env.advance(3.0)

    assert env.data.qpos[2] > 0.75
