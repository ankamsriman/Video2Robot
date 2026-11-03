"""Using each arm for the glasses on its own side."""

from pathlib import Path

import numpy as np
import pytest

from src.reasoning.fit_layout import arm_for_each_glass, first_goes_on_left
from src.simulation.robot_env import RobotEnv, load_motion_config, motion_for_arm
from src.simulation.scene import build_model, load_scene_config

CONFIG = Path("configs/cups_real.yaml")


def test_each_glass_uses_the_arm_on_its_own_side() -> None:
    starts = [np.array((0.3, 0.2)), np.array((0.3, -0.2)), np.array((0.4, 0.05))]

    assert arm_for_each_glass(starts) == ("left", "right", "left")


def test_a_glass_on_the_centre_line_uses_the_right_arm() -> None:
    assert arm_for_each_glass([np.array((0.3, 0.0))]) == ("right",)


@pytest.mark.parametrize(
    ("arms", "expected"),
    [
        (("left", "right", "left"), True),
        (("right", "left", "left"), False),
        (("right", "right", "left"), True),
        (("left", "left", "right"), False),
    ],
)
def test_the_lower_glasses_never_make_an_arm_cross_the_tower(
    arms: tuple[str, str, str], expected: bool
) -> None:
    assert first_goes_on_left(arms) is expected


@pytest.mark.integration
def test_the_robot_hands_over_to_its_other_arm() -> None:
    scene = load_scene_config(CONFIG)
    layout = {"cup_a": (0.33, 0.25), "cup_b": (0.33, -0.25), "cup_c": (0.45, 0.25)}
    motion = motion_for_arm(load_motion_config(CONFIG), "right")
    env = RobotEnv(build_model(scene, layout), motion)
    right_hand = env.hand_position

    env.use_arm("left")

    assert env.motion.arm == "left"
    assert env.hand_position[1] > 0
    assert np.linalg.norm(env.hand_position - np.array(env.motion.home_position)) < 0.03
    assert env.move_hand(np.array((0.35, 0.2, 1.0)))
    site = env.model.site("right_grasp").id
    assert env.data.site_xpos[site][2] < right_hand[2]
