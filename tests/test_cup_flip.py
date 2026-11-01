"""The flipped cup: geometry of the hand's twist, and what counts as success."""

import numpy as np
import pytest

from src.simulation.cup_flip import (
    flip_failures,
    hand_rotation,
    rotation_between,
    slerp_rotation,
)
from src.verification.triangle import CupPose, TriangleConfig

TRIANGLE = TriangleConfig(
    center=(0.47, -0.13),
    cup_spacing=0.085,
    cup_height=0.09,
    rim_radius=0.04,
    top_radius=0.028,
    position_tolerance=0.02,
    max_tilt_degrees=10.0,
)
TABLE = 0.72
TOP_CUP = CupPose(position=(0.47, -0.13, 0.855), up=(0.0, 0.0, 1.0))
FLIPPED = CupPose(position=(0.47, -0.13, 0.945), up=(0.0, 0.0, -1.0))


def test_a_correctly_flipped_cup_has_no_failures() -> None:
    assert flip_failures(FLIPPED, TOP_CUP, TRIANGLE) == ()


def test_a_cup_on_top_that_was_not_flipped_fails() -> None:
    unflipped = CupPose(FLIPPED.position, (0.0, 0.0, 1.0))

    assert "top cup isn't flipped" in flip_failures(unflipped, TOP_CUP, TRIANGLE)


def test_a_flipped_cup_that_fell_off_fails() -> None:
    fallen = CupPose((0.60, -0.13, 0.765), (0.0, 0.0, -1.0))

    failures = flip_failures(fallen, TOP_CUP, TRIANGLE)

    assert "fourth cup isn't resting on the top cup" in failures
    assert "fourth cup isn't centred on the top cup" in failures


def test_a_badly_tilted_flipped_cup_fails() -> None:
    tilt = np.radians(40)
    tipped = CupPose(FLIPPED.position, (float(np.sin(tilt)), 0.0, -float(np.cos(tilt))))

    assert "top cup isn't flipped" in flip_failures(tipped, TOP_CUP, TRIANGLE)


def test_side_grip_puts_the_palm_beside_the_cup() -> None:
    rotation = hand_rotation(np.pi / 2)

    assert rotation[:, 0] == pytest.approx([1, 0, 0], abs=1e-9)
    assert rotation[:, 1] == pytest.approx([0, 1, 0], abs=1e-9)


def test_after_the_twist_the_palm_is_on_the_other_side() -> None:
    assert hand_rotation(-np.pi / 2)[:, 1] == pytest.approx([0, -1, 0], abs=1e-9)


def test_halfway_through_the_twist_the_palm_faces_down() -> None:
    start, end = hand_rotation(np.pi / 2), hand_rotation(-np.pi / 2)
    middle = hand_rotation(0.0)

    assert slerp_rotation(start, end, 0.5) == pytest.approx(middle, abs=1e-9)


def test_rotation_between_turns_one_direction_into_another() -> None:
    start = np.array([0.2, -0.1, -0.97])
    start /= np.linalg.norm(start)

    turned = rotation_between(start, np.array([0.0, 0.0, -1.0])) @ start

    assert turned == pytest.approx([0, 0, -1], abs=1e-9)


def test_rotation_between_identical_directions_does_nothing() -> None:
    same = np.array([0.0, 0.0, 1.0])

    assert rotation_between(same, same) == pytest.approx(np.eye(3))


@pytest.mark.integration
def test_g1_flips_the_fourth_cup_onto_the_triangle() -> None:
    from scripts.flip_trial import run_trial

    result = run_trial((101, False))

    assert result["success"], result
