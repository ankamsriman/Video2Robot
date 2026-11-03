"""Turning a hand's path in a video into a carrying path, using a made-up camera."""

import numpy as np

from src.perception.hand_trajectory import (
    CarryLimits,
    carry_window,
    follow_moving_hand,
    is_believable,
    point_on_carry_plane,
    profile_from_pixels,
    robot_carry_path,
)
from src.perception.table_view import table_view

WIDTH, DEPTH = 0.80, 0.50
SIZE = (1620, 1080)
LIMITS = CarryLimits(min_height=0.20, max_height=0.38, max_speed=0.20, blend=0.15)


def _camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lens = np.array([[1900.0, 0, 810], [0, 1900.0, 540], [0, 0, 1]])
    eye = np.array([1.3, 0.44, 0.97])
    forward = (np.array([0.25, 0.40, 0.0]) - eye) / np.linalg.norm(
        np.array([0.25, 0.40, 0.0]) - eye
    )
    right = np.cross(forward, [0.0, 0.0, -1.0])
    right /= np.linalg.norm(right)
    rotation = np.vstack((right, np.cross(forward, right), forward))
    return lens, rotation, -rotation @ eye


def _project(point: tuple[float, float, float]) -> np.ndarray:
    lens, rotation, translation = _camera()
    seen = lens @ (rotation @ np.array(point) + translation)
    return seen[:2] / seen[2]


def _view():
    corners = np.array(
        [_project(p) for p in ((0, 0, 0), (0, WIDTH, 0), (DEPTH, WIDTH, 0), (DEPTH, 0, 0))]
    )
    return table_view(corners, WIDTH, DEPTH, SIZE)


START, END = np.array((0.20, 0.60)), np.array((0.30, 0.35))


def _arc(fraction: float, height: float) -> np.ndarray:
    place = START + fraction * (END - START)
    return _project((place[0], place[1], height))


def test_a_point_on_the_carry_plane_is_recovered_from_its_pixel() -> None:
    along, height = point_on_carry_plane(_view(), _arc(0.4, 0.25), START, END)

    assert abs(along - 0.4) < 0.01
    assert abs(height - 0.25) < 0.005


def test_a_known_arc_becomes_a_rising_and_falling_profile() -> None:
    fractions = np.linspace(0, 1, 31)
    heights = 0.08 + 0.2 * np.sin(np.pi * fractions)
    pixels = np.array([_arc(f, h) for f, h in zip(fractions, heights, strict=True)])

    profile = profile_from_pixels(pixels, 30.0, _view(), START, END)

    assert profile.along[0] == 0.0
    assert profile.along[-1] == 1.0
    assert np.all(np.diff(profile.along) >= 0)
    assert abs(profile.height[15] - 0.28) < 0.02
    assert profile.height[15] > profile.height[1]
    assert abs(profile.seconds - 1.0) < 0.01


def test_a_noisy_backward_step_never_moves_the_hand_backwards() -> None:
    fractions = np.array([0.0, 0.3, 0.2, 0.6, 1.0])
    pixels = np.array([_arc(f, 0.25) for f in fractions])

    profile = profile_from_pixels(pixels, 30.0, _view(), START, END)

    assert np.all(np.diff(profile.along) >= 0)


def _profile(height: float = 0.3):
    from src.perception.hand_trajectory import CarryProfile

    return CarryProfile(np.linspace(0, 1, 21), np.full(21, height), 1.0)


def test_the_robot_path_joins_its_start_and_end_exactly() -> None:
    start, end = np.array((0.3, 0.2, 0.96)), np.array((0.45, -0.05, 0.96))

    path = robot_carry_path(_profile(), start, end, 0.72, LIMITS)

    assert np.allclose(path.points[0], start)
    assert np.allclose(path.points[-1], end)


def test_a_path_too_low_is_raised_to_clear_the_glasses_and_reported() -> None:
    start, end = np.array((0.3, 0.2, 0.96)), np.array((0.45, -0.05, 0.96))

    path = robot_carry_path(_profile(0.05), start, end, 0.72, LIMITS)

    assert path.points[:, 2].min() >= 0.72 + LIMITS.min_height - 1e-9
    assert path.limited_fraction == 1.0


def test_a_path_within_limits_keeps_the_hands_height_in_the_middle() -> None:
    start, end = np.array((0.3, 0.2, 0.96)), np.array((0.45, -0.05, 0.96))

    path = robot_carry_path(_profile(0.32), start, end, 0.72, LIMITS)

    assert abs(path.points[10, 2] - (0.72 + 0.32)) < 1e-6
    assert path.limited_fraction == 0.0


def test_a_fast_hand_is_slowed_to_the_robots_safe_speed() -> None:
    start, end = np.array((0.3, 0.2, 0.96)), np.array((0.45, -0.05, 0.96))

    path = robot_carry_path(_profile(), start, end, 0.72, LIMITS)

    steps = np.linalg.norm(np.diff(path.points, axis=0), axis=1)
    speed = steps.max() / (path.seconds / (len(path.points) - 1))
    assert speed <= LIMITS.max_speed + 1e-6
    assert path.seconds >= 1.0


def test_a_carry_starting_or_going_too_high_is_not_believed() -> None:
    from src.perception.hand_trajectory import CarryProfile

    along = np.linspace(0, 1, 3)

    assert is_believable(CarryProfile(along, np.array((0.07, 0.3, 0.06)), 1.0), 0.15, 0.45)
    assert not is_believable(CarryProfile(along, np.array((0.34, 0.3, 0.06)), 1.0), 0.15, 0.45)
    assert not is_believable(CarryProfile(along, np.array((0.07, 0.6, 0.06)), 1.0), 0.15, 0.45)


def test_the_moving_hand_is_followed_not_the_idle_one() -> None:
    moving = np.column_stack((np.linspace(100, 300, 11), np.full(11, 200.0)))
    idle = np.tile((900.0, 700.0), (11, 1))
    both = np.stack((idle, moving), axis=1)
    both[5, 1] = np.nan

    path = follow_moving_hand(both, np.array((100.0, 200.0)), max_jump=120.0)

    assert np.allclose(path[0], (100, 200))
    assert np.allclose(path[10], (300, 200))
    assert np.allclose(path[5], (200, 200))


def test_the_carry_window_runs_from_the_start_spot_to_the_end_spot() -> None:
    path = np.column_stack((np.linspace(0, 100, 101), np.zeros(101)))

    first, last = carry_window(path, np.array((20.0, 0.0)), np.array((80.0, 0.0)))

    assert (first, last) == (20, 80)


def test_the_carry_ends_where_the_hand_settles_not_where_it_first_passes() -> None:
    xs = np.concatenate((np.linspace(0, 80, 41), np.full(10, 95.0), np.full(10, 82.0)))
    path = np.column_stack((xs, np.zeros(61)))

    _, last = carry_window(path, np.array((0.0, 0.0)), np.array((80.0, 0.0)), 5.0)

    assert last == 60
