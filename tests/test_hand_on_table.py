"""Placing a tracked palm on the real table, so the robot's hand can go to the same spot."""

import numpy as np
import pytest

from src.perception.table_view import table_view
from src.teleop.hand_on_table import (
    PalmRay,
    fit_depth_factor,
    palm_ray,
    palm_spot,
    table_to_robot,
)

WIDTH, DEPTH = 0.80, 0.50
# A camera like the user's phone: across the table, looking down at it.
CORNERS = np.array([[420.0, 980.0], [1500.0, 980.0], [1330.0, 640.0], [590.0, 640.0]])
VIEW = table_view(CORNERS, WIDTH, DEPTH, (1920, 1080))


def _hand_at(
    point: np.ndarray, hand_size: float = 0.09
) -> tuple[np.ndarray, np.ndarray]:
    """A flat made-up hand facing the camera at a table position, seen in pixels and metres."""
    offsets = np.array(
        [[0.0, 0.0], [0.03, 0.04], [0.035, 0.08], [0.0, 0.09], [-0.035, 0.08]]
    ) * (hand_size / 0.09)
    offsets = offsets - offsets.mean(axis=0)
    side = 1.0 if VIEW.camera[2] > 0 else -1.0
    toward_camera = VIEW.camera - np.array([point[0], point[1], side * point[2]])
    toward_camera /= np.linalg.norm(toward_camera)
    across = np.cross(toward_camera, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)
    up = np.cross(across, toward_camera)
    centre = np.array([point[0], point[1], side * point[2]])
    points = [centre + a * across + b * up for a, b in offsets]
    pixels = np.array([VIEW.pixel(p[:2], side * p[2]) for p in points])
    camera_axes = VIEW.rotation
    world = np.array([camera_axes @ (p - centre) for p in points])
    return pixels, world


def test_a_palm_is_placed_where_it_really_is() -> None:
    truth = np.array([0.25, 0.30, 0.12])
    pixels, world = _hand_at(truth)

    spot = palm_spot(palm_ray(VIEW, pixels, world, range(5)), 1.0, VIEW)

    assert spot == pytest.approx(truth, abs=0.01)


def test_a_bigger_hand_than_assumed_needs_a_factor_above_one() -> None:
    truth = np.array([0.25, 0.30, 0.12])
    pixels, world = _hand_at(truth, hand_size=0.10)
    # The tracker reports every hand at the usual size, so this one seems nearer.
    world = world * 0.09 / 0.10

    near = palm_spot(palm_ray(VIEW, pixels, world, range(5)), 1.0, VIEW)
    fixed = palm_spot(palm_ray(VIEW, pixels, world, range(5)), 0.10 / 0.09, VIEW)

    assert np.linalg.norm(near - truth) > 0.05
    assert fixed == pytest.approx(truth, abs=0.01)


def test_the_factor_is_fitted_from_where_the_glasses_stood() -> None:
    glasses = [np.array([0.2, 0.2]), np.array([0.3, 0.5]), np.array([0.35, 0.3])]
    rays = []
    for glass in glasses:
        pixels, world = _hand_at(np.array([*glass, 0.12]), hand_size=0.10)
        rays.append([palm_ray(VIEW, pixels, world * 0.9, range(5))])

    factor, misses = fit_depth_factor(rays, glasses, VIEW)

    assert factor == pytest.approx(1 / 0.9, abs=0.03)
    assert max(misses) < 0.02


def test_a_glass_with_no_hand_seen_is_skipped() -> None:
    glass = np.array([0.2, 0.2])
    pixels, world = _hand_at(np.array([*glass, 0.12]))

    factor, misses = fit_depth_factor(
        [[], [palm_ray(VIEW, pixels, world, range(5))]], [glass, glass], VIEW
    )

    assert factor == pytest.approx(1.0, abs=0.03)
    assert len(misses) == 1


def test_table_positions_become_robot_positions() -> None:
    robot = table_to_robot(np.array([0.2, 0.5, 0.1]), 0.16, 0.8, 0.72)

    assert robot == pytest.approx((0.36, 0.1, 0.82))


def test_a_ray_is_frozen() -> None:
    ray = PalmRay(origin=np.zeros(3), reach=np.ones(3))

    with pytest.raises(AttributeError):
        ray.origin = np.ones(3)  # type: ignore[misc]
