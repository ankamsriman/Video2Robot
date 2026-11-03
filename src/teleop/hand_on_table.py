"""Place a tracked palm on the real table, so the robot's hand can go to the same spot.

The table's corners and measured size say where the camera is (`TableView`). A pixel then
fixes a line from the camera through the palm, and how big the hand looks says how far along
that line it is: MediaPipe gives the hand's size in metres, and the picture gives it in
pixels. MediaPipe assumes a usual hand size, so one factor per video corrects the distance.
It is fitted on where the glasses stood when the hand took them, which uses the video only.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.perception.table_view import TableView
from src.teleop.hand_retarget import image_scale

PALM_POINTS = (0, 5, 9, 17)
DEPTH_FACTORS = tuple(np.round(np.arange(0.70, 1.41, 0.01), 2))


@dataclass(frozen=True)
class PalmRay:
    """The camera's position, and the step to the palm at the distance the hand's size gives."""

    origin: np.ndarray
    reach: np.ndarray


def palm_ray(
    view: TableView,
    pixels: np.ndarray,
    world: np.ndarray,
    palm: Iterable[int] = PALM_POINTS,
) -> PalmRay:
    """Where the palm is, along the line from the camera, from one frame's hand."""
    depth = view.lens[0, 0] / image_scale(pixels, world)
    centre = pixels[list(palm)].mean(axis=0)
    direction = np.linalg.inv(view.lens) @ np.array([*centre, 1.0])
    return PalmRay(view.camera, view.rotation.T @ (depth * direction))


def palm_spot(ray: PalmRay, factor: float, view: TableView) -> np.ndarray:
    """The palm as (forward, left, up) in metres on the table, with the distance corrected."""
    point = ray.origin + factor * ray.reach
    side = 1.0 if view.camera[2] > 0 else -1.0
    return np.array([point[0], point[1], side * point[2]])


def _miss(rays: Sequence[PalmRay], glass: np.ndarray, factor: float, view) -> float:
    spots = np.array([palm_spot(ray, factor, view)[:2] for ray in rays])
    return float(np.linalg.norm(np.median(spots, axis=0) - glass))


def fit_depth_factor(
    rays_at_grabs: Sequence[Sequence[PalmRay]],
    glasses: Sequence[np.ndarray],
    view: TableView,
) -> tuple[float, list[float]]:
    """The distance factor that puts the palm on each glass as it was taken, and the misses.

    `rays_at_grabs` holds, for each glass, the palm in the frames just before it moved. A
    glass with no hand seen then is skipped. The factor minimises the median miss.
    """
    pairs = [
        (rays, glass)
        for rays, glass in zip(rays_at_grabs, glasses, strict=True)
        if rays
    ]
    if not pairs:
        return 1.0, []
    best = min(
        DEPTH_FACTORS,
        key=lambda factor: float(
            np.median([_miss(rays, glass, factor, view) for rays, glass in pairs])
        ),
    )
    return float(best), [_miss(rays, glass, best, view) for rays, glass in pairs]


def table_to_robot(
    spot: np.ndarray, near_edge_x: float, table_width: float, top_height: float
) -> np.ndarray:
    """A (forward, left, up) table position as a simulator position, robot mid-edge."""
    return np.array(
        [spot[0] + near_edge_x, spot[1] - table_width / 2, spot[2] + top_height]
    )
