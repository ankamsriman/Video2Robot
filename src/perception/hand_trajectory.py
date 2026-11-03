"""Turn the path of the person's hand in a video into a path for the robot's hand.

One camera can't give depth from a pixel alone. Each carry's start and end are known, so the
hand is assumed to travel in the upright plane between them, and the camera's line through
the hand's pixel meets that plane at one point. So the sideways path isn't copied: only how
high the hand went, and how its speed changed along the way. This is an estimate, with no
ground truth to check it against.
"""

from dataclasses import dataclass

import numpy as np

from src.keyframe_selector.speed_analyzer import fill_gaps
from src.perception.table_view import TableView

SMOOTHING_FRAMES = 5


@dataclass(frozen=True)
class CarryLimits:
    """What keeps a copied path safe, in metres, seconds and fractions of the carry."""

    min_height: float
    max_height: float
    max_speed: float
    blend: float


@dataclass(frozen=True)
class CarryProfile:
    """How far along the carry the hand was, and how high above the table, per video frame."""

    along: np.ndarray
    height: np.ndarray
    seconds: float


@dataclass(frozen=True)
class RobotCarry:
    points: np.ndarray
    seconds: float
    limited_fraction: float


def point_on_carry_plane(
    view: TableView, pixel: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[float, float]:
    """Where a pixel's line meets the upright plane from start to end: (along, height)."""
    ray = view.rotation.T @ (np.linalg.inv(view.lens) @ np.array([pixel[0], pixel[1], 1.0]))
    direction = end - start
    normal = np.array([-direction[1], direction[0]])
    denominator = float(normal @ ray[:2])
    if abs(denominator) < 1e-9:
        raise ValueError("The camera is looking along this carry, so depth can't be found")
    distance = float(normal @ (start - view.camera[:2])) / denominator
    point = view.camera + distance * ray
    along = float((point[:2] - start) @ direction) / float(direction @ direction)
    side = 1.0 if view.camera[2] > 0 else -1.0
    return along, float(side * point[2])


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < SMOOTHING_FRAMES:
        return values
    pad = SMOOTHING_FRAMES // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(SMOOTHING_FRAMES) / SMOOTHING_FRAMES, mode="valid")


def profile_from_pixels(
    pixels: np.ndarray, fps: float, view: TableView, start: np.ndarray, end: np.ndarray
) -> CarryProfile:
    """Turn a hand's pixels during one carry into progress and height over time."""
    points = np.array([point_on_carry_plane(view, p, start, end) for p in pixels])
    along = np.maximum.accumulate(_smooth(points[:, 0]))
    span = along[-1] - along[0]
    along = (along - along[0]) / span if span > 0 else np.linspace(0.0, 1.0, len(along))
    return CarryProfile(along, _smooth(points[:, 1]), (len(pixels) - 1) / fps)


def is_believable(profile: CarryProfile, start_height: float, max_height: float) -> bool:
    """Whether a carry's heights make sense: starting near the glass, never too high.

    A carry straight towards the camera lies almost along the camera's line of sight, so
    its depth can't be judged, and the heights come out wrong.
    """
    return bool(
        profile.height[0] <= start_height and float(profile.height.max()) <= max_height
    )


def robot_carry_path(
    profile: CarryProfile,
    hand_start: np.ndarray,
    hand_end: np.ndarray,
    table_height: float,
    limits: CarryLimits,
) -> RobotCarry:
    """Fit a profile between the robot's hand positions, within what's safe for it.

    The ends blend into the robot's own start and end heights, so the grip and placing steps
    are unchanged. The result reports how much of the hand's height had to be limited.
    """
    wanted = profile.height
    limited = np.clip(wanted, limits.min_height, limits.max_height)
    along = profile.along
    edges = np.minimum(along, 1.0 - along) / limits.blend
    weight = np.clip(edges, 0.0, 1.0)
    # The robot's own heights take over at the ends, so only the middle counts as limited.
    middle = weight >= 1.0
    changed = np.abs(limited - wanted) > 1e-9
    was_limited = float(np.mean(changed[middle])) if middle.any() else 1.0
    weight = weight * weight * (3 - 2 * weight)
    flat = hand_start[2] + along * (hand_end[2] - hand_start[2])
    heights = (1 - weight) * flat + weight * (table_height + limited)
    sideways = hand_start[:2] + np.outer(along, hand_end[:2] - hand_start[:2])
    points = np.column_stack((sideways, heights))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    frame_time = profile.seconds / max(len(points) - 1, 1)
    fastest = float(steps.max()) / frame_time if len(steps) and frame_time > 0 else 0.0
    slow_down = max(fastest / limits.max_speed, 1.0)
    return RobotCarry(points, profile.seconds * slow_down, was_limited)


def follow_moving_hand(
    hands: np.ndarray, begin: np.ndarray, max_jump: float
) -> np.ndarray:
    """Follow the hand nearest the last known spot, from frame to frame.

    `hands` holds two pixel positions per frame, NaN where a hand wasn't seen. The biggest
    hand in the picture isn't always the moving one.
    """
    path = np.full((len(hands), 2), np.nan)
    last = begin
    for frame, candidates in enumerate(hands):
        gaps = np.linalg.norm(candidates - last, axis=1)
        if np.all(np.isnan(gaps)):
            continue
        nearest = int(np.nanargmin(gaps))
        if gaps[nearest] <= max_jump:
            path[frame] = candidates[nearest]
            last = candidates[nearest]
    return fill_gaps(path)


def carry_window(
    path: np.ndarray,
    start_pixel: np.ndarray,
    end_pixel: np.ndarray,
    settle_pixels: float = 0.0,
) -> tuple[int, int]:
    """The frames where the hand is closest to the start spot, then last near the end spot.

    A hand can pass close to the end spot on its way there, so the carry ends at the last
    frame within `settle_pixels` of the closest it ever came.
    """
    first = int(np.argmin(np.linalg.norm(path - start_pixel, axis=1)))
    gaps = np.linalg.norm(path[first:] - end_pixel, axis=1)
    last = first + int(np.nonzero(gaps <= gaps.min() + settle_pixels)[0][-1])
    if last <= first:
        raise ValueError("The hand never moved from the start spot towards the end spot")
    return first, last
