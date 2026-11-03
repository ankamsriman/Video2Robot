"""Add starting glasses the bright-top reader missed, using points from a vision model.

A model points to every glass in an early frame. Any it finds that the reader didn't becomes
a starting spot. When that glass left is measured without the model: a small patch of the
picture at its spot keeps looking like the first frame until the glass is taken away.
"""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from src.perception.glass_tracker import Spot

Point = tuple[float, float]


def missing_points(
    points: Sequence[Point],
    spots: Sequence[Spot],
    fps: float,
    early_seconds: float,
    near_pixels: float,
) -> list[Point]:
    """The model's points with no spot from the reader near them at any time.

    A glass the reader only noticed late was still found, so it isn't missing. See
    `seen_from_start` for that case.
    """
    return [
        point
        for point in points
        if all(np.hypot(point[0] - s.x, point[1] - s.y) > near_pixels for s in spots)
    ]


def seen_from_start(
    points: Sequence[Point],
    spots: Sequence[Spot],
    fps: float,
    early_seconds: float,
    near_pixels: float,
) -> tuple[Spot, ...]:
    """Spots the reader noticed late, at a place the model saw a glass from the start.

    That glass was there all along, so its spot starts at the beginning.
    """
    updated = []
    for spot in spots:
        late = spot.first / fps > early_seconds
        at_point = any(np.hypot(p[0] - spot.x, p[1] - spot.y) <= near_pixels for p in points)
        updated.append(replace(spot, first=0) if late and at_point else spot)
    return tuple(updated)


def departure_frame(
    change: np.ndarray, threshold: float, fps: float, settle_seconds: float
) -> int | None:
    """The last frame where the patch still matched, if it then stays changed.

    A hand passing in front changes the patch only briefly, so a change counts only if it
    lasts at least `settle_seconds` to the end of the video, or until it matches again. The
    changes are smoothed over half a second first, so one noisy frame can't count as a match.
    """
    window = max(int(fps / 2), 1)
    if len(change) >= window:
        padded = np.pad(change, window // 2, mode="edge")
        change = np.convolve(padded, np.ones(window) / window, mode="valid")[: len(change)]
    matched = np.nonzero(change <= threshold)[0]
    if len(matched) == 0:
        return None
    last = int(matched[-1])
    if len(change) - 1 - last < settle_seconds * fps:
        return None
    return last


def rescued_spot(point: Point, departure: int) -> Spot:
    return Spot(float(point[0]), float(point[1]), 0, departure, departure + 1)


def patch_change(
    frames_gray: np.ndarray, point: Point, half: int, reference: np.ndarray
) -> float:
    """Mean difference between a small patch now and the same patch at the start."""
    x, y = int(point[0]), int(point[1])
    patch = frames_gray[max(y - half, 0) : y + half, max(x - half, 0) : x + half]
    return float(np.mean(np.abs(patch.astype(float) - reference)))
