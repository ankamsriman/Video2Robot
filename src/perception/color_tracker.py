"""Follow objects by colour.

For each item, it reports the middle of the bottom edge of its coloured area, because
that's the part touching the table, which is what the marker mapping needs.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

CLEANUP_KERNEL = np.ones((3, 3), np.uint8)
GAP_KERNEL = np.ones((7, 7), np.uint8)
HUE_MAX = 180
SATURATION_VALUE_MAX = 255


@dataclass(frozen=True)
class ColorSpec:
    """A colour range. A hue range whose start exceeds its end wraps round, as red does."""

    name: str
    hue: tuple[int, int]
    min_saturation: int
    min_value: int


def _mask(hsv: np.ndarray, spec: ColorSpec) -> np.ndarray:
    low, high = spec.hue
    ranges = [(low, high)] if low <= high else [(low, HUE_MAX), (0, high)]
    combined = np.zeros(hsv.shape[:2], np.uint8)
    for start, end in ranges:
        lower = np.array([start, spec.min_saturation, spec.min_value], np.uint8)
        upper = np.array([end, SATURATION_VALUE_MAX, SATURATION_VALUE_MAX], np.uint8)
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
    # Remove specks, then join parts of one object split by noise or glare.
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, CLEANUP_KERNEL)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, GAP_KERNEL)


def _largest_blob(mask: np.ndarray) -> tuple[np.ndarray, int] | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best, int(stats[best, cv2.CC_STAT_AREA])


def _bottom_middle(blob: np.ndarray) -> tuple[float, float]:
    # Using each column's lowest pixel, not only the lowest row, because a slightly tilted
    # edge has one corner as its lowest row, which is half the object's width off-centre.
    columns = np.nonzero(blob.any(axis=0))[0]
    from_bottom = np.argmax(blob[::-1, columns], axis=0)
    bottoms = blob.shape[0] - 1 - from_bottom
    return (float(columns.mean()), float(np.median(bottoms)))


def track_colors(
    frames: Iterable[np.ndarray],
    specs: list[ColorSpec],
    min_area: int,
    area_fraction: float,
) -> dict[str, np.ndarray]:
    """Return each item's bottom-middle pixel per frame.

    A frame gives NaN when the item is missing, or when too much of it is hidden compared
    with its largest size so far, because a partly hidden edge would mislead the position.
    """
    points: dict[str, list[tuple[float, float]]] = {spec.name: [] for spec in specs}
    largest: dict[str, int] = {spec.name: 0 for spec in specs}
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for spec in specs:
            found = _largest_blob(_mask(hsv, spec))
            usable = found is not None and found[1] >= min_area
            if usable:
                largest[spec.name] = max(largest[spec.name], found[1])
            visible = usable and found[1] >= area_fraction * largest[spec.name]
            points[spec.name].append(
                _bottom_middle(found[0]) if visible else (np.nan, np.nan)
            )
    return {name: np.array(track, dtype=float) for name, track in points.items()}
